"""评测脚本：跑黄金评测集，输出指标报告。

用法：
    uv run python eval/run_eval.py --mode fake   # 内存库+Fake模型：仅验证链路可跑通
    uv run python eval/run_eval.py --mode real   # 生产 PG+真实模型+Chroma：真实指标
    uv run python eval/run_eval.py --mode real --limit 3   # 只跑前 3 条
"""
import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

UPLOADS_DIR = Path(__file__).resolve().parents[1] / 'uploads'
GOLDEN_SET = Path(__file__).resolve().parent / 'golden_set.json'
REPORT_DIR = Path(__file__).resolve().parent / 'reports'

# 诚实"无法回答"标记（信息不足类题目允许的回答形态）
HONEST_UNANSWERABLE = (
    '未提及', '没有相关', '无法', '未写', '没有写', '不存在',
    '不知道', '不得而知', '原文没有', '并未说明', '没有信息',
    '没有答案', '未找到', '没有记录', '未说明',
)


def _load_golden() -> list[dict]:
    data = json.loads(GOLDEN_SET.read_text(encoding='utf-8'))
    return data['cases']


def _make_client(mode: str) -> tuple[TestClient, object]:
    if mode == 'real':
        from reading_assistant.api import create_app
        return TestClient(create_app()), None
    # fake：内存库 + Fake 模型
    from langchain_core.embeddings import Embeddings
    from reading_assistant.api import create_app
    from reading_assistant.storage import create_db_engine, create_session_factory, init_db
    from reading_assistant.storage.vector_store import InMemoryVectorStore

    class FakeEmbeddings(Embeddings):
        def embed_query(self, text):
            return [1.0, 0.0]

        def embed_documents(self, texts):
            return [[1.0, 0.0] for _ in texts]

    class FakeLLM:
        def invoke(self, prompt):
            return SimpleNamespace(content='这是 fake 模式的测试回答。')

    engine = create_db_engine('sqlite:///:memory:')
    init_db(engine)
    app = create_app(
        session_factory=create_session_factory(engine),
        vector_store=InMemoryVectorStore(),
        llm=FakeLLM(),
        embedding_model=FakeEmbeddings(),
        upload_dir=Path('uploads'),
    )
    return TestClient(app), engine


def _upload_books(client: TestClient) -> dict[str, int]:
    """上传 uploads/ 下所有书，返回 filename -> document_id 映射。"""
    mapping = {}
    for path in sorted(UPLOADS_DIR.glob('*')):
        if path.suffix.lower() not in ('.txt', '.epub', '.pdf', '.docx'):
            continue
        resp = client.post(
            '/api/documents/upload',
            files={'file': (path.name, path.read_bytes(), 'text/plain')},
        )
        if resp.status_code == 201:
            mapping[path.name] = resp.json()['id']
    return mapping


def _answer_hits(answer: str | None, keywords: list[str]) -> bool:
    if not answer:
        return False
    return any(k in answer for k in keywords)


def _is_honest_unanswerable(answer: str | None, needs_clarification: bool) -> bool:
    if needs_clarification:
        return True
    if not answer:
        return False
    return any(m in answer for m in HONEST_UNANSWERABLE)


def _run_case(client: TestClient, case: dict, doc_map: dict) -> dict:
    session_id = client.post('/api/sessions').json()['session_id']
    doc_name = case.get('document')
    doc_ids = [doc_map[doc_name]] if doc_name and doc_name in doc_map else []
    start = time.perf_counter()
    resp = client.post(
        f'/api/sessions/{session_id}/messages',
        json={'question': case['question'], 'document_ids': doc_ids},
    )
    latency_ms = (time.perf_counter() - start) * 1000
    body = resp.json()
    answer = body.get('answer') or ''
    citations = body.get('citations') or []
    needs_clar = body.get('needs_clarification', False)

    if case.get('expect_unanswerable'):
        passed = _is_honest_unanswerable(answer, needs_clar)
        fail_reason = '' if passed else '应诚实说明无法回答/走HITL，实际给出了内容或未识别'
    else:
        keywords = case.get('expect_keywords') or []
        if keywords:
            passed = _answer_hits(answer, keywords)
            fail_reason = '' if passed else f"期望含 {keywords}"
        else:
            # 无金标关键词（自定义题）：能给出非空回答即视为通过
            passed = bool(answer)
            fail_reason = '' if passed else '未能生成回答'

    return {
        'id': case['id'],
        'question': case['question'],
        'document': doc_name or '(全库)',
        'pass': passed,
        'answer': answer[:80],
        'citations': len(citations),
        'needs_clarification': needs_clar,
        'latency_ms': round(latency_ms),
        'fail_reason': fail_reason,
    }


def run(mode: str = 'fake', limit: int = 0, cases: list[dict] | None = None) -> dict:
    """运行评测（fake=链路冒烟 / real=真实模型），返回报告 dict，供 API 与 CLI 复用。

    cases 为空时加载黄金集；也可传入自定义题目（question/expect_keywords 等）。
    """
    if cases is None:
        cases = _load_golden()
    else:
        cases = [dict(c) for c in cases]
        for i, c in enumerate(cases):
            c.setdefault('id', f'custom-{i + 1}')
            c.setdefault('document', None)
            c.setdefault('expect_keywords', [])
            c.setdefault('expect_unanswerable', False)
    if limit:
        cases = cases[: limit]

    client, engine = _make_client(mode)
    print(f'[eval] mode={mode} cases={len(cases)} 上传书籍中…')
    with client:
        doc_map = _upload_books(client)
        print(f'[eval] 文档映射: {doc_map}')

        results = []
        for case in cases:
            r = _run_case(client, case, doc_map)
            results.append(r)
            flag = 'PASS' if r['pass'] else 'FAIL'
            print(f"  [{flag}] {r['id']:12s} ({r['latency_ms']:5d}ms cit={r['citations']}) "
                  f"{r['question'][:24]}")
            if not r['pass'] and r['fail_reason']:
                print(f"          ↳ {r['fail_reason']} | answer: {r['answer']}")

    answerable = [r for r in results if not cases[results.index(r)].get('expect_unanswerable')]
    unans = [r for r in results if cases[results.index(r)].get('expect_unanswerable')]
    n_pass = sum(1 for r in results if r['pass'])
    n_unans_pass = sum(1 for r in unans if r['pass'])
    n_cited = sum(1 for r in answerable if r['citations'] > 0)
    avg_latency = sum(r['latency_ms'] for r in results) / len(results) if results else 0

    summary = {
        'mode': mode,
        'total': len(results),
        'passed': n_pass,
        'pass_rate': round(n_pass / len(results), 3) if results else 0,
        'answerable_accuracy': round(
            sum(1 for r in answerable if r['pass']) / len(answerable), 3
        ) if answerable else None,
        'unanswerable_recognition': round(n_unans_pass / len(unans), 3) if unans else None,
        'citation_coverage': round(n_cited / len(answerable), 3) if answerable else None,
        'avg_latency_ms': round(avg_latency),
    }
    print('\n=== 汇总 ===')
    for k, v in summary.items():
        print(f'  {k}: {v}')

    REPORT_DIR.mkdir(exist_ok=True)
    report_path = REPORT_DIR / f'eval_report_{mode}_{time.strftime("%Y%m%d_%H%M%S")}.json'
    report_path.write_text(
        json.dumps({'summary': summary, 'results': results}, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    print(f'\n报告已保存: {report_path}')
    if engine is not None:
        engine.dispose()
    return {'summary': summary, 'results': results, 'report_path': str(report_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description='RAG 问答黄金评测')
    parser.add_argument('--mode', choices=['fake', 'real'], default='fake')
    parser.add_argument('--limit', type=int, default=0, help='只跑前 N 条（0=全部）')
    args = parser.parse_args()
    run(mode=args.mode, limit=args.limit)


if __name__ == '__main__':
    main()
