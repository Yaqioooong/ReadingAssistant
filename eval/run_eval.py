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

# 评测语料目录（eval/uploads，仅作读取源）。
# 注意：不能指向仓库根 uploads/——那是应用的上传落盘目录，
# 每次上传都会写入 uuid 前缀拷贝，语料源若与之重合会自我复制、指数膨胀。
UPLOADS_DIR = Path(__file__).resolve().parent / 'uploads'
GOLDEN_SET = Path(__file__).resolve().parent / 'golden_set.json'
REPORT_DIR = Path(__file__).resolve().parent / 'reports'

# 诚实"无法回答"标记（信息不足类题目允许的回答形态）。
# ⚠️ 这是纯词表匹配，**漏词 = 假阴性**：模型明明诚实拒答了，却因为措辞不在表里被判 FAIL。
# 2026-09-17 扩集时实测踩到：「没有提到」「没有交代」不在原表内，
# 两道信息不足题（牛魔王的师父 / 铁扇公主的父亲）模型都答了
# 「原文片段中没有提到…」，却被判成「应诚实说明无法回答，实际给出了内容」。
# 注意区分：只有 expect_unanswerable 的题才走这份词表，所以放宽不会让可答题蒙混过关。
HONEST_UNANSWERABLE = (
    '未提及', '没有相关', '无法', '未写', '没有写', '不存在',
    '不知道', '不得而知', '原文没有', '并未说明', '没有信息',
    '没有答案', '未找到', '没有记录', '未说明',
    # --- 2026-09-17 补：模型实际使用的高频「明确表示缺失」措辞 ---
    '没有提到', '没有提', '没有提及', '没有交代', '未交代', '并未交代',
    '没有说明', '未明确', '无从得知', '原文未', '没有记载', '未曾提及',
)


def _load_golden(path: Path | None = None) -> list[dict]:
    source = path or GOLDEN_SET
    data = json.loads(source.read_text(encoding='utf-8'))
    return data['cases']


def _make_client(mode: str) -> tuple[TestClient, object]:
    if mode == 'real':
        from reading_assistant.api import create_app
        return TestClient(create_app()), None
    # fake：内存库 + Fake 模型
    import tempfile

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
        upload_dir=Path(tempfile.mkdtemp(prefix='ra_eval_fake_')),
    )
    return TestClient(app), engine


def _upload_books(client: TestClient) -> dict[str, int]:
    """上传 uploads/ 下所有书，返回 filename -> document_id 映射。"""
    mapping = {}
    for path in sorted(UPLOADS_DIR.glob('*')):
        if path.suffix.lower() not in ('.txt', '.epub', '.pdf', '.docx'):
            continue
        # 保险丝：历史遗留的 uuid 前缀累积会让文件名逼近系统上限，
        # 超长源文件直接跳过并告警，避免整次评测崩溃（Errno 63）。
        if len(path.name) > 120:
            print(f'[eval] 跳过超长文件 ({len(path.name)} 字符): {path.name[:60]}...')
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
    multi_names = case.get('documents') or []
    if multi_names:
        doc_ids = [doc_map[n] for n in multi_names if n in doc_map]
        doc_name = ','.join(multi_names)
    else:
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
        'session_id': session_id,
    }


def run(
    mode: str = 'fake',
    limit: int = 0,
    cases: list[dict] | None = None,
    golden_path: Path | None = None,
    keep_sessions: bool = False,
) -> dict:
    """运行评测（fake=链路冒烟 / real=真实模型），返回报告 dict，供 API 与 CLI 复用。

    cases 为空时加载黄金集（可用 golden_path 指定其他评测文件，如多文档用例）；
    也可传入自定义题目（question/expect_keywords 等）。
    """
    if cases is None:
        cases = _load_golden(golden_path)
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

        # 清场：评测会话会产生 awaiting 澄清任务，默认跑完即删，避免污染库
        if not keep_sessions:
            removed = 0
            for sid in {r.get('session_id') for r in results if r.get('session_id')}:
                try:
                    removed += int(client.delete(f'/api/sessions/{sid}').status_code == 204)
                except Exception:  # noqa: BLE001 清理失败不影响评测结论
                    pass
            print(f'[清理] 已删除评测会话 {removed} 个（含其澄清任务）')

    answerable = [r for r in results if not cases[results.index(r)].get('expect_unanswerable')]
    unans = [r for r in results if cases[results.index(r)].get('expect_unanswerable')]
    n_pass = sum(1 for r in results if r['pass'])
    n_unans_pass = sum(1 for r in unans if r['pass'])
    n_cited = sum(1 for r in answerable if r['citations'] > 0)
    avg_latency = sum(r['latency_ms'] for r in results) / len(results) if results else 0

    # 生成层规则指标（附加口径，原字段一律不变）
    from eval.metrics import abstention_metrics, answer_relevancy

    _records = [
        {
            'expect_unanswerable': bool(
                (cases[index] if index < len(cases) else {}).get('expect_unanswerable')
            ),
            'answer': r.get('answer', ''),
        }
        for index, r in enumerate(results)
    ]
    _abstention = abstention_metrics(_records)
    _relevancy_scores = [
        answer_relevancy(r.get('question', ''), r.get('answer', '')) for r in results
    ]
    _avg_relevancy = (
        round(sum(_relevancy_scores) / len(_relevancy_scores), 4)
        if _relevancy_scores
        else None
    )

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
        'abstention_rate': _abstention['abstention_rate'],
        'false_refusal_rate': _abstention['false_refusal_rate'],
        'hallucination_rate': _abstention['hallucination_rate'],
        'answer_relevancy': _avg_relevancy,
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
    parser.add_argument(
        '--golden', type=Path, default=None, help='自定义黄金集路径（默认 golden_set.json）'
    )
    parser.add_argument('--keep-sessions', action='store_true',
                        help='保留评测会话（默认跑完删除，避免污染库）')
    args = parser.parse_args()
    run(mode=args.mode, limit=args.limit, golden_path=args.golden,
        keep_sessions=args.keep_sessions)


if __name__ == '__main__':
    main()
