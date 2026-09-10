"""多轮意图评测：按 multi_turn_gold.json 逐会话回放，校验检索门路由与历史回忆。

用法：
    uv run python eval/run_intent_eval.py --mode fake   # 内存库+Fake 路由模型：链路自检
    uv run python eval/run_intent_eval.py --mode real   # 生产 PG+真实模型：真实门控准确率
    uv run python eval/run_intent_eval.py --mode fake --limit 2
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

GOLDEN = Path(__file__).resolve().parent / 'multi_turn_gold.json'
UPLOADS_DIR = Path(__file__).resolve().parent / 'uploads'
REPORT_DIR = Path(__file__).resolve().parent / 'reports'


def _load_golden() -> list[dict]:
    return json.loads(GOLDEN.read_text(encoding='utf-8'))['cases']


def _ask(client: TestClient, sid: str, turn: dict) -> dict:
    payload = {'question': turn['question']}
    if turn.get('clarification'):
        payload['clarification'] = turn['clarification']
    return client.post(f'/api/sessions/{sid}/messages', json=payload).json()


class FakeRouterLLM:
    """fake 模式的意图路由模型：按关键词给 gate 标签；context 轮回显记录中最后一条用户提问。"""

    _HISTORY_HINTS = ('上一个', '问过哪些', '前两', '前几', '问了什么', '问过什么', '第一个问题')
    _CHAT_HINTS = ('你好', 'hello', 'hi', '谢谢', '你是谁')

    def invoke(self, prompt: str) -> SimpleNamespace:
        if '意图分类任务' in prompt:
            question = self._current_question(prompt)
            if any(h in question for h in self._HISTORY_HINTS):
                return SimpleNamespace(content='history')
            if any(h.lower() in question.lower() for h in self._CHAT_HINTS):
                return SimpleNamespace(content='chat')
            return SimpleNamespace(content='book')
        if '不需要检索书籍内容' in prompt:
            last_q = self._last_user_question(prompt)
            return SimpleNamespace(content=f'你上一个问题是：「{last_q}」')
        return SimpleNamespace(content='这是 fake 模式的多轮评测回答。')

    @staticmethod
    def _current_question(prompt: str) -> str:
        part = prompt.split('【当前提问】', 1)[-1]
        return next((ln.strip() for ln in part.splitlines() if ln.strip()), '')

    @staticmethod
    def _last_user_question(prompt: str) -> str:
        part = prompt.split('【对话记录】', 1)[-1].split('【当前提问】', 1)[0]
        last = ''
        for line in part.splitlines():
            line = line.strip()
            if line.startswith('用户：'):
                last = line[len('用户：'):]
        return last


class FakeEmb:
    def embed_query(self, text: str):
        return [0.9, 0.44]

    def embed_documents(self, texts: list[str]):
        return [[1.0, 0.0] for _ in texts]


def _make_client(mode: str) -> tuple[TestClient, dict[str, int]]:
    if mode == 'real':
        from reading_assistant.api import create_app
        client = TestClient(create_app())
        mapping: dict[str, int] = {}
        with client:
            for path in sorted(UPLOADS_DIR.glob('*')):
                if path.suffix.lower() in ('.txt', '.epub', '.pdf', '.docx'):
                    resp = client.post('/api/documents/upload', files={'file': path.open('rb')})
                    if resp.status_code == 200:
                        mapping[path.name] = resp.json().get('document_id')
        return client, mapping
    # fake：内存库 + Fake 路由模型
    from reading_assistant.api import create_app
    from reading_assistant.storage import create_db_engine, create_session_factory, init_db
    from reading_assistant.storage.models import Document
    from reading_assistant.storage.vector_store import InMemoryVectorStore, StoredChunk

    engine = create_db_engine('sqlite:///:memory:')
    init_db(engine)
    factory = create_session_factory(engine)
    with factory() as session:
        session.add(Document(filename='d1.txt', title='d1', file_hash='f1', content_hash='c1'))
        session.commit()
    store = InMemoryVectorStore()
    store.add(
        [
            StoredChunk(
                id='d1-0',
                text='文档一内容',
                metadata={'document_id': 1, 'chapter': '正文', 'chapter_index': 0},
                embedding=[1.0, 0.0],
            )
        ]
    )
    app = create_app(
        session_factory=factory,
        vector_store=store,
        llm=FakeRouterLLM(),
        embedding_model=FakeEmb(),
        upload_dir=Path(tempfile.mkdtemp(prefix='ra_intent_fake_')),
    )
    return TestClient(app), {}


def run(mode: str, limit: int, keep_sessions: bool = False) -> dict:
    cases = _load_golden()
    if limit > 0:
        cases = cases[:limit]
    client, doc_map = _make_client(mode)

    case_results: list[dict] = []
    created_sessions: list[str] = []
    route_ok = 0
    content_ok = 0
    content_total = 0
    confusion: dict[str, dict[str, int]] = {}
    with client:
        for case in cases:
            sid = client.post('/api/sessions').json()['session_id']
            created_sessions.append(sid)
            turns = case['turns']
            per_turn = []
            case_pass = True
            for idx, turn in enumerate(turns):
                resp = _ask(client, sid, turn)
                expected = turn['route']
                actual = resp.get('intent') or 'book'
                ok = actual == expected
                route_ok += int(ok)
                case_pass = case_pass and ok
                confusion.setdefault(expected, {}).setdefault(actual, 0)
                confusion[expected][actual] += 1
                record = {
                    'turn': idx + 1,
                    'question': turn['question'][:40],
                    'expected': expected,
                    'actual': actual,
                    'ok': ok,
                }
                if 'in_answer' in turn:
                    content_total += 1
                    answer = resp.get('answer') or ''
                    hit = bool(answer) and turn['in_answer'] in answer
                    content_ok += int(hit)
                    record['in_answer_hit'] = hit
                    case_pass = case_pass and hit
                per_turn.append(record)
            case_results.append({'id': case['id'], 'pass': case_pass, 'turns': per_turn})

        # 清场：评测会话是抛头（会产生 awaiting 澄清任务），默认跑完即删，
        # 避免污染生产库、也避免任务在前端以「待补充」卡片形式回流。
        if not keep_sessions:
            removed = 0
            for sid in created_sessions:
                try:
                    resp = client.delete(f'/api/sessions/{sid}')
                    removed += int(resp.status_code == 204)
                except Exception:  # noqa: BLE001 清理失败不影响评测结论
                    pass
            print(f'[清理] 已删除评测会话 {removed}/{len(created_sessions)} 个（含其澄清任务）')

    total = sum(len(c['turns']) for c in case_results)
    summary = {
        'mode': mode,
        'case_count': len(case_results),
        'turn_total': total,
        'route_accuracy': route_ok / total if total else 0.0,
        'content_accuracy': content_ok / content_total if content_total else None,
        'confusion': confusion,
    }
    report_path = REPORT_DIR / f'intent_report_{mode}.json'
    REPORT_DIR.mkdir(exist_ok=True)
    payload = {'summary': summary, 'cases': case_results}
    report_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    return summary, case_results


def main() -> None:
    parser = argparse.ArgumentParser(description='多轮意图评测')
    parser.add_argument('--mode', choices=('fake', 'real'), default='fake')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--keep-sessions', action='store_true',
                        help='保留评测会话（默认跑完删除，避免污染库）')
    args = parser.parse_args()

    summary, case_results = run(args.mode, args.limit, args.keep_sessions)
    content_acc = summary['content_accuracy']
    content_txt = '—' if content_acc is None else f'{content_acc:.2%}'
    print(f"[intent-eval {summary['mode']}] cases={summary['case_count']} "
          f"turns={summary['turn_total']} route_acc={summary['route_accuracy']:.2%} "
          f"content_acc={content_txt}")
    for case in case_results:
        marks = []
        for t in case['turns']:
            flag = '✓' if t['ok'] else f"✗{t['expected']}→{t['actual']}"
            mark = f"t{t['turn']}:{flag}"
            if 'in_answer_hit' in t:
                mark += ':内容✓' if t['in_answer_hit'] else ':内容✗'
            marks.append(mark)
        marks = ' '.join(marks)
        print(f"  {case['id']:<24} {'PASS' if case['pass'] else 'FAIL'}  {marks}")
    conf = summary['confusion']
    if conf:
        print('  route 混淆矩阵(expected→actual):')
        for expected, row in conf.items():
            print(f"    {expected:<8} " + ', '.join(f'{a}:{n}' for a, n in sorted(row.items())))


if __name__ == '__main__':
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        print(f'[intent-eval] 运行失败: {exc}', file=sys.stderr)
        raise
