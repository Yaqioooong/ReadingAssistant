"""检索 agent 专项评测：多问场景能否解决 + 单问是否被拖累。

与其它评测层的分工：
- `run_retrieval_eval` 量**单次向量检索**的召回 —— 本集里的 ordinal/compound/pronoun
  在那一层是**结构性无解**的（2026-09-21 实测：min_score=0、top_k=200 都捞不到），
  放进检索集只会污染指标；所以它们单独在这里量「端到端能不能答对」。
- `run_eval`（生成层）量通用问答质量。本集**只量 agent 的增量价值与成本**。

两类验收标准：
1. `ordinal` / `compound` / `pronoun`：**要答对**（`expect_contains` 出现在回答里）。
2. `simple`：**agent 不得触发**（`must_not_use_agent`）—— 守住「兼容单问」的成本底线。

用法：
    uv run python eval/run_agent_eval.py --mode real
    uv run python eval/run_agent_eval.py --mode real --limit 4
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

GOLD = Path(__file__).resolve().parent / 'agent_gold.json'
REPORT_DIR = Path(__file__).resolve().parent / 'reports'
DOC_ID = 7  # 西游记（本集全部题目都在这本书内）


def _load_golden() -> list[dict]:
    return json.loads(GOLD.read_text(encoding='utf-8'))['cases']


def _build_env():
    """内存库 + 生产向量库/模型：不写生产库，但检索与模型都是真的。

    必须种入 documents：检索白名单 `list_indexed_document_ids` 为空时，
    retrieve 的命中会被全部滤掉 —— 探针会得出「agent 到处触发」的假象
    （2026-09-21 实际踩到，浪费了一轮解读）。
    """
    from reading_assistant.runtime import get_session_factory, get_vector_store
    from reading_assistant.storage import (
        create_db_engine,
        create_session_factory,
        init_db,
        list_documents,
    )
    from reading_assistant.storage.models import Document

    with get_session_factory()() as session:
        real_docs = [
            (d.id, d.filename, d.content_hash, d.index_status)
            for d in list_documents(session)
        ]

    engine = create_db_engine('sqlite:///:memory:')
    init_db(engine)
    factory = create_session_factory(engine)
    with factory() as session:
        for doc_id, filename, content_hash, status in real_docs:
            session.add(Document(
                id=doc_id, filename=filename, title=filename[:40],
                file_hash=f'f{doc_id}', content_hash=content_hash,
                index_status=status,
            ))
        session.commit()
    return factory, get_vector_store()


def _ask(factory, vector_store, question: str, history: list[dict] | None, thread: str) -> dict:
    """跑一题。**每题一个独立会话**（见下）。

    ⚠️ 曾经所有用例共用 ``ChatSession(id=1)``，于是 gold 的 per-case ``history``
    会往同一个会话里累加 —— 上一题的上下文泄漏进下一题。
    后果实测（2026-09-22）：``ag-pronoun-01`` 把「寅将军」写进历史后，
    ``ag-pronoun-02``（gold 期望「孙悟空」）拿到的历史里多了「寅将军」，
    agent 遂把「他」当成寅将军、去第十三回找「收服」，必然答错。
    而 gold 给每题单独写 ``history`` 就是要**控制**上下文，泄漏直接废掉了这个控制。
    改用独立会话后，每题的上下文才等于 gold 声明的那一段。
    """
    from reading_assistant.graph.qa import build_qa_graph
    from reading_assistant.storage.models import ChatMessage, ChatSession

    with factory() as session:
        chat = ChatSession()
        session.add(chat)
        session.commit()
        session_id = chat.id
        for message in (history or []):
            session.add(ChatMessage(
                session_id=session_id, role=message['role'], content=message['content'],
            ))
        session.commit()

    graph = build_qa_graph(factory, vector_store)
    return graph.invoke(
        {'question': question, 'session_id': session_id, 'document_ids': [DOC_ID]},
        config={'configurable': {'thread_id': thread}},
    )


def run(mode: str, limit: int) -> dict:
    if mode != 'real':
        print('[agent-eval] 本集只跑 real 模式：fake 模型没有工具能力，agent 必然不触发，'
              '测不出任何结论。', file=sys.stderr)
        sys.exit(2)
    cases = _load_golden()
    if limit > 0:
        cases = cases[:limit]
    factory, vector_store = _build_env()

    results = []
    for case in cases:
        started = time.perf_counter()
        try:
            state = _ask(factory, vector_store, case['question'],
                         case.get('history'), thread=f"ag-{case['id']}")
            error = ''
        except Exception as exc:  # noqa: BLE001 单题失败不该中断整轮评测
            state, error = {}, f'{type(exc).__name__}: {exc}'
        answer = str(state.get('answer') or '')
        actions = [t.get('action') for t in (state.get('agent_trace') or [])]
        contain_ok = (case.get('expect_contains') or '') in answer if answer else False
        if case.get('must_not_use_agent'):
            passed = bool(answer) and contain_ok and not state.get('agent_used')
        else:
            passed = contain_ok
        results.append({
            'id': case['id'], 'class': case['class'], 'question': case['question'][:44],
            'agent_used': bool(state.get('agent_used')),
            'agent_steps': state.get('agent_steps') or 0,
            'actions': actions,
            'contain_ok': contain_ok,
            'passed': passed,
            'needs_clarification': bool(state.get('needs_clarification')),
            'cost_ms': round((time.perf_counter() - started) * 1000),
            'error': error,
        })
        flag = 'PASS' if passed else 'FAIL'
        print(f"  {case['id']:<16} {flag}  类={case['class']:<8} "
              f"agent={'是' if results[-1]['agent_used'] else '否'} "
              f"步={results[-1]['agent_steps']} 命中={contain_ok} "
              f"{('动作=' + str(actions[:4])) if actions else ''}")

    by_class: dict[str, dict] = {}
    for r in results:
        bucket = by_class.setdefault(r['class'], {'n': 0, 'pass': 0, 'agent': 0, 'steps': 0})
        bucket['n'] += 1
        bucket['pass'] += int(r['passed'])
        bucket['agent'] += int(r['agent_used'])
        bucket['steps'] += r['agent_steps']

    total = len(results)
    passed_total = sum(r['passed'] for r in results)
    simple = [r for r in results if r['class'] == 'simple']
    summary = {
        'mode': mode,
        'total': total,
        'passed': passed_total,
        'pass_rate': passed_total / total if total else 0.0,
        'agent_triggered': sum(r['agent_used'] for r in results),
        'simple_regression': sum(1 for r in simple if r['agent_used']),
        'avg_agent_steps': (
            sum(r['agent_steps'] for r in results if r['agent_used'])
            / max(1, sum(1 for r in results if r['agent_used']))
        ),
        'action_distribution': dict(Counter(
            a for r in results for a in r['actions']
        )),
        'by_class': {
            name: {
                'n': b['n'], 'pass_rate': b['pass'] / b['n'] if b['n'] else 0.0,
                'trigger_rate': b['agent'] / b['n'] if b['n'] else 0.0,
                'avg_steps': b['steps'] / b['n'] if b['n'] else 0.0,
            }
            for name, b in sorted(by_class.items())
        },
    }
    REPORT_DIR.mkdir(exist_ok=True)
    (REPORT_DIR / f'agent_report_{mode}.json').write_text(
        json.dumps({'summary': summary, 'cases': results}, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    return summary, results


def main() -> None:
    parser = argparse.ArgumentParser(description='检索 agent 专项评测')
    parser.add_argument('--mode', choices=('real', 'fake'), default='real')
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    summary, _ = run(mode=args.mode, limit=args.limit)
    print(f"\n[agent-eval {summary['mode']}] 共 {summary['total']} 题 "
          f"通过 {summary['passed']} ({summary['pass_rate']:.1%})")
    print(f"  agent 触发 {summary['agent_triggered']} 题，"
          f"平均 {summary['avg_agent_steps']:.1f} 步")
    print(f"  ⚠️ 单问题误触发 agent: {summary['simple_regression']} 题（必须为 0）")
    print(f"  动作分布 {summary['action_distribution']}")
    for name, stat in summary['by_class'].items():
        print(f"  [{name:<8}] 通过 {stat['pass_rate']:.1%}  触发 {stat['trigger_rate']:.0%}  "
              f"平均 {stat['avg_steps']:.1f} 步")
    report_path = REPORT_DIR / f"agent_report_{summary['mode']}.json"
    print(f"\n报告：{report_path}")


if __name__ == '__main__':
    main()
