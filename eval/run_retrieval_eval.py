"""检索级评测：vector（纯稠密）vs hybrid（BM25+稠密 RRF）召回对比。

**两个检索口径同时跑**（见 ``SCOPE_FILTERED`` / ``SCOPE_CROSS`` 注释）：
- ``filtered``：带 ``document_id``，对应「指定文档内检索」
- ``cross``   ：不带过滤，对应 ``qa.py`` 在 ``document_id=None`` 时的跨书库真实路径

只跑 ``filtered`` 会掩盖 HNSW 图退化（搜索空间缩到单文档，图断裂不可见）——
实测一个已退化的索引在 filtered 下 14/14 满分、cross 下仅 11/14。
用 ``--no-cross`` 可跳过（不建议，会失去该项检测能力）。

依赖 eval/retrieval_gold.json（先跑 build_retrieval_gold.py 生成）。
指标：Recall@N / Hit@N / MRR。

用法：
    uv run python -m eval.build_retrieval_gold --mode real   # 先生成 gold
    uv run python -m eval.run_retrieval_eval --mode real     # 真实模型对比（推荐）
    uv run python -m eval.run_retrieval_eval --mode fake     # 冒烟（gold 语义弱）
    uv run python -m eval.run_retrieval_eval --mode real --no-cross   # 只跑带过滤口径

输出：eval/reports/retrieval_eval_{mode}_{ts}.json + 控制台对比表
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from fastapi.testclient import TestClient

from eval.metrics import case_retrieval_metrics
from eval.run_eval import _upload_books

EVAL_DIR = Path(__file__).resolve().parent
REPORT_DIR = EVAL_DIR / 'reports'
GOLD_FILE = EVAL_DIR / 'retrieval_gold.json'
TOP_K_VALUES = (5, 10, 20)

# 检索口径。必须两个都跑，原因见下。
#   filtered —— 带 document_id，对应「指定文档内检索」
#   cross    —— 不带过滤，对应 qa.py 在 document_id=None 时的**跨书库**真实路径
#
# 为什么不能只跑 filtered：带 document_id 时 Chroma 会把搜索空间缩到单个文档，
# HNSW 图断裂被掩盖。实测（2026-09-17，1520 chunks）一个已退化的索引在 filtered
# 口径下 14/14 满分，在 cross 口径下只有 11/14 —— 只有 cross 能检出退化。
# 反之，cross 口径也覆盖了 filtered 覆盖不到的真实场景（跨书库提问）。
SCOPE_FILTERED = 'filtered'
SCOPE_CROSS = 'cross'
SYSTEMS = ('vector', 'hybrid')


def _load_gold(path: Path | None = None) -> list[dict]:
    source = path or GOLD_FILE
    data = json.loads(source.read_text(encoding='utf-8'))
    return data['cases']


def _gold_ids_from_contains(
    case: dict, doc_map: dict[str, int], chunks_all: list
) -> list[str]:
    """按 must_contain 短语在原文中定位 gold chunk（非引用弱标注）。

    用于词面鲁棒性评测集：phrase 命中的 chunk 即应当召回的 ground truth。
    """
    phrase = case.get('contains')
    if not phrase:
        return []
    doc_ids: list[int] = []
    names = case.get('documents') or ([case['document']] if case.get('document') else [])
    for name in names:
        if name in doc_map:
            doc_ids.append(doc_map[name])
    return [
        chunk.id
        for chunk in chunks_all
        if phrase in (chunk.text or '')
        and (not doc_ids or chunk.metadata.get('document_id') in doc_ids)
    ]


def _make_real_env() -> tuple[list[dict], object, object]:
    """real 模式：生产 app 上传书籍 + 生产向量库（chroma）+ 真实 embedding。"""
    from reading_assistant.api import create_app
    from reading_assistant.model.factory import get_embedding_model
    from reading_assistant.storage.vector_store import create_vector_store

    client = TestClient(create_app())
    with client:
        doc_map = _upload_books(client)
    store = create_vector_store()  # 与 app 共用同一 persist 目录/collection
    embedding = get_embedding_model()
    return doc_map, store, embedding


def _make_fake_env() -> tuple[list[dict], object, object]:
    """fake 模式：内存库 + FakeEmbeddings（向量退化，仅链路冒烟）。"""
    from langchain_core.embeddings import Embeddings

    from reading_assistant.api import create_app
    from reading_assistant.storage import create_db_engine, create_session_factory, init_db
    from reading_assistant.storage.vector_store import InMemoryVectorStore

    class FakeEmbeddings(Embeddings):
        def embed_query(self, text):
            return [1.0, 0.0]

        def embed_documents(self, texts):
            return [[1.0, 0.0] for _ in texts]

    engine = create_db_engine('sqlite:///:memory:')
    init_db(engine)
    store = InMemoryVectorStore()
    app = create_app(
        session_factory=create_session_factory(engine),
        vector_store=store,
        embedding_model=FakeEmbeddings(),
        upload_dir=Path('uploads'),
    )
    client = TestClient(app)
    with client:
        doc_map = _upload_books(client)
    return doc_map, store, FakeEmbeddings()


def _retrieved_ids(retriever, question: str, document_id: int | None, top_n: int) -> list[str]:
    hits = retriever.retrieve(question, top_k=top_n, document_id=document_id)
    return [hit.chunk_id for hit in hits]


def _case_metrics(retrieved: list[str], gold: list[str]) -> dict:
    gold_set = set(gold)
    hits = [cid for cid in retrieved if cid in gold_set]
    n = len(retrieved)
    hit_count = len(hits)
    recall = hit_count / len(gold_set) if gold_set else 0.0
    first_rank = None
    for index, cid in enumerate(retrieved, start=1):
        if cid in gold_set:
            first_rank = index
            break
    mrr = 1.0 / first_rank if first_rank else 0.0
    k = n or 1
    extra = case_retrieval_metrics(retrieved, gold, ks=(k,))
    return {
        'hit': 1 if hit_count else 0,
        'recall': recall,
        'mrr': mrr,
        'n': n,
        'ndcg': extra[f'ndcg@{k}'],
        'precision': extra[f'precision@{k}'],
        'map': extra[f'map@{k}'],
    }


def _empty_summaries() -> dict:
    """按 (system, top_k) 构造累加器。"""
    return {
        system: {
            n: {
                'hit': 0,
                'recall': 0.0,
                'mrr': 0.0,
                'ndcg': 0.0,
                'precision': 0.0,
                'map': 0.0,
                'cases': 0,
            }
            for n in TOP_K_VALUES
        }
        for system in SYSTEMS
    }


def _aggregate(summaries: dict, n_cases: int) -> dict:
    """把累加器转成均值 dict。"""
    aggregated: dict = {}
    for system in SYSTEMS:
        vals = {}
        for n in TOP_K_VALUES:
            s = summaries[system][n]
            vals[f'recall@{n}'] = round(s['recall'] / n_cases, 4)
            vals[f'hit@{n}'] = round(s['hit'] / n_cases, 4)
            vals[f'mrr@{n}'] = round(s['mrr'] / n_cases, 4)
            vals[f'ndcg@{n}'] = round(s['ndcg'] / n_cases, 4)
            vals[f'precision@{n}'] = round(s['precision'] / n_cases, 4)
            vals[f'map@{n}'] = round(s['map'] / n_cases, 4)
        aggregated[system] = vals
    return aggregated


def _print_table(title: str, aggregated: dict) -> None:
    print(f'\n=== {title} ===')
    head = f"{'':8s} {'Recall@5':>10s} {'Recall@10':>11s} {'Recall@20':>11s}"
    head += f" {'Hit@5':>7s} {'Hit@10':>8s} {'Hit@20':>8s} {'MRR':>6s}"
    head += f" {'NDCG@5':>8s} {'NDCG@10':>8s} {'MAP@5':>7s} {'P@5':>6s}"
    print(head)
    for system in SYSTEMS:
        vals = aggregated[system]
        print(f"{system:8s} {vals['recall@5']:>10.3f} {vals['recall@10']:>11.3f} "
              f"{vals['recall@20']:>11.3f} {vals['hit@5']:>7.2f} {vals['hit@10']:>8.2f} "
              f"{vals['hit@20']:>8.2f} {vals['mrr@5']:>6.3f} "
              f"{vals['ndcg@5']:>8.3f} {vals['ndcg@10']:>8.3f} "
              f"{vals['map@5']:>7.3f} {vals['precision@5']:>6.3f}")


def run(
    mode: str = 'real',
    gold_file: Path | None = None,
    cross_scope: bool = True,
) -> dict:
    gold = _load_gold(gold_file)
    doc_map, store, embedding = _make_real_env() if mode == 'real' else _make_fake_env()

    from reading_assistant.rag import HybridRetriever, Retriever

    vector_r = Retriever(store, embedding)
    hybrid_r = HybridRetriever(store, embedding)

    def resolve_document_id(case: dict) -> int | None:
        if case.get('document') and case['document'] in doc_map:
            return doc_map[case['document']]
        return case.get('document_id')

    summaries = _empty_summaries()
    cross_summaries = _empty_summaries() if cross_scope else None
    scopes = [SCOPE_FILTERED] + ([SCOPE_CROSS] if cross_scope else [])
    per_case: list[dict] = []
    print(f'[retrieval-eval] mode={mode} gold_cases={len(gold)} '
          f'topk={TOP_K_VALUES} scopes={scopes}')
    chunks_all = store.all_chunks()
    for case in gold:
        question = case['question']
        doc_id = resolve_document_id(case)
        gold_ids = list(case.get('chunk_ids') or [])
        if not gold_ids:
            gold_ids = _gold_ids_from_contains(case, doc_map, chunks_all)
        if not gold_ids:
            print(f"  [skip] {case['id']:12s} 无 gold（短语未命中或文档缺失），不计入统计")
            continue
        row = {'id': case['id'], 'gold_chunks': len(gold_ids), 'doc': case['document']}
        for scope in scopes:
            buckets = summaries if scope == SCOPE_FILTERED else cross_summaries
            # filtered 带 document_id；cross 不带过滤
            scope_doc = doc_id if scope == SCOPE_FILTERED else None
            suffix = '' if scope == SCOPE_FILTERED else '_cross'
            for system, retriever in (('vector', vector_r), ('hybrid', hybrid_r)):
                retrieved = _retrieved_ids(
                    retriever, question, scope_doc, max(TOP_K_VALUES)
                )
                for n in TOP_K_VALUES:
                    m = _case_metrics(retrieved[:n], gold_ids)
                    cell = buckets[system][n]
                    cell['hit'] += m['hit']
                    cell['recall'] += m['recall']
                    cell['mrr'] += m['mrr']
                    cell['ndcg'] += m['ndcg']
                    cell['precision'] += m['precision']
                    cell['map'] += m['map']
                    cell['cases'] += 1
                row[f'{system}{suffix}_mrr'] = round(
                    _case_metrics(retrieved, gold_ids)['mrr'], 3
                )
                row[f'{system}{suffix}_ids'] = retrieved[:5]
        per_case.append(row)
        line = f"  {case['id']:12s} gold={len(gold_ids):2d}"
        line += (f" | vector MRR={row['vector_mrr']:.3f}"
                 f" hybrid MRR={row['hybrid_mrr']:.3f}")
        if cross_scope:
            line += (f" | cross: v={row['vector_cross_mrr']:.3f}"
                     f" h={row['hybrid_cross_mrr']:.3f}")
        print(line)

    n_cases = len(per_case) or 1
    aggregated = {'mode': mode, 'cases': len(per_case)}
    aggregated.update(_aggregate(summaries, n_cases))
    _print_table('汇总（均值）· 指定文档内检索（带 document_id）', aggregated)

    cross_aggregated = None
    if cross_scope:
        cross_aggregated = {'mode': mode, 'cases': len(per_case)}
        cross_aggregated.update(_aggregate(cross_summaries, n_cases))
        _print_table('汇总（均值）· 跨书库检索（无过滤）', cross_aggregated)
        print('  ↑ cross 口径对应 qa.py 在 document_id=None 时的真实检索路径。')
        print('    带 document_id 会把搜索空间缩到单文档，掩盖 HNSW 图断裂 ——')
        print('    实测退化索引 filtered 14/14、cross 仅 11/14，故两者都要看。')

    payload: dict = {'summary': aggregated, 'per_case': per_case}
    if cross_aggregated is not None:
        # 单独放顶层键：compare_reports.py 会把 summary 里的 dict 型字段当作
        # 「系统」展平，嵌进 summary 会污染下游对比。
        payload['cross_summary'] = cross_aggregated

    REPORT_DIR.mkdir(exist_ok=True)
    report_path = REPORT_DIR / f'retrieval_eval_{mode}_{time.strftime("%Y%m%d_%H%M%S")}.json'
    report_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    print(f'\n报告已保存: {report_path}')
    result = {
        'summary': aggregated,
        'per_case': per_case,
        'report_path': str(report_path),
    }
    if cross_aggregated is not None:
        result['cross_summary'] = cross_aggregated
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description='检索级评测：vector vs hybrid')
    parser.add_argument('--mode', choices=['fake', 'real'], default='real')
    parser.add_argument(
        '--gold-file',
        type=Path,
        default=GOLD_FILE,
        help='评测集路径（默认 retrieval_gold.json；支持 contains 词面定位用例）',
    )
    parser.add_argument(
        '--no-cross',
        dest='cross',
        action='store_false',
        help='跳过跨书库口径（默认两个口径都跑；跳过会丢失 HNSW 退化检测能力）',
    )
    args = parser.parse_args()
    run(mode=args.mode, gold_file=args.gold_file, cross_scope=args.cross)


if __name__ == '__main__':
    main()
