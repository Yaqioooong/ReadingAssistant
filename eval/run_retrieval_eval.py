"""检索级评测：vector（纯稠密）vs hybrid（BM25+稠密 RRF）召回对比。

依赖 eval/retrieval_gold.json（先跑 build_retrieval_gold.py 生成）。
指标：Recall@N / Hit@N / MRR。

用法：
    uv run python -m eval.build_retrieval_gold --mode real   # 先生成 gold
    uv run python -m eval.run_retrieval_eval --mode real     # 真实模型对比（推荐）
    uv run python -m eval.run_retrieval_eval --mode fake     # 冒烟（gold 语义弱）

输出：eval/reports/retrieval_eval_{mode}_{ts}.json + 控制台对比表
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from fastapi.testclient import TestClient

from eval.run_eval import _upload_books

EVAL_DIR = Path(__file__).resolve().parent
REPORT_DIR = EVAL_DIR / 'reports'
GOLD_FILE = EVAL_DIR / 'retrieval_gold.json'
TOP_K_VALUES = (5, 10, 20)


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
    return {'hit': 1 if hit_count else 0, 'recall': recall, 'mrr': mrr, 'n': n}


def run(mode: str = 'real', gold_file: Path | None = None) -> dict:
    gold = _load_gold(gold_file)
    doc_map, store, embedding = _make_real_env() if mode == 'real' else _make_fake_env()

    from reading_assistant.rag import HybridRetriever, Retriever

    vector_r = Retriever(store, embedding)
    hybrid_r = HybridRetriever(store, embedding)

    def resolve_document_id(case: dict) -> int | None:
        if case.get('document') and case['document'] in doc_map:
            return doc_map[case['document']]
        return case.get('document_id')

    summaries = {
        'vector': {n: {'hit': 0, 'recall': 0.0, 'mrr': 0.0, 'cases': 0} for n in TOP_K_VALUES},
        'hybrid': {n: {'hit': 0, 'recall': 0.0, 'mrr': 0.0, 'cases': 0} for n in TOP_K_VALUES},
    }
    per_case: list[dict] = []
    print(f'[retrieval-eval] mode={mode} gold_cases={len(gold)} topk={TOP_K_VALUES}')
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
        for system, retriever in (('vector', vector_r), ('hybrid', hybrid_r)):
            retrieved = _retrieved_ids(retriever, question, doc_id, max(TOP_K_VALUES))
            for n in TOP_K_VALUES:
                m = _case_metrics(retrieved[:n], gold_ids)
                summaries[system][n]['hit'] += m['hit']
                summaries[system][n]['recall'] += m['recall']
                summaries[system][n]['mrr'] += m['mrr']
                summaries[system][n]['cases'] += 1
            row[system + '_mrr'] = round(_case_metrics(retrieved, gold_ids)['mrr'], 3)
            row[system + '_ids'] = retrieved[:5]
        per_case.append(row)
        v_mrr = row['vector_mrr']
        h_mrr = row['hybrid_mrr']
        line = f"  {case['id']:12s} gold={len(gold_ids):2d}"
        line += f" | vector MRR={v_mrr:.3f} hybrid MRR={h_mrr:.3f}"
        print(line)

    n_cases = len(per_case) or 1
    print('\n=== 汇总（均值） ===')
    head = f"{'':8s} {'Recall@5':>10s} {'Recall@10':>11s} {'Recall@20':>11s}"
    head += f" {'Hit@5':>7s} {'Hit@10':>8s} {'Hit@20':>8s} {'MRR':>6s}"
    print(head)
    aggregated = {'mode': mode, 'cases': len(per_case)}
    for system in ('vector', 'hybrid'):
        vals = {}
        for n in TOP_K_VALUES:
            s = summaries[system][n]
            vals[f'recall@{n}'] = round(s['recall'] / n_cases, 4)
            vals[f'hit@{n}'] = round(s['hit'] / n_cases, 4)
            vals[f'mrr@{n}'] = round(s['mrr'] / n_cases, 4)
        aggregated[system] = vals
        print(f"{system:8s} {vals['recall@5']:>10.3f} {vals['recall@10']:>11.3f} "
              f"{vals['recall@20']:>11.3f} {vals['hit@5']:>7.2f} {vals['hit@10']:>8.2f} "
              f"{vals['hit@20']:>8.2f} {vals['mrr@5']:>6.3f}")

    REPORT_DIR.mkdir(exist_ok=True)
    report_path = REPORT_DIR / f'retrieval_eval_{mode}_{time.strftime("%Y%m%d_%H%M%S")}.json'
    report_path.write_text(
        json.dumps(
            {'summary': aggregated, 'per_case': per_case},
            ensure_ascii=False,
            indent=2,
        ),
        encoding='utf-8',
    )
    print(f'\n报告已保存: {report_path}')
    return {'summary': aggregated, 'per_case': per_case, 'report_path': str(report_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description='检索级评测：vector vs hybrid')
    parser.add_argument('--mode', choices=['fake', 'real'], default='real')
    parser.add_argument(
        '--gold-file',
        type=Path,
        default=GOLD_FILE,
        help='评测集路径（默认 retrieval_gold.json；支持 contains 词面定位用例）',
    )
    args = parser.parse_args()
    run(mode=args.mode, gold_file=args.gold_file)


if __name__ == '__main__':
    main()
