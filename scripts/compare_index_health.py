"""对比两个向量索引的健康度（跨书库检索口径）。

为什么需要专门的口径
--------------------
``eval/run_retrieval_eval.py`` 走 ``retriever.retrieve(..., document_id=...)``，
**带 document_id 过滤** —— Chroma 会把搜索空间缩到单个文档，HNSW 图断裂被掩盖。
而 ``graph/qa.py`` 在 ``document_id is None``（跨书库提问）时**不带过滤**，
在全库范围内搜索，这才是暴露图退化的路径。

实测（2026-09-17，1520 chunks）：
    评测口径（带过滤）     ：修复前后均 14/14 → 无法检出退化
    跨书库口径（无过滤）   ：0.786 → 1.000  → 退化暴露且重建后修复

用法
----
    # 对比「当前索引」与「备份的旧索引」
    .venv/bin/python scripts/compare_index_health.py <旧索引目录>

    # 只体检当前索引
    .venv/bin/python scripts/compare_index_health.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# 两代 id 归一化：旧 doc{d}-{i} / 新 doc{d}-{hash8}-{i} → (d, i)
_CHUNK_ID_RE = re.compile(r'^doc(\d+)-(?:[0-9a-f]{8}-)?(\d+)$')


def _normalize(chunk_id: str) -> tuple[int, int] | None:
    match = _CHUNK_ID_RE.match(chunk_id or '')
    return (int(match.group(1)), int(match.group(2))) if match else None


class _ReadOnlyView:
    """把裸 Chroma collection 包成 Retriever 所需的只读 VectorStore 视图。"""

    def __init__(self, collection) -> None:
        self._collection = collection

    def query(self, embedding, top_k: int = 6, where: dict | None = None):
        from reading_assistant.storage.vector_store import SearchHit

        result = self._collection.query(
            query_embeddings=[embedding], n_results=top_k, where=where
        )
        ids = (result.get('ids') or [[]])[0]
        distances = (result.get('distances') or [[]])[0]
        metadatas = (result.get('metadatas') or [[]])[0]
        documents = (result.get('documents') or [[]])[0]
        return [
            SearchHit(
                id=chunk_id,
                # 统一到余弦：L2 + 归一化向量 → cos = 1 - dist/2
                score=1.0 - distances[i] / 2.0,
                metadata=metadatas[i] or {},
                text=documents[i] or '',
            )
            for i, chunk_id in enumerate(ids)
        ]

    def count(self) -> int:
        return self._collection.count()

    def all_chunks(self):
        return []

    def add(self, chunks) -> None:
        raise NotImplementedError('只读视图')

    def delete(self, document_id: int) -> None:
        raise NotImplementedError('只读视图')

    def delete_stale(self, document_id: int, keep_content_hash: str, keep_ids=None) -> int:
        raise NotImplementedError('只读视图')


def _evaluate(collection, cases, embedder, top_k: int) -> tuple[int, list[str]]:
    from reading_assistant.rag import Retriever

    retriever = Retriever(_ReadOnlyView(collection), embedder)
    hits = 0
    misses: list[str] = []
    for case in cases:
        got = {
            _normalize(hit.chunk_id)
            for hit in retriever.retrieve(case['question'], top_k=top_k)
        }
        want = {_normalize(cid) for cid in case.get('chunk_ids') or []}
        if got & want:
            hits += 1
        else:
            misses.append(case['id'])
    return hits, misses


def _open(path: Path, collection_name: str):
    import chromadb

    client = chromadb.PersistentClient(path=str(path))
    return client.get_collection(name=collection_name)


def main() -> None:
    parser = argparse.ArgumentParser(description='对比向量索引健康度（跨书库检索口径）')
    parser.add_argument('old_path', nargs='?', help='旧索引目录（不给则只体检当前索引）')
    parser.add_argument('--gold', default='eval/retrieval_gold.json')
    parser.add_argument('--top-k', type=int, default=20)
    args = parser.parse_args()

    from reading_assistant.config import get_settings
    from reading_assistant.runtime import get_embedding_model

    settings = get_settings()
    gold_path = Path(args.gold)
    if not gold_path.is_file():
        print(f'gold 集不存在：{gold_path}', file=sys.stderr)
        raise SystemExit(1)
    cases = json.loads(gold_path.read_text(encoding='utf-8')).get('cases', [])
    if not cases:
        print('gold 集为空', file=sys.stderr)
        raise SystemExit(1)

    embedder = get_embedding_model()
    print(f'口径：跨书库检索（无 document_id 过滤），top-{args.top_k}，{len(cases)} 例\n')

    if args.old_path:
        old = _open(Path(args.old_path), settings.chroma_collection_name)
        hits, misses = _evaluate(old, cases, embedder, args.top_k)
        print(f'  旧索引 {args.old_path}')
        print(f'    命中 {hits}/{len(cases)} = {hits / len(cases):.3f}   未命中={misses}')

    current = _open(Path(settings.chroma_persist_path), settings.chroma_collection_name)
    hits, misses = _evaluate(current, cases, embedder, args.top_k)
    print(f'  当前索引 {settings.chroma_persist_path}')
    print(f'    命中 {hits}/{len(cases)} = {hits / len(cases):.3f}   未命中={misses}')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f'执行失败：{exc}', file=sys.stderr)
        raise SystemExit(1) from exc
