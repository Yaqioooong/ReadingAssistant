"""向量库查看工具：分块总量/文档分布、抽样内容、单块详情、相似度检索。

用法（仓库根目录执行）：
    uv run python scripts/inspect_vector_store.py stats              # 总量 + 按文档分布(含书名)
    uv run python scripts/inspect_vector_store.py peek -n 5          # 抽样看分块内容
    uv run python scripts/inspect_vector_store.py peek -n 5 --doc 3  # 只看某文档的分块
    uv run python scripts/inspect_vector_store.py get doc1-0         # 单块全文 + 向量信息
    uv run python scripts/inspect_vector_store.py search "罗辑的咒语" -k 5
        # 上一行为向量检索(需 embedding API)

裸看 SQLite（Chroma 持久化文件）：
    sqlite3 rag/chroma_db/chroma.sqlite3 "select count(*) from embeddings;"
    sqlite3 rag/chroma_db/chroma.sqlite3 \
      "select string_value, count(*) from embedding_metadata where key='document_id' group by 1;"
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter

from reading_assistant.config import get_settings
from reading_assistant.storage.vector_store import create_vector_store


def _doc_titles() -> dict[int, str]:
    """尽力从关系库取文档 id → 书名；库不可用时退化为空表。"""
    try:
        from sqlalchemy import select

        from reading_assistant.storage import (
            create_db_engine,
            create_session_factory,
        )
        from reading_assistant.storage.models import Document

        engine = create_db_engine()
        factory = create_session_factory(engine)
        with factory() as session:
            return {
                doc.id: doc.title or doc.filename
                for doc in session.scalars(select(Document))
            }
    except Exception:  # noqa: BLE001 关系库未启动等情况
        return {}


def cmd_stats(_: argparse.Namespace) -> None:
    settings = get_settings()
    store = create_vector_store()
    chunks = store.all_chunks()
    titles = _doc_titles()
    print(f'后端={settings.vector_store_backend} 集合={settings.chroma_collection_name}')
    print(f'持久化目录={settings.chroma_persist_dir}')
    print(f'分块总数={store.count()}（全量拉取={len(chunks)}）')
    dist = Counter(str(c.metadata.get('document_id')) for c in chunks)
    print('按文档分布：')
    for doc_id, num in dist.most_common():
        title = titles.get(int(doc_id)) if doc_id.isdigit() else None
        print(f'  document_id={doc_id:<4} 分块={num:<5} {f"《{title}》" if title else ""}')


def cmd_peek(args: argparse.Namespace) -> None:
    store = create_vector_store()
    titles = _doc_titles()
    chunks = store.all_chunks()
    if args.doc is not None:
        chunks = [c for c in chunks if c.metadata.get('document_id') == args.doc]
    if not chunks:
        print('没有匹配的分块')
        return
    for chunk in chunks[: args.n]:
        doc_id = chunk.metadata.get('document_id')
        title = titles.get(doc_id) if isinstance(doc_id, int) else None
        preview = chunk.text.replace('\n', ' ')[:120]
        print(f'[{chunk.id}] doc={doc_id}'
              f'{f"《{title}》" if title else ""} '
              f"chapter={chunk.metadata.get('chapter')} block={chunk.metadata.get('block_type')}")
        print(f'    {preview}')


def cmd_get(args: argparse.Namespace) -> None:
    store = create_vector_store()
    for chunk in store.all_chunks():
        if chunk.id == args.chunk_id:
            embedding = getattr(chunk, 'embedding', None)
            print(f'id={chunk.id}')
            print(f'metadata={chunk.metadata}')
            print(f'向量维度={len(embedding) if embedding else "—"} '
                  f'前8维={[round(float(x), 4) for x in (embedding or [])[:8]]}')
            print('全文：')
            print(chunk.text)
            return
    print(f'未找到分块：{args.chunk_id}')


def cmd_search(args: argparse.Namespace) -> None:
    from reading_assistant.model.factory import get_embedding_model

    store = create_vector_store()
    embedding = get_embedding_model().embed_query(args.text)
    hits = store.query(embedding, top_k=args.k)
    print(f'query="{args.text}" top{args.k}（chroma 后端 score=1/(1+距离)，仅作排序参考）')
    for rank, hit in enumerate(hits, start=1):
        preview = hit.text.replace('\n', ' ')[:100]
        print(f'{rank}. score={hit.score:.4f} id={hit.id} meta={hit.metadata}')
        print(f'   {preview}')


def main() -> None:
    parser = argparse.ArgumentParser(description='向量库查看工具')
    sub = parser.add_subparsers(dest='cmd', required=True)

    sub.add_parser('stats', help='分块总量 + 按文档分布')

    p_peek = sub.add_parser('peek', help='抽样查看分块内容')
    p_peek.add_argument('-n', type=int, default=5, help='条数（默认 5）')
    p_peek.add_argument('--doc', type=int, default=None, help='只看指定 document_id')

    p_get = sub.add_parser('get', help='查看单个分块详情')
    p_get.add_argument('chunk_id')

    p_search = sub.add_parser('search', help='向量检索（需要 embedding API）')
    p_search.add_argument('text')
    p_search.add_argument('-k', type=int, default=5)

    args = parser.parse_args()
    handlers = {'stats': cmd_stats, 'peek': cmd_peek, 'get': cmd_get, 'search': cmd_search}
    try:
        handlers[args.cmd](args)
    except Exception as exc:  # noqa: BLE001
        print(f'执行失败：{exc}', file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == '__main__':
    main()
