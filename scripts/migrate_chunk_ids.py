"""chunk id 迁移：位置型 → 内容寻址 + 版本化（**不重新 embedding**）。

旧 id：``doc{document_id}-{index}``
新 id：``doc{document_id}-{content_hash[:8]}-{index}``，并给 metadata 补 ``content_hash``。

做法：读全量 chunk（含已持久化的向量）→ 用 ``documents.content_hash`` 计算新
id → upsert 新 id（复用原向量，**不调用 embedding API**）→ 删除旧 id。

删除旧版本走 :meth:`VectorStore.delete_stale`：新版本写入后再删「``content_hash``
不等于当前版本」的块（历史块缺该键，也会被清理）。脚本幂等：再次运行时会发现
``new_id == old_id`` 而跳过。

用法（仓库根目录）：
    .venv/bin/python scripts/migrate_chunk_ids.py            # dry-run
    .venv/bin/python scripts/migrate_chunk_ids.py --apply    # 执行
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from sqlalchemy import select

from reading_assistant.rag.chunking import make_chunk_id
from reading_assistant.storage import create_db_engine, create_session_factory
from reading_assistant.storage.models import Document, QaCacheEntry
from reading_assistant.storage.vector_store import StoredChunk, create_vector_store


def _chunk_index(chunk_id: str) -> int:
    """从旧 id（``doc{doc}-{index}`` 或已是新格式）尾部解析 index。"""
    tail = chunk_id.rsplit('-', 1)[-1]
    return int(tail) if tail.isdigit() else -1


def _document_hashes() -> dict[int, str]:
    """读取 documents.id → content_hash；关系库不可用时退出。"""
    engine = create_db_engine()
    factory = create_session_factory(engine)
    with factory() as session:
        rows = session.execute(select(Document.id, Document.content_hash)).all()
        return {doc_id: content_hash for doc_id, content_hash in rows}



_NEW_CHUNK_ID_RE = re.compile(r'^doc\d+-[0-9a-f]{8}-\d+$')
_OLD_CHUNK_ID_RE = re.compile(r'^doc(\d+)-(\d+)$')


def _migrate_cache_citations(
    hashes: dict[int, str], apply: bool
) -> tuple[int, int]:
    """同步迁移问答缓存里 citations 的 chunk_id 引用。

    缓存条目绑定 ``content_hash``，但 citations 里嵌的是**当时的 chunk id**。
    chunk id 从位置型改为版本化后，旧引用会指向已不存在的块（悬空引用），
    使答案里的「跳转到出处」失效 —— 故派生引用必须与 chunk 本体一并迁移。

    返回 ``(受影响记录数, 受影响引用数)``。已在用新格式的引用会被跳过（幂等）。
    """
    engine = create_db_engine()
    factory = create_session_factory(engine)
    changed_entries = 0
    changed_citations = 0
    with factory() as session:
        for entry in session.execute(select(QaCacheEntry)).scalars().all():
            citations = entry.citations or []
            if not isinstance(citations, list):
                continue
            updated = []
            touched = False
            for citation in citations:
                if not isinstance(citation, dict):
                    updated.append(citation)
                    continue
                chunk_id = citation.get('chunk_id') or ''
                match = _OLD_CHUNK_ID_RE.match(chunk_id)
                if _NEW_CHUNK_ID_RE.match(chunk_id) or match is None:
                    updated.append(citation)
                    continue
                document_id, index = (int(group) for group in match.groups())
                content_hash = hashes.get(document_id)
                if content_hash is None:
                    updated.append(citation)
                    continue
                updated.append(
                    {**citation, 'chunk_id': make_chunk_id(document_id, content_hash, index)}
                )
                touched = True
                changed_citations += 1
            if touched:
                changed_entries += 1
                if apply:
                    entry.citations = updated
        if apply:
            session.commit()
    return changed_entries, changed_citations



# 活跃评测数据：含 gold / 负例的 chunk id 引用，必须与库同步。
# 刻意不含 eval/reports/*：历史报告是过去结果的快照，应保持原样。
_EVAL_ARTIFACTS = (
    'eval/retrieval_gold.json',
    'eval/retrieval_hard_neg.json',
)


def _rewrite_ids_in_json(node, hashes: dict[int, str]) -> tuple[object, int]:
    """递归替换 JSON 结构中所有旧格式 chunk id 字符串。

    只替换**精确匹配** ``doc{did}-{idx}`` 且 ``did`` 能在文档表里查到的字符串，
    避免误伤正文文本里的相似片段。
    """
    changed = 0
    if isinstance(node, dict):
        out = {}
        for key, value in node.items():
            new_value, delta = _rewrite_ids_in_json(value, hashes)
            out[key] = new_value
            changed += delta
        return out, changed
    if isinstance(node, list):
        items = []
        for value in node:
            new_value, delta = _rewrite_ids_in_json(value, hashes)
            items.append(new_value)
            changed += delta
        return items, changed
    if isinstance(node, str):
        match = _OLD_CHUNK_ID_RE.match(node)
        if match is not None and not _NEW_CHUNK_ID_RE.match(node):
            document_id, index = (int(group) for group in match.groups())
            content_hash = hashes.get(document_id)
            if content_hash is not None:
                return make_chunk_id(document_id, content_hash, index), 1
    return node, changed


def _migrate_eval_artifacts(hashes: dict[int, str], apply: bool) -> tuple[int, int]:
    """迁移评测集里的 chunk id 引用；返回 ``(涉及文件数, 替换条数)``。"""
    files = 0
    total = 0
    for relative in _EVAL_ARTIFACTS:
        path = Path(relative)
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError):
            continue
        updated, changed = _rewrite_ids_in_json(payload, hashes)
        if changed:
            files += 1
            total += changed
            if apply:
                path.write_text(
                    json.dumps(updated, ensure_ascii=False, indent=2) + '\n',
                    encoding='utf-8',
                )
    return files, total


def main() -> None:
    parser = argparse.ArgumentParser(description='迁移 chunk id 为内容寻址 + 版本化')
    parser.add_argument('--apply', action='store_true', help='真正执行（默认 dry-run）')
    args = parser.parse_args()

    hashes = _document_hashes()
    store = create_vector_store()
    chunks = store.all_chunks()

    new_chunks: list[StoredChunk] = []
    affected: dict[int, str] = {}  # document_id -> 当前版本 content_hash
    skipped = 0
    for chunk in chunks:
        document_id = (chunk.metadata or {}).get('document_id')
        content_hash = hashes.get(document_id) if isinstance(document_id, int) else None
        index = _chunk_index(chunk.id)
        if content_hash is None or index < 0:
            skipped += 1
            continue
        affected[document_id] = content_hash
        new_id = make_chunk_id(document_id, content_hash, index)
        if new_id == chunk.id:
            continue  # 已迁移
        new_chunks.append(
            StoredChunk(
                id=new_id,
                text=chunk.text,
                metadata={**chunk.metadata, 'content_hash': content_hash},
                embedding=chunk.embedding,  # 复用原向量，不重新 embedding
            )
        )

    print(f'chunk 总数={len(chunks)} 待迁移={len(new_chunks)} '
          f'涉及文档={len(affected)} 跳过(未知文档/无法解析)={skipped}')
    for chunk in new_chunks[:5]:
        print(f'  示例：{chunk.metadata.get("document_id")} -> {chunk.id}')

    if not args.apply:
        entries, citations = _migrate_cache_citations(hashes, apply=False)
        print(f'缓存引用：{entries} 条记录 / {citations} 个引用待迁移')
        files, count = _migrate_eval_artifacts(hashes, apply=False)
        print(f'评测集引用：{files} 个文件 / {count} 个 id 待迁移')
        print('dry-run：未做任何修改（加 --apply 执行）')
        return

    # 顺序契约：先 upsert 新版本 → 再 delete_stale 旧版本
    if new_chunks:
        store.add(new_chunks)
    removed = 0
    for document_id, content_hash in affected.items():
        removed += store.delete_stale(document_id, content_hash)
    print(f'完成：写入 {len(new_chunks)} 个新 id，删除 {removed} 个旧版本块')

    # 派生引用：问答缓存 citations 里的 chunk_id 必须一并迁移
    entries, citations = _migrate_cache_citations(hashes, apply=True)
    print(f'缓存引用：{entries} 条记录 / {citations} 个引用已迁移')

    files, count = _migrate_eval_artifacts(hashes, apply=True)
    print(f'评测集引用：{files} 个文件 / {count} 个 id 已迁移')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f'执行失败：{exc}', file=sys.stderr)
        raise SystemExit(1) from exc
