"""索引完整性：版本化 chunk id 与孤儿向量清理（P1-2）。

覆盖 2026-09-17 修复的缺陷：reindex 只 upsert 不 delete，重切后 chunk 数
变少则尾部旧向量残留 —— 检索会返回「幽灵 chunk」（元数据正常、无法区分）。

对应实测故障（见 docs/fix-backlog.md）：
  doc5 chunk_size 800→2000 重切，181→66 块，尾部 115 个旧向量残留；
  用孤儿原文检索 top-6，6 条里 5 条是幽灵，元数据完全正常、无法区分。

> 分数尺度（P2-3）与 RRF 不变量（P2-2）的覆盖在同一个文件的后续提交中加入。
"""

from __future__ import annotations

from langchain_core.embeddings import Embeddings

from reading_assistant.rag import make_chunk_id
from reading_assistant.storage.vector_store import (
    ChromaVectorStore,
    InMemoryVectorStore,
    StoredChunk,
)


class FakeEmbeddings(Embeddings):
    """恒定向量，避免测试依赖真实模型。"""

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


def _chunk(
    cid: str,
    doc_id: int,
    content_hash: str | None,
    text: str = '片段',
    embedding: list[float] | None = None,
) -> StoredChunk:
    """构造 chunk；content_hash=None 模拟迁移前的位置型 chunk（缺该 metadata 键）。"""
    meta: dict = {'document_id': doc_id}
    if content_hash is not None:
        meta['content_hash'] = content_hash
    return StoredChunk(
        id=cid, text=text, metadata=meta, embedding=embedding or [1.0, 0.0]
    )


# ─────────────────────── 版本化 chunk id ───────────────────────


class TestMakeChunkId:
    def test_format_is_document_version_index(self) -> None:
        assert make_chunk_id(7, 'a' * 64, 3) == 'doc7-aaaaaaaa-3'

    def test_same_index_different_version_never_collides(self) -> None:
        """核心不变量：同一位置、不同版本必须落在不同 id 命名空间。

        这是「先写新版本、再删旧版本」能够安全生效的前提。
        """
        assert make_chunk_id(7, 'a' * 64, 3) != make_chunk_id(7, 'b' * 64, 3)

    def test_empty_hash_still_distinct_from_legacy_positional_id(self) -> None:
        """历史位置型 id 为 doc7-3；即便 hash 缺失也不能与之撞车，
        否则新写入会静默覆盖旧 chunk（正是 P1-2 的根因）。"""
        assert make_chunk_id(7, '', 3) != 'doc7-3'

    def test_none_hash_does_not_raise(self) -> None:
        assert make_chunk_id(7, None, 0).startswith('doc7-')

    def test_deterministic(self) -> None:
        """同样输入必须产出同样 id —— 幂等重跑的基础。"""
        assert make_chunk_id(7, 'abc12345', 9) == make_chunk_id(7, 'abc12345', 9)


# ─────────────────────── delete_stale（内存后端） ───────────────────────


class TestDeleteStaleInMemory:
    def test_removes_previous_version_only(self) -> None:
        store = InMemoryVectorStore()
        store.add([_chunk(f'doc1-oldoldol-{i}', 1, 'oldoldol') for i in range(3)])
        store.add([_chunk(f'doc1-newnewne-{i}', 1, 'newnewne') for i in range(2)])

        removed = store.delete_stale(1, 'newnewne')

        assert removed == 3
        assert {c.id for c in store.all_chunks()} == {
            'doc1-newnewne-0',
            'doc1-newnewne-1',
        }

    def test_legacy_chunks_without_content_hash_are_removed(self) -> None:
        """迁移前的位置型 chunk 缺 content_hash 键 —— 必须也能被清掉。

        这正是不使用 Chroma ``$ne`` where 过滤的原因：对缺失键的行为不可靠。
        """
        store = InMemoryVectorStore()
        store.add([_chunk('doc1-0', 1, None), _chunk('doc1-1', 1, None)])
        store.add([_chunk('doc1-newnewne-0', 1, 'newnewne')])

        removed = store.delete_stale(1, 'newnewne', keep_ids=['doc1-newnewne-0'])

        assert removed == 2
        assert {c.id for c in store.all_chunks()} == {'doc1-newnewne-0'}

    def test_other_documents_untouched(self) -> None:
        """清理必须按 document_id 隔离，不能误伤别的书。"""
        store = InMemoryVectorStore()
        store.add([_chunk('doc1-oldoldol-0', 1, 'oldoldol')])
        store.add([_chunk('doc2-otheroth-0', 2, 'otheroth')])
        store.add([_chunk('doc1-newnewne-0', 1, 'newnewne')])

        store.delete_stale(1, 'newnewne')

        assert {c.id for c in store.all_chunks()} == {
            'doc1-newnewne-0',
            'doc2-otheroth-0',
        }

    def test_idempotent(self) -> None:
        store = InMemoryVectorStore()
        store.add([_chunk('doc1-newnewne-0', 1, 'newnewne')])

        assert store.delete_stale(1, 'newnewne') == 0
        assert store.delete_stale(1, 'newnewne') == 0
        assert store.count() == 1

    def test_shrinking_rechunk_leaves_no_ghost(self) -> None:
        """复现原始故障规模：181 → 66 块，尾部 115 个旧向量必须被清掉。"""
        store = InMemoryVectorStore()
        old_count, new_count = 181, 66
        store.add(
            [_chunk(f'doc5-oldoldol-{i}', 5, 'oldoldol') for i in range(old_count)]
        )
        new_ids = [f'doc5-newnewne-{i}' for i in range(new_count)]
        store.add([_chunk(cid, 5, 'newnewne') for cid in new_ids])

        removed = store.delete_stale(5, 'newnewne', keep_ids=new_ids)

        assert removed == old_count
        assert store.count() == new_count

    def test_no_ghost_is_retrievable_after_cleanup(self) -> None:
        """清理后，旧边界之外的 id 不应再出现在检索结果里。"""
        store = InMemoryVectorStore()
        store.add(
            [_chunk(f'doc5-oldoldol-{i}', 5, 'oldoldol', text=f'旧文本{i}') for i in range(10)]
        )
        new_ids = [f'doc5-newnewne-{i}' for i in range(3)]
        store.add(
            [_chunk(cid, 5, 'newnewne', text=f'新文本{i}') for i, cid in enumerate(new_ids)]
        )

        store.delete_stale(5, 'newnewne', keep_ids=new_ids)
        hit_ids = {h.id for h in store.query([1.0, 0.0], top_k=20)}

        assert hit_ids == set(new_ids)

    def test_keep_ids_takes_precedence_over_hash(self) -> None:
        """显式 keep_ids 优先于 content_hash 匹配。"""
        store = InMemoryVectorStore()
        # 两个 chunk 的 hash 都与 keep_hash 不同，但其中一个在 keep_ids 里
        store.add([_chunk('doc1-xxxxxxx1-0', 1, 'xxxxxxx1')])
        store.add([_chunk('doc1-yyyyyyy1-0', 1, 'yyyyyyy1')])

        removed = store.delete_stale(1, 'zzzzzzz1', keep_ids=['doc1-xxxxxxx1-0'])

        assert removed == 1
        assert {c.id for c in store.all_chunks()} == {'doc1-xxxxxxx1-0'}


# ─────────────────────── delete_stale（Chroma 后端） ───────────────────────


class TestDeleteStaleChroma:
    def test_removes_previous_version(self, tmp_path) -> None:
        store = ChromaVectorStore(
            persist_dir=str(tmp_path), collection_name='test_delete_stale'
        )
        store.add([_chunk(f'doc1-oldoldol-{i}', 1, 'oldoldol') for i in range(3)])
        store.add([_chunk('doc1-newnewne-0', 1, 'newnewne')])
        assert store.count() == 4

        removed = store.delete_stale(1, 'newnewne', keep_ids=['doc1-newnewne-0'])

        assert removed == 3
        assert store.count() == 1
        assert {c.id for c in store.all_chunks()} == {'doc1-newnewne-0'}
