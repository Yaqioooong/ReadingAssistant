"""索引完整性：版本化 chunk id、孤儿向量清理、分数尺度、RRF 不变量。

覆盖 2026-09-17 修复的三处缺陷：
- P1-2 孤儿向量：内容寻址 + 版本化 id + delete_stale
- P2-3 分数尺度：dense 与 bm25 统一到余弦（原先 0.45 跨两种量纲）
- P2-2 RRF 不变量：k > pool - 2，否则融合语义静默翻转

对应实测故障（见 docs/fix-backlog.md）：
  doc5 chunk_size 800→2000 重切，181→66 块，尾部 115 个旧向量残留；
  用孤儿原文检索 top-6，6 条里 5 条是幽灵，元数据完全正常、无法区分。
"""

from __future__ import annotations

import pytest
from langchain_core.embeddings import Embeddings

from reading_assistant.rag import HybridRetriever, make_chunk_id
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


# ─────────────────────── 分数尺度统一（P2-3） ───────────────────────


class TestDistanceToCosine:
    """Chroma 返回距离，本项目统一换算成余弦后再套 min_score。"""

    def test_l2_space_matches_measured_conversion(self, tmp_path) -> None:
        """实测基准：归一化向量的 L2 距离 0.7134 ↔ 余弦 0.6433。

        （旧公式 1/(1+dist) 会给出 0.5837，与 BM25 支路的余弦不同量纲。）
        """
        store = ChromaVectorStore(
            persist_dir=str(tmp_path), collection_name='test_score_l2'
        )
        assert store._distance_to_cosine(0.7134) == pytest.approx(0.6433, abs=1e-4)

    def test_l2_zero_distance_is_max_cosine(self, tmp_path) -> None:
        store = ChromaVectorStore(
            persist_dir=str(tmp_path), collection_name='test_score_zero'
        )
        assert store._distance_to_cosine(0.0) == pytest.approx(1.0)

    def test_l2_orthogonal_is_zero_cosine(self, tmp_path) -> None:
        """归一化正交向量 L2 距离 = 2 → 余弦 0（旧公式会给 0.333）。"""
        store = ChromaVectorStore(
            persist_dir=str(tmp_path), collection_name='test_score_ortho'
        )
        assert store._distance_to_cosine(2.0) == pytest.approx(0.0, abs=1e-6)

    def test_cosine_space_uses_unit_distance(self, tmp_path) -> None:
        store = ChromaVectorStore(
            persist_dir=str(tmp_path), collection_name='test_score_cos'
        )
        store._space = 'cosine'
        assert store._distance_to_cosine(0.3) == pytest.approx(0.7)

    def test_result_clamped_to_valid_cosine_range(self, tmp_path) -> None:
        store = ChromaVectorStore(
            persist_dir=str(tmp_path), collection_name='test_score_clamp'
        )
        assert store._distance_to_cosine(99.0) == pytest.approx(-1.0)

    def test_query_returns_cosine_scale(self, tmp_path) -> None:
        """端到端：查询分数必须是余弦，而不是 1/(1+距离)。"""
        store = ChromaVectorStore(
            persist_dir=str(tmp_path), collection_name='test_query_scale'
        )
        store.add([_chunk('doc1-aaaaaaa1-0', 1, 'aaaaaaa1', embedding=[1.0, 0.0])])

        hit = store.query([1.0, 0.0], top_k=1)[0]

        assert hit.score == pytest.approx(1.0, abs=1e-5)

    def test_query_orthogonal_scores_zero_not_one_third(self, tmp_path) -> None:
        """回归：正交向量在余弦尺度下应是 0；旧公式会给 1/(1+2)=0.333。"""
        store = ChromaVectorStore(
            persist_dir=str(tmp_path), collection_name='test_query_ortho'
        )
        store.add([_chunk('doc1-aaaaaaa1-0', 1, 'aaaaaaa1', embedding=[0.0, 1.0])])

        hit = store.query([1.0, 0.0], top_k=1)[0]

        assert hit.score == pytest.approx(0.0, abs=1e-5)


# ─────────────────────── RRF 不变量（P2-2） ───────────────────────


class TestRrfInvariant:
    """双路最差分 2/(k+pool) > 单路最优分 1/(k+1) ⟺ k > pool - 2。

    不满足时「双路共现优先」的语义静默翻转：单路 rank1 不再必然输给双路共现。
    """

    @pytest.mark.parametrize(
        ('pool', 'k'),
        [(50, 1), (50, 10), (50, 47), (50, 48), (50, 49), (50, 60), (50, 200), (70, 60)],
    )
    def test_theorem_dual_beats_single_iff_k_gt_pool_minus_2(
        self, pool: int, k: int
    ) -> None:
        dual_worst = 2 / (k + pool)
        single_best = 1 / (k + 1)
        assert (dual_worst > single_best) == (k > pool - 2)

    def test_boundary_is_exclusive(self) -> None:
        """k == pool - 2 时不成立（是临界点，不是满足点）。"""
        pool = 50
        k = pool - 2  # 48
        assert 2 / (k + pool) == pytest.approx(1 / (k + 1))

    def test_default_config_satisfies_invariant(self, settings) -> None:
        assert settings.rrf_k > settings.hybrid_pool_size - 2, (
            'config/chroma.yml 的 rrf_k 与 hybrid_pool_size 组合破坏了 RRF 不变量'
        )

    def test_broken_invariant_emits_warning(self, monkeypatch) -> None:
        """pool=70 且 rrf_k=60 → 60 <= 68 → 必须告警。"""
        warnings: list[str] = []

        class _Recorder:
            def warning(self, *args, **kwargs) -> None:
                warnings.append(str(args[0]))

        monkeypatch.setattr('reading_assistant.rag.retriever.logger', _Recorder())
        store = InMemoryVectorStore()
        store.add([_chunk('doc1-aaaaaaa1-0', 1, 'aaaaaaa1')])

        HybridRetriever(store, FakeEmbeddings(), pool_size=70, rrf_k=60)

        assert warnings, 'pool=70 / rrf_k=60 破坏了不变量，必须告警'

    def test_sound_config_emits_no_warning(self, monkeypatch) -> None:
        warnings: list[str] = []

        class _Recorder:
            def warning(self, *args, **kwargs) -> None:
                warnings.append(str(args[0]))

        monkeypatch.setattr('reading_assistant.rag.retriever.logger', _Recorder())
        store = InMemoryVectorStore()
        store.add([_chunk('doc1-aaaaaaa1-0', 1, 'aaaaaaa1')])

        HybridRetriever(store, FakeEmbeddings(), pool_size=50, rrf_k=60)

        assert not warnings, '默认组合（k=60 > 48）不应告警'

    def test_fuse_rrf_ranks_dual_hit_first_under_sound_config(self) -> None:
        """在满足不变量的配置下，双路共现必须排在单路强命中之前。"""
        from reading_assistant.rag import fuse_rrf

        dense = ['a', 'b', 'x']      # x 在 dense 排第 3
        bm25 = ['c', 'd', 'e', 'x']  # x 在 bm25 排第 4
        fused = fuse_rrf([dense, bm25], k=60)

        assert fused[0] == 'x', '双路共现的 x 应排第一'
