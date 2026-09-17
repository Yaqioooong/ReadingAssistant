"""P2 向量库适配器测试：内存实现 + Chroma 本地持久化实现。"""

from pathlib import Path
from types import SimpleNamespace

from reading_assistant.storage.vector_store import (
    ChromaVectorStore,
    InMemoryVectorStore,
    StoredChunk,
    create_vector_store,
)


def _chunks() -> list[StoredChunk]:
    return [
        StoredChunk(
            id='c1', text='人物张三出场', metadata={'document_id': 1}, embedding=[1.0, 0.0]
        ),
        StoredChunk(id='c2', text='事件发展', metadata={'document_id': 1}, embedding=[0.0, 1.0]),
        StoredChunk(id='c3', text='另一本书', metadata={'document_id': 2}, embedding=[1.0, 1.0]),
    ]


class TestInMemoryVectorStore:
    def test_add_query_and_filter(self) -> None:
        store = InMemoryVectorStore()
        store.add(_chunks())

        hits = store.query([1.0, 0.0], top_k=2)
        assert [hit.id for hit in hits] == ['c1', 'c3']

        filtered = store.query([1.0, 0.0], top_k=5, where={'document_id': 2})
        assert [hit.id for hit in filtered] == ['c3']
        assert store.count() == 3

    def test_delete_removes_only_matching_document(self) -> None:
        store = InMemoryVectorStore()
        store.add(_chunks())

        store.delete(1)

        assert store.count() == 1
        hits = store.query([1.0, 0.0], top_k=5)
        assert [hit.id for hit in hits] == ['c3']

    def test_delete_missing_document_is_noop(self) -> None:
        store = InMemoryVectorStore()
        store.add(_chunks())

        store.delete(999)

        assert store.count() == 3
        assert len(store.query([1.0, 0.0], top_k=5)) == 3


class TestChromaVectorStore:
    def test_add_query_and_count(self, tmp_path: Path) -> None:
        store = ChromaVectorStore(
            persist_dir=str(tmp_path / 'chroma'),
            collection_name='test_collection',
        )
        store.add(_chunks())

        hits = store.query([1.0, 0.0], top_k=2)
        assert len(hits) == 2
        assert hits[0].id == 'c1'
        assert hits[0].metadata['document_id'] == 1
        assert store.count() == 3

    def test_reopen_persists_data(self, tmp_path: Path) -> None:
        persist_dir = str(tmp_path / 'chroma')
        ChromaVectorStore(persist_dir=persist_dir, collection_name='persist_coll').add(_chunks())

        reopened = ChromaVectorStore(persist_dir=persist_dir, collection_name='persist_coll')
        assert reopened.count() == 3

    def test_delete_removes_only_matching_document(self, tmp_path: Path) -> None:
        store = ChromaVectorStore(
            persist_dir=str(tmp_path / 'chroma'), collection_name='del_coll'
        )
        store.add(_chunks())

        store.delete(1)

        assert store.count() == 1
        hits = store.query([1.0, 0.0], top_k=5)
        assert [hit.id for hit in hits] == ['c3']


class TestFactory:
    def test_create_memory_backend(self) -> None:
        assert isinstance(create_vector_store('memory'), InMemoryVectorStore)

    def test_create_chroma_backend_uses_configured_path(self, tmp_path: Path, monkeypatch) -> None:
        settings = SimpleNamespace(
            chroma_persist_path=tmp_path / 'chroma', chroma_collection_name='test'
        )
        monkeypatch.setattr('reading_assistant.storage.vector_store.get_settings', lambda: settings)
        assert isinstance(create_vector_store('chroma'), ChromaVectorStore)


class TestScoreTypeContract:
    """分数类型契约：检索分数必须是内建 ``float``。

    2026-09-17 实测踩坑：Chroma 的 distance 是 **numpy 标量**，换算成余弦后仍是
    ``numpy.float32/float64``，混进 QA state 的 ``chunks[*]['score']`` 后，
    LangGraph checkpoint 用 ormsgpack 序列化会抛
    ``Type is not msgpack serializable: numpy.float64``
    → 接口 500，且只在该片段落入最终 top-k 时间歇发生（n_results 小时 Chroma
    反而返回内建 float，所以极难复现）。

    eval 扩集后每个新题都走真实检索路径，这个坑直接把整套端到端评测打死，
    才被暴露出来。这里把它钉在源头。
    """

    def _store(self, space: str) -> ChromaVectorStore:
        # 不连库：只验证换算函数，__new__ 绕过 __init__ 里的 Chroma 连接
        store = ChromaVectorStore.__new__(ChromaVectorStore)
        store._space = space
        return store

    def test_l2_distance_with_numpy_input_returns_builtin_float(self) -> None:
        import numpy as np

        score = self._store('l2')._distance_to_cosine(np.float64(0.5))
        assert type(score) is float

    def test_cosine_distance_with_numpy_float32_returns_builtin_float(self) -> None:
        import numpy as np

        score = self._store('cosine')._distance_to_cosine(np.float32(0.25))
        assert type(score) is float

    def test_score_is_msgpack_serializable(self) -> None:
        """最贴近故障现场的断言：分数要能被 checkpoint 的序列化器处理。"""
        import numpy as np
        import ormsgpack

        score = self._store('l2')._distance_to_cosine(np.float64(0.3))
        assert ormsgpack.packb({'score': score})

    def test_score_is_clamped_to_valid_cosine_range(self) -> None:
        store = self._store('l2')
        assert store._distance_to_cosine(0.0) == 1.0
        assert store._distance_to_cosine(4.0) == -1.0

    def test_all_chunks_embeddings_are_builtin_floats(self, tmp_path: Path) -> None:
        """all_chunks() 的向量必须收敛为内建 float。

        它喂给 BM25 索引，稀疏路用这些向量现算余弦补分数；numpy 标量会一路
        传到 QA state 的 ``score``，让 checkpoint 序列化抛 TypeError。
        """
        store = ChromaVectorStore(
            persist_dir=str(tmp_path / 'chroma_float'),
            collection_name='float_coll',
        )
        store.add(_chunks())
        for chunk in store.all_chunks():
            for value in chunk.embedding or []:
                assert type(value) is float
