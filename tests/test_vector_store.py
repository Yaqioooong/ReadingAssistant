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
