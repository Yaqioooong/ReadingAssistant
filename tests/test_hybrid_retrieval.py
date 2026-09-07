"""混合检索测试：RRF 融合、BM25 索引、all_chunks 适配。"""

from types import SimpleNamespace

from langchain_core.embeddings import Embeddings

from reading_assistant.rag import HybridRetriever, Retriever, create_retriever, fuse_rrf
from reading_assistant.rag.bm25_index import BM25Index
from reading_assistant.storage.vector_store import (
    ChromaVectorStore,
    InMemoryVectorStore,
    StoredChunk,
)


class FakeEmbeddings(Embeddings):
    """固定向量：区分文档1与文档2的语义方向。"""

    def __init__(self, vector: list[float] | None = None) -> None:
        self._vector = vector or [1.0, 0.0]

    def embed_query(self, text: str) -> list[float]:
        return list(self._vector)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [list(self._vector) for _ in texts]


def _store() -> InMemoryVectorStore:
    store = InMemoryVectorStore()
    store.add(
        [
            StoredChunk(
                id='doc1-0',
                text='张三喜欢李四，两人在宴会上相遇。',
                metadata={'document_id': 1, 'chapter': '第一章', 'chapter_index': 0},
                embedding=[1.0, 0.0],
            ),
            StoredChunk(
                id='doc1-1',
                text='王五想要追求张三，朱六劝他放弃。',
                metadata={'document_id': 1, 'chapter': '第二章', 'chapter_index': 1},
                embedding=[0.8, 0.2],
            ),
            StoredChunk(
                id='doc2-0',
                text='少白公在一九六六年失去权力。',
                metadata={'document_id': 2, 'chapter': '第一章', 'chapter_index': 0},
                embedding=[0.0, 1.0],
            ),
            StoredChunk(
                id='doc2-1',
                text='舵手打倒少白公，是务实路线的选择。',
                metadata={'document_id': 2, 'chapter': '第二章', 'chapter_index': 1},
                embedding=[0.2, 0.8],
            ),
        ]
    )
    return store


class TestFuseRrf:
    def test_orders_by_reciprocal_rank(self) -> None:
        # b 在两路都靠前（rank1/rank2），应领先只在单路靠前的 x
        fused = fuse_rrf([['b', 'a', 'x'], ['b', 'x', 'a']], k=60)
        assert fused[0] == 'b'

    def test_union_and_empty_rankings(self) -> None:
        assert fuse_rrf([['a', 'b'], []], k=60)[:2] == ['a', 'b']
        assert fuse_rrf([[], []]) == []


class TestBM25Index:
    def test_build_and_lexical_query(self) -> None:
        store = _store()
        index = BM25Index(store)
        hits = dict(index.query('一九六六年 权力', top_k=5))
        assert 'doc2-0' in hits  # 词汇命中（稠密可能弱）

    def test_document_filter(self) -> None:
        store = _store()
        index = BM25Index(store)
        ids = [cid for cid, _ in index.query('张三', top_k=5, document_id=1)]
        assert all(cid.startswith('doc1') for cid in ids)
        assert 'doc2-1' not in ids

    def test_refresh_after_add(self) -> None:
        store = _store()
        index = BM25Index(store)
        store.add(
            [
                StoredChunk(
                    id='doc1-2',
                    text='张三在第三章放下了酒杯。',
                    metadata={'document_id': 1, 'chapter': '第三章', 'chapter_index': 2},
                    embedding=[1.0, 0.0],
                )
            ]
        )
        ids = [cid for cid, _ in index.query('酒杯', top_k=5)]
        assert 'doc1-2' in ids  # add 后版本自增，懒重建应感知

    def test_chunk_info(self) -> None:
        store = _store()
        index = BM25Index(store)
        info = index.chunk_info('doc2-0')
        assert info is not None and '少白公' in info.text


class TestAllChunks:
    def test_in_memory_returns_all(self) -> None:
        store = _store()
        chunks = store.all_chunks()
        assert {c.id for c in chunks} == {'doc1-0', 'doc1-1', 'doc2-0', 'doc2-1'}
        assert all(c.embedding is not None for c in chunks)

    def test_chroma_returns_all(self, tmp_path) -> None:
        store = ChromaVectorStore(persist_dir=str(tmp_path / 'c'), collection_name='test_coll')
        store.add(
            [
                StoredChunk(
                    id='a', text='内容一', metadata={'document_id': 1}, embedding=[1.0, 0.0]
                ),
                StoredChunk(
                    id='b', text='内容二', metadata={'document_id': 2}, embedding=[0.0, 1.0]
                ),
            ]
        )
        chunks = store.all_chunks()
        assert {c.id for c in chunks} == {'a', 'b'}
        assert {c.text for c in chunks} == {'内容一', '内容二'}


class TestHybridRetriever:
    def test_hybrid_finds_lexical_match(self) -> None:
        store = _store()
        retriever = HybridRetriever(store, FakeEmbeddings([0.5, 0.5]), pool_size=10, rrf_k=60)
        hits = retriever.retrieve('一九六六年 少白公 权力', top_k=3)
        ids = [h.chunk_id for h in hits]
        assert 'doc2-0' in ids
        assert all(h.score >= 0.45 for h in hits)

    def test_respects_document_id(self) -> None:
        store = _store()
        retriever = HybridRetriever(store, FakeEmbeddings([1.0, 0.0]), pool_size=10)
        hits = retriever.retrieve('张三', top_k=5, document_id=1)
        assert hits and all(h.document_id == 1 for h in hits)

    def test_min_score_gate_keeps_poor_semantic_out(self) -> None:
        # doc2-1 语义方向(0.2,0.8)与查询(1.0,0.0)余弦低，即使 BM25 命中也被闸门过滤
        store = _store()
        retriever = HybridRetriever(store, FakeEmbeddings([1.0, 0.0]), pool_size=20)
        hits = retriever.retrieve('舵手 少白公 务实', top_k=10)
        assert all(h.score >= 0.45 for h in hits)

    def test_metadata_preserved(self) -> None:
        store = _store()
        retriever = HybridRetriever(store, FakeEmbeddings([1.0, 0.0]), pool_size=10)
        hit = retriever.retrieve('张三 李四 宴会', top_k=1)[0]
        assert hit.chapter is not None and hit.text


class TestCreateRetriever:
    def test_default_mode_is_vector(self, monkeypatch) -> None:
        settings = SimpleNamespace(
            retrieval_mode='vector',
            cache_enabled=False,
            cache_max_entries=100,
        )
        monkeypatch.setattr('reading_assistant.rag.retriever.get_settings', lambda: settings)
        assert isinstance(create_retriever(_store(), FakeEmbeddings()), Retriever)

    def test_hybrid_mode(self, monkeypatch) -> None:
        settings = SimpleNamespace(
            retrieval_mode='hybrid',
            cache_enabled=False,
            cache_max_entries=100,
            hybrid_pool_size=50,
            rrf_k=60,
            bm25_tokenizer='jieba',
        )
        monkeypatch.setattr('reading_assistant.rag.retriever.get_settings', lambda: settings)
        assert isinstance(create_retriever(_store(), FakeEmbeddings()), HybridRetriever)
