"""上下文管理与检索修复测试：标识符兜底、BM25 词面等价。"""

from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient
from langchain_core.embeddings import Embeddings

from reading_assistant.api import create_app
from reading_assistant.rag import HybridRetriever, Retriever
from reading_assistant.rag.bm25_index import BM25Index
from reading_assistant.storage import (
    create_db_engine,
    create_session_factory,
    init_db,
)
from reading_assistant.storage.vector_store import InMemoryVectorStore, StoredChunk


class OrthoEmb(Embeddings):
    """查询沿 [1,0] 方向；与表格块 [0,1] 正交 → 语义通道必然全灭，逼出兜底。"""

    def embed_query(self, text):
        return [1.0, 0.0]

    def embed_documents(self, texts):
        return [[1.0, 0.0] for _ in texts]


class FakeLLM:
    def invoke(self, prompt):
        return SimpleNamespace(content='P-002 的上市时间是 2023 年 9 月 22 日。')


def _store_with_table() -> InMemoryVectorStore:
    store = InMemoryVectorStore()
    store.add(
        [
            StoredChunk(
                id='doc1-0',
                text='| 产品ID | 上市时间 |\\n| --- | --- |\\n| P-002 | 2023-09-22 |',
                metadata={'document_id': 1, 'chapter': '正文', 'chapter_index': 0},
                embedding=[0.0, 1.0],
            ),
            StoredChunk(
                id='doc2-0',
                text='张三喜欢李四。',
                metadata={'document_id': 2, 'chapter': '正文', 'chapter_index': 0},
                embedding=[1.0, 0.0],
            ),
        ]
    )
    return store


class TestIdentifierFallback:
    def test_vector_mode_recovers_variant(self) -> None:
        store = _store_with_table()
        retriever = Retriever(store, OrthoEmb())
        hits = retriever.retrieve('P002上市时间是什么时候？', document_id=1)
        assert hits and 'P-002' in hits[0].text

    def test_hyphen_and_spaced_variants(self) -> None:
        store = _store_with_table()
        retriever = Retriever(store, OrthoEmb())
        for variant in ('P002 上市时间', 'P-002 什么时候上市', 'p 002 上架日期'):
            hits = retriever.retrieve(variant, document_id=1)
            assert hits, f'变体未兜底命中: {variant}'

    def test_hybrid_mode_fallback(self) -> None:
        store = _store_with_table()
        retriever = HybridRetriever(store, OrthoEmb(), pool_size=5, rrf_k=60)
        hits = retriever.retrieve('P002上市时间是什么时候？', document_id=1)
        assert hits and 'P-002' in hits[0].text

    def test_no_identifier_no_fallback(self) -> None:
        store = _store_with_table()
        retriever = Retriever(store, OrthoEmb())
        hits = retriever.retrieve('这本书讲了什么爱情故事？', document_id=2)
        # 纯中文语义题，方向正交 → 不应靠兜底捞书1表格
        assert all(h.document_id == 2 for h in hits)


class TestBM25Canonical:
    def test_query_variant_hits_doc_token(self) -> None:
        store = _store_with_table()
        index = BM25Index(store)
        hits = dict(index.query('P002 上市时间', top_k=5))
        assert 'doc1-0' in hits

class _VariantEmb(Embeddings):
    """让 Q1/Q2 向量接近但不达全局语义阈值，逼出“标识符复用”通道。"""

    def embed_query(self, text: str) -> list[float]:
        return [0.98, 0.2] if '产品' in (text or '') else [0.8, 0.6]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


class TestCacheIdentifierReuse:
    def test_variant_question_reuses_cached_answer(self, tmp_path: Path) -> None:
        from reading_assistant.storage.models import Document

        engine = create_db_engine('sqlite:///:memory:')
        init_db(engine)
        factory = create_session_factory(engine)
        with factory() as session:
            session.add(
                Document(
                    filename='测试表格.docx',
                    title='测试表格',
                    file_hash='h-t1',
                    content_hash='c-t1',
                )
            )
            session.commit()
        store = InMemoryVectorStore()
        store.add(
            [
                StoredChunk(
                    id='doc1-0',
                    text='| 产品ID | 上市时间 |\n| --- | --- |\n| P-002 | 2023-09-22 |',
                    metadata={'document_id': 1, 'chapter': '正文', 'chapter_index': 0},
                    embedding=[1.0, 0.0],
                )
            ]
        )
        app = create_app(
            session_factory=factory,
            vector_store=store,
            llm=FakeLLM(),
            embedding_model=_VariantEmb(),
            upload_dir=tmp_path / 'uploads',
        )
        client = TestClient(app)
        with client:
            session_id = client.post('/api/sessions').json()['session_id']
            first = client.post(
                f'/api/sessions/{session_id}/messages',
                json={'question': 'P-002产品的上市时间是什么时候？', 'document_ids': [1]},
            ).json()
            assert first['answer'] and first['citations']
            second = client.post(
                f'/api/sessions/{session_id}/messages',
                json={'question': 'P002上市时间是什么时候？', 'document_ids': [1]},
            ).json()
            assert second['needs_clarification'] is False
            assert second['answer'] == first['answer'], '变体问题应复用上一次缓存回答'
            assert second['citations'] == first['citations']
