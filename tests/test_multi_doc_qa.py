"""多文档问答测试：并行 fan-out 合流、书名标注、单文档零回归。"""

from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient
from langchain_core.embeddings import Embeddings

from reading_assistant.api import create_app
from reading_assistant.storage import (
    create_db_engine,
    create_session_factory,
    init_db,
)
from reading_assistant.storage.models import Document
from reading_assistant.storage.vector_store import InMemoryVectorStore, StoredChunk


class FakeEmbeddings(Embeddings):
    """固定向量，保证所有 chunk 与查询同空间、可通过 min_score 闸门。"""

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


class FakeLLM:
    def invoke(self, prompt):
        return SimpleNamespace(content='根据原文可以回答这个问题，这是测试回答。')


def _make_env() -> tuple[TestClient, InMemoryVectorStore, int, int]:
    engine = create_db_engine('sqlite:///:memory:')
    init_db(engine)
    factory = create_session_factory(engine)
    with factory() as session:
        session.add(
            Document(
                filename='书A.txt', title='书A', file_hash='hash-a',
                content_hash='content-a', index_status='indexed',
            )
        )
        session.add(
            Document(
                filename='书B.txt', title='书B', file_hash='hash-b',
                content_hash='content-b', index_status='indexed',
            )
        )
        session.flush()
        ids = [d.id for d in (session.get(Document, 1), session.get(Document, 2))]
        session.commit()
    id_a, id_b = ids
    store = InMemoryVectorStore()
    store.add(
        [
            StoredChunk(
                id=f'doc{id_a}-0',
                text='甲说：乙喜欢丙。',
                metadata={'document_id': id_a, 'chapter': '第一章', 'chapter_index': 0},
                embedding=[1.0, 0.0],
            ),
            StoredChunk(
                id=f'doc{id_b}-0',
                text='丁的记录：戊在1949年掌权。',
                metadata={'document_id': id_b, 'chapter': '第一章', 'chapter_index': 0},
                embedding=[1.0, 0.0],
            ),
        ]
    )
    app = create_app(
        session_factory=factory,
        vector_store=store,
        llm=FakeLLM(),
        embedding_model=FakeEmbeddings(),
        upload_dir=Path('uploads'),
    )
    return TestClient(app), store, id_a, id_b


def _ask(client: TestClient, question: str, doc_ids: list[int]) -> dict:
    session_id = client.post('/api/sessions').json()['session_id']
    body = client.post(
        f'/api/sessions/{session_id}/messages',
        json={'question': question, 'document_ids': doc_ids},
    ).json()
    return body


class TestMultiDocQA:
    def test_multi_doc_merges_both_sources(self) -> None:
        client, _store, id_a, id_b = _make_env()
        with client:
            body = _ask(client, '书A和书B各自提到了什么？', [id_a, id_b])
        assert body['needs_clarification'] is False
        assert body['answer']
        docs = {c['document'] for c in body['citations']}
        assert docs == {'书A.txt', '书B.txt'}, f'合流应包含两本书，实际 {docs}'

    def test_multi_doc_citations_carry_document(self) -> None:
        client, _store, id_a, id_b = _make_env()
        with client:
            body = _ask(client, '谁喜欢谁？', [id_a, id_b])
        assert all(c['document'] in ('书A.txt', '书B.txt') for c in body['citations'])
        assert len(body['citations']) >= 2

    def test_single_doc_keeps_legacy_behavior(self) -> None:
        client, _store, id_a, _id_b = _make_env()
        with client:
            body = _ask(client, '乙喜欢谁？', [id_a])
        docs = {c['document'] for c in body['citations']}
        assert docs == {'书A.txt'}, f'单文档只应命中书A，实际 {docs}'
