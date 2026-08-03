"""P5 FastAPI 接口测试。"""

from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from langchain_core.embeddings import Embeddings

from reading_assistant.api import create_app
from reading_assistant.storage import (
    HitlTask,
    create_db_engine,
    create_session_factory,
    init_db,
)
from reading_assistant.storage.vector_store import InMemoryVectorStore, StoredChunk


class FakeEmbeddings(Embeddings):
    """固定向量的假 Embedding。"""

    def __init__(self, vector: tuple[float, float] = (1.0, 0.0)) -> None:
        self._vector = list(vector)

    def embed_query(self, text: str) -> list[float]:
        return list(self._vector)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [list(self._vector) for _ in texts]


class FakeLLM:
    """返回固定内容的假 LLM。"""

    def __init__(self, content: str = '这是基于原文的测试回答。') -> None:
        self._content = content

    def invoke(self, prompt: str):
        return SimpleNamespace(content=self._content)


def _make_app(tmp_path: Path, store: InMemoryVectorStore):
    engine = create_db_engine('sqlite:///:memory:')
    init_db(engine)
    session_factory = create_session_factory(engine)
    app = create_app(
        session_factory=session_factory,
        vector_store=store,
        llm=FakeLLM(),
        embedding_model=FakeEmbeddings([1.0, 0.0]),
        upload_dir=tmp_path / 'uploads',
    )
    return app, engine


@pytest.fixture
def populated_store() -> InMemoryVectorStore:
    store = InMemoryVectorStore()
    store.add(
        [
            StoredChunk(
                id='doc1-0',
                text='张三在第一章出场。',
                metadata={'document_id': 1, 'chapter': '第一章', 'chapter_index': 0},
                embedding=[1.0, 0.0],
            )
        ]
    )
    return store


@pytest.fixture
def client(tmp_path: Path, populated_store: InMemoryVectorStore):
    app, engine = _make_app(tmp_path, populated_store)
    with TestClient(app) as test_client:
        yield test_client
    engine.dispose()


def _upload_txt(
    client: TestClient,
    filename: str = 'book.txt',
    content: str = '第一章 开端\n正文内容。\n',
):
    return client.post(
        '/api/documents/upload',
        files={'file': (filename, content.encode('utf-8'), 'text/plain')},
    )


class TestDocuments:
    def test_upload_and_list(self, client: TestClient) -> None:
        response = _upload_txt(client)

        assert response.status_code == 201
        body = response.json()
        assert body['id'] is not None
        assert body['filename'] == 'book.txt'
        assert body['chunk_count'] > 0
        assert body['duplicate'] is False

        documents = client.get('/api/documents')
        assert documents.status_code == 200
        assert len(documents.json()) == 1

    def test_upload_duplicate_returns_same_document(self, client: TestClient) -> None:
        first = _upload_txt(client).json()

        second = _upload_txt(client).json()

        assert second['duplicate'] is True
        assert second['id'] == first['id']

    def test_upload_unsupported_format(self, client: TestClient) -> None:
        response = client.post(
            '/api/documents/upload',
            files={'file': ('book.mobi', b'data', 'application/octet-stream')},
        )

        assert response.status_code == 422
        assert '不支持的' in response.json()['detail']

    def test_upload_corrupted_file(self, client: TestClient) -> None:
        response = client.post(
            '/api/documents/upload',
            files={'file': ('broken.epub', b'not an epub', 'application/octet-stream')},
        )

        assert response.status_code == 422


class TestSessions:
    def test_create_and_list_sessions(self, client: TestClient) -> None:
        created = client.post('/api/sessions')

        assert created.status_code == 201
        session_id = created.json()['session_id']
        assert isinstance(session_id, str)

        sessions = client.get('/api/sessions').json()
        assert len(sessions) == 1
        assert sessions[0]['id'] == session_id
        assert sessions[0]['created_at']

    def test_messages_of_new_session_empty(self, client: TestClient) -> None:
        session_id = client.post('/api/sessions').json()['session_id']

        response = client.get(f'/api/sessions/{session_id}/messages')

        assert response.status_code == 200
        assert response.json() == []

    def test_nonexistent_session_returns_404(self, client: TestClient) -> None:
        assert client.get('/api/sessions/999/messages').status_code == 404
        response = client.post('/api/sessions/999/messages', json={'question': '你好'})
        assert response.status_code == 404


class TestAsk:
    def test_ask_returns_answer_and_records_messages(self, client: TestClient) -> None:
        session_id = client.post('/api/sessions').json()['session_id']

        response = client.post(
            f'/api/sessions/{session_id}/messages',
            json={'question': '张三是谁', 'document_ids': [1]},
        )

        assert response.status_code == 200
        body = response.json()
        assert body['needs_clarification'] is False
        assert body['answer'] == '这是基于原文的测试回答。'
        assert body['citations'][0]['chapter'] == '第一章'
        assert body['citations'][0]['excerpt']

        messages = client.get(f'/api/sessions/{session_id}/messages').json()
        assert [message['role'] for message in messages] == ['user', 'assistant']
        assert messages[-1]['meta']['citations']

    def test_ask_empty_question_returns_422(self, client: TestClient) -> None:
        session_id = client.post('/api/sessions').json()['session_id']

        response = client.post(f'/api/sessions/{session_id}/messages', json={'question': ''})

        assert response.status_code == 422


class TestHitl:
    def _empty_store_client(self, tmp_path: Path):
        app, engine = _make_app(tmp_path, InMemoryVectorStore())
        test_client = TestClient(app)
        return test_client, engine

    def test_ask_without_chunks_creates_hitl_task(self, tmp_path: Path) -> None:
        client, engine = self._empty_store_client(tmp_path)
        with client:
            session_id = client.post('/api/sessions').json()['session_id']

            response = client.post(
                f'/api/sessions/{session_id}/messages',
                json={'question': '完全没有检索结果的提问'},
            )

            assert response.status_code == 200
            body = response.json()
            assert body['needs_clarification'] is True
            assert body['hitl_task_id'] is not None

            tasks = client.get('/api/hitl/tasks', params={'session_id': int(session_id)})
            assert tasks.status_code == 200
            assert tasks.json()[0]['status'] == HitlTask.STATUS_AWAITING
        engine.dispose()

    def test_submit_and_reject_clarification(self, tmp_path: Path) -> None:
        client, engine = self._empty_store_client(tmp_path)
        with client:
            session_id = client.post('/api/sessions').json()['session_id']
            task_id = client.post(
                f'/api/sessions/{session_id}/messages',
                json={'question': '无结果提问'},
            ).json()['hitl_task_id']

            submitted = client.post(
                f'/api/hitl/tasks/{task_id}/submit', json={'clarification': '补充信息'}
            )
            assert submitted.status_code == 200
            assert submitted.json()['status'] == HitlTask.STATUS_APPROVED

            rejected = client.post(f'/api/hitl/tasks/{task_id}/reject')
            assert rejected.status_code == 400  # 已批准，不可再拒绝

            missing = client.post('/api/hitl/tasks/999/submit', json={'clarification': 'x'})
            assert missing.status_code == 404
        engine.dispose()
