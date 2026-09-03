"""P5 FastAPI 接口测试。"""

from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from langchain_core.embeddings import Embeddings

from reading_assistant.api import create_app
from sqlalchemy import select

from reading_assistant.storage import (
    Document,
    HitlTask,
    QaCacheEntry,
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


class FakeNoInfoLLM(FakeLLM):
    """模拟 LLM 判定原文信息不足。"""

    def __init__(self) -> None:
        super().__init__(content='原文中没有相关信息。')


class _ToolResponse:
    """带 tool_calls 的伪模型响应。"""

    def __init__(self, name: str, args: dict) -> None:
        self.content = ''
        self.tool_calls = [
            {'name': name, 'args': args, 'id': 'call_1', 'type': 'tool_call'}
        ]


class FakeToolLLM(FakeLLM):
    """模拟通过工具调用表态的 LLM。"""

    def __init__(self, tool_name: str, args: dict) -> None:
        super().__init__()
        self._tool_name = tool_name
        self._tool_args = args

    def bind_tools(self, tools):
        return self

    def invoke(self, prompt: str):
        return _ToolResponse(self._tool_name, self._tool_args)


def _make_app(tmp_path: Path, store: InMemoryVectorStore, llm=None):
    engine = create_db_engine('sqlite:///:memory:')
    init_db(engine)
    session_factory = create_session_factory(engine)
    app = create_app(
        session_factory=session_factory,
        vector_store=store,
        llm=llm or FakeLLM(),
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

    def test_reindex_document_keeps_chunk_count(self, client: TestClient) -> None:
        uploaded = _upload_txt(client).json()

        response = client.post(f"/api/documents/{uploaded['id']}/reindex")

        assert response.status_code == 200
        body = response.json()
        assert body['id'] == uploaded['id']
        assert body['chunk_count'] == uploaded['chunk_count']

        # 重复 reindex 应幂等
        again = client.post(f"/api/documents/{uploaded['id']}/reindex")
        assert again.status_code == 200
        assert again.json()['chunk_count'] == uploaded['chunk_count']

    def test_reindex_marks_document_indexed(self, tmp_path: Path) -> None:
        app, engine = _make_app(tmp_path, InMemoryVectorStore())
        session_factory = create_session_factory(engine)
        with TestClient(app) as client:
            uploaded = _upload_txt(client).json()

            response = client.post(f"/api/documents/{uploaded['id']}/reindex")
            assert response.status_code == 200

            with session_factory() as session:
                document = session.get(Document, uploaded['id'])
                assert document.index_status == 'indexed'
        engine.dispose()

    def test_reindex_missing_document_404(self, client: TestClient) -> None:
        response = client.post('/api/documents/999/reindex')

        assert response.status_code == 404


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

    def test_delete_session_removes_messages(self, client: TestClient) -> None:
        session_id = client.post('/api/sessions').json()['session_id']
        client.post(
            f'/api/sessions/{session_id}/messages',
            json={'question': '张三是谁', 'document_ids': [1]},
        )

        response = client.delete(f'/api/sessions/{session_id}')

        assert response.status_code == 204
        assert client.get(f'/api/sessions/{session_id}/messages').status_code == 404
        assert client.get('/api/sessions').json() == []

    def test_delete_session_removes_hitl_tasks(self, tmp_path: Path) -> None:
        app, engine = _make_app(tmp_path, InMemoryVectorStore())
        with TestClient(app) as client:
            session_id = client.post('/api/sessions').json()['session_id']
            client.post(
                f'/api/sessions/{session_id}/messages', json={'question': '无结果提问'}
            )
            tasks = client.get(
                '/api/hitl/tasks', params={'session_id': int(session_id)}
            ).json()
            assert len(tasks) == 1

            assert client.delete(f'/api/sessions/{session_id}').status_code == 204
            remaining = client.get(
                '/api/hitl/tasks', params={'session_id': int(session_id)}
            ).json()
            assert remaining == []
        engine.dispose()

    def test_delete_missing_session_404(self, client: TestClient) -> None:
        assert client.delete('/api/sessions/999').status_code == 404


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

    def test_cache_hit_returns_citations(self, client: TestClient) -> None:
        session_id = client.post('/api/sessions').json()['session_id']
        payload = {'question': '张三是谁', 'document_ids': [1]}

        first = client.post(f'/api/sessions/{session_id}/messages', json=payload)
        assert first.status_code == 200
        assert len(first.json()['citations']) >= 1

        # 同一问题第二次提问应命中缓存，且引用必须完整返回
        second = client.post(f'/api/sessions/{session_id}/messages', json=payload)
        assert second.status_code == 200
        body = second.json()
        assert body['answer'] == '这是基于原文的测试回答。'
        assert body['citations'] == first.json()['citations']
        assert body['citations'][0]['excerpt']

    def test_poisoned_cache_is_invalidated(self, tmp_path: Path) -> None:
        """缓存里混入'信息不足'式回答时，命中后应丢弃并重新回答。"""
        app, engine = _make_app(tmp_path, InMemoryVectorStore())
        session_factory = create_session_factory(engine)
        with TestClient(app) as client:
            doc = _upload_txt(client).json()
            session_id = client.post('/api/sessions').json()['session_id']
            payload = {'question': '张三是谁', 'document_ids': [doc['id']]}

            first = client.post(f'/api/sessions/{session_id}/messages', json=payload)
            assert first.status_code == 200

            # 污染缓存：把刚生成的缓存回答改成"信息不足"式文本
            with session_factory() as session:
                entry = session.scalar(select(QaCacheEntry))
                assert entry is not None
                entry.answer = '原文中没有相关信息。'
                session.commit()

            # 再问同一问题：应拦截无效缓存并重新走 LLM 回答
            again = client.post(f'/api/sessions/{session_id}/messages', json=payload)
            assert again.status_code == 200
            body = again.json()
            assert body['needs_clarification'] is False
            assert body['answer'] == '这是基于原文的测试回答。'
            assert body['citations'] != []

            # 缓存应已重建为有效回答
            with session_factory() as session:
                entry = session.scalar(select(QaCacheEntry))
                assert entry.answer == '这是基于原文的测试回答。'
        engine.dispose()

    def test_same_question_across_documents_no_conflict(self, tmp_path: Path) -> None:
        app, engine = _make_app(tmp_path, InMemoryVectorStore())
        with TestClient(app) as client:
            first = _upload_txt(
                client, filename='a.txt', content='第一章 甲\n内容甲。\n'
            ).json()
            second = _upload_txt(
                client, filename='b.txt', content='第一章 乙\n内容乙。\n'
            ).json()
            session_id = client.post('/api/sessions').json()['session_id']

            # 同一问题在文档 A 上问（生成缓存）
            r1 = client.post(
                f'/api/sessions/{session_id}/messages',
                json={'question': '内容是什么', 'document_ids': [first['id']]},
            )
            assert r1.status_code == 200

            # 同样的问题在文档 B 上问：不应触发 question_hash 唯一约束冲突
            r2 = client.post(
                f'/api/sessions/{session_id}/messages',
                json={'question': '内容是什么', 'document_ids': [second['id']]},
            )
            assert r2.status_code == 200
            assert len(r2.json()['citations']) >= 1
        engine.dispose()


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

    def test_llm_no_info_creates_hitl_task(self, tmp_path: Path) -> None:
        """检索到片段但 LLM 判定信息不足时，自动创建澄清任务且不记录干瘪回答。"""
        app, engine = _make_app(tmp_path, InMemoryVectorStore(), llm=FakeNoInfoLLM())
        with TestClient(app) as client:
            doc = _upload_txt(client).json()
            session_id = client.post('/api/sessions').json()['session_id']

            response = client.post(
                f'/api/sessions/{session_id}/messages',
                json={'question': '片段中没有答案的细节问题', 'document_ids': [doc['id']]},
            )

            assert response.status_code == 200
            body = response.json()
            assert body['needs_clarification'] is True
            assert body['hitl_task_id'] is not None
            assert body['answer'] is None

            # LLM 那句"原文中没有相关信息"不应写入聊天记录
            messages = client.get(f'/api/sessions/{session_id}/messages').json()
            assert [m['role'] for m in messages] == ['user']

            tasks = client.get(
                '/api/hitl/tasks', params={'session_id': int(session_id)}
            ).json()
            assert tasks[0]['status'] == HitlTask.STATUS_AWAITING
        engine.dispose()

    def test_tool_request_clarification_creates_hitl(self, tmp_path: Path) -> None:
        """模型调用 request_clarification 工具 → 自动创建澄清任务。"""
        app, engine = _make_app(
            tmp_path,
            InMemoryVectorStore(),
            llm=FakeToolLLM(
                'request_clarification',
                {'missing_info': '需要少白公与舵手关系的完整原文'},
            ),
        )
        with TestClient(app) as client:
            doc = _upload_txt(client).json()
            session_id = client.post('/api/sessions').json()['session_id']

            response = client.post(
                f'/api/sessions/{session_id}/messages',
                json={'question': '细节问题', 'document_ids': [doc['id']]},
            )

            assert response.status_code == 200
            body = response.json()
            assert body['needs_clarification'] is True
            assert body['hitl_task_id'] is not None
            assert body['answer'] is None
            assert len(body['citations']) == 0

            messages = client.get(f'/api/sessions/{session_id}/messages').json()
            assert [m['role'] for m in messages] == ['user']
        engine.dispose()

    def test_tool_final_answer_returns_answer(self, tmp_path: Path) -> None:
        """模型调用 final_answer 工具 → 正常回答并记录。"""
        app, engine = _make_app(
            tmp_path,
            InMemoryVectorStore(),
            llm=FakeToolLLM('final_answer', {'answer': '根据原文，张三出场了。'}),
        )
        with TestClient(app) as client:
            doc = _upload_txt(client).json()
            session_id = client.post('/api/sessions').json()['session_id']

            response = client.post(
                f'/api/sessions/{session_id}/messages',
                json={'question': '张三是谁', 'document_ids': [doc['id']]},
            )

            assert response.status_code == 200
            body = response.json()
            assert body['needs_clarification'] is False
            assert body['answer'] == '根据原文，张三出场了。'
            assert len(body['citations']) >= 1

            messages = client.get(f'/api/sessions/{session_id}/messages').json()
            assert [m['role'] for m in messages] == ['user', 'assistant']
            assert messages[-1]['content'] == '根据原文，张三出场了。'
        engine.dispose()
