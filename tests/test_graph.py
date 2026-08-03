"""P4 LangGraph 流水线测试：入库、问答与 HITL 状态流转。"""

from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.embeddings import Embeddings
from sqlalchemy import select

from reading_assistant.graph import build_ingest_graph, build_qa_graph
from reading_assistant.storage import (
    ChatMessage,
    ChatSession,
    HitlTask,
    create_db_engine,
    create_hitl_task,
    create_session_factory,
    init_db,
    reject_hitl_task,
    submit_hitl_clarification,
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


@pytest.fixture
def session_factory():
    engine = create_db_engine('sqlite:///:memory:')
    init_db(engine)
    yield create_session_factory(engine)
    engine.dispose()


@pytest.fixture
def session(session_factory):
    session = session_factory()
    yield session
    session.close()


def _write_book(
    tmp_path: Path,
    chapters: tuple[str, ...] = ('第一章 开端\n正文一。', '第二章 发展\n正文二。'),
) -> Path:
    path = tmp_path / 'book.txt'
    path.write_text('\n'.join(chapters) + '\n', encoding='utf-8')
    return path


def _populated_store(document_id: int = 1) -> InMemoryVectorStore:
    store = InMemoryVectorStore()
    store.add(
        [
            StoredChunk(
                id=f'doc{document_id}-0',
                text='张三在第一章出场。',
                metadata={
                    'document_id': document_id,
                    'chapter': '第一章',
                    'chapter_index': 0,
                },
                embedding=[1.0, 0.0],
            ),
            StoredChunk(
                id=f'doc{document_id}-1',
                text='李四在第二章被追捕。',
                metadata={
                    'document_id': document_id,
                    'chapter': '第二章',
                    'chapter_index': 1,
                },
                embedding=[0.0, 1.0],
            ),
        ]
    )
    return store


class TestIngestGraph:
    def test_indexes_book(self, session_factory, tmp_path: Path) -> None:
        store = InMemoryVectorStore()
        graph = build_ingest_graph(session_factory, store, embedding_model=FakeEmbeddings())
        book = _write_book(tmp_path)

        result = graph.invoke(
            {'book_path': str(book)},
            config={'configurable': {'thread_id': 'ingest-1'}},
        )

        assert result['duplicate'] is False
        assert result['document_id'] is not None
        assert result['chunk_count'] > 0
        assert store.count() == result['chunk_count']

    def test_duplicate_short_circuits(self, session_factory, tmp_path: Path) -> None:
        store = InMemoryVectorStore()
        graph = build_ingest_graph(session_factory, store, embedding_model=FakeEmbeddings())
        book = _write_book(tmp_path)
        first = graph.invoke(
            {'book_path': str(book)},
            config={'configurable': {'thread_id': 'ingest-2'}},
        )

        second = graph.invoke(
            {'book_path': str(book)},
            config={'configurable': {'thread_id': 'ingest-3'}},
        )

        assert second['duplicate'] is True
        assert second['document_id'] == first['document_id']
        assert store.count() == first['chunk_count']


class TestQaGraph:
    def _make_session(self, session) -> int:
        chat = ChatSession(title='测试')
        session.add(chat)
        session.commit()
        return chat.id

    def test_answers_with_citations_and_records_messages(self, session_factory, session) -> None:
        session_id = self._make_session(session)
        graph = build_qa_graph(
            session_factory,
            _populated_store(),
            llm=FakeLLM(),
            embedding_model=FakeEmbeddings([1.0, 0.0]),
        )

        result = graph.invoke(
            {'question': '张三是谁', 'session_id': session_id},
            config={'configurable': {'thread_id': 'qa-1'}},
        )

        assert result['answer'] == '这是基于原文的测试回答。'
        assert result['needs_clarification'] is False
        assert result['citations']
        messages = session.scalars(select(ChatMessage).order_by(ChatMessage.id)).all()
        assert {message.role for message in messages} == {'user', 'assistant'}
        assert messages[-1].meta['citations']

    def test_hitl_when_no_chunks(self, session_factory, session) -> None:
        session_id = self._make_session(session)
        graph = build_qa_graph(
            session_factory,
            InMemoryVectorStore(),
            llm=FakeLLM(),
            embedding_model=FakeEmbeddings([1.0, 0.0]),
        )

        result = graph.invoke(
            {'question': '完全没有检索结果的提问', 'session_id': session_id},
            config={'configurable': {'thread_id': 'qa-2'}},
        )

        assert result['needs_clarification'] is True
        assert result['hitl_task_id'] is not None
        assert result.get('answer') is None
        task = session.get(HitlTask, result['hitl_task_id'])
        assert task.status == HitlTask.STATUS_AWAITING
        messages = session.scalars(select(ChatMessage).order_by(ChatMessage.id)).all()
        assert [message.role for message in messages] == ['user']

    def test_resume_with_clarification_answers(self, session_factory) -> None:
        graph = build_qa_graph(
            session_factory,
            InMemoryVectorStore(),
            llm=FakeLLM('补充后的回答。'),
            embedding_model=FakeEmbeddings([1.0, 0.0]),
        )

        result = graph.invoke(
            {'question': '某问题', 'clarification': '补充：请结合张三出场回答'},
            config={'configurable': {'thread_id': 'qa-3'}},
        )

        assert result['needs_clarification'] is False
        assert result['answer'] == '补充后的回答。'
        assert result.get('hitl_task_id') is None


class TestHitlTransitions:
    def test_awaiting_to_approved(self, session) -> None:
        task = create_hitl_task(session, session_id=None, question='信息不足')
        assert task.status == HitlTask.STATUS_AWAITING

        updated = submit_hitl_clarification(session, task.id, '补充信息')

        assert updated.status == HitlTask.STATUS_APPROVED
        assert updated.clarification == '补充信息'

    def test_awaiting_to_rejected(self, session) -> None:
        task = create_hitl_task(session, session_id=None, question='信息不足')

        updated = reject_hitl_task(session, task.id)

        assert updated.status == HitlTask.STATUS_REJECTED

    def test_approved_cannot_be_rejected_again(self, session) -> None:
        task = create_hitl_task(session, session_id=None, question='信息不足')
        submit_hitl_clarification(session, task.id, '补充')

        with pytest.raises(ValueError, match='状态'):
            reject_hitl_task(session, task.id)

    def test_missing_task_raises(self, session) -> None:
        with pytest.raises(ValueError, match='不存在'):
            submit_hitl_clarification(session, 999, 'x')
