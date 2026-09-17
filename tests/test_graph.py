"""P4 LangGraph 流水线测试：入库、问答与 HITL 状态流转。"""

from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.embeddings import Embeddings
from sqlalchemy import select

from reading_assistant.graph import (
    build_ingest_graph,
    build_qa_graph,
    interrupt_payload,
)
from reading_assistant.storage import (
    ChatMessage,
    ChatSession,
    Document,
    HitlTask,
    create_db_engine,
    create_hitl_task,
    create_session_factory,
    get_document,
    init_db,
    list_indexed_document_ids,
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

    def test_duplicate_upload_keeps_document_indexed(
        self, session_factory, tmp_path: Path
    ) -> None:
        """回归：重复上传（dup=True 且未 force）不得把文档打回 indexing。

        ``route_after_add`` 在重复时直接 END，没有任何节点会再把 index_status 写回
        indexed；而 QA 检索白名单 ``list_indexed_document_ids()`` 只认 indexed。
        一旦被误置，这本书在问答里就等于不存在：检索 hit=0 → judge 判「信息不足」
        → 转 HITL，日志里只有一行 HITL，看不到任何异常。
        eval 每次运行都会重传全部语料，所以这条路径每轮都会被踩到。
        """
        store = InMemoryVectorStore()
        graph = build_ingest_graph(session_factory, store, embedding_model=FakeEmbeddings())
        book = _write_book(tmp_path)
        first = graph.invoke(
            {'book_path': str(book)},
            config={'configurable': {'thread_id': 'ingest-idx-1'}},
        )
        document_id = first['document_id']

        graph.invoke(
            {'book_path': str(book)},
            config={'configurable': {'thread_id': 'ingest-idx-2'}},
        )

        session = session_factory()
        try:
            assert get_document(session, document_id).index_status == 'indexed'
            assert document_id in list_indexed_document_ids(session)
        finally:
            session.close()

    def test_force_reindex_overrides_duplicate_lease(
        self, session_factory, tmp_path: Path
    ) -> None:
        """force=True 时即使命中 file_hash 也要重新走 chunk_and_index（保持 indexed）。"""
        store = InMemoryVectorStore()
        graph = build_ingest_graph(session_factory, store, embedding_model=FakeEmbeddings())
        book = _write_book(tmp_path)
        first = graph.invoke(
            {'book_path': str(book)},
            config={'configurable': {'thread_id': 'ingest-force-1'}},
        )

        graph.invoke(
            {'book_path': str(book), 'force': True},
            config={'configurable': {'thread_id': 'ingest-force-2'}},
        )

        session = session_factory()
        try:
            document = get_document(session, first['document_id'])
            assert document.index_status == 'indexed'
            assert document.index_started_at is not None
        finally:
            session.close()


class TestQaGraph:
    def _make_session(self, session) -> int:
        chat = ChatSession(title='测试')
        session.add(chat)
        session.commit()
        return chat.id

    def test_answers_with_citations_and_records_messages(self, session_factory, session) -> None:
        session_id = self._make_session(session)
        # 检索层只放行 index_status=='indexed' 的文档，故语料对应文档须先标记为已索引
        with session_factory() as seed:
            seed.add(
                Document(
                    filename='book.txt',
                    title='book',
                    file_hash='f1',
                    content_hash='c1',
                    index_status='indexed',
                )
            )
            seed.commit()
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

        print(graph.get_graph().draw_mermaid())

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

        # 真挂起后：节点返回值不进 state，任务 id 只能从挂起载荷里取
        payload = interrupt_payload(result)
        assert payload is not None, '图应在 create_hitl 处挂起'
        assert result.get('answer') is None
        task_id = payload['hitl_task_id']
        task = session.get(HitlTask, task_id)
        assert task.status == HitlTask.STATUS_AWAITING
        # 恢复指针必须落库，否则重启/换进程后就接不上了
        assert task.thread_id == 'qa-2'
        # 用户的问题在**轮次开头**就记下了（挂起时 record 不会执行）
        messages = session.scalars(select(ChatMessage).order_by(ChatMessage.id)).all()
        assert [message.role for message in messages] == ['user']
        assert '没有检索结果的提问' in messages[0].content

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

    def test_resume_from_interrupt_continues_not_restarts(self, session_factory) -> None:
        """真中断-恢复：从挂起点继续，而不是带着 clarification 从头重跑。

        这是本项目的核心契约之一。与“重跑”的可观测差异：
        - 恢复时不需要客户端重发问题，图自带原来的 document_ids / 历史；
        - ``record`` 只在恢复后执行一次（挂起时不会写 assistant 消息）；
        - 恢复后回到 retrieve 重检索，而不是重跑 gate/context_load/summarize。
        """
        from langgraph.types import Command

        from reading_assistant.graph import interrupt_task_id

        with session_factory() as seed:
            seed.add(
                Document(
                    filename='book.txt',
                    title='book',
                    file_hash='f1',
                    content_hash='c1',
                    index_status='indexed',
                )
            )
            seed.commit()

        graph = build_qa_graph(
            session_factory,
            InMemoryVectorStore(),          # 空库 → 无片段 → 判「信息不足」
            llm=FakeLLM('恢复后的回答。'),
            embedding_model=FakeEmbeddings([1.0, 0.0]),
        )
        config = {'configurable': {'thread_id': 'qa-resume-1'}}
        first = graph.invoke({'question': '书里没有的东西？', 'session_id': None}, config=config)

        task_id = interrupt_task_id(first)
        assert task_id is not None
        # 挂起中：state 停在 create_hitl，还有待执行节点
        assert graph.get_state(config).next, '挂起时应有待执行节点'

        second = graph.invoke(Command(resume='我说的是第三章'), config=config)

        assert second.get('answer') == '恢复后的回答。'
        assert second['needs_clarification'] is False
        # 恢复后跑到 END，不再挂起
        assert not graph.get_state(config).next

    def test_interrupt_is_not_reissued_after_clarification(self, session_factory) -> None:
        """澄清已用过之后再判信息不足，不得再次挂起（否则构成死循环）。"""
        from langgraph.types import Command

        from reading_assistant.graph import interrupt_task_id

        graph = build_qa_graph(
            session_factory,
            InMemoryVectorStore(),
            llm=FakeLLM('仍然不足。'),
            embedding_model=FakeEmbeddings([1.0, 0.0]),
        )
        config = {'configurable': {'thread_id': 'qa-resume-2'}}
        first = graph.invoke({'question': 'q', 'session_id': None}, config=config)
        assert interrupt_task_id(first) is not None

        second = graph.invoke(Command(resume='补充'), config=config)
        # 关键：第二遍不得再挂起
        assert interrupt_task_id(second) is None
        assert not graph.get_state(config).next


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
