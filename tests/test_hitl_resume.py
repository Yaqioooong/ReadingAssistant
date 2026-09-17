"""HITL 真「中断-恢复」测试。

这是本项目最容易退化的契约，退化了也不会报错 —— 只会静默变回「重跑」，
所以这里把每个必要条件都钉住：

| 必要条件 | 缺了会怎样 |
| --- | --- |
| 图在 ``create_hitl`` 处 ``interrupt()`` | 跑到 END，没有断点可续 |
| ``thread_id`` 落库（``HitlTask.thread_id``） | 换个进程/请求就找不到断点 |
| saver 与图**单例** | 每请求重建 = 每请求换 saver，断点永远查不到 |
| 任务创建幂等 | ``interrupt()`` 恢复时节点重跑 → 重复建任务 |
| 问题在轮次开头记账 | 挂起时 ``record`` 不执行 → 用户的问题丢失 |
| 跑完回收 checkpoint | 单例图 + InMemorySaver 每请求堆 37KB，永不回收 |
| 澄清用尽后不再挂起 | 再判不足就二次挂起 → 死循环 |
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from langchain_core.embeddings import Embeddings
from langgraph.checkpoint.memory import InMemorySaver

from reading_assistant.api import create_app
from reading_assistant.graph import interrupt_payload, interrupt_task_id
from reading_assistant.storage import (
    Document,
    HitlTask,
    create_db_engine,
    create_hitl_task,
    create_session_factory,
    init_db,
)
from reading_assistant.storage.vector_store import InMemoryVectorStore, StoredChunk


class FakeEmbeddings(Embeddings):
    def __init__(self, vector=(1.0, 0.0)) -> None:
        self._vector = list(vector)

    def embed_query(self, text):
        return list(self._vector)

    def embed_documents(self, texts):
        return [list(self._vector) for _ in texts]


class FakeLLM:
    def __init__(self, content: str = '这是基于原文的测试回答。') -> None:
        self._content = content
        self.calls = 0

    def invoke(self, prompt):
        self.calls += 1
        return SimpleNamespace(content=self._content)


def _make_app(tmp_path: Path, store: InMemoryVectorStore, llm=None, seed_doc=True):
    engine = create_db_engine('sqlite:///:memory:')
    init_db(engine)
    factory = create_session_factory(engine)
    if seed_doc:
        # 检索层白名单只放行 index_status=='indexed' 的文档；
        # 不建这行，检索结果会被过滤空 → 每道题都被判信息不足，测不到「正常回答」路径
        with factory() as session:
            session.add(
                Document(
                    filename='book.txt',
                    title='book',
                    file_hash='f1',
                    content_hash='c1',
                    index_status='indexed',
                    chunk_count=1,
                )
            )
            session.commit()
    app = create_app(
        session_factory=factory,
        vector_store=store,
        llm=llm or FakeLLM(),
        embedding_model=FakeEmbeddings(),
        upload_dir=tmp_path / 'uploads',
        # 独立 saver：默认的 checkpointer 是进程级单例，会让各测试的 checkpoint 串味
        checkpointer=InMemorySaver(),
    )
    return app, engine, factory


def _populated_store() -> InMemoryVectorStore:
    store = InMemoryVectorStore()
    store.add([
        StoredChunk(
            id='doc1-0',
            text='第一章 张三与李四。',
            metadata={'document_id': 1, 'chapter': '第一章', 'chapter_index': 0},
            embedding=[1.0, 0.0],
        )
    ])
    return store


def _ask(client: TestClient, sid, question: str, docs=None) -> dict:
    return client.post(
        f'/api/sessions/{sid}/messages',
        json={'question': question, 'document_ids': docs or []},
    ).json()


class TestSuspension:
    """提问阶段：真挂起，且把恢复指针落库。"""

    def test_ask_suspends_with_interrupt_payload(self, tmp_path: Path) -> None:
        app, engine, _ = _make_app(tmp_path, InMemoryVectorStore())
        with TestClient(app) as client:
            sid = client.post('/api/sessions').json()['session_id']
            body = _ask(client, sid, '书里没写的东西？')

            assert body['needs_clarification'] is True
            assert body['answer'] is None
            assert body['hitl_task_id'] is not None
            assert body['resumable'] is True
            assert body['thread_id'].startswith('qa-')
        engine.dispose()

    def test_thread_id_is_persisted_on_task(self, tmp_path: Path) -> None:
        """恢复指针必须落库 —— 否则跨进程/重启就接不上了。"""
        app, engine, factory = _make_app(tmp_path, InMemoryVectorStore())
        with TestClient(app) as client:
            sid = client.post('/api/sessions').json()['session_id']
            body = _ask(client, sid, '书里没写的东西？')
            with factory() as session:
                task = session.get(HitlTask, body['hitl_task_id'])
                assert task.thread_id == body['thread_id']
                assert task.status == HitlTask.STATUS_AWAITING
        engine.dispose()

    def test_question_is_recorded_even_while_suspended(self, tmp_path: Path) -> None:
        """挂起时 ``record`` 不会执行 —— 问题必须在轮次开头就记下，否则丢失。"""
        app, engine, _ = _make_app(tmp_path, InMemoryVectorStore())
        with TestClient(app) as client:
            sid = client.post('/api/sessions').json()['session_id']
            _ask(client, sid, '会被挂起的问题？')
            messages = client.get(f'/api/sessions/{sid}/messages').json()
            assert [m['role'] for m in messages] == ['user']
            assert messages[0]['content'] == '会被挂起的问题？'
        engine.dispose()


class TestResume:
    """提交澄清：从断点继续，而不是重跑。"""

    def test_submit_resumes_and_returns_answer(self, tmp_path: Path) -> None:
        app, engine, _ = _make_app(tmp_path, InMemoryVectorStore(), llm=FakeLLM('恢复后的回答。'))
        with TestClient(app) as client:
            sid = client.post('/api/sessions').json()['session_id']
            body = _ask(client, sid, '书里没写的东西？')

            resp = client.post(
                f'/api/hitl/tasks/{body["hitl_task_id"]}/submit',
                json={'clarification': '我说的是第三章'},
            )
            assert resp.status_code == 200
            out = resp.json()
            assert out['resumed'] is True
            assert out['answer'] == '恢复后的回答。'
            assert out['status'] == HitlTask.STATUS_APPROVED
        engine.dispose()

    def test_resume_does_not_repay_upstream_stages(self, tmp_path: Path) -> None:
        """恢复不得重跑前段（gate / context_load / summarize）。

        可观测口径：LLM 调用次数。旧「重跑」路径下 gate 会再调一次模型；
        真恢复只多出 answer 那一次。
        """
        llm = FakeLLM('答案。')
        app, engine, _ = _make_app(tmp_path, InMemoryVectorStore(), llm=llm)
        with TestClient(app) as client:
            sid = client.post('/api/sessions').json()['session_id']
            body = _ask(client, sid, '问题？')
            after_ask = llm.calls

            client.post(
                f'/api/hitl/tasks/{body["hitl_task_id"]}/submit',
                json={'clarification': '补充'},
            )

            # 恢复只应新增「生成回答」这一次；若 gate 也被重跑就会多出一次
            assert llm.calls - after_ask <= 1
        engine.dispose()

    def test_clarification_is_recorded_as_user_message(self, tmp_path: Path) -> None:
        app, engine, _ = _make_app(tmp_path, InMemoryVectorStore())
        with TestClient(app) as client:
            sid = client.post('/api/sessions').json()['session_id']
            body = _ask(client, sid, '问题？')
            client.post(
                f'/api/hitl/tasks/{body["hitl_task_id"]}/submit',
                json={'clarification': '第三章'},
            )
            messages = client.get(f'/api/sessions/{sid}/messages').json()
            contents = [m['content'] for m in messages]
            assert contents[0] == '问题？'
            assert any('补充说明' in c and '第三章' in c for c in contents)
            assert [m['role'] for m in messages][-1] == 'assistant'
        engine.dispose()

    def test_repeated_submit_is_rejected(self, tmp_path: Path) -> None:
        """任务只能被 approved 一次 —— 否则会重复恢复。"""
        app, engine, _ = _make_app(tmp_path, InMemoryVectorStore())
        with TestClient(app) as client:
            sid = client.post('/api/sessions').json()['session_id']
            body = _ask(client, sid, '问题？')
            url = f'/api/hitl/tasks/{body["hitl_task_id"]}/submit'
            assert client.post(url, json={'clarification': '第一次'}).status_code == 200
            assert client.post(url, json={'clarification': '第二次'}).status_code == 400
        engine.dispose()

    def test_resume_after_normal_answer_is_not_possible(self, tmp_path: Path) -> None:
        """正常回答的轮次 checkpoint 已被回收 —— 不存在可恢复的断点（且不该报错）。"""
        app, engine, factory = _make_app(tmp_path, _populated_store())
        with TestClient(app) as client:
            sid = client.post('/api/sessions').json()['session_id']
            body = _ask(client, sid, '张三是谁', docs=[1])
            assert body['needs_clarification'] is False
            thread_id = body['thread_id']

            # 造一个指向已回收 thread 的任务：应降级为重跑，而不是假装恢复成功
            with factory() as session:
                orphan = create_hitl_task(
                    session, session_id=int(sid), question='q', thread_id=thread_id,
                )
                session.commit()
                orphan_id = orphan.id
            out = client.post(
                f'/api/hitl/tasks/{orphan_id}/submit', json={'clarification': 'x'}
            ).json()
            assert out['resumed'] is False
            assert out['answer'] is None
        engine.dispose()

    def test_task_without_thread_id_degrades_gracefully(self, tmp_path: Path) -> None:
        """加列前的旧任务没有 thread_id → 降级为重跑，不报错、不假装恢复。"""
        app, engine, factory = _make_app(tmp_path, InMemoryVectorStore())
        with TestClient(app) as client:
            sid = client.post('/api/sessions').json()['session_id']
            with factory() as session:
                legacy = create_hitl_task(session, session_id=int(sid), question='旧任务')
                session.commit()
                legacy_id = legacy.id
            out = client.post(
                f'/api/hitl/tasks/{legacy_id}/submit', json={'clarification': 'x'}
            ).json()
            assert out['resumed'] is False
            assert out['status'] == HitlTask.STATUS_APPROVED
        engine.dispose()


class TestCheckpointHygiene:
    """checkpoint 的生命周期：跑完即回收，挂起的保留。"""

    def test_finished_thread_checkpoint_is_discarded(self, tmp_path: Path) -> None:
        """回归：跑完的 thread 不留 checkpoint。

        单例图 + InMemorySaver 若不做回收，每请求堆 ~37KB 永不释放
        （实测 200 次请求积 7.2MB）。thread_id 一次性只保证「查不到」，不保证「不存」。
        """
        app, engine, _ = _make_app(tmp_path, _populated_store())
        with TestClient(app) as client:
            sid = client.post('/api/sessions').json()['session_id']
            for _ in range(3):
                body = _ask(client, sid, '张三是谁', docs=[1])
                assert body['needs_clarification'] is False
            graph = app.state.qa_graph
            saver = graph.checkpointer
            assert len(saver.storage) == 0, '正常回答的 thread 应立即回收'
        engine.dispose()

    def test_suspended_thread_is_retained(self, tmp_path: Path) -> None:
        """挂起中的 thread 必须保留 —— 那是恢复的唯一依据。"""
        app, engine, _ = _make_app(tmp_path, InMemoryVectorStore())
        with TestClient(app) as client:
            sid = client.post('/api/sessions').json()['session_id']
            _ask(client, sid, '会被挂起的问题？')
            assert len(app.state.qa_graph.checkpointer.storage) == 1
        engine.dispose()


class TestGraphSingleton:
    """图必须按 app 单例：恢复要求「同一张图 + 同一个 saver」。"""

    def test_graph_is_reused_across_requests(self, tmp_path: Path) -> None:
        app, engine, _ = _make_app(tmp_path, InMemoryVectorStore())
        with TestClient(app) as client:
            sid = client.post('/api/sessions').json()['session_id']
            _ask(client, sid, 'q1')
            first = app.state.qa_graph
            _ask(client, sid, 'q2')
            assert app.state.qa_graph is first
        engine.dispose()


class TestInterruptHelpers:
    def test_payload_is_none_when_not_interrupted(self) -> None:
        assert interrupt_payload({}) is None
        assert interrupt_task_id({'answer': 'x'}) is None

    def test_task_id_is_read_from_interrupt_payload(self) -> None:
        class _Interrupt:
            value = {'hitl_task_id': 7, 'question': 'q'}

        assert interrupt_task_id({'__interrupt__': [_Interrupt()]}) == 7


@pytest.mark.postgres
def test_cross_process_resume_with_postgres(tmp_path: Path, monkeypatch) -> None:
    """跨进程恢复：两个独立 saver + 两张图，模拟两个 worker。

    这条是「真中断-恢复」的最终验收 —— 单进程内存 saver 只能证明逻辑对，
    只有独立的 PG 连接能证明「换进程也接得上」。
    """
    from langgraph.types import Command

    from reading_assistant.graph import build_qa_graph
    from reading_assistant.graph.checkpointer import (
        close_checkpointer,
        create_postgres_checkpointer,
        discard_thread_if_finished,
    )
    from reading_assistant.storage import create_db_engine as _engine

    class _Emb(FakeEmbeddings):
        pass

    engine = _engine()
    init_db(engine)
    factory = create_session_factory(engine)
    from uuid import uuid4

    # qa_cache 是共享的 PG 表：固定题干会命中上次跑出的缓存答案（甚至走语义通道，
    # 相似度 ≥0.95 就复用），于是根本不挂起。这里直接关掉缓存，让断点语义成为唯一变量。
    from reading_assistant.config import get_settings

    monkeypatch.setenv('CACHE_ENABLED', 'false')
    get_settings.cache_clear()

    thread_id = f'qa-xproc-{uuid4().hex}'
    question = f'跨进程问题 {uuid4().hex}？'

    saver_a = create_postgres_checkpointer()
    graph_a = build_qa_graph(
        factory, InMemoryVectorStore(), llm=FakeLLM('跨进程回答。'),
        embedding_model=_Emb(), checkpointer=saver_a,
    )
    try:
        first = graph_a.invoke(
            {'question': question, 'session_id': None},
            config={'configurable': {'thread_id': thread_id}},
        )
        assert interrupt_task_id(first) is not None

        # 换一个 saver + 换一张图 = 另一个进程
        saver_b = create_postgres_checkpointer()
        graph_b = build_qa_graph(
            factory, InMemoryVectorStore(), llm=FakeLLM('跨进程回答。'),
            embedding_model=_Emb(), checkpointer=saver_b,
        )
        try:
            assert graph_b.get_state({'configurable': {'thread_id': thread_id}}).next
            resumed = graph_b.invoke(
                Command(resume='补充'), config={'configurable': {'thread_id': thread_id}}
            )
            assert resumed['answer'] == '跨进程回答。'
            assert discard_thread_if_finished(graph_b, thread_id) is True
        finally:
            close_checkpointer(saver_b)
    finally:
        close_checkpointer(saver_a)
        engine.dispose()
        monkeypatch.delenv('CACHE_ENABLED', raising=False)
        get_settings.cache_clear()
