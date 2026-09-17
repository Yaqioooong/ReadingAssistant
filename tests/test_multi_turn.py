"""多轮上下文 M1 测试：检索门路由、历史回忆、闲聊直答、书追问注入历史。"""

import json
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient
from langchain_core.embeddings import Embeddings
from sqlalchemy import select

from reading_assistant.api import create_app
from reading_assistant.storage import (
    create_db_engine,
    create_session_factory,
    init_db,
)
from reading_assistant.storage.models import ChatSession, Document, QaCacheEntry
from reading_assistant.storage.vector_store import InMemoryVectorStore, StoredChunk

BOOK_REPLY = 'P-002 的上市时间是 2023 年 9 月 22 日。'


class RouterLLM:
    """按 prompt 内容路由：gate 提示按当前问题返回标签；
    context 提示返回预设回复；书问题返回固定答案。"""

    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.gate_calls = 0
        self.context_calls = 0
        self.summarize_calls = 0
        self.context_reply = '你上一个问题是：「张三喜欢谁？」'

    def invoke(self, prompt: str) -> SimpleNamespace:
        self.prompts.append(prompt)
        if '意图分类任务' in prompt:
            self.gate_calls += 1
            return SimpleNamespace(content=self._gate_tag(prompt))
        if '对话摘要任务' in prompt:
            self.summarize_calls += 1
            return SimpleNamespace(content='早期对话摘要：较早的几轮书问题。')
        if '不需要检索书籍内容' in prompt:
            self.context_calls += 1
            return SimpleNamespace(content=self.context_reply)
        return SimpleNamespace(content=BOOK_REPLY)

    @staticmethod
    def _gate_tag(prompt: str) -> str:
        part = prompt.split('【当前提问】', 1)[-1]
        question = next((ln.strip() for ln in part.splitlines() if ln.strip()), '')
        history_hints = ('上一个问题', '前两个问题', '问过哪些', '问了什么', '第一个问题')
        if any(t in question for t in history_hints):
            return 'history'
        if any(t.lower() in question.lower() for t in ('你好', 'hello', 'hi', '谢谢')):
            return 'chat'
        return 'book'


class TurnEmb(Embeddings):
    """按问题内容给出不同向量：保证与文档块相似(≥检索阈值)而彼此区分(<缓存阈值)。"""

    def embed_query(self, text: str) -> list[float]:
        if '张三刚才' in text:
            return [0.5, 0.87]
        if '李四喜欢' in text:
            return [0.6, 0.8]
        if '王五喜欢' in text:
            return [0.5, 0.87]
        if '赵六喜欢' in text:
            return [0.7, 0.71]
        if '张三喜欢' in text:
            return [0.99, 0.1]
        return [0.9, 0.44]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


def _make_env(tmp_path: Path, llm: RouterLLM) -> tuple[TestClient, object]:
    engine = create_db_engine('sqlite:///:memory:')
    init_db(engine)
    factory = create_session_factory(engine)
    with factory() as session:
        session.add_all(
            [
                Document(
                    filename='d1.docx', title='d1', file_hash='f1',
                    content_hash='c1', index_status='indexed',
                ),
                Document(
                    filename='d2.docx', title='d2', file_hash='f2',
                    content_hash='c2', index_status='indexed',
                ),
            ]
        )
        session.commit()
    store = InMemoryVectorStore()
    store.add(
        [
            StoredChunk(
                id=f'd{i}-0',
                text=f'文档{i}内容',
                metadata={'document_id': i, 'chapter': '正文', 'chapter_index': 0},
                embedding=[1.0, 0.0],
            )
            for i in (1, 2)
        ]
    )
    app = create_app(
        session_factory=factory,
        vector_store=store,
        llm=llm,
        embedding_model=TurnEmb(),
        upload_dir=tmp_path / 'uploads',
    )
    return TestClient(app), factory


def _ask(client: TestClient, sid: str, question: str, doc: int | None = None) -> dict:
    payload = {'question': question}
    if doc is not None:
        payload['document_ids'] = [doc]
    return client.post(f'/api/sessions/{sid}/messages', json=payload).json()


class TestMultiTurnContext:
    def test_history_recall_answers_from_record(self, tmp_path: Path) -> None:
        llm = RouterLLM()
        client, _ = _make_env(tmp_path, llm)
        with client:
            sid = client.post('/api/sessions').json()['session_id']
            first = _ask(client, sid, '张三喜欢谁？', doc=1)
            assert first['answer'] and first['needs_clarification'] is False
            assert llm.gate_calls == 1, '首问也要过检索门(否则 hello 会掉进书链路)'

            meta = _ask(client, sid, '我上一个问题是什么？')
            assert '张三喜欢谁？' in meta['answer'], meta
            assert meta['citations'] == []
            assert meta['needs_clarification'] is False
            assert meta['intent'] == 'history', '历史回忆轮应暴露 history 意图'
            assert llm.gate_calls == 2

            msgs = client.get(f'/api/sessions/{sid}/messages').json()
            user_msgs = [m['content'] for m in msgs if m['role'] == 'user']
            assert user_msgs == ['张三喜欢谁？', '我上一个问题是什么？'], '元问题轮次也应写回记录'

    def test_first_turn_greeting_is_chat(self, tmp_path: Path) -> None:
        """回归:会话第一问发 hello,应走 chat 直答,不得被判信息不足建 HITL。"""
        llm = RouterLLM()
        llm.context_reply = '你好呀，想从书里了解点什么？'
        client, factory = _make_env(tmp_path, llm)
        with client:
            sid = client.post('/api/sessions').json()['session_id']
            resp = _ask(client, sid, 'hello')
            assert resp['intent'] == 'chat', resp
            assert resp['answer'] == '你好呀，想从书里了解点什么？'
            assert resp['needs_clarification'] is False
            assert resp.get('hitl_task_id') is None, '寒暄不应创建澄清任务'
            assert resp['citations'] == []
            with factory() as session:
                assert list(session.scalars(select(QaCacheEntry))) == [], '寒暄不入缓存'

    def test_chitchat_skips_retrieval(self, tmp_path: Path) -> None:
        llm = RouterLLM()
        llm.context_reply = '你好呀，想从书里了解点什么？'
        client, _ = _make_env(tmp_path, llm)
        with client:
            sid = client.post('/api/sessions').json()['session_id']
            assert _ask(client, sid, '张三喜欢谁？', doc=1)['answer']
            r = _ask(client, sid, '你好')
            assert r['answer'] == '你好呀，想从书里了解点什么？'
            assert r['intent'] == 'chat'
            assert r['citations'] == []
            assert r['needs_clarification'] is False
            assert llm.context_calls == 1

    def test_book_followup_injects_history(self, tmp_path: Path) -> None:
        llm = RouterLLM()
        client, _ = _make_env(tmp_path, llm)
        with client:
            sid = client.post('/api/sessions').json()['session_id']
            assert _ask(client, sid, '张三喜欢谁？', doc=1)['answer']
            follow = _ask(client, sid, '张三刚才那句话是什么意思？', doc=1)
            assert follow['answer'], '书问题应照常走检索回答'
            assert follow['needs_clarification'] is False
            answer_prompts = [
                p for p in llm.prompts if '原文片段' in p and '张三刚才那句话' in p
            ]
            assert answer_prompts, '第二轮书问题应进入 answer 节点'
            assert '最近对话' in answer_prompts[0], 'answer prompt 应注入对话记录'

    def test_context_turns_do_not_pollute_cache(self, tmp_path: Path) -> None:
        llm = RouterLLM()
        client, factory = _make_env(tmp_path, llm)
        with client:
            sid = client.post('/api/sessions').json()['session_id']
            assert _ask(client, sid, '张三喜欢谁？', doc=1)['answer']
            assert _ask(client, sid, '我上一个问题是什么？')['answer']
            assert _ask(client, sid, '你好')['answer']
            assert _ask(client, sid, '李四喜欢谁？', doc=2)['answer']
            with factory() as session:
                rows = list(session.scalars(select(QaCacheEntry)))
            assert len(rows) == 2, '只有书问题产生缓存，history/chat 不得入库'

    def test_repeat_book_question_still_cached(self, tmp_path: Path) -> None:
        """多轮下同问题重问仍应命中缓存(有历史不破坏单轮缓存语义)。"""
        llm = RouterLLM()
        client, factory = _make_env(tmp_path, llm)
        with client:
            sid = client.post('/api/sessions').json()['session_id']
            question = '张三喜欢谁？'
            first = _ask(client, sid, question, doc=1)
            assert _ask(client, sid, '你好')['answer']
            third = _ask(client, sid, question, doc=1)
            assert third['answer'] == first['answer'], '同问同文档应命中缓存'
            assert third['intent'] == 'book'
            with factory() as session:
                rows = list(session.scalars(select(QaCacheEntry)))
            assert len(rows) == 1, '重问应命中缓存而不是新增条目'


class TestLongSessionSummary:
    """M3 长会话：超过窗口后滚动增量摘要注入(更早对话摘要段)与落库。"""

    def test_overflow_turns_trigger_incremental_summary(self, tmp_path: Path) -> None:
        llm = RouterLLM()
        client, factory = _make_env(tmp_path, llm)
        with client:
            sid = client.post('/api/sessions').json()['session_id']
            turns = [
                ('张三喜欢谁？', 1),
                ('李四喜欢谁？', 2),
                ('王五喜欢谁？', 1),
                ('赵六喜欢谁？', 2),
                ('张三喜欢谁？', 1),  # 重复首问 → 命中缓存,但触发摘要
            ]
            for q, doc in turns:
                resp = _ask(client, sid, q, doc=doc)
                assert resp['answer'], q
            assert llm.summarize_calls >= 1, '消息超窗口后应触发滚动摘要'
            assert any('更早对话摘要' in p for p in llm.prompts), 'gate/context prompt 应注入摘要段'

            early = _ask(client, sid, '我第一个问题是什么？')
            assert early['intent'] == 'history'
            assert early['answer'] and early['needs_clarification'] is False

            with factory() as session:
                chat = session.get(ChatSession, int(sid))
                assert chat is not None and chat.summary, '摘要应落库'
                data = json.loads(chat.summary)
                assert isinstance(data, dict) and 'upto' in data and data['text']
            assert early['answer']  # 历史轮次照常作答
