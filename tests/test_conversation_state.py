"""会话结构化状态：确定性合并 + 坏数据不炸 + 落库往返 + 旧摘要列收敛。

这套状态取代滚动摘要，靠的是三条硬性质：
1. **零 LLM 成本** —— 字段全部来自流水线已有信号（本文件的单测里没有任何模型调用）；
2. **确定性** —— ``merge_state`` 是纯函数，模型不参与记忆生成；
3. **幂等** —— 字段是本轮输入的函数，``turn_count`` 从消息表数出而非自增。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from reading_assistant.graph.state import (
    ConversationState,
    from_json,
    load_conversation_state,
    merge_state,
    save_conversation_state,
    to_json,
)
from reading_assistant.storage import ChatSession, create_db_engine, create_session_factory, init_db


class TestMergeState:
    """``merge_state`` 是这套记忆**唯一**的写入路径，语义必须钉死。"""

    def test_none_prev_starts_from_empty(self) -> None:
        merged = merge_state(None, {'last_question': '问'})
        assert merged.last_question == '问'
        assert merged.active_documents == []

    def test_empty_patch_keeps_prev(self) -> None:
        prev = ConversationState(active_documents=['a'], last_question='q', turn_count=3)
        assert merge_state(prev, {}) == prev
        assert merge_state(prev, None) == prev

    def test_none_and_empty_values_do_not_overwrite(self) -> None:
        """非空才覆盖。

        否则一轮「没检索到东西」的追问就会把 active_documents 清空 ——
        而那恰恰是最需要它还记着的场景。
        """
        prev = ConversationState(active_documents=['a'], last_question='q')
        merged = merge_state(
            prev, {'active_documents': [], 'last_question': '', 'last_intent': None}
        )
        assert merged == prev

    def test_non_empty_value_overwrites(self) -> None:
        prev = ConversationState(last_intent='book')
        assert merge_state(prev, {'last_intent': 'chat'}).last_intent == 'chat'

    def test_unknown_keys_are_ignored(self) -> None:
        """库里可能存着旧版本写的字段 —— 字段增删不能让写入炸掉。

        （这是面向客服扩展槽位时的守门人：加 order_id 不能让旧记录读不出。）
        """
        merged = merge_state(None, {'order_id': 'A123', 'last_intent': 'book'})
        assert merged.last_intent == 'book'
        assert not hasattr(merged, 'order_id')

    def test_does_not_mutate_input(self) -> None:
        prev = ConversationState(last_question='q')
        merged = merge_state(prev, {'last_question': 'r'})
        assert prev.last_question == 'q'
        assert merged.last_question == 'r'


class TestSerialization:
    def test_roundtrip(self) -> None:
        state = ConversationState(active_documents=['a'], last_question='q', turn_count=2)
        assert from_json(to_json(state)) == state

    def test_json_keeps_chinese_readable(self) -> None:
        """ensure_ascii=False —— 库里要能直接看懂，排查时才不用先解码。"""
        assert '张三喜欢谁？' in to_json(ConversationState(last_question='张三喜欢谁？'))

    @pytest.mark.parametrize('raw', ['{不是 json', '[1,2,3]', '', None, 'null'])
    def test_bad_data_returns_none_instead_of_raising(self, raw: str | None) -> None:
        """读路径绝不能因为一条脏记录把整道题带崩。

        旧格式、半截写入、人工改库在本项目都真实发生过。
        """
        assert from_json(raw) is None

    def test_unknown_fields_dropped_on_read(self) -> None:
        state = from_json('{"last_intent": "book", "legacy_field": 1}')
        assert state is not None
        assert state.last_intent == 'book'


class TestPersistenceAndMigration:
    @staticmethod
    def _factory(tmp_path: Path, name: str = 'state.db'):
        engine = create_db_engine(f'sqlite:///{tmp_path / name}')
        init_db(engine)
        return create_session_factory(engine)

    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        factory = self._factory(tmp_path)
        with factory() as session:
            chat = ChatSession(title='t')
            session.add(chat)
            session.commit()
            save_conversation_state(
                session, chat.id, ConversationState(active_documents=['x'], turn_count=1)
            )
            session.commit()
            loaded = load_conversation_state(session, chat.id)
        assert loaded is not None
        assert loaded.active_documents == ['x']
        assert loaded.turn_count == 1

    def test_load_missing_session_returns_none(self, tmp_path: Path) -> None:
        factory = self._factory(tmp_path)
        with factory() as session:
            assert load_conversation_state(session, 99999) is None

    def test_save_missing_session_is_silent(self, tmp_path: Path) -> None:
        """会话不存在时静默跳过，不制造幽灵行。"""
        factory = self._factory(tmp_path)
        with factory() as session:
            save_conversation_state(session, 99999, ConversationState())
            session.commit()
            assert session.get(ChatSession, 99999) is None

    def test_load_on_empty_column_returns_none(self, tmp_path: Path) -> None:
        factory = self._factory(tmp_path)
        with factory() as session:
            chat = ChatSession(title='t')
            session.add(chat)
            session.commit()
            assert load_conversation_state(session, chat.id) is None

    def test_migration_purges_deprecated_summary(self, tmp_path: Path) -> None:
        """收敛式迁移：每次启动都把废弃的滚动摘要清掉。

        迁移的职责是**把库收敛到目标形态**，不是「只在首次正确」。
        留着非空的旧摘要是本项目最忌讳的形态 —— 两处都在、没人知道该信哪个
        （前科：文档打回 indexing、前端没 rebuild、shell 旧 key 盖 .env）。
        """
        factory = self._factory(tmp_path, 'migration.db')
        with factory() as session:
            chat = ChatSession(title='t', summary=json.dumps({'upto': 3, 'text': '旧摘要'}))
            session.add(chat)
            session.commit()
            sid = chat.id

        # 再跑一次 init_db：真实场景就是服务重启
        init_db(factory.kw['bind'])
        with factory() as session:
            reloaded = session.get(ChatSession, sid)
            assert reloaded is not None
            assert reloaded.summary is None, '旧摘要列必须被清空'
            # 列本身保留（不 DROP —— 用户有活库，破坏性操作要单独走）
            assert reloaded.state is None

    def test_state_and_summary_are_separate_columns(self, tmp_path: Path) -> None:
        factory = self._factory(tmp_path, 'cols.db')
        with factory() as session:
            chat = ChatSession(title='t')
            session.add(chat)
            session.commit()
            save_conversation_state(session, chat.id, ConversationState(last_intent='book'))
            session.commit()
            reloaded = session.get(ChatSession, chat.id)
            assert reloaded is not None
            assert reloaded.state
            assert reloaded.summary is None
