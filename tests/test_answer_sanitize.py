"""模型自由文本的出口清洗：工具参数是唯一没有护栏的通道。

## 现场（2026-09-24，用户把回答贴回来问「这类格式错误应该如何规避」）

回答正文末尾带着 ``</answer>\\n</invoke>\\n``。全仓 grep（含 ``config/prompt.yml``）为空
→ 不是我们写进去的，是模型附加在**工具参数字符串**末尾的。
库里三条同形态（``id=2874`` 回答正文、``id=2787/2861``「信息不足」说明文本）——
``final_answer`` 与 ``request_clarification`` **两个工具都中招**。

## 为什么这不是「难看」而是缺陷

带标签的文本写进 ``chat_messages`` 后，此后每轮 prompt 的历史里都带着它，
模型继续模仿自己上一轮的格式 —— **自我强化的反馈环**。
所以本文件同时钉住两半：出口（写库/返回）**拦新的**、入口（组 prompt）**中和存量**。
"""
from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient
from sqlalchemy import select

from reading_assistant.storage.models import ChatMessage, QaCacheEntry
from reading_assistant.storage.vector_store import InMemoryVectorStore
from reading_assistant.utils.answer_text import sanitize_model_text
from tests.test_api import FakeToolLLM, _make_app, _upload_txt

# 库里 id=2874 那条，逐字照抄（截去正文中段，只保留末尾的残渣形态）
RECORDED = '路旁闪出的女子是个妖精，即"毒敌山琵琶洞"的女怪。[1]</answer>\n</invoke>\n'


class TestSanitizeModelText:
    def test_strips_the_recorded_artifact(self) -> None:
        """实测形态：两层标签 + 换行，剥干净且不碰正文。"""
        assert sanitize_model_text(RECORDED) == (
            '路旁闪出的女子是个妖精，即"毒敌山琵琶洞"的女怪。[1]'
        )

    def test_strips_leading_and_namespaced_and_multiple_layers(self) -> None:
        for raw, want in [
            ('<answer>正文</answer>', '正文'),
            ('正文</answer>\n</invoke>', '正文'),
            ('\n\n正文\n\n</answer>\n</invoke>\n\n', '正文'),
            ('正文</antml:invoke>', '正文'),          # 允许命名空间前缀
            ('<answer>正文</answer>\n</invoke>', '正文'),
            ('正文<answer>', '正文'),                  # 未闭合的开标签也是残渣
        ]:
            assert sanitize_model_text(raw) == want, raw

    def test_is_idempotent(self) -> None:
        once = sanitize_model_text(RECORDED)
        assert sanitize_model_text(once) == once

    def test_clean_text_is_returned_untouched(self) -> None:
        """干净文本必须**原对象**返回——这个函数要无脑挂在所有出口上。

        断言 `is` 而不是 `==`：返回原对象意味着连空白都没有被顺手改掉。
        """
        for clean in ['这是一条正常回答。[1][2]', '没有尖括号的回答', '']:
            assert sanitize_model_text(clean) is clean

    def test_interior_occurrence_is_left_alone(self) -> None:
        """正文中间的同类标签是**内容**不是残渣 —— 只剥首尾，绝不做全局替换。

        依据：正文里会出现引号、破折号、`[1]`，也可能引用含尖括号的原文；
        在中间做「去掉所有 <...>」的正则会误伤正文。
        """
        text = '原文里有一句 </invoke> 这样的词，但它出现在句子中间。'
        assert sanitize_model_text(text) is text

    def test_unknown_tags_are_not_stripped(self) -> None:
        """按**白名单**认标签：不认识的尖括号一律不碰。

        收得越宽越容易误伤，宁可漏。
        """
        for text in ['<foo>正文</foo>', '正文 <div>', '<person>张三</person>']:
            assert sanitize_model_text(text) is text

    def test_all_artifact_falls_back_to_original(self) -> None:
        """整条都是标签 → 回退原文并告警。

        本函数只管格式，不做「这条回答是不是没内容」的判断；
        更不能把一个非空回答变成空字符串（那等于把判断权从调用方手里拿走）。
        """
        assert sanitize_model_text('</invoke>') == '</invoke>'

    def test_falsy_passes_through(self) -> None:
        assert sanitize_model_text(None) is None
        assert sanitize_model_text('') == ''


class TestAnswerPathIsSanitized:
    """出口：模型污染的工具参数不得到达响应 / messages 表 / QA 缓存。"""

    def _ask(self, tmp_path: Path, answer: str):
        app, engine = _make_app(
            tmp_path, InMemoryVectorStore(), llm=FakeToolLLM('final_answer', {'answer': answer})
        )
        with TestClient(app) as client:
            doc = _upload_txt(client).json()
            sid = client.post('/api/sessions').json()['session_id']
            body = client.post(
                f'/api/sessions/{sid}/messages',
                json={'question': '张三是谁', 'document_ids': [doc['id']]},
            ).json()
            return body, engine, sid

    def test_api_response_is_clean(self, tmp_path: Path) -> None:
        body, engine, _ = self._ask(tmp_path, RECORDED)
        assert body['answer'] == '路旁闪出的女子是个妖精，即"毒敌山琵琶洞"的女怪。[1]'
        engine.dispose()

    def test_stored_message_and_cache_are_clean(self, tmp_path: Path) -> None:
        """三处去向必须一起干净：只修响应 = 把残渣留在库里继续教坏模型。"""
        from reading_assistant.storage import create_session_factory

        body, engine, sid = self._ask(tmp_path, RECORDED)
        assert body['answer']
        with create_session_factory(engine)() as session:
            stored = [
                m.content
                for m in session.scalars(
                    select(ChatMessage).where(ChatMessage.role == 'assistant')
                )
            ]
            assert stored, '应当落库一条助手消息'
            assert all('</invoke>' not in c and '</answer>' not in c for c in stored), stored
            cached = [e.answer for e in session.scalars(select(QaCacheEntry))]
            assert all('</invoke>' not in (a or '') for a in cached), cached
        engine.dispose()


class TestHistoryIsSanitized:
    """入口：存量的带标签历史不得再进 prompt（否则出口修好了、模型还在照抄）。"""

    def test_polluted_history_does_not_reach_the_prompt(self, tmp_path: Path) -> None:
        import json

        from reading_assistant.storage import create_session_factory
        from reading_assistant.storage.models import ChatSession

        seen: list[str] = []

        class RecordingLLM(FakeToolLLM):
            def invoke(self, prompt):
                seen.append(json.dumps(prompt, ensure_ascii=False, default=str))
                return super().invoke(prompt)

        app, engine = _make_app(
            tmp_path,
            InMemoryVectorStore(),
            llm=RecordingLLM('final_answer', {'answer': '干净回答。'}),
        )
        with TestClient(app) as client:
            doc = _upload_txt(client).json()
            sid = client.post('/api/sessions').json()['session_id']
            # 手工种一条**上一轮**的污染历史（模拟修复前存下来的行）
            with create_session_factory(engine)() as session:
                session.add(
                    ChatMessage(session_id=sid, role='assistant', content=RECORDED)
                )
                session.add(
                    ChatMessage(session_id=sid, role='user', content='上一轮用户消息')
                )
                session.commit()
            assert session.get(ChatSession, sid) is not None  # noqa: F841 存在性自检

            seen.clear()
            client.post(
                f'/api/sessions/{sid}/messages',
                json={'question': '张三是谁', 'document_ids': [doc['id']]},
            )

        joined = '\n'.join(seen)
        assert seen, '模型应当收到 prompt'
        assert '</invoke>' not in joined, '污染的存量历史被原样喂进了 prompt'
        assert '</answer>' not in joined
        assert '路旁闪出的女子是个妖精' in joined, '只该剥标签，正文必须保留'
        engine.dispose()


class TestLegacyPollutedCacheIsNeutralized:
    """存量：修复**之前**写进 QA 缓存的条目里已经有带标签的（实测 3 条）。

    「修好写路径」只保证此后不再新增，管不住已经躺在缓存里的旧条目 ——
    命中时它们会把残渣原样带回来。所以命中路径也要过同一道清洗。
    """

    CLEAN = '这是基于原文的测试回答。'

    def test_polluted_cache_hit_returns_clean_answer(self) -> None:
        from reading_assistant.storage.models import Document
        from reading_assistant.storage.service import normalize_question, sha256_hex
        from tests.test_cache_version_key import QUESTION, _Env

        env = _Env()
        env.add_document(1, 'a1')
        # 手工种一条**修复前**形态的缓存（answer 带残渣），哈希按生产算法算，保证走精确命中
        question_hash = sha256_hex(
            normalize_question(QUESTION) + '|' + normalize_question('') + '|' + '1'
        )
        with env.factory() as session:
            assert session.get(Document, 1).content_hash == 'a1'
            session.add(
                QaCacheEntry(
                    question_raw=QUESTION,
                    question_normalized=normalize_question(QUESTION),
                    question_hash=question_hash,
                    question_embedding=[],
                    answer=self.CLEAN + '</answer>\n</invoke>\n',
                    citations=[],
                    needs_clarification=False,
                    document_id=1,
                    content_hash='a1',
                )
            )
            session.commit()

        result = env.ask(env.graph(), document_id=1, turn='legacy-cache')
        assert result.get('cache_hit') is True, '这条应当是缓存命中路径，否则本测试没测到东西'
        assert result['answer'] == self.CLEAN


class TestHistoryViewIsSanitized:
    """用户可见面：app 里往回翻的聊天记录。

    用户就是把这条贴回来问的 —— 所以「回答正文修干净了」还不够，
    历史列表里那几条存量必须也看不见残渣。
    """

    def test_polluted_stored_message_is_clean_in_history(self, tmp_path: Path) -> None:
        from reading_assistant.storage import create_session_factory

        app, engine = _make_app(
            tmp_path, InMemoryVectorStore(), llm=FakeToolLLM('final_answer', {'answer': 'x'})
        )
        with TestClient(app) as client:
            sid = client.post('/api/sessions').json()['session_id']
            with create_session_factory(engine)() as session:
                session.add(ChatMessage(session_id=sid, role='assistant', content=RECORDED))
                session.commit()

            rows = client.get(f'/api/sessions/{sid}/messages').json()

        assert rows, '历史接口应当返回那条消息'
        content = rows[-1]['content']
        assert '</answer>' not in content and '</invoke>' not in content, content
        assert '路旁闪出的女子是个妖精' in content, '只该剥标签，正文必须保留'
        engine.dispose()

    def test_history_does_not_mutate_user_messages(self, tmp_path: Path) -> None:
        """用户消息是用户的输入，不替他们改写（哪怕里面就有这种标签）。"""
        from reading_assistant.storage import create_session_factory

        user_text = '这个回答后面为什么带 </answer></invoke>？'
        app, engine = _make_app(
            tmp_path, InMemoryVectorStore(), llm=FakeToolLLM('final_answer', {'answer': 'x'})
        )
        with TestClient(app) as client:
            sid = client.post('/api/sessions').json()['session_id']
            with create_session_factory(engine)() as session:
                session.add(ChatMessage(session_id=sid, role='user', content=user_text))
                session.commit()
            rows = client.get(f'/api/sessions/{sid}/messages').json()

        assert rows[-1]['content'] == user_text
        engine.dispose()
