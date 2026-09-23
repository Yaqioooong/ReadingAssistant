"""对话式查询改写（CQR）：指代消解。

回归的是一条具体故障：``retrieve`` 的 query 就是原始问题字符串，
**对话历史只喂给「作答」、没喂给「检索」** —— 于是「它的特点是什么？」里的「它」
对检索器就是空气，答案所在那一章根本捞不上来。

本文件钉死的三条硬性质：
1. **只在含代词的追问上触发** —— 简单问题必须零改动（否则无辜实体被塞进 query，稀释信号）。
2. **确定性纯函数** —— 节点会被 LangGraph 重放，非确定性改写会让重放产生不同查询。
3. **缓存键用改写后的查询** —— 否则同一句「它的特点是什么？」在两个上文里共用缓存、
   直接返回上一个上下文的答案（这是引入改写前就有的隐患）。
"""
from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from reading_assistant.graph.rewrite import (
    collect_candidates,
    detect_pronoun,
    extract_entities,
    resolve_query,
)
from reading_assistant.graph.state import ConversationState, merge_state  # noqa: F401  (存在性自检)

# 真实失败用例（eval/agent_gold.json 的 ag-pronoun-01/02），逐字照抄
H_PRONOUN_01 = [
    {'role': 'user', 'content': '西游记中唐僧师徒遇到的第一个妖怪是什么？'},
    {'role': 'assistant', 'content': '是双叉岭上的老虎精寅将军。'},
]
H_PRONOUN_02 = [
    {'role': 'user', 'content': '白骨精的故事讲了什么？'},
    {'role': 'assistant', 'content': '白骨精三次变化想吃唐僧，被孙悟空识破并打死。'},
]


class TestPronounDetection:
    def test_detects_common_pronouns(self) -> None:
        for q in ['它的特点是什么？', '他是怎么被收服的？', '她的结局如何？',
                  '这个妖怪的特点是什么？', '其特点是什么？', '它们之间有什么关系？']:
            assert detect_pronoun(q), q

    def test_does_not_fire_on_plain_questions(self) -> None:
        """简单问题必须零触发 —— 这是「不拖累既有单问」的前提。"""
        for q in ['唐僧师徒遇到的第一个妖怪是什么？',
                  '孙悟空是怎么被压到五行山下的？',
                  '白骨精的故事讲了什么？',
                  'P-002 的上市时间是什么时候？',
                  '寅将军是什么妖怪？']:
            assert detect_pronoun(q) == [], q

    def test_substring_false_positive_is_rejected(self) -> None:
        """「应该」里的「该」不能算代词。

        判据必须是「单独成词 **且** 词性为 r」——
        拿 `in` 做子串匹配会在这里误触发，让简单问题也走上改写路径。
        """
        assert detect_pronoun('我应该怎么办？') == []
        assert detect_pronoun('应该先做哪一步？') == []


class TestEntityExtraction:
    def test_merges_adjacent_name_tokens(self) -> None:
        """相邻名词块要并起来：单看词性只会得到「黄眉」+「大王」两个碎块。"""
        assert '黄眉大王' in extract_entities('黄眉大王本是弥勒佛座下司磬的童儿。')

    def test_extracts_entity_split_by_adjective(self) -> None:
        """「老虎精寅将军」→ 寅将军。

        jieba 词典里没有「寅将军」（切成 寅(mg) + 将军(n)），
        靠「形容词 '精' 断块、'mg'+'n' 合块」才把它拼回来 ——
        这条正是 ag-pronoun-01 能否被救回来的关键。
        """
        assert '寅将军' in extract_entities('是双叉岭上的老虎精寅将军。')

    def test_generic_words_are_filtered(self) -> None:
        got = extract_entities('西游记中唐僧师徒遇到的第一个妖怪是什么？')
        assert '妖怪' not in got
        assert '西游记' not in got

    def test_empty_input(self) -> None:
        assert extract_entities('') == []

    def test_candidates_prefer_recent_messages(self) -> None:
        """由近及远 —— 指代指向的几乎总是最近提到的那个东西。"""
        history = [
            {'role': 'user', 'content': '很早以前问过孙悟空的事。'},
            {'role': 'assistant', 'content': '后来聊到了寅将军。'},
        ]
        got = collect_candidates(history, limit=2)
        assert got and got[0] == '寅将军'


class TestResolveQuery:
    def test_resolves_pronoun_01_from_assistant_reply(self) -> None:
        """实体只在**助手回复**里 —— 用户的问句是「第一个妖怪是什么」，里面没有实体。"""
        resolved, entities = resolve_query('这个妖怪的特点是什么？', H_PRONOUN_01)
        assert '寅将军' in entities
        assert '寅将军' in resolved
        assert resolved.startswith('这个妖怪的特点是什么？'), '追加而非替换'

    def test_resolves_pronoun_02_candidates(self) -> None:
        resolved, entities = resolve_query('他是怎么被收服的？', H_PRONOUN_02)
        # 候选集合而非单一实体：jieba 抽不出「寅将军」这类词，集合能容错
        assert '孙悟空' in entities
        assert '白骨精' in entities

    def test_appends_instead_of_replacing(self) -> None:
        """代词本身不携带信息，但整句其余部分（「特点」「怎么被收服」）才是问题核心。

        替换只会让查询更短更糊；追加是纯增量信号。
        """
        question = '它的特点是什么？'
        resolved, _ = resolve_query(question, H_PRONOUN_01)
        assert question in resolved

    def test_plain_question_is_untouched(self) -> None:
        """零改动、零开销 —— 简单问题不该被改写波及。"""
        q = '唐僧师徒遇到的第一个妖怪是什么？'
        assert resolve_query(q, H_PRONOUN_01) == (q, [])

    def test_pronoun_without_history_is_untouched(self) -> None:
        """首轮就说「它」→ 抽不出实体 → 保持原查询。

        硬塞空列表只会让日志出现「改写成功但没变化」的假信号。
        """
        q = '它是什么？'
        assert resolve_query(q, []) == (q, [])
        assert resolve_query(q, None) == (q, [])

    def test_disabled_switch(self) -> None:
        q = '它的特点是什么？'
        assert resolve_query(q, H_PRONOUN_01, enabled=False) == (q, [])

    def test_is_deterministic(self) -> None:
        """必须可重放：节点会在 LangGraph 重放，非确定性改写会让同一轮产生不同查询。"""
        runs = [resolve_query('这个妖怪的特点是什么？', H_PRONOUN_01) for _ in range(3)]
        assert runs[0] == runs[1] == runs[2]

    def test_max_entities_caps_candidates(self) -> None:
        resolved, entities = resolve_query(
            '它的特点是什么？', H_PRONOUN_01, max_entities=2
        )
        assert len(entities) <= 2
        assert resolved  # 仍然改写


class TestGraphIntegration:
    """端到端：缓存键必须绑定改写后的查询，否则跨上下文串答案。"""

    @staticmethod
    def _client(tmp_path: Path, llm):
        from reading_assistant.api import create_app
        from reading_assistant.storage import create_db_engine, create_session_factory, init_db
        from reading_assistant.storage.models import Document
        from reading_assistant.storage.vector_store import InMemoryVectorStore, StoredChunk

        engine = create_db_engine('sqlite:///:memory:')
        init_db(engine)
        factory = create_session_factory(engine)
        with factory() as session:
            session.add_all([
                Document(filename='d1.docx', title='d1', file_hash='f1',
                         content_hash='c1', index_status='indexed'),
            ])
            session.commit()
        store = InMemoryVectorStore()
        store.add([
            StoredChunk(id='d1-0', text='寅将军内容',
                        metadata={'document_id': 1, 'chapter': '正文', 'chapter_index': 0},
                        embedding=[1.0, 0.0]),
        ])
        emb = _Emb()
        app = create_app(session_factory=factory, vector_store=store, llm=llm,
                         embedding_model=emb, upload_dir=tmp_path / 'uploads')
        return TestClient(app), emb

    def test_pronoun_turn_exposes_entities_and_rewrites(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        from reading_assistant.config import get_settings
        from tests.test_multi_turn import RouterLLM

        # 必须先禁缓存：本轮要验的是「改写后的查询真的进了检索」，
        # 而缓存一旦命中就**根本不跑 retrieve** —— 变量会被缓存吃掉。
        # （这正是本项目反复踩的那个坑：跑 A/B 或统计先 CACHE_ENABLED=false。）
        monkeypatch.setenv('CACHE_ENABLED', 'false')
        get_settings.cache_clear()

        llm = RouterLLM()
        client, emb = self._client(tmp_path, llm)
        with client:
            sid = client.post('/api/sessions').json()['session_id']
            client.post(f'/api/sessions/{sid}/messages',
                        json={'question': '寅将军是什么妖怪？', 'document_ids': [1]})
            resp = client.post(f'/api/sessions/{sid}/messages',
                               json={'question': '它的特点是什么？',
                                     'document_ids': [1]}).json()
        # 观测字段透出（先行指标：cqr 命中率可监控）
        assert resp.get('cqr_entities'), '含代词的追问应透出消解出的实体'
        assert '寅将军' in resp['cqr_entities']
        # 且改写后的文本真的进了**检索调用**（而不是只算了不用）。
        # ⚠️ 断言对象是检索器收到的 query，不是 llm.prompts —— 检索走 embedding，
        # 根本不经过 chat 模型。这一点我第一次就断言错了。
        assert any('上文相关实体' in q for q in emb.queries)


class _Emb:
    """恒定向量：让检索稳定命中，避免测试依赖真实 embedding。

    顺带记录收到的 query —— 这是**唯一**能证明「改写进了检索」的观测点。
    """

    def __init__(self) -> None:
        self.queries: list[str] = []

    def embed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        return [1.0, 0.0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


class TestOwnAnchorGate:
    """问句自带专名时**不改写**。

    依据是实测：ag-ordinal-04「唐僧收孙悟空为徒后…它的技能是什么？」里
    既有专名「孙悟空」又有代词「它」，改写触发后追加了一批上文实体 →
    该题由 PASS 翻成 FAIL（「多问合一互相稀释」这一已知失败模式）。
    """

    def test_question_with_proper_noun_is_untouched(self) -> None:
        q = '唐僧收孙悟空为徒后，孙悟空消灭的第一个妖精是谁？它的技能是什么？'
        assert detect_pronoun(q), '这句确实含代词，是有意义的对照'
        assert resolve_query(q, H_PRONOUN_01) == (q, [])

    def test_pronoun_only_question_still_rewrites(self) -> None:
        """闸门不能把真正需要改写的追问也挡掉 —— 这几条必须仍然触发。"""
        for q in ['它的特点是什么？', '他是怎么被收服的？', '这个妖怪的特点是什么？']:
            resolved, entities = resolve_query(q, H_PRONOUN_01)
            assert resolved != q, q
            assert entities, q

    def test_verb_mis_tagged_as_noun_does_not_count_as_anchor(self) -> None:
        """判据不能用 'n'：jieba 把「收服」标成名词，用它当锚点会让 pronoun-02 漏改写。"""
        q = '他是怎么被收服的？'
        assert resolve_query(q, H_PRONOUN_02)[0] != q


class TestCandidateRecall:
    """候选抽取的**召回**，而不是排序。

    这条线的由来（2026-09-23 实测）：
    「哪一个是正确的指称」这件事，我试过三种信号都失败 ——
    原始共现计数、逐候选假设检索分数、谓语同义族共现，**全都偏向最频繁的实体**
    （例：皈依 ∩ 唐僧 = 16 > ∩ 孙悟空 = 7）。
    「语料在场块数最少者优先」在已见过的 2 条上 2/2，但在我**事先写好**的 6 条上 0/6
    —— 纯属巧合，已否决。
    真正能量化改进的是**召回**：按生产配置（只注入 4 个候选）指称在候选集里的
    覆盖率只有 4/8。下面钉死修好的那几条。
    """

    def test_older_message_not_crowded_out_by_long_reply(self) -> None:
        """较新的**长回复**不得把名额吃光 —— 指称可能只在较早的提问里。

        依据：ag-pronoun-07 的「黑熊精」只出现在用户问句里，
        而助手那条回复自带 5~6 个实体。
        """
        history = [
            {'role': 'user', 'content': '黑风山的黑熊精是什么来历？'},
            {'role': 'assistant',
             'content': '黑风山黑风洞的妖怪偷了唐僧的锦襕袈裟，与孙悟空相斗，'
                        '善使黑缨枪，观音院的金池长老也牵涉其中。'},
        ]
        got = collect_candidates(history, limit=6)
        assert any('黑熊' in x for x in got), f'较早提问里的指称被挤掉了：{got}'
        assert len(got) <= 6

    def test_round_robin_interleaves_messages(self) -> None:
        """轮转：两条消息各出第一个，再各出第二个。"""
        history = [
            {'role': 'user', 'content': '甲山的甲怪是什么？'},
            {'role': 'assistant', 'content': '乙山乙洞的乙怪和丙山丙洞的丙怪都与此有关。'},
        ]
        got = collect_candidates(history, limit=6)
        # 助理（较新）的第一个应排在用户（较旧）的第一个之前
        assert got[0].startswith('乙'), got
        # 但用户的实体必须在助理的后续实体之前出现
        assert any('甲' in x for x in got[:3]), got

    def test_state_word_tag_is_recognised(self) -> None:
        """'z'（状态词）要收：jieba 把「黑熊」标成 z。

        不收就会把「黑熊精」这类名字拦腰切断 —— 实测 ag-pronoun-07 因此完全抽不到指称。
        """
        got = extract_entities('黑熊精偷了锦襕袈裟。')
        assert any('黑熊' in x for x in got), got

    def test_proper_noun_blocks_rank_before_plain_nouns(self) -> None:
        """含专名词性的块排在纯通用名词块之前。

        依据：纯 `n` 的块大量是「干娘/童子/下界」这类通用词，
        而指称几乎总是真专名（九尾狐/寅将军）。
        """
        got = extract_entities('他们还有个干娘，是压龙山的九尾狐。')
        assert any('九尾狐' in x for x in got), got
        assert got.index(next(x for x in got if '九尾狐' in x)) < got.index('干娘'), got

    def test_default_cap_meets_measured_floor(self) -> None:
        """配置下限护栏：实测指称召回 4→6/8（cap=4）、7/8（cap=6）。

        把 cap 调回 4 会让召回退化，这条测试就是为了拦住它。
        """
        from reading_assistant.config import get_settings

        assert get_settings().cqr_max_entities >= 6

    def test_simple_question_candidates_unchanged(self) -> None:
        """简单问题的抽取不受影响（不引入新垃圾）。"""
        got = extract_entities('唐僧师徒遇到的第一个妖怪是什么？')
        assert '唐僧' in got
        assert '妖怪' not in got and '西游记' not in got
