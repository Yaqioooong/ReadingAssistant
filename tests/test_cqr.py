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
    detect_descriptive_reference,
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

# 2026-09-24 端到端失败那一轮的上下文，逐字照抄（含中间那轮闲聊，
# 因为它会往候选里掺「助手」「hello」—— 噪声是在场的，判据必须扛得住）
H_DEFINITE_NP = [
    {'role': 'user', 'content': (
        '西游记中师徒四人到达女儿国时发生了什么事？国王对唐僧做了什么？唐僧如何处理的？'
    )},
    {'role': 'assistant', 'content': (
        '师徒四人到女儿国时，女王先是远送他们，随后在分别时公然挽留唐僧：她称愿以一国之富招'
        '唐僧为夫，让他登位称君，自己做王后。唐僧对此的处理是：他慢下龙车，向女王拱手告辞。'
        '八戒也上前撒泼，把女王吓得跌入辇驾之中；随后沙僧趁机把唐僧抢出人群，扶他上马。'
        '不料路旁又闪出一个女子，一阵旋风把唐僧摄走，不知去向。'
    )},
    {'role': 'user', 'content': 'hello'},
    {'role': 'assistant', 'content': (
        '你好！我是你的阅读助手，随时可以帮你解答关于书中内容的问题。'
    )},
]
Q_DEFINITE_NP = '路旁闪出的女子是谁？'


def _lcs_len(a: str, b: str) -> int:
    """最长公共子串长度。只用来证明判据**不是**它（见下）。"""
    best = 0
    prev = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        cur = [0] * (len(b) + 1)
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                cur[j] = prev[j - 1] + 1
                best = max(best, cur[j])
        prev = cur
    return best


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


class TestDefiniteNpAnaphora:
    """定指名词短语回指 —— 2026-09-24 实测的端到端失败。

    用户先问了女儿国那段（回答正确、引用了第五十四回），紧接着追问
    「路旁闪出的女子是谁？」。这句的指代是**定指名词短语**「路旁闪出的女子」，
    不是代词，旧闸门 `if not detect_pronoun(question): return question, []` 直接返回，
    **上文一个字都没进 query**：

        retrieve('路旁闪出的女子是谁？')                     ->  0 hits  ← 首检 0 命中
        retrieve('女儿国 路旁闪出的女子是谁')                  -> 16 hits，正解排第 2
        retrieve('路旁闪出的女子是谁？（上文相关实体：助手、hello、唐僧、王后）')
                                                            -> 11 hits，正解 rank1

    首检 0 命中把一道一次检索就能答的题推进了检索 agent；agent 6 步预算耗尽
    （`stopped=budget finished=False`）后仍判「信息不足」——
    而正确答案（蝎子精，毒敌山琵琶洞，第五十五回）就在语料里。
    """

    def test_gate_fires_on_definite_np_followup(self) -> None:
        resolved, entities = resolve_query(Q_DEFINITE_NP, H_DEFINITE_NP)
        assert resolved != Q_DEFINITE_NP, '定指回指必须触发改写（这就是那个 0 命中的根因）'
        assert resolved.startswith(Q_DEFINITE_NP)
        assert '（上文相关实体：' in resolved
        assert entities, '触发改写就必须真的并入实体'

    def test_echo_fires_where_longest_common_substring_cannot(self) -> None:
        """判据必须是**内容词回声**，不能是字符级重叠/最长公共子串。

        我原先就是这么想的，实测被证伪：上文那句是「路旁**又**闪出**一个**女子」，
        问句是「路旁闪**出**的女**子**是谁」，最长公共子串只有 2 字
        —— 被「又」「一个」「的」「是谁」打断。
        谁把判据"简化"成公共子串，第一条断言就会红。
        """
        source = H_DEFINITE_NP[1]['content']
        assert _lcs_len(Q_DEFINITE_NP, source) <= 2, '前提变了：现在真的有长公共子串了'
        assert detect_descriptive_reference(Q_DEFINITE_NP, H_DEFINITE_NP) == ['女子']

    def test_no_echo_is_untouched(self) -> None:
        """上文里没有回声名词 → 不改写（简单问题零改动零开销）。"""
        q = '猪八戒的钉钯有什么来历？'
        assert resolve_query(q, H_DEFINITE_NP) == (q, [])

    def test_without_history_is_untouched(self) -> None:
        assert resolve_query(Q_DEFINITE_NP, []) == (Q_DEFINITE_NP, [])
        assert resolve_query(Q_DEFINITE_NP, None) == (Q_DEFINITE_NP, [])

    def test_own_anchor_blocks_the_new_path_too(self) -> None:
        """「自带专名」这道闸门必须同时管住新路径。

        依据 `ag-ordinal-04`：那次问句自带专名，改写把上文实体追加进去后由 PASS 翻成 FAIL。
        """
        q = '女儿国国王对唐僧说了什么？'
        assert detect_descriptive_reference(q, H_DEFINITE_NP) == []
        assert resolve_query(q, H_DEFINITE_NP) == (q, [])

    def test_anaphora_switch_is_independent_of_pronoun_path(self) -> None:
        """单变量 A/B 的开关：关掉新路径，代词路径必须原样工作。"""
        pronoun_q = '它的特点是什么？'
        assert resolve_query(
            pronoun_q, H_PRONOUN_01, anaphora_enabled=False
        )[0] != pronoun_q
        assert resolve_query(
            Q_DEFINITE_NP, H_DEFINITE_NP, anaphora_enabled=False
        ) == (Q_DEFINITE_NP, [])


class TestDemonstrativeClassifierPronouns:
    """「指示词 + 量词」也是代词，但**不在 PRONOUNS 词表里** → 旧实现整类漏检。

    jieba 把 `那段`/`这本`/`这部`/`那种` 都标成 `r`，而 `detect_pronoun` 只认词表，
    于是 `detect_pronoun('你刚才说的那段再解释一下')` 返回 `[]`
    —— 与「代词漏检」后果完全一样（gold `mt-11-adversarial-impersonal-followup` 正是这个形态）。
    """

    def test_detects_demonstrative_classifier_forms(self) -> None:
        for q, want in [
            ('你刚才说的那段再解释一下', ['那段']),
            ('这本讲了什么？', ['这本']),
            ('这部小说讲了什么？', ['这部']),
            ('那种妖怪有什么特点？', ['那种']),
        ]:
            assert detect_pronoun(q) == want, q

    def test_known_ceiling_is_jieba_segmentation(self) -> None:
        """已知上限，**如实钉住**，免得后来人以为这一档已经覆盖全了。

        形态档只能认出 jieba **切成了一个 r 词**的形态，切法不听话的就漏：
            '那部书怎么样？' -> 那/r + 部书/n     （没切成 '那部'）
            '那本书呢？'     -> 那本书/nr        （连词性都不是 r）
        要再往上收就得引入裸 `这`/`那` 当指示词的规则，但那会放宽一批
        目前没有实测依据的形态 —— 留作后续，不在本次范围。
        """
        assert detect_pronoun('那部书怎么样？') == []
        assert detect_pronoun('那本书呢？') == []

    def test_adverbial_na_me_is_not_a_pronoun(self) -> None:
        """`那么`/`这么`/`那样` 词性也是 r，但它们是副词性的 —— 绝不能判成指代。

        所以形态档用**显式量词集合**卡第二个字（'么'/'样' 不在集合里，天然挡掉）。
        """
        for q in ['那么，答案是什么？', '这么写对不对？', '那样做可以吗？']:
            assert detect_pronoun(q) == [], q


class TestAnaphoraRouting:
    """定指回指 → 必须显式放行 agent。

    这一条是被 A/B 实测逼出来的（2026-09-24，各 3 次）：

        闸门修好之前：retrieve 0 命中 → judge 判信息不足 → 进 agent → 2/3 答对蝎子精、1/3 要澄清
        闸门修好之后（未接线）：retrieve 11 命中 → judge 认为证据充足 → **不进 agent**
                              → 输出「原文片段没有交代那女子的身份」**3/3 全错**
        闸门修好 + 放行：3/3 答对、0/3 澄清

    根因是「有片段」≠「够用」：那 11 条命中里确实有**正确的第 61 章**，
    但命中的那一块正文里没出现「蝎子精」—— 名字在 61 章的另一块，
    问句与它之间没有字面桥梁（词汇鸿沟，不是排序问题）。
    而 `read_chapter` 恰恰能一次读到那一块。所以这一路必须绕过 judge 的「有片段即充足」。
    """

    def test_pronoun_path_is_not_the_anaphora_path(self) -> None:
        """代词问句的指称实体已经进了 query，检索层能解决 —— 不该走这条路由。

        两者混为一谈会让所有代词追问都多烧一次 agent（CQR 省下的成本又还回去）。
        """
        from reading_assistant.graph.rewrite import is_descriptive_anaphora

        assert is_descriptive_anaphora('它的特点是什么？', H_PRONOUN_01) is False
        assert is_descriptive_anaphora(Q_DEFINITE_NP, H_DEFINITE_NP) is True

    def test_flag_is_false_without_history_or_echo(self) -> None:
        from reading_assistant.graph.rewrite import is_descriptive_anaphora

        assert is_descriptive_anaphora(Q_DEFINITE_NP, []) is False
        assert is_descriptive_anaphora(Q_DEFINITE_NP, None) is False
        assert is_descriptive_anaphora('猪八戒的钉钯有什么来历？', H_DEFINITE_NP) is False


class TestGraphCarriesAnaphoraFlag:
    """端到端：`cqr_anaphora` 必须真的进图 state —— 它就是路由判据。"""

    def _graph(self, tmp_path: Path):
        from reading_assistant.storage import create_db_engine, create_session_factory, init_db
        from reading_assistant.storage.models import Document
        from reading_assistant.storage.vector_store import InMemoryVectorStore, StoredChunk

        engine = create_db_engine('sqlite:///:memory:')
        init_db(engine)
        factory = create_session_factory(engine)
        with factory() as session:
            session.add(Document(filename='d1.epub', title='d1', file_hash='f1',
                                 content_hash='c1', index_status='indexed'))
            session.commit()
        store = InMemoryVectorStore()
        store.add([
            StoredChunk(id='d1-0', text='西游记 女儿国 女子 旋风 唐僧',
                        metadata={'document_id': 1, 'chapter': '第五十四回', 'chapter_index': 60},
                        embedding=[1.0, 0.0]),
        ])
        return factory, store

    def _invoke(self, factory, store, question: str, history: list[dict], thread: str) -> dict:
        from reading_assistant.graph.qa import build_qa_graph
        from reading_assistant.storage.models import ChatMessage, ChatSession

        with factory() as session:
            chat = ChatSession()
            session.add(chat)
            session.commit()
            sid = chat.id
            for msg in history:
                session.add(ChatMessage(session_id=sid, role=msg['role'], content=msg['content']))
            session.commit()
        graph = build_qa_graph(factory, store)
        return graph.invoke(
            {'question': question, 'session_id': sid, 'document_ids': [1]},
            config={'configurable': {'thread_id': thread}},
        )

    def test_definite_np_followup_sets_the_flag(self, tmp_path: Path) -> None:
        factory, store = self._graph(tmp_path)
        state = self._invoke(
            factory, store, Q_DEFINITE_NP, H_DEFINITE_NP, 't-anaphora-on'
        )
        assert state.get('cqr_anaphora') is True, (
            '定指回指没置位 → judge 会因为「有片段」放行到 answer，'
            '而那条路实测 3/3 答错（见 TestAnaphoraRouting 的 A/B 数字）'
        )
        assert state.get('cqr_entities')

    def test_plain_question_never_sets_the_flag(self, tmp_path: Path) -> None:
        factory, store = self._graph(tmp_path)
        state = self._invoke(
            factory, store, '女儿国国王对唐僧说了什么？', H_DEFINITE_NP, 't-anaphora-off'
        )
        assert state.get('cqr_anaphora') is False

