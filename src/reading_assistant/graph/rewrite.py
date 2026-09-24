"""对话式查询改写（CQR）：把「它／这个」换成上文里的实体。

## 问题定位
``retrieve`` 的 query 就是原始问题字符串 —— 对话历史只喂给「作答」、**没喂给「检索」**。
于是「它的特点是什么？」里的「它」对检索器就是空气：历史里明明躺着「寅将军」，
但那段历史**从没进过检索**，所以答案所在的那一章根本捞不上来。
表现为「模型说片段里没找到」—— 其实不是模型笨，是它压根没拿到东西。

## 归属：这是 CQR，不是普通 query rewriting
- **CQR（多轮）**：指代消解、省略补全 —— 解决「上下文依赖」。
- **普通改写（单轮）**：同义扩展、拼写纠错、HyDE —— 解决「词汇鸿沟」。
两者是**不同的失败模式**（换词 vs 换指称），benchmark 与调参口径都不同，
混成一个模块会让后续无从判断是哪一类在退化。本模块**只做前者**。

## 为什么规则优先、零 LLM
与 ``graph/intent.py`` 的级联同一思路：前两层零 LLM，只有都拿不准才付一次模型调用。
代词检测 + 实体抽取在这类文本上都可用确定性规则解决，而**确定性是硬要求**：
节点会被 LangGraph 重放，非确定性改写会让同一轮重放出不同的查询。

## 为什么给「候选集合」而不是「一个实体」
实测（2026-09-22）jieba 词典缺「寅将军」（切成 ``寅(mg)+将军(n)``），
任何「挑一个」的策略都会在它上面失败。而同一句回复里的「双叉岭」是第十三回回目里的词，
**候选集合能容错**：只要有一个候选能落到正确章节就够。
这也正好咬合 agent 已有的多候选 ``grep("A,B,C")`` 能力。
"""
from __future__ import annotations

from functools import lru_cache

from reading_assistant.utils.logger_handler import get_logger

logger = get_logger('rewrite')

# 只有这些词、**且词性为 r** 才算代词性指代。
# 词性必须一起卡：否则「我应该怎么办」里的「该」会误触发，
# 让简单问题也走上改写路径（白白把无辜的实体塞进检索 query）。
PRONOUNS: frozenset[str] = frozenset({
    '它', '他', '她', '牠', '祂',
    '它们', '他们', '她们',
    '这个', '那个', '这位', '那位', '这些', '那些',
    '此', '该', '其', '这东西', '这人', '这事', '此物', '这只', '那只',
})

# 「指示词 + 量词」组合的形态档：那个/这位/这段/这本/这部/那种…
# 只列**量词性**的第二个字。刻意不含 么/样/边/里/儿 ——
# 「那么/这么/那样/这样/那边」也是 r，但它们是副词/方位性的，不是指代。
_DEMONSTRATIVE_HEADS: frozenset[str] = frozenset({'这', '那'})
_DEMONSTRATIVE_CLASSIFIERS: frozenset[str] = frozenset(
    set('个位些只种本部段篇章回条件次名群张把条')
)

# 组块用的「像名字」的词性。刻意**不含** 'a'（形容词）：
# 「老虎(nr) 精(a) 寅(mg) 将军(n)」——正是让 'a' 断开，
# 「寅将军」才作为一个完整块被合并出来（实测）。
#
# 含 'z'（状态词）：jieba 把「黑熊」这类标成 z，不含就会把「黑熊精」拦腰切断，
# 实测 ag-pronoun-07 因此**完全抽不到指称**。代价是可能混入个别状态词，
# 但它们会被排到「弱候选」之后，且实测噪声率没有恶化。
_NAME_FLAGS: frozenset[str] = frozenset({'nr', 'nrt', 'ns', 'nz', 'nt', 'mg', 'n', 'eng', 'z'})

# 单条历史消息内最多抽几个候选。放宽是为了**召回**：
# 实测指称常排在一条长回复的第 7~11 位，旧上限 6 会让它整条漏掉
# （ag-pronoun-05 的「九尾狐」就是这么丢的）。
_PER_MESSAGE_LIMIT = 20

# 通用词：出现在块里几乎一定不是「用户指代的那个东西」。
_GENERIC: frozenset[str] = frozenset({
    '妖怪', '妖精', '妖魔', '鬼怪', '神仙', '人物',
    '特点', '特征', '故事', '来历', '结局', '师傅', '徒弟', '技能', '本事',
    '原因', '结果', '问题', '东西', '地方', '方法', '办法', '情况', '内容',
    '时候', '名字', '过程', '细节', '关系', '区别', '作用', '意义',
    '西游记', '中', '上', '下', '里', '本', '他们', '师徒', '之后', '以前',
})

# 「自带锚点」的词性。刻意**不含** 'n'：'n' 会把「收服」这类被误标成名词的动词也算进来，
# 用它当判据会让真正的代词追问（「他是怎么被收服的？」）被误判成「自带实体」而漏改写。
_ANCHOR_FLAGS: frozenset[str] = frozenset({'nr', 'nrt', 'ns', 'nz', 'nt', 'mg'})

_MIN_ENTITY_CHARS = 2
_MAX_ENTITY_CHARS = 8


@lru_cache(maxsize=1)
def _posseg():
    """延迟导入 jieba.posseg（首用加载词典约 0.26s，不能拖慢进程启动）。"""
    import jieba.posseg as pseg

    return pseg


def detect_pronoun(question: str) -> list[str]:
    """返回问题里的代词性指代词；没有则空列表。

    代词必须**单独成词且词性为 r**：拿 `in` 做子串匹配会把「应该」判成「该」。

    除词表外还按**形态**补一档：「指示词 + 量词」（那个/这位/那段/这本/这部/那种…）。
    依据（2026-09-24 实测）：jieba 把 `那段`/`这本`/`这部`/`那种` 都标成 `r`，
    但它们**不在** `PRONOUNS` 里 → ``detect_pronoun('你刚才说的那段再解释一下')`` 返回 ``[]``，
    后果与「代词漏检」完全一样（gold `mt-11-adversarial-impersonal-followup` 正是这个形态）。
    用**显式量词集合**而不是「只要 r 且以 这/那 开头」：`那么`/`这么`/`那样` 也是 `r`，
    但它们是副词性的，绝不能判成指代（'么'/'样' 不在量词集合里，天然挡掉）。
    """
    if not question:
        return []
    pseg = _posseg()
    return [
        word
        for word, flag in pseg.cut(question)
        if flag == 'r'
        and (
            word in PRONOUNS
            or (
                len(word) >= 2
                and word[0] in _DEMONSTRATIVE_HEADS
                and word[1] in _DEMONSTRATIVE_CLASSIFIERS
            )
        )
    ]


def has_own_anchor(question: str) -> bool:
    """问句是否自带专名锚点。

    **自带锚点时不改写**：句内代词（「…孙悟空…它的技能是什么？」）大概率指的就是
    句内那个实体，上下文里再捞一批只会往 query 里掺噪声 ——
    而「多问合一互相稀释」正是本项目实测过的失败模式。

    实测（2026-09-22）：``ag-ordinal-04`` 的问句里有关键词「孙悟空」和代词「它」，
    改写触发后把上文实体追加进去 → 该题由 PASS 翻成 FAIL。
    """
    if not question:
        return False
    pseg = _posseg()
    return any(flag in _ANCHOR_FLAGS for _, flag in pseg.cut(question))


def extract_entities(text: str, limit: int = 6) -> list[str]:
    """从一段文本里抽候选实体。

    做法：把**连续**的「像名字」的词并成一个块（遇到动词/助词/标点就断开），
    再按长度与通用词表过滤。合并是关键 —— 单看词性会得到「黄眉」+「大王」两个碎块，
    合并后才是用户认识的「黄眉大王」。
    """
    if not text:
        return []
    pseg = _posseg()
    found: list[tuple[str, bool]] = []  # (块, 是否含专名词性)
    run: list[str] = []
    strong = False

    def flush() -> None:
        nonlocal strong
        if run:
            name = ''.join(run)
            if (
                _MIN_ENTITY_CHARS <= len(name) <= _MAX_ENTITY_CHARS
                and name not in _GENERIC
            ):
                found.append((name, strong))
            run.clear()
            strong = False

    for word, flag in pseg.cut(text):
        if flag in _NAME_FLAGS and word not in _GENERIC:
            run.append(word)
            if flag in _ANCHOR_FLAGS:
                strong = True
        else:
            flush()
    flush()
    # 含专名词性的块排前面（稳定排序，保持块内原有先后）。
    # 依据：纯 `n` 组成的块大量是「干娘/童子/宝贝/下界」这类通用词，
    # 而指称几乎总是真专名（九尾狐/黑熊精/寅将军）。
    found.sort(key=lambda item: not item[1])
    return [name for name, _ in found][:limit]


def collect_candidates(
    history: list[dict] | None, limit: int = 4
) -> list[str]:
    """从对话历史里收集候选实体，**由近及远**、去重保序。

    由近及远很重要：指代指向的几乎总是最近提到的那个东西。
    助手回复也参与抽取 —— 而且往往是**唯一**的来源：
    「这个妖怪的特点是什么？」的前一轮，实体（寅将军）只出现在**助手的回复**里
    （用户的问句是「第一个妖怪是什么？」，里面没有实体）。
    """
    if not history or limit <= 0:
        return []
    # 逐条消息抽取（由近及远），再**轮转**合并：每条消息先各出一个，再各出第二个……
    #
    # 不用「抽干最新一条再轮到上一条」：最新那条往往是**很长的助手回复**，
    # 它一个人就能把名额吃光，而指称可能落在较早的提问里
    # （实测 ag-pronoun-07 的「黑熊精」只在用户问句里出现过）。
    per_message = [
        extract_entities(str(msg.get('content') or ''), limit=_PER_MESSAGE_LIMIT)
        for msg in reversed(history)
    ]
    picked: list[str] = []
    for rank in range(_PER_MESSAGE_LIMIT):
        for names in per_message:
            if rank >= len(names):
                continue
            name = names[rank]
            if name not in picked:
                picked.append(name)
                if len(picked) >= limit:
                    return picked
    return picked


# 定指描述的回声窗口：只看最近几条上文。
_ANAPHORA_WINDOW = 4

# 能当「指称中心」的普通名词词性。刻意**不含** nr/nrt（专名）：
# 问句自带专名说明句内可解，由 has_own_anchor 排除，不该走回指路径。
_REFERENT_FLAGS: frozenset[str] = frozenset({'n', 'ns', 'nt', 'nz', 'ng'})


def detect_descriptive_reference(
    question: str, history: list[dict] | None, window: int = _ANAPHORA_WINDOW
) -> list[str]:
    """问句是否在**回指上文提到过的普通名词**（定指描述）；返回回声词列表。

    背景（2026-09-24 实测的端到端失败）：用户先问了女儿国那段，接着追问
    「路旁闪出的女子是谁？」。这句的指代是**定指名词短语**「路旁闪出的女子」，
    不是代词 —— jieba 切作 `路旁/s 闪出/v 的/uj 女子/n 是/v 谁/r`，
    ``detect_pronoun`` 返回 ``[]``（「谁」词性是 r 但不在词表），
    于是上文一个字都没进 query：

        retrieve('路旁闪出的女子是谁？')                       ->  0 hits  ← 首检 0 命中
        retrieve('女儿国 路旁闪出的女子是谁')                    -> 16 hits，正解排第 2
        retrieve('路旁闪出的女子是谁？（上文相关实体：助手、hello、唐僧、王后）')
                                                              -> 11 hits，正解 rank1/rank2

    ⚠️ **不能用最长公共子串或字符级重叠做判据**（先试过，实测不成立）：
    上文那句是「路旁**又**闪出**一个**女子」，问句是「路旁闪**出**的女**子**是谁」，
    最长公共子串只有 2 字（「路旁」「闪出」「女子」），被「又」「一个」「的」「是谁」打断。
    所以判据必须是**内容词回声**：问句里的普通名词，字面出现在最近的上文里。

    刻意保持窄：只认**非通用**普通名词（``_GENERIC`` 里的「妖怪」「特点」等不算）；
    ``has_own_anchor`` 的句子一律排除；没有上文就永不触发。
    依据是 `ag-ordinal-04` 的前车之鉴 —— 那次问句自带专名，改写把上文实体追加进去后
    由 PASS 翻成 FAIL，所以「自带专名」这条闸门必须同时管住新路径。
    """
    if not question or not history:
        return []
    if has_own_anchor(question):
        return []
    pseg = _posseg()
    nouns = [
        word
        for word, flag in pseg.cut(question)
        if flag in _REFERENT_FLAGS and len(word) >= 2 and word not in _GENERIC
    ]
    if not nouns:
        return []
    recent = ' '.join(str(msg.get('content') or '') for msg in history[-window:])
    if not recent:
        return []
    return [word for word in nouns if word in recent]


def is_descriptive_anaphora(question: str, history: list[dict] | None) -> bool:
    """本轮是否该走「定指描述回指」路径（供 qa.py 决定要不要先给 agent 一次机会）。

    与 ``resolve_query`` 里的判据同源，抽出来单独用是为了不搅动它的返回值契约
    （2 元组被 20 多条测试与 API 观测依赖）。

    **为什么路由需要知道这件事**（2026-09-24 实测）：
    修好闸门后 `retrieve` 从 0 命中变成 11 命中，但命中的是「路旁闪出／女王远送」**那一段场景**，
    而写着名字的那一块（第五十五回「毒敌山琵琶洞…蝎子精」）**没有被检索到** ——
    问句与名字之间没有字面桥梁，这是词汇鸿沟，不是排序问题。
    更糟的是「有片段」会让 `judge` 认为证据充足，**反而把 agent 挡在门外**，
    而 agent 的 `read_chapter` 正是唯一能读到那一块的手段（实测 A/B：让 agent 跑时 2/3 答对）。
    所以这类追问要显式允许 agent 介入。
    """
    if not question:
        return False
    if detect_pronoun(question):
        return False  # 代词路径另算：指称实体本身已进 query，检索层就能解决
    return bool(detect_descriptive_reference(question, history))


def resolve_query(
    question: str,
    history: list[dict] | None,
    enabled: bool = True,
    max_entities: int = 4,
    anaphora_enabled: bool = True,
) -> tuple[str, list[str]]:
    """把含代词的追问改写成自带实体的查询。

    返回 ``(resolved_question, entities)``；无需改写时原样返回 ``(question, [])``。

    **追加而非替换**：代词本身不携带信息，但整句的其余部分（「特点」「怎么被收服」）
    才是问题的核心。替换只会让查询更短更糊；追加是纯增量信号。
    """
    if not enabled or not question:
        return question, []
    echo: list[str] = []
    if not detect_pronoun(question):
        # 代词路径未命中，再看**定指描述回指**（2026-09-24 新增）。
        # 两条路径互斥，汇合后共用同一套抽取与追加逻辑。
        if not anaphora_enabled:
            return question, []
        echo = detect_descriptive_reference(question, history)
        if not echo:
            return question, []  # 简单问题零改动、零开销
    if has_own_anchor(question):
        # 句内已有实体，代词在句内就能解 —— 借上文只会掺噪声、稀释真正的信号
        logger.info('问答[改写] 问句自带专名，句内可解 → 不改写')
        return question, []
    entities = collect_candidates(history, limit=max_entities)
    if not entities:
        # 检测到代词但历史里抽不出任何实体（例如首轮就说「它」）→ 不改写。
        # 硬塞空列表只会让日志出现「改写成功但没变化」的假信号。
        logger.info('问答[改写] 命中代词但历史无可抽实体，保持原查询')
        return question, []
    # 只给候选，**不给操作指令**。曾试过在这里加一句「若首选与谓语语义不合请依次试其余候选」，
    # 动机是实测轨迹里 agent 自己说出了「白骨精是被打死的，并非收服」却没换候选 ——
    # 但加了以后无效且有害（各 3 次）：
    #   ag-pronoun-02 仍 1/3（没救回来）；
    #   ag-pronoun-01 的 agent 重新被触发（步数 0/0 → 0/4/3），把 CQR 省下的 agent 成本又还了回去。
    # 结论：靠 prompt 催「多试几个候选」不解决问题，反而削弱 CQR 在检索层已经拿到的收益。
    resolved = f'{question}（上文相关实体：{"、".join(entities)}）'
    if echo:
        logger.info('问答[改写] 命中定指描述回指 回声=%s → 实体=%s', echo, entities)
    else:
        logger.info('问答[改写] %s → 实体=%s', question, entities)
    return resolved, entities
