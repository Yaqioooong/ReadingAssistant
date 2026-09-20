"""意图识别三级级联：规则快通道 → 向量原型 → LLM 兜底。

## 为什么这样分层

检索门每轮都在回答同一个问题：「这轮要不要检索书籍」。现状是**每轮都问一次 LLM**，
而绝大多数轮次的答案就是 book（正常读书会话里几乎全是书问题）—— 这是可以省掉的钱。

分层的依据是**代价不对称**：

- 判成 book 却判错了：最多多检索一次，白花一点检索成本，用户无感；
- 判成 chat/history 却判错了：问题被直接吞掉，用户对着一个闲聊式回答发懵。

前者廉价、后者昂贵，于是三条铁律：

1. **fallback 方向恒为 book** —— 拿不准一律进检索，宁可多花钱也不吞问题；
2. **向量原型层只敢自动判 book，绝不敢自动判 chat/history** ——
   「确认安全默认值」是廉价的，「押注偏离默认值」是昂贵的；
3. chat/history 的自动判定只交给规则层，且规则一律**高精度**构造：
   宁可漏判（漏了会掉到 L2/L3 兜底），绝不误判。

历史教训见 docs/design-intent-recognition.md：首轮发 "hello" 掉进书问答链路、
检索为空、被判「信息不足」并弹出澄清任务 —— 规则层存在的理由就是这个。
回归锁在 tests/test_multi_turn.py::test_first_turn_greeting_is_chat。
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Sequence
from typing import Literal, NamedTuple

from reading_assistant.utils.logger_handler import get_logger

logger = get_logger('intent')

Intent = Literal['book', 'history', 'chat']

# ============================================================ L1 · 规则快通道

# 归一化：去掉空白与常见标点。**不**去书名号 —— 《》是 L1 判定 book 的强信号。
_PUNCT_RE = re.compile(r'[\s，。！？!?,.、~～…·"\'“”‘’()（）:：;；]+')

# 纯寒暄不会长；超过这个长度就当成「带内容」，规则直接让路给 L2/L3。
_CHAT_MAX_LEN = 12

_CHAT_PATTERNS: tuple[str, ...] = (
    # 问候（长词在前，减少回溯）
    r'(?:你好|您好|哈喽|嗨|hi|hello|hey|在吗|在不在|早上好|早安|中午好|下午好|晚上好|晚安|早)'
    r'(?:呀|啊|哦|呐|鸭|呢|哈|~)*',
    # 致谢
    r'(?:太|好|非常|真的)?(?:谢谢|多谢|感谢|thx|thanks|thankyou|3q)'
    r'(?:你|您|楼主|亲)?(?:的)?(?:回答|回复|解答|解释|帮忙|指点|建议)?(?:啦|了|哦|呀|哈)*',
    # 道别
    r'(?:再见|拜拜|bye|回见|下次聊|先这样)(?:啦|了|哦|呀|哈)*',
    # 自指 / 能力询问
    r'(?:你|您)(?:是|叫)(?:谁|什么|啥)(?:呀|啊|哦|呢|哈)*',
    r'(?:你|您)(?:能|会|可以)(?:做|干|帮)(?:什么|啥|点啥|些什么|什么忙)(?:呀|啊|哦|呢|哈)*',
    # 应答
    r'(?:好的|好|嗯+|哦+|收到|明白|知道了|ok|okay|行|可以)(?:的|啦|了|哦|呀|呢|哈)*',
    # 辛苦
    r'辛苦(?:你|您|了|啦)*',
    # 笑
    r'(?:哈哈+|嘿嘿+|呵呵+|hhh+)(?:哈)*',
)
_CHAT_RE = re.compile('^(?:' + '|'.join(_CHAT_PATTERNS) + ')$')

# 元语言问法（history）。**每条都要求第一人称**（我/咱）或整串锚定的极短形式 ——
# 这是「提到刚才/上一条 ≠ history」的执行方式：只要消息里没有第一人称主语，
# 「刚才说的那个计划，执行者是谁？」这类书追问就一条规则都撞不上。
_HISTORY_PATTERNS: tuple[tuple[str, str], ...] = (
    (
        # ① 「我上一个问题是什么」：元语言名词 + 疑问词。序数词不强制，
        #    因为「我的问题是什么」同样是元问题。
        r'^(?:请问|麻烦你|那|对了|还有|所以|嗯|好)*?(?:我|咱)(?:们)?[^\n]{0,8}?'
        r'(?:问题|提问|问的)[^\n]{0,6}?(?:是|为|叫)?(?:什么|啥|哪些|哪几个)',
        '我(被)问的问题是什么',
    ),
    (
        # ② 「我(刚才|之前|上次)(问|说)了什么」
        r'^(?:请问|麻烦你|那|对了|还有|所以|嗯|好)*?(?:我|咱)(?:们)?[^\n]{0,4}?'
        r'(?:刚才|刚刚|之前|前面|上次|上一?[轮条回遍次])'
        r'[^\n]{0,3}?(?:问|说|提)(?:过|了|的)?[^\n]{0,3}?(?:什么|啥|哪些)',
        '我刚才问了什么',
    ),
    (
        # ③ 「我问过/说过什么」
        r'^(?:请问|那|嗯)*?(?:我|咱)(?:们)?(?:问过|说过|提过)[^\n]{0,6}?(?:什么|啥|哪些|问题)',
        '我问过什么',
    ),
    (
        # ④ 「我第 N 个问题」—— qa_meta 当年漏判的「第 N 个」序数词表在这里补齐
        r'^(?:请问|那)?(?:我|咱)(?:们)?第[一二三四五六七八九十百千\d]+'
        r'(?:个|条|道|轮|次|回|句|遍)(?:问题|提问)?',
        '我第 N 个问题',
    ),
    (
        # ⑤ 「再上一个呢 / 上一个呢」—— 省略主语的追问。整串锚定且极短，
        #    因此不要求第一人称也不会误伤（「上一个章节讲了什么」撞不上）。
        r'^(?:那|那么|还有|再|接着)?(?:再)?(?:上|前)[一二三四五六七八九十\d]*'
        r'(?:个|条|道|轮|次|回|句|遍)?(?:呢|是什么|是啥|是哪个)\??$',
        '再上一个呢',
    ),
    (
        # ⑥ 「把刚才的问题再说一遍」
        r'^(?:请|麻烦你)?(?:把|将)?(?:(?:我|你|咱)(?:们)?)?(?:刚才|之前|上次|上一条|上一个)'
        r'[^\n]{0,4}?(?:问题|话|提问)[^\n]{0,4}?(?:再|重)(?:说|讲|重复|念|发)',
        '复述刚才的问题',
    ),
    (
        # ⑦ 「你重复一下我问过的话」—— 祈使式复述，主语是「你」而不是「我」
        #   结尾锚定：'再说一下我上一个问题里提到的计划' 是书追问，必须撞不上
        r'^(?:你|请|麻烦你|劳驾)?(?:重复|复述|再念|重念|再说|重说|重讲)'
        r'[^\n]{0,6}?(?:我|咱)(?:们)?[^\n]{0,4}?(?:问题|话|提问)'
        r'(?:呢|呀|啊|吧|的)?\??$',
        '重复我问过的话',
    ),
    (
        # ⑧ 「我们聊到哪了」
        r'^(?:那)?(?:我们|咱们|咱)(?:聊|说|讲)到[^\n]{0,4}?(?:哪|什么)(?:了|儿)?',
        '聊到哪了',
    ),
)
_HISTORY_RES = tuple((re.compile(pattern), label) for pattern, label in _HISTORY_PATTERNS)


def _normalize_text(question: str) -> str:
    """去空白与标点（保留书名号），供规则整体匹配。"""
    return _PUNCT_RE.sub('', question or '')


def _match_history(text: str) -> str | None:
    """命中返回规则标签（供日志归因），未命中返回 None。"""
    for regex, label in _HISTORY_RES:
        if regex.match(text):
            return label
    return None


def _match_chat(text: str) -> bool:
    if len(text) > _CHAT_MAX_LEN:
        # 纯寒暄不会长。长消息一律让路 —— 这是「谢谢，那《三体》里…」这类
        # 寒暄开头 + 书内容的混合消息不被误判成 chat 的第一道闸。
        return False
    if '《' in text or '》' in text:
        return False
    return bool(_CHAT_RE.match(text.lower()))


def rule_intent(question: str) -> tuple[Intent | None, str]:
    """L1 规则层：高精度快通道。返回 (意图 | None, 归因标签)。

    None 表示「规则不敢判」，交给 L2/L3 —— **漏判是允许的，误判是不允许的**。
    """
    text = _normalize_text(question)
    if not text:
        return None, ''
    # history 先判：元语言问法比寒暄具体，避免「我上一个问题是什么」被闲聊词表抢走
    label = _match_history(text)
    if label:
        return 'history', 'h:' + label
    if _match_chat(text):
        return 'chat', 'c:chitchat'
    # book 的强字面信号：书名号。带《》的消息不可能只是寒暄或元问题，
    # 一定是在拿某本书的内容提问（chat/history 的句式都不带书名号）。
    if '《' in text and '》' in text:
        return 'book', 'b:title'
    return None, ''


# ============================================================ L2 · 向量原型

# 每类意图的「原型句」。原型不参与训练，只做最近邻参照 —— 加意图 = 加例句。
INTENT_PROTOTYPES: dict[str, tuple[str, ...]] = {
    'book': (
        '《三体》中罗辑的咒语指的是什么？',
        '这本书里主人公最后怎么样了？',
        '张三喜欢谁？',
        '面壁计划是谁提出的？',
        '原文里提到的那个计划执行者是谁？',
        '关于章北海的情节有哪些？',
        '这一段讲了什么内容？',
        '文档里怎么描述这个事件的？',
        '作者为什么这么安排结局？',
        '这句话是什么意思？',
        '书里有没有提到过这个人？',
        '请介绍一下文中的主要人物关系。',
        '这部分和前面那部分有什么关系？',
        '罗辑最后做了什么决定？',
        '这段话的出处是哪里？',
    ),
    'history': (
        '我上一个问题是什么？',
        '我刚才问了什么？',
        '我前面问过哪些问题？',
        '再上一个呢？',
        '我第一个问题是什么？',
        '把刚才的问题再说一遍',
        '我问过关于罗辑的问题吗？',
        '我们聊到哪了？',
        '我之前的提问都有哪些？',
        '你重复一下我问过的话',
    ),
    'chat': (
        '你好',
        'hello',
        '谢谢你的回答',
        '你是谁？',
        '再见',
        '你能做什么？',
        '嗯嗯',
        '早上好',
        '辛苦了',
        '哈哈',
    ),
}


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """余弦相似度；任一为空或维度不一致返回 0（退化安全）。"""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class PrototypeRouter:
    """向量原型路由：原型句向量化后取最近邻，做**单向**判定。

    ⚠️ 只会输出 book 或 None：

    - 输出 book = 「这就是在问书内容，不必再花钱问 LLM」；
    - 输出 None = 「不敢确认是 book」→ 交 L3 LLM 兜底。

    它**不会**输出 chat/history。原因见模块 docstring：押注偏离安全默认值太贵，
    那类判断只交给高精度规则层。
    """

    def __init__(
        self,
        embed_documents: Callable[[list[str]], list[list[float]]],
        prototypes: dict[str, tuple[str, ...]] | None = None,
        threshold: float = 0.80,
        margin: float = 0.04,
    ) -> None:
        self.threshold = threshold
        self.margin = margin
        self._prototypes = prototypes or INTENT_PROTOTYPES
        self._vectors: dict[str, list[list[float]]] = {}
        texts: list[str] = []
        owners: list[str] = []
        for intent, samples in self._prototypes.items():
            for sample in samples:
                owners.append(intent)
                texts.append(sample)
        embedded = embed_documents(texts)
        for intent, vector in zip(owners, embedded):
            self._vectors.setdefault(intent, []).append(list(vector))

    @property
    def ready(self) -> bool:
        return bool(self._vectors.get('book'))

    def score(self, vector: Sequence[float]) -> dict[str, float]:
        """每类取该类原型的最高余弦分。"""
        return {
            intent: max((_cosine(vector, proto) for proto in protos), default=0.0)
            for intent, protos in self._vectors.items()
        }

    def route(self, vector: Sequence[float]) -> tuple[Intent | None, float]:
        """高置信确认 book；否则 (None, book 分) 交上层兜底。"""
        scores = self.score(vector)
        book = scores.get('book', 0.0)
        others = max(scores.get('chat', 0.0), scores.get('history', 0.0))
        if book >= self.threshold and (book - others) >= self.margin:
            return 'book', book
        return None, book


# ============================================================ 级联编排


class IntentDecision(NamedTuple):
    """快通道判定结果。intent is None 表示需要上层 LLM 兜底。"""

    intent: Intent | None
    channel: str  # rule | prototype | ''（未命中）
    score: float | None = None
    vector: list[float] | None = None
    rule_label: str = ''


def route_fast(
    question: str,
    router: PrototypeRouter | None,
    embed: Callable[[], list[float]] | None = None,
    *,
    rules_enabled: bool = True,
) -> IntentDecision:
    """L1 → L2，**零 LLM 调用**。命中不了就返回 intent=None 交给 L3。

    embed 是惰性回调：只有真的走到 L2 才会算向量 ——
    规则已经命中的轮次一次 embedding 都不花。
    """
    if rules_enabled:
        intent, label = rule_intent(question)
        if intent:
            return IntentDecision(intent, 'rule', None, None, label)

    if router is None or not router.ready or embed is None:
        return IntentDecision(None, '')

    vector = embed()
    intent, score = router.route(vector)
    if intent:
        return IntentDecision(intent, 'prototype', score, vector)
    # 未确认是 book，但向量已经算出来了 —— 带出去给 cache_check 复用，别白算
    return IntentDecision(None, '', score, vector)


def build_prototype_router(
    embed_documents: Callable[[list[str]], list[list[float]]],
) -> PrototypeRouter | None:
    """按配置构建原型路由；阈值 <=0 停用；构建失败降级为停用（不影响主链路）。"""
    from reading_assistant.config import get_settings

    settings = get_settings()
    threshold = settings.intent_prototype_threshold
    if threshold <= 0:
        logger.info('意图[原型层] 已停用(threshold=%.2f)，意图判定回落 LLM', threshold)
        return None
    try:
        router = PrototypeRouter(
            embed_documents,
            threshold=threshold,
            margin=settings.intent_prototype_margin,
        )
    except Exception:  # noqa: BLE001 —— 原型层是优化项，不能拖垮主链路
        logger.exception('意图[原型层] 构建失败，降级为 LLM 兜底')
        return None
    if not router.ready:
        return None
    logger.info(
        '意图[原型层] 就绪 threshold=%.2f margin=%.2f 原型分布=%s',
        threshold,
        settings.intent_prototype_margin,
        {k: len(v) for k, v in router._vectors.items()},  # noqa: SLF001
    )
    return router
