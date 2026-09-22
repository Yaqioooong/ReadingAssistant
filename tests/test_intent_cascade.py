"""意图识别三级级联单测：L1 规则高精度、L2 原型退化安全与单向性、级联 fallback 方向。

这些测试守的是**代价不对称**这条设计主轴：

- 判成 book 判错 = 多检索一次（廉价）；
- 判成 chat/history 判错 = 问题被吞掉（昂贵）。

所以 L1 必须高精度（宁可漏判），L2 必须**只敢判 book**，拿不准一律交 L3 兜底。
"""

from __future__ import annotations

import pytest

from reading_assistant.graph.intent import (
    PrototypeRouter,
    route_fast,
    rule_intent,
)

# --------------------------------------------------------------- L1 规则层

HISTORY_CASES = [
    '请问我上一个问题是什么?',
    '那对了麻烦你…我上次问了啥?',
    '我上一轮问了什么?',
    '我上一回问了什么?',
    '我第一个问题问的是什么?',
    '再上一个呢?',
    '我上一个问题是什么？',
    '我第一个问题是什么？',
    '我刚才问了什么？',
    '我问过关于罗辑的问题吗？',
    '我们聊到哪了？',
    '把刚才的问题再说一遍',
    '你重复一下我问过的话',
    '重复一下我上一个问题',
]

CHAT_CASES = [
    '你好',
    'hello',
    '谢谢你的回答',
    '你是谁？',
    '嗯嗯',
    '太谢谢你了',
    '哈哈',
    '早上好',
    '辛苦了',
    '你能做什么？',
    '再见',
]

# 最贵的一类错误：书内容问题被打成 chat/history。这里全部必须**不被规则命中**。
BOOK_CASES = [
    '我上一个问题里问的那个人是谁？',      # 问的是书里的人，不是问记录
    '刚才说的那个计划，执行者是谁？',      # design doc 明写的边界
    '那第一个计划又是什么？',
    '你好，请问面壁计划是什么？',          # 寒暄词 + 书内容
    '你刚才说的那段再解释一下',
    '张三喜欢谁？',
    '上一个章节讲了什么？',                # 「上一个」但问的是章节
    '我上一轮问的罗辑是谁？',
    '把刚才说的那个计划再讲一遍',
    '再说一下我上一个问题里提到的计划',    # 元问题外壳 + 书内容内核
    '把这个章节的内容重复一下',
]


@pytest.mark.parametrize('question', HISTORY_CASES)
def test_rule_routes_history(question: str) -> None:
    assert rule_intent(question)[0] == 'history', question


@pytest.mark.parametrize('question', CHAT_CASES)
def test_rule_routes_chat(question: str) -> None:
    assert rule_intent(question)[0] == 'chat', question


@pytest.mark.parametrize('question', BOOK_CASES)
def test_rule_never_swallows_book_questions(question: str) -> None:
    """规则层宁可漏判（返回 None 交给 L2/L3），也绝不能把书问题判成 chat/history。"""
    assert rule_intent(question)[0] != 'chat', question
    assert rule_intent(question)[0] != 'history', question


def test_rule_routes_book_by_title_marker() -> None:
    """书名号是 book 的强字面信号，且优先于同句里的寒暄词。"""
    assert rule_intent('《三体》中罗辑的咒语指的是什么？')[0] == 'book'
    assert rule_intent('谢谢，那《三体》里罗辑最后怎么了？')[0] == 'book'


# --------------------------------------------------------------- L2 原型层


class _TableEmbed:
    """按查表返回原型向量，便于构造正交/退化场景。"""

    def __init__(self, table: dict[str, list[float]]) -> None:
        self._table = table

    def __call__(self, texts: list[str]) -> list[list[float]]:
        return [self._table[t] for t in texts]


_PROTOS = {'book': ('p_book',), 'history': ('p_hist',), 'chat': ('p_chat',)}
_TABLE = {'p_book': [1.0, 0.0], 'p_hist': [0.0, 1.0], 'p_chat': [0.0, -1.0]}


def _router(**kwargs) -> PrototypeRouter:
    return PrototypeRouter(_TableEmbed(_TABLE), _PROTOS, **kwargs)


def test_prototype_never_returns_chat_or_history() -> None:
    """核心不变量：原型层只输出 book 或 None —— 结构化禁止它吞问题。"""
    router = _router(threshold=0.75, margin=0.08)
    for vector in (
        [1.0, 0.0], [0.0, 1.0], [0.0, -1.0], [0.5, 0.5],
        [0.9, 0.9], [0.0, 0.0], [-1.0, 0.0], [0.99, 0.01],
    ):
        intent, _score = router.route(vector)
        assert intent in (None, 'book'), (vector, intent)


def test_prototype_confirms_book_when_confident() -> None:
    intent, score = _router(threshold=0.75, margin=0.08).route([1.0, 0.0])
    assert intent == 'book'
    assert score == pytest.approx(1.0)


def test_prototype_below_threshold_defers() -> None:
    assert _router(threshold=0.75, margin=0.08).route([0.6, 0.8])[0] is None


def test_prototype_tie_defers_even_above_threshold() -> None:
    """book 分很高但 chat/history 同样高（无领先）→ 不敢判，交 LLM。"""
    assert _router(threshold=0.75, margin=0.08).route([0.9, 0.9])[0] is None


def test_prototype_degrades_safely_on_flat_embeddings() -> None:
    """退化 embedding（所有文本同向量）必须不产生置信判定 —— 落到 L3 兜底。

    测试与评测里的 FakeEmbeddings 正是这种退化形态，这条保证了
    换掉真模型时级联会安全退化为「原样走 LLM」，而不是乱判。
    """
    flat = PrototypeRouter(
        lambda texts: [[1.0, 0.0] for _ in texts], _PROTOS,
        threshold=0.75, margin=0.08,
    )
    assert flat.route([1.0, 0.0])[0] is None


# --------------------------------------------------------------- 级联编排


def test_cascade_rule_hit_does_not_embed_or_llm() -> None:
    """规则命中的轮次连 embedding 都不该花 —— embed 是惰性回调。"""
    calls = {'embed': 0}

    def _embed() -> list[float]:
        calls['embed'] += 1
        return [1.0, 0.0]

    decision = route_fast('我上一个问题是什么？', _router(), _embed)
    assert decision.intent == 'history'
    assert decision.channel == 'rule'
    assert calls['embed'] == 0, '规则命中不应触发 embedding'


def test_cascade_defers_to_llm_when_unsure() -> None:
    """前两层都拿不准 → intent=None，由上层 LLM 兜底（fallback 方向仍是 book）。"""
    decision = route_fast('张三喜欢谁？', _router(), lambda: [0.6, 0.8])
    assert decision.intent is None
    assert decision.channel == ''


def test_cascade_soft_prototype_still_feeds_back_vector() -> None:
    """原型不确认时向量仍要带出去，供 cache_check 复用，不白算。"""
    decision = route_fast('张三喜欢谁？', _router(), lambda: [0.6, 0.8])
    assert decision.vector == [0.6, 0.8]


def test_cascade_without_router_defers() -> None:
    assert route_fast('张三喜欢谁？', None, lambda: [1.0, 0.0]).intent is None


def test_cascade_rules_can_be_disabled() -> None:
    decision = route_fast('我上一个问题是什么？', None, None, rules_enabled=False)
    assert decision.intent is None, '停用规则层后应完全交给 LLM'


def test_cascade_never_produces_chat_or_history_without_rules() -> None:
    """除规则层外，任何通道都不允许产出 chat/history。"""
    router = _router(threshold=0.75, margin=0.08)
    for vector in ([1.0, 0.0], [0.0, 1.0], [0.9, 0.9]):
        decision = route_fast('随便问一句书里的内容', router, lambda v=vector: v)
        assert decision.intent in (None, 'book')
