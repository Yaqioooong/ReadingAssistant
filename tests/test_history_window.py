"""历史窗口：token 预算 + 整条保留 + 单条按 token 截断。

取代原「超出 6 条消息就滚动摘要」方案。这里几乎每条断言都对应旧方案的一个具体缺陷：
- 条数窗口 → 与真实开销脱钩（长回答短回答一视同仁）
- 字符截断 → 中文里字符数远小于 token 数，长回答被过早砍掉
- 滚动摘要 → 每轮一次 LLM、且在 prompt 前缀里每轮都变（破坏 prefix cache）
"""
from __future__ import annotations

from reading_assistant.graph.qa import (
    _HISTORY_ROLE_OVERHEAD,
    _clip_tokens,
    _count_tokens,
    _render_history,
    _select_window,
)


def _row(role: str, text: str) -> tuple[str, str]:
    return (role, text)


def _fit_budget(*texts: str) -> int:
    """刚好装得下这些文本的预算（含角色前缀开销）。"""
    return sum(_count_tokens(t) + _HISTORY_ROLE_OVERHEAD for t in texts)


class TestTokenHelpers:
    def test_count_tokens_is_monotonic(self) -> None:
        assert _count_tokens('你好') > 0
        assert _count_tokens('你好' * 10) > _count_tokens('你好')

    def test_clip_leaves_short_text_untouched(self) -> None:
        text = '很短的一句话'
        assert _clip_tokens(text, 100) == text

    def test_clip_truncates_long_text(self) -> None:
        text = '西游记' * 500
        clipped = _clip_tokens(text, 20)
        assert clipped.endswith('…')
        assert _count_tokens(clipped) < _count_tokens(text) / 10

    def test_zero_cap_means_no_limit(self) -> None:
        text = '西游记' * 500
        assert _clip_tokens(text, 0) == text


class TestSelectWindow:
    """窗口选择：从**最新**往回整条收，超预算即停。"""

    def test_keeps_everything_within_budget(self) -> None:
        rows = [_row('user', '一'), _row('assistant', '二'), _row('user', '三')]
        picked = _select_window(rows, 10_000)
        assert [p['content'] for p in picked] == ['一', '二', '三']

    def test_returns_chronological_order(self) -> None:
        """最早在前 —— ``_render_history`` 依赖这个顺序。"""
        rows = [_row('user', '早'), _row('assistant', '晚')]
        picked = _select_window(rows, 10_000)
        assert [p['role'] for p in picked] == ['user', 'assistant']

    def test_drops_oldest_first_when_over_budget(self) -> None:
        """超预算时丢的是**最早**的，保留最近的 —— 这是「窗口」的定义。"""
        rows = [
            _row('user', '第1条消息'),
            _row('assistant', '第2条消息'),
            _row('user', '第3条消息'),
        ]
        picked = _select_window(rows, _fit_budget('第2条消息', '第3条消息'))
        assert [p['content'] for p in picked] == ['第2条消息', '第3条消息']

    def test_never_truncates_a_message_in_half(self) -> None:
        """整条收或整条不收：半条截断的语义不可预测、也无从断言。"""
        rows = [_row('user', '第一条'), _row('assistant', '第二条')]
        picked = _select_window(rows, _fit_budget('第二条'))
        assert [p['content'] for p in picked] == ['第二条']

    def test_always_keeps_the_newest_message(self) -> None:
        """最新一条无条件保留，哪怕它自己就超预算。

        否则一条长回答会把窗口清成空的，等于把「刚说过什么」整个丢掉 ——
        而「刚说过什么」恰恰是指代消解最需要的东西。
        单条长度另有 ``history_msg_token_cap`` 兜底，不会失控。
        """
        rows = [_row('user', '很长' * 500)]
        assert len(_select_window(rows, 0)) == 1

    def test_empty_rows_gives_empty_window(self) -> None:
        assert _select_window([], 1000) == []

    def test_none_content_does_not_crash(self) -> None:
        """库里可能有 content 为空的历史行（旧数据/异常写入）。"""
        picked = _select_window([_row('assistant', None)], 100)  # type: ignore[arg-type]
        assert len(picked) == 1


class TestRenderHistory:
    def test_empty_history_renders_empty(self) -> None:
        assert _render_history(None) == ''
        assert _render_history([]) == ''

    def test_roles_are_labelled(self) -> None:
        text = _render_history(
            [{'role': 'user', 'content': '问'}, {'role': 'assistant', 'content': '答'}]
        )
        assert text == '用户：问\n助手：答'

    def test_per_message_cap_applies(self) -> None:
        long_text = '西游记' * 1000
        text = _render_history([{'role': 'assistant', 'content': long_text}], cap=20)
        assert text.endswith('…')
        assert len(text) < len(long_text) / 10
