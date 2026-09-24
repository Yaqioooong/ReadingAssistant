"""历史窗口预算 A/B：用**生产库真实消息**量「各预算下到底截断多少会话」。

为什么要这个探针：``history_token_budget`` 是可被环境变量覆盖的配置
（``HISTORY_TOKEN_BUDGET=16000``），本来就是可 A/B 的 —— 但**没有任何现成夹具能看见
它的改动**：``eval/multi_turn_gold.json`` 最长 6 轮，而窗口装得下更多，
budget 调到 32000 端到端指标会逐字相同（不是改动无害，是夹具看不见）。

更关键的是，``scripts/probe_window_capacity.py`` 用的是**合成文本**
（助手单条 520 token），而生产库真实助手消息 p50 只有 55 token —— 高估近 10 倍。
拿合成文本推「4000 只装 7 轮」会得出与现实相反的结论。所以容量必须用真实序列量。

跑法：``uv run python scripts/probe_context_budget_ab.py``
      ``uv run python scripts/probe_context_budget_ab.py --budgets 4000,8000,16000,32000``

判读：
- 「被截断的会话占比」是**决策指标**。预算只需覆盖真实流量里最长的那个会话；
  覆盖率已是 100% 时再调大，是纯粹的上限空转（不花钱，但也没有收益）。
- 「每轮中位开销」用于把预算换算成轮数 —— 注意它来自真实消息长度，不是合成文本。
"""
from __future__ import annotations

import argparse
import collections
from pathlib import Path

from sqlalchemy import func, select

from reading_assistant.config import get_settings
from reading_assistant.graph.qa import (
    _HISTORY_ROLE_OVERHEAD,
    _count_tokens,
    _select_window,
)
from reading_assistant.storage import create_db_engine, create_session_factory
from reading_assistant.storage.models import ChatMessage

DEFAULT_BUDGETS = (4000, 8000, 16000, 32000)


def _pct(sorted_vals: list[int], p: float) -> int:
    if not sorted_vals:
        return 0
    return sorted_vals[min(len(sorted_vals) - 1, int(len(sorted_vals) * p))]


def main() -> None:
    parser = argparse.ArgumentParser(description='历史窗口预算 A/B（真实消息序列）')
    parser.add_argument(
        '--budgets',
        default=','.join(str(b) for b in DEFAULT_BUDGETS),
        help='逗号分隔的预算列表',
    )
    parser.add_argument('--env-file', default=None, help='备用的 .env 路径（默认用配置里的）')
    args = parser.parse_args()
    if args.env_file:
        # 必须在 get_settings 之前落进进程环境，否则 @lru_cache 读到的是 .env 那份
        import os

        for line in Path(args.env_file).read_text(encoding='utf-8').splitlines():
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                k, v = line.split('=', 1)
                os.environ.setdefault(k.strip(), v.strip())
    budgets = [int(b) for b in args.budgets.split(',') if b.strip()]

    settings = get_settings()
    engine = create_db_engine(settings.database_url)
    factory = create_session_factory(engine)
    with factory() as db:
        messages = list(db.scalars(select(ChatMessage).order_by(ChatMessage.id)))
        per_session = db.execute(
            select(ChatMessage.session_id, func.count(ChatMessage.id)).group_by(
                ChatMessage.session_id
            )
        ).all()

    if not messages:
        print('库里没有消息：先在 app 里产生一些会话再跑本探针')
        return

    by_session: dict[int, list[ChatMessage]] = collections.defaultdict(list)
    for m in messages:
        by_session[m.session_id].append(m)
    counts = sorted(c for _, c in per_session)
    user_tok = sorted(_count_tokens(m.content or '') for m in messages if m.role == 'user')
    asst_tok = sorted(_count_tokens(m.content or '') for m in messages if m.role == 'assistant')

    print(f'当前生效预算：history_token_budget={settings.history_token_budget} '
          f'history_msg_token_cap={settings.history_msg_token_cap}')
    print(f'（可用 HISTORY_TOKEN_BUDGET=… 环境变量覆盖后重跑本探针对比）\n')
    print(f'真实流量：{len(counts)} 个会话 / {len(messages)} 条消息')
    print(f'  每会话消息条数：p50={_pct(counts, .5)} p90={_pct(counts, .9)} '
          f'p99={_pct(counts, .99)} max={counts[-1]}')
    print(f'  用户消息 token：p50={_pct(user_tok, .5)} p90={_pct(user_tok, .9)} max={user_tok[-1]}')
    print(f'  助手消息 token：p50={_pct(asst_tok, .5)} p90={_pct(asst_tok, .9)} max={asst_tok[-1]}')
    # 合成文本（旧探针用的形态）与真实分布差多少 —— 这是「别用合成文本推容量」的证据
    synth_turn = 520 + 28 + 2 * _HISTORY_ROLE_OVERHEAD
    real_p50 = _pct(asst_tok, .5) + _pct(user_tok, .5) + 2 * _HISTORY_ROLE_OVERHEAD
    real_p90 = _pct(asst_tok, .9) + _pct(user_tok, .9) + 2 * _HISTORY_ROLE_OVERHEAD
    print(f'  每轮开销：真实 p50≈{real_p50} token、真实 p90≈{real_p90} token、'
          f'旧探针合成文本≈{synth_turn} token（高估 {synth_turn / real_p50:.1f}×）')
    print()

    header = f'{"budget":>8} | {"截断会话":>12} | {"中位保留":>8} | {"p10 保留":>8} | 最长会话'
    print(header)
    print('-' * len(header))
    for budget in budgets:
        truncated = 0
        kept_counts: list[int] = []
        longest = (0, 0)  # (原条数, 保留条数)
        for rows_msgs in by_session.values():
            rows = [(m.role, m.content or '') for m in rows_msgs]
            kept = _select_window(rows, budget)
            if len(kept) < len(rows):
                truncated += 1
            kept_counts.append(len(kept))
            if len(rows) > longest[0]:
                longest = (len(rows), len(kept))
        kept_counts.sort()
        ratio = f'{truncated}/{len(counts)} = {truncated / len(counts) * 100:.1f}%'
        print(
            f'{budget:>8} | {ratio:>12} | {_pct(kept_counts, .5):>6} 条 | '
            f'{_pct(kept_counts, .1):>6} 条 | {longest[0]} 条只留 {longest[1]} 条'
        )

    print(
        '\n判读：预算是**上限不是配额** —— 短会话永远碰不到它，调大对它们零成本；\n'
        '      只有真长到那一档的会话才付费，而那些会话恰恰需要上下文。\n'
        '      所以「截断占比」降到 0 之后，再往上调只是空转上限，既不加分也不花钱。'
    )


if __name__ == '__main__':
    main()
