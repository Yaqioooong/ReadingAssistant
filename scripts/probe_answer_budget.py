"""答案 prompt 的检索片段预算：真实语料下片段侧到底吃掉多少 token。

背景：上下文管理原先只有一半 —— 历史侧有 ``history_token_budget``（见
``scripts/probe_window_capacity.py``），而注入答案的检索片段**一个 token 都不数**，
只有 ``top_k`` / ``max_chunks`` 这种按条数的限制。语料一长，prompt 总量就没有上界。

本探针读真实向量库，量单块 token 分布，并模拟 ``_select_answer_chunks``
在预算下裁剪前后的条数与 token 数。默认预算从 settings 读
（``answer_chunk_token_budget``），改配置后可直接复跑对比。

跑法：``uv run python scripts/probe_answer_budget.py``
"""
from __future__ import annotations

from reading_assistant.config import get_settings
from reading_assistant.graph.qa import (
    _CHUNK_LABEL_OVERHEAD,
    _count_tokens,
    _select_answer_chunks,
)
from reading_assistant.storage.vector_store import create_vector_store


def _pct(sorted_vals: list[int], p: float) -> int:
    if not sorted_vals:
        return 0
    return sorted_vals[min(len(sorted_vals) - 1, int(len(sorted_vals) * p))]


def _as_chunks(texts: list[str]) -> list[dict]:
    return [{'chunk_id': f'c{i}', 'text': t} for i, t in enumerate(texts)]


def main() -> None:
    settings = get_settings()
    chunk_objs = create_vector_store().all_chunks()
    toks = sorted(_count_tokens(c.text) + _CHUNK_LABEL_OVERHEAD for c in chunk_objs)
    n = len(toks)
    if not n:
        print('向量库为空：先入库语料再跑本探针')
        return

    print(f'语料：{n} 块（后端 {settings.vector_store_backend}）')
    print(
        f'单块 token（含 {_CHUNK_LABEL_OVERHEAD} 前缀开销）：'
        f'min={toks[0]} p50={_pct(toks, 0.5)} p90={_pct(toks, 0.9)} '
        f'p99={_pct(toks, 0.99)} max={toks[-1]} mean={sum(toks) / n:.0f}'
    )
    print(f'配置：top_k={settings.top_k} · 片段预算={settings.answer_chunk_token_budget} token')
    print(f'      （对照：历史窗口预算={settings.history_token_budget} token）\n')

    # 按 token 排序后再取窗口：chunk_objs 的顺序是库内顺序，
    # 直接切尾部会得到「任意 16 块」而不是「最大的 16 块」
    pairs = sorted(
        (_count_tokens(c.text) + _CHUNK_LABEL_OVERHEAD, c.text) for c in chunk_objs
    )
    mid = n // 2
    k = settings.top_k
    shapes = {
        f'中位窗口（{k} 块）': [text for _, text in pairs[max(0, mid - k // 2): mid + k // 2]],
        f'最坏窗口（最大的 {k} 块）': [text for _, text in pairs[-k:]],
        f'理论极值（{k} × max）': [pairs[-1][1]] * k,
    }
    for label, picked_texts in shapes.items():
        chunks = _as_chunks(picked_texts)
        before = sum(_count_tokens(c['text']) + _CHUNK_LABEL_OVERHEAD for c in chunks)
        kept = _select_answer_chunks(chunks, settings.answer_chunk_token_budget)
        after = sum(_count_tokens(c['text']) + _CHUNK_LABEL_OVERHEAD for c in kept)
        flag = '未裁剪' if len(kept) == len(chunks) else f'裁掉 {len(chunks) - len(kept)} 条'
        print(
            f'{label}：{len(chunks)} 条 / {before} token → {len(kept)} 条 / {after} token（{flag}）'
        )

    print(
        f'\n→ 片段侧预算 {settings.answer_chunk_token_budget} 的效果：'
        f'覆盖实测最坏窗口即视为护栏生效且不损失证据；'
        f'若上面出现「裁掉 N 条」，说明语料/top_k 已增长到需要重新评估该配置。'
    )


if __name__ == '__main__':
    main()
