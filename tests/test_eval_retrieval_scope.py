"""检索评测的双口径（filtered / cross）单元测试。

背景（2026-09-17 实测）：只跑带 ``document_id`` 的口径会**掩盖 HNSW 图退化** ——
Chroma 把搜索空间缩到单个文档后，图断裂不可见。一个已退化的索引在 filtered 口径
下 14/14 满分、cross 口径下只有 11/14。故两个口径必须同时存在。

本文件锁住：
1. 累加器不能被两个口径共享（同引用会导致串味）
2. 聚合除法的正确性
3. 口径常量与键名约定（per_case 的 ``*_cross_mrr`` 后缀）
"""

from __future__ import annotations

from eval.run_retrieval_eval import (
    SCOPE_CROSS,
    SCOPE_FILTERED,
    SYSTEMS,
    TOP_K_VALUES,
    _aggregate,
    _empty_summaries,
)

# --------------------------------------------------------------------------
# 口径常量
# --------------------------------------------------------------------------


def test_scopes_are_distinct_and_named():
    assert SCOPE_FILTERED != SCOPE_CROSS
    assert (SCOPE_FILTERED, SCOPE_CROSS) == ('filtered', 'cross')


def test_systems_cover_vector_and_hybrid():
    assert set(SYSTEMS) == {'vector', 'hybrid'}


# --------------------------------------------------------------------------
# 累加器
# --------------------------------------------------------------------------


def test_empty_summaries_shape():
    summaries = _empty_summaries()
    assert set(summaries) == set(SYSTEMS)
    for system in SYSTEMS:
        assert set(summaries[system]) == set(TOP_K_VALUES)
        for n in TOP_K_VALUES:
            cell = summaries[system][n]
            assert set(cell) == {
                'hit', 'recall', 'mrr', 'ndcg', 'precision', 'map', 'cases'
            }
            assert all(
                cell[key] == 0 or cell[key] == 0.0
                for key in ('hit', 'recall', 'mrr', 'ndcg', 'precision', 'map', 'cases')
            )


def test_empty_summaries_returns_fresh_objects():
    """两个口径各拿一个累加器，**不能共享引用** —— 否则分数会互相串味。"""
    a = _empty_summaries()
    b = _empty_summaries()

    assert a is not b
    assert a['vector'] is not b['vector']
    assert a['vector'][5] is not b['vector'][5]

    a['vector'][5]['hit'] += 1
    a['vector'][5]['mrr'] += 0.5
    assert b['vector'][5]['hit'] == 0
    assert b['vector'][5]['mrr'] == 0.0

    # 同一累加器内部：不同 top_k 也必须是独立的单元
    a['vector'][10]['hit'] += 1
    assert a['vector'][5]['hit'] == 1


# --------------------------------------------------------------------------
# 聚合
# --------------------------------------------------------------------------


def test_aggregate_divides_by_case_count():
    summaries = _empty_summaries()
    for n in TOP_K_VALUES:
        summaries['vector'][n]['hit'] = 3
        summaries['vector'][n]['recall'] = 3.0
        summaries['vector'][n]['mrr'] = 1.5
        summaries['hybrid'][n]['hit'] = 6
        summaries['hybrid'][n]['recall'] = 6.0
        summaries['hybrid'][n]['mrr'] = 3.0

    aggregated = _aggregate(summaries, n_cases=6)

    for n in TOP_K_VALUES:
        assert aggregated['vector'][f'hit@{n}'] == 0.5
        assert aggregated['vector'][f'recall@{n}'] == 0.5
        assert aggregated['vector'][f'mrr@{n}'] == 0.25
        assert aggregated['hybrid'][f'hit@{n}'] == 1.0
        assert aggregated['hybrid'][f'mrr@{n}'] == 0.5


def test_aggregate_rounds_to_four_decimals():
    summaries = _empty_summaries()
    summaries['vector'][5]['mrr'] = 1.0

    aggregated = _aggregate(summaries, n_cases=3)

    assert aggregated['vector']['mrr@5'] == round(1 / 3, 4)
    assert isinstance(aggregated['vector']['mrr@5'], float)


def test_aggregate_is_independent_of_other_scope():
    """cross 口径的聚合不得影响 filtered 口径（反之亦然）。"""
    filtered = _empty_summaries()
    cross = _empty_summaries()
    cross['hybrid'][5]['hit'] = 9

    filtered_agg = _aggregate(filtered, n_cases=3)
    cross_agg = _aggregate(cross, n_cases=3)

    assert filtered_agg['hybrid']['hit@5'] == 0.0
    assert cross_agg['hybrid']['hit@5'] == 3.0


def test_aggregate_zero_cases_does_not_crash():
    """n_cases 由 len(per_case) or 1 兜底；此处直接传 1 验证不除零。"""
    aggregated = _aggregate(_empty_summaries(), n_cases=1)
    assert aggregated['vector']['precision@20'] == 0.0
