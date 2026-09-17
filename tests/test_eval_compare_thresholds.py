"""回归门禁的显著性判定：按指标族的量纲选阈值口径。

背景（2026-09-17 实测）
----------------------
全局 ``--threshold`` 是**绝对值**（默认 0.02），只适配 0~1 的比率型指标。
对 ``avg_latency_ms``（毫秒）而言 ``abs(delta) >= 0.02`` 恒成立 —— 历史 6 次
real 运行里 **5/5 次相邻对比都被标记**，而同期 ``pass_rate`` 恒为 1.0 从未变化。
假警报率 100%，会把回归门禁变成狼来了。

修法：噪声指标族走「相对 + 绝对同时超阈」；比率型指标行为保持不变。
"""

from __future__ import annotations

import pytest

from eval.compare_reports import (
    _exceeds_threshold,
    _threshold_override,
    detect_type,
    diff_metrics,
    extract_metrics,
)

DEFAULT_THRESHOLD = 0.02

# 历史 real 运行的 avg_latency_ms 实测序列（同数据集、同 gold、质量全程 1.0）
HISTORICAL_LATENCIES = (1421, 774, 932, 687, 581, 710)


# --------------------------------------------------------------------------
# 噪声指标族（延迟）
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ('base', 'new'),
    list(zip(HISTORICAL_LATENCIES, HISTORICAL_LATENCIES[1:])),
)
def test_historical_latency_jitter_is_not_significant(base: int, new: int) -> None:
    """回归用例：历史自然波动（最大 ±45%）全部不得判为显著。

    这些波动对应的质量指标（pass_rate / hallucination_rate）全程未变，
    若被判为回归即为假警报。
    """
    delta = float(new - base)
    assert not _exceeds_threshold('e2e.avg_latency_ms', base, delta, DEFAULT_THRESHOLD)


@pytest.mark.parametrize(
    ('base', 'new'),
    [
        (581, 1200),   # 2.1x
        (581, 1421),   # 2.4x
        (581, 1800),   # 3.1x
        (774, 1600),   # 2.1x
    ],
)
def test_real_latency_degradation_is_significant(base: int, new: int) -> None:
    """真正的延迟劣化（2 倍以上）必须被拦下 —— 修阈值不能修成瞎子。"""
    delta = float(new - base)
    assert _exceeds_threshold('e2e.avg_latency_ms', base, delta, DEFAULT_THRESHOLD)


def test_latency_requires_both_relative_and_absolute() -> None:
    """相对与绝对必须同时超阈。"""
    # 相对超 50% 但绝对不足 200ms → 不显著
    assert not _exceeds_threshold('e2e.avg_latency_ms', 100, 160, DEFAULT_THRESHOLD)
    # 绝对超 200ms 但相对不足 50% → 不显著
    assert not _exceeds_threshold('e2e.avg_latency_ms', 5000, 300, DEFAULT_THRESHOLD)
    # 两者都超 → 显著
    assert _exceeds_threshold('e2e.avg_latency_ms', 1000, 700, DEFAULT_THRESHOLD)


def test_latency_improvement_also_needs_both() -> None:
    """改善方向同样按该口径判定（否则会误报「大幅提速」）。"""
    assert not _exceeds_threshold('e2e.avg_latency_ms', 1421, -647, DEFAULT_THRESHOLD)
    assert _exceeds_threshold('e2e.avg_latency_ms', 2000, -1500, DEFAULT_THRESHOLD)


def test_bare_and_qualified_metric_names_both_match() -> None:
    """``avg_latency_ms`` 与 ``e2e.avg_latency_ms`` 都要命中同一策略。"""
    assert _threshold_override('avg_latency_ms') is not None
    assert _threshold_override('e2e.avg_latency_ms') is not None
    assert _threshold_override('avg_latency_ms') == _threshold_override('e2e.avg_latency_ms')


# --------------------------------------------------------------------------
# 比率型指标：行为必须与修前一致
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ('metric', 'base', 'new', 'expected'),
    [
        ('e2e.hallucination_rate', 0.0, 0.05, True),      # 幻觉率上升 → 显著
        ('e2e.false_refusal_rate', 0.0, 0.01, False),     # 低于阈值 → 不显著
        ('e2e.pass_rate', 1.0, 0.95, True),               # 通过率下降 → 显著
        ('e2e.pass_rate', 1.0, 0.999, False),             # 波动小于阈值 → 不显著
        ('retrieval.vector.recall@5', 1.0, 0.98, True),
        ('retrieval.hybrid.precision@5', 0.9524, 1.0, True),
        ('intent.route_accuracy', 1.0, 0.99, False),
    ],
)
def test_ratio_metrics_keep_global_absolute_threshold(
    metric: str, base: float, new: float, expected: bool
) -> None:
    delta = new - base
    assert _exceeds_threshold(metric, base, delta, DEFAULT_THRESHOLD) is expected


def test_ratio_metrics_have_no_override() -> None:
    for metric in ('e2e.pass_rate', 'retrieval.vector.recall@5', 'intent.route_accuracy'):
        assert _threshold_override(metric) is None


# --------------------------------------------------------------------------
# 端到端：门禁不再被延迟噪声触发
# --------------------------------------------------------------------------


def _report(latency: float, pass_rate: float = 1.0) -> dict:
    """构造最小 e2e 报告（``detect_type`` 依赖顶层 results 键）。"""
    return {
        'summary': {
            'mode': 'real',
            'cases': 15,
            'pass_rate': pass_rate,
            'avg_latency_ms': latency,
        },
        'results': [],
    }


def test_gate_not_triggered_by_latency_jitter_alone() -> None:
    """回归用例：质量未变、仅延迟波动 → 不得判定为回归。"""
    base = _report(581.0)
    new = _report(710.0)

    rows = diff_metrics(
        extract_metrics(base, detect_type(base)),
        extract_metrics(new, detect_type(new)),
        DEFAULT_THRESHOLD,
    )
    by_metric = {row['metric']: row for row in rows}

    assert by_metric['avg_latency_ms']['is_regression'] is False
    assert by_metric['avg_latency_ms']['direction'] == '='
    assert by_metric['pass_rate']['is_regression'] is False


def test_gate_still_triggered_by_real_latency_degradation() -> None:
    base = _report(581.0)
    new = _report(1800.0)
    rows = diff_metrics(
        extract_metrics(base, detect_type(base)),
        extract_metrics(new, detect_type(new)),
        DEFAULT_THRESHOLD,
    )
    by_metric = {row['metric']: row for row in rows}

    assert by_metric['avg_latency_ms']['is_regression'] is True
    assert by_metric['avg_latency_ms']['direction'] == 'DOWN'  # 越低越好 → 变差记 DOWN
