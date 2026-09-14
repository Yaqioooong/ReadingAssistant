"""指标路由：缓存体系的可观测性看板数据源。

聚合口径见 docs/cache-metrics-contract.md。选在 Python 侧聚合而非纯 SQL，
是为了同时兼容 SQLite(测试) 与 PostgreSQL(生产) 的日期/JSON 差异，
且窗口内样本量很小（万级以下），代价可忽略。
"""

import math
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from reading_assistant.api.deps import get_db_session
from reading_assistant.config import get_settings
from reading_assistant.storage import QaCacheEntry, QaFeedback, QaRequestEvent

router = APIRouter(prefix='/api/stats', tags=['stats'])

_TOP_HITS_LIMIT = 10
_EXPIRING_SOON_DAYS = 3  # 距 TTL 到期不足这些天，视为"临近过期"
_NEAR_THRESHOLD_GAP = 0.01  # 与阈值距离小于该值，视为"贴近阈值"


def _as_utc(value: datetime | None) -> datetime | None:
    """统一时区：SQLite 返回 naive，PostgreSQL 返回 aware。"""
    if value is None:
        return None
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


def _percentile(values: list[float], q: float) -> float | None:
    """线性插值分位数；空样本返回 None。"""
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _mean(values: list[int]) -> int | None:
    return round(sum(values) / len(values)) if values else None


def _ratio(numerator: int, denominator: int) -> float | None:
    """分母为 0 时返回 None —— 前端据此显示「—」而不是误导性的 0%。"""
    return numerator / denominator if denominator else None


def _cache_summary(events: list[QaRequestEvent]) -> tuple[dict, dict]:
    """请求级汇总 + 通道分布。"""
    hits = [e for e in events if e.cache_hit]
    misses = [e for e in events if not e.cache_hit]
    hit_latency = [e.latency_ms for e in hits if e.latency_ms is not None]
    miss_latency = [e.latency_ms for e in misses if e.latency_ms is not None]
    avg_hit, avg_miss = _mean(hit_latency), _mean(miss_latency)
    saved_ms = (avg_miss - avg_hit) * len(hits) if avg_hit and avg_miss else None

    by_channel = {
        'exact': 0, 'semantic': 0, 'identifier': 0,
        'miss': 0, 'disabled': 0, 'skipped': 0,
    }
    for event in events:
        key = event.cache_channel
        if key in by_channel:
            by_channel[key] += 1

    # 可缓存请求 = 全量 - 不适用缓存(多文档/关闭) - 未走缓存检查(闲聊/历史)。
    # 用全量作命中率分母会被闲聊稀释（它们永远不命中），导致指标失真。
    cacheable = len(events) - by_channel['disabled'] - by_channel['skipped']
    summary = {
        'total_requests': len(events),
        'cacheable_requests': cacheable,
        'hit_total': len(hits),
        'miss_total': len(misses),
        'hit_rate': _ratio(len(hits), len(events)),
        'hit_rate_cacheable': _ratio(len(hits), cacheable),
        'avg_latency_hit_ms': avg_hit,
        'avg_latency_miss_ms': avg_miss,
        'saved_ms_total': saved_ms,
        'saved_calls': len(hits),
    }
    return summary, by_channel


def _semantic_stats(events: list[QaRequestEvent], threshold: float) -> dict:
    """相似度分布 —— 阈值调优的依据。

    分位口径只取**语义通道命中**样本（与 by_channel.semantic 一致）；
    ``near_threshold`` 统计未命中但距阈值不足 0.01 的"擦肩而过"次数，
    是下调阈值能多拿多少命中的直接预估。
    """
    # 只取语义通道命中样本：与 by_channel.semantic 口径一致
    # （标识符通道的相似度被 0.90 下限约束，单独看意义不大）
    hit_scores = [
        e.cache_similarity for e in events
        if e.cache_channel == 'semantic' and e.cache_similarity is not None
    ]
    near_miss = sum(
        1 for e in events
        if not e.cache_hit
        and e.cache_similarity is not None
        and 0 <= threshold - e.cache_similarity < _NEAR_THRESHOLD_GAP
    )
    return {
        'count': len(hit_scores),
        'p50': _percentile(hit_scores, 0.5),
        'p95': _percentile(hit_scores, 0.95),
        'min': min(hit_scores) if hit_scores else None,
        'max': max(hit_scores) if hit_scores else None,
        'threshold': threshold,
        'near_threshold': near_miss,
    }


def _feedback_stats(feedbacks: list[QaFeedback]) -> dict:
    """误命中率 vs 非缓存错误率 —— 缓存质量的核心对照。"""
    up = sum(1 for f in feedbacks if f.vote == 'up')
    down = sum(1 for f in feedbacks if f.vote == 'down')
    cached = [f for f in feedbacks if f.cache_hit]
    fresh = [f for f in feedbacks if not f.cache_hit]
    cached_down = sum(1 for f in cached if f.vote == 'down')
    fresh_down = sum(1 for f in fresh if f.vote == 'down')
    return {
        'up': up,
        'down': down,
        'total': len(feedbacks),
        'cache_hit_feedback': len(cached),
        'cache_hit_down': cached_down,
        'mis_hit_rate': _ratio(cached_down, len(cached)),
        'non_cache_feedback': len(fresh),
        'non_cache_down': fresh_down,
        'non_cache_error_rate': _ratio(fresh_down, len(fresh)),
    }


def _entry_stats(entries: list[QaCacheEntry], ttl_days: int) -> dict:
    """存量缓存的健康度：冷热、类型、临近过期与热门榜。"""
    now = datetime.now(timezone.utc)
    expiring = 0
    for entry in entries:
        created = _as_utc(entry.created_at)
        if created is None:
            continue
        if (now - created) > timedelta(days=max(ttl_days - _EXPIRING_SOON_DAYS, 0)):
            expiring += 1

    top = sorted(entries, key=lambda e: e.hit_count or 0, reverse=True)
    top_hits = [
        {
            'question': entry.question_raw,
            'hit_count': entry.hit_count or 0,
            'last_hit_at': _as_utc(entry.last_hit_at).isoformat() if entry.last_hit_at else None,
        }
        for entry in top[:_TOP_HITS_LIMIT]
        if (entry.hit_count or 0) > 0
    ]
    return {
        'total': len(entries),
        'cold': sum(1 for e in entries if not e.hit_count),
        'hitl': sum(1 for e in entries if e.needs_clarification),
        'with_citations': sum(1 for e in entries if e.citations),
        'expiring_soon': expiring,
        'total_hits': sum(e.hit_count or 0 for e in entries),
        'top_hits': top_hits,
    }


def _daily_stats(events: list[QaRequestEvent], days: int) -> list[dict]:
    """按天补齐的命中/未命中趋势（缺失日期补 0，便于前端直接画柱）。"""
    today = datetime.now(timezone.utc).date()
    buckets: dict[str, dict[str, int]] = {}
    for offset in range(days - 1, -1, -1):
        buckets[(today - timedelta(days=offset)).isoformat()] = {'hit': 0, 'miss': 0}
    for event in events:
        created = _as_utc(event.created_at)
        if created is None:
            continue
        key = created.date().isoformat()
        if key in buckets:
            buckets[key]['hit' if event.cache_hit else 'miss'] += 1
    return [{'date': key, **value} for key, value in buckets.items()]


@router.get('/cache')
def cache_stats(
    days: int = Query(7, ge=1, le=90, description='统计窗口天数'),
    session: Session = Depends(get_db_session),
):
    """缓存体系指标总览：请求级归因、相似度分布、脏缓存、反馈与存量健康度。"""
    settings = get_settings()
    since = datetime.now(timezone.utc) - timedelta(days=days)

    events = [
        e for e in session.scalars(select(QaRequestEvent))
        if (_as_utc(e.created_at) or since) >= since
    ]
    feedbacks = [
        f for f in session.scalars(select(QaFeedback))
        if (_as_utc(f.created_at) or since) >= since
    ]
    entries = list(session.scalars(select(QaCacheEntry)))

    summary, by_channel = _cache_summary(events)
    invalidated = sum(1 for e in events if e.cache_invalidated)

    return {
        'window_days': days,
        'summary': summary,
        'by_channel': by_channel,
        'semantic': _semantic_stats(events, settings.cache_similarity_threshold),
        'invalidated': {
            'count': invalidated,
            'rate': _ratio(invalidated, len(events)),
        },
        'feedback': _feedback_stats(feedbacks),
        'entries': _entry_stats(entries, settings.cache_ttl_days),
        'daily': _daily_stats(events, days),
    }
