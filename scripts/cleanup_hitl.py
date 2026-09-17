"""清理僵尸 HITL 任务（长时间卡在 awaiting 的澄清任务）。

背景：评测脚本与会话调试会在库里留下大量 awaiting 任务；前端未接通澄清交互前
它们无法被推进，接通后又会以「待补充」卡片形式回到会话里，需要定期清理。

用法：
    uv run python scripts/cleanup_hitl.py                 # 预演：只统计，不删
    uv run python scripts/cleanup_hitl.py --apply         # 执行删除
    uv run python scripts/cleanup_hitl.py --apply --reject  # 改为标记 rejected（保留记录）
    uv run python scripts/cleanup_hitl.py --apply --older-than-days 3   # 只清更早的
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone

from sqlalchemy import select

from reading_assistant.storage import create_db_engine, create_session_factory
from reading_assistant.storage.models import HitlTask


def _stats(tasks: list[HitlTask]) -> dict[str, int]:
    out: dict[str, int] = {}
    for t in tasks:
        out[t.status] = out.get(t.status, 0) + 1
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description='清理僵尸 HITL 任务')
    parser.add_argument('--apply', action='store_true', help='真正执行（默认只预演）')
    parser.add_argument('--reject', action='store_true',
                        help='只标记为 rejected，不删除记录')
    parser.add_argument('--older-than-days', type=float, default=0,
                        help='只处理创建时间早于 N 天的任务（默认 0 = 全部等待中的）')
    args = parser.parse_args()

    factory = create_session_factory(create_db_engine())
    with factory() as session:
        all_tasks = list(session.scalars(select(HitlTask)))
        awaiting = [t for t in all_tasks if t.status == HitlTask.STATUS_AWAITING]
        targets = awaiting
        if args.older_than_days > 0:
            cutoff = datetime.now(timezone.utc) - timedelta(days=args.older_than_days)
            targets = [
                t for t in awaiting
                if t.created_at is not None
                and (t.created_at if t.created_at.tzinfo else
                     t.created_at.replace(tzinfo=timezone.utc)) < cutoff
            ]

        print(f'全部任务 {len(all_tasks)} 个，状态分布 {_stats(all_tasks)}')
        print(f'待处理(awaiting) {len(awaiting)} 个，本次命中 {len(targets)} 个')
        if not targets:
            print('没有需要清理的任务。')
            return

        by_day: dict[str, int] = {}
        for t in targets:
            key = t.created_at.strftime('%Y-%m-%d') if t.created_at else '未知'
            by_day[key] = by_day.get(key, 0) + 1
        print('命中任务按日期：', dict(sorted(by_day.items())))

        if not args.apply:
            print('\n[预演] 未做任何修改。加 --apply 执行。')
            return

        if args.reject:
            for t in targets:
                t.status = HitlTask.STATUS_REJECTED
            action = '标记为 rejected'
        else:
            for t in targets:
                session.delete(t)
            action = '删除'
        session.commit()
        remain = list(session.scalars(select(HitlTask)))
        print(f'\n已{action} {len(targets)} 个任务。')
        print(f'剩余 {len(remain)} 个，状态分布 {_stats(remain)}')


if __name__ == '__main__':
    main()
