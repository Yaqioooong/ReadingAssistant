"""索引状态对账：清洗卡死在 ``indexing`` 的残留记录。

默认 dry-run，列出将变更的记录；``--apply`` 才写入。

⚠️ 这是【3】检索过滤 `index_status` 的前置步骤：历史库里 3/6 文档卡在
``indexing`` 但向量其实完整，不先清洗就会在检索层被误杀。

用法（仓库根目录）：
    .venv/bin/python scripts/reconcile_index_status.py             # dry-run
    .venv/bin/python scripts/reconcile_index_status.py --apply     # 执行
    .venv/bin/python scripts/reconcile_index_status.py --lease-timeout 0 --apply
"""

from __future__ import annotations

import argparse
import sys

from reading_assistant.storage import create_db_engine, create_session_factory, init_db
from reading_assistant.storage.reconcile import (
    DEFAULT_LEASE_TIMEOUT_SECONDS,
    reconcile_index_status,
)
from reading_assistant.storage.vector_store import create_vector_store


def main() -> None:
    parser = argparse.ArgumentParser(description='索引状态对账（租约 + 向量库实况核对）')
    parser.add_argument('--apply', action='store_true', help='真正写入（默认 dry-run）')
    parser.add_argument(
        '--lease-timeout',
        type=int,
        default=DEFAULT_LEASE_TIMEOUT_SECONDS,
        help=f'租约超时秒数（默认 {DEFAULT_LEASE_TIMEOUT_SECONDS}）',
    )
    args = parser.parse_args()

    engine = create_db_engine()
    # 幂等确保 schema：create_all + 轻量加列（含 index_started_at）。
    # 缺此步时，旧库上本脚本会因列不存在而直接报错。
    init_db(engine)
    factory = create_session_factory(engine)
    store = create_vector_store()

    actions = reconcile_index_status(
        factory,
        store,
        lease_timeout_seconds=args.lease_timeout,
        apply=args.apply,
    )
    if not actions:
        print(f'无卡死记录（租约 {args.lease_timeout}s）')
        return

    mode = '已应用' if args.apply else 'dry-run（未写入）'
    print(f'{mode}：{len(actions)} 条将变更')
    for action in actions:
        print(
            f'  doc={action.document_id} 《{action.filename}》 '
            f'{action.previous_status} → {action.new_status} '
            f'（{action.reason}）'
        )
    if not args.apply:
        print('加 --apply 执行')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f'执行失败：{exc}', file=sys.stderr)
        raise SystemExit(1) from exc
