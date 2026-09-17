"""索引状态对账：把卡死的 ``indexing`` 记录拉回 ``indexed`` / ``failed``。

背景
----
``index_status`` 的状态机为 ``pending → indexing → indexed | failed``。进程在
「向量已写、PG 状态未提交」的窗口内崩溃时，不走 ``except``，连 ``failed`` 都
写不上，记录会**永久卡在 indexing**（本仓库实测 3/6 文档如此）。

策略（租约 + 对账）
-------------------
仅处理「``indexing`` 且租约过期」的记录（``index_started_at`` 为空，或早于
``now - lease_timeout``）；再用**向量库实况**核对内容是否完整：

- 统计向量库中 ``document_id`` 匹配且 ``content_hash == doc.content_hash`` 的
  chunk 数，与 ``doc.chunk_count`` 比对；
- 一致且非空 → 修正为 ``indexed``；否则 → ``failed``。

该函数**幂等**：执行后记录不再是 ``indexing``，重复调用不再产生动作。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from reading_assistant.storage.models import Document
from reading_assistant.storage.vector_store import VectorStore
from reading_assistant.utils.logger_handler import get_logger

logger = get_logger('reconcile')

#: 索引租约超时（秒）。进入 indexing 超过该时长仍未被标记完成，视为崩溃残留。
DEFAULT_LEASE_TIMEOUT_SECONDS = 600


@dataclass
class ReconcileAction:
    """一条将被（或已被）修正的对账动作，供 dry-run 打印。"""

    document_id: int
    filename: str
    previous_status: str
    new_status: str
    expected_chunks: int
    actual_chunks: int
    reason: str


def _as_utc(value: datetime | None) -> datetime | None:
    """把 naive（SQLite）时间视为 UTC，与 PG 的 aware 时间可比。"""
    if value is None:
        return None
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


def _lease_expired(started_at: datetime | None, now: datetime, lease_timeout: int) -> bool:
    """租约是否过期：未记录开始时间视为过期（历史崩溃残留）。"""
    started = _as_utc(started_at)
    if started is None:
        return True
    return started < now - timedelta(seconds=lease_timeout)


def _count_current_chunks(
    vector_store: VectorStore, document_id: int, content_hash: str
) -> int:
    """统计向量库中属于该文档**当前版本**的 chunk 数。"""
    matched = 0
    for chunk in vector_store.all_chunks():
        metadata = chunk.metadata or {}
        if (
            metadata.get('document_id') == document_id
            and metadata.get('content_hash') == content_hash
        ):
            matched += 1
    return matched


def reconcile_index_status(
    session_factory: sessionmaker[Session],
    vector_store: VectorStore,
    lease_timeout_seconds: int = DEFAULT_LEASE_TIMEOUT_SECONDS,
    apply: bool = False,
) -> list[ReconcileAction]:
    """对账卡死的 ``indexing`` 记录；``apply=False`` 时只做 dry-run。

    返回待/已修正的动作列表（``apply=True`` 时即实际写入的动作）。
    """
    now = datetime.now(timezone.utc)
    actions: list[ReconcileAction] = []
    with session_factory() as session:
        candidates = list(
            session.scalars(select(Document).where(Document.index_status == 'indexing'))
        )
        for document in candidates:
            if not _lease_expired(document.index_started_at, now, lease_timeout_seconds):
                continue  # 租约未过期：可能仍在正常入库，跳过
            expected = int(document.chunk_count or 0)
            actual = _count_current_chunks(vector_store, document.id, document.content_hash)
            healthy = expected > 0 and actual == expected
            new_status = 'indexed' if healthy else 'failed'
            reason = (
                f'向量完整（{actual}/{expected}）'
                if healthy
                else f'向量不完整或为空（{actual}/{expected}）'
            )
            actions.append(
                ReconcileAction(
                    document_id=document.id,
                    filename=document.filename,
                    previous_status=document.index_status,
                    new_status=new_status,
                    expected_chunks=expected,
                    actual_chunks=actual,
                    reason=reason,
                )
            )
            if apply:
                document.index_status = new_status
        if apply and actions:
            session.commit()

    for action in actions:
        logger.info(
            '对账[索引状态] doc=%s file=%s %s → %s（%s）',
            action.document_id,
            action.filename,
            action.previous_status,
            action.new_status,
            action.reason,
        )
    if not actions:
        logger.info('对账[索引状态] 无卡死记录（租约 %ds）', lease_timeout_seconds)
    return actions
