"""文档 repository：入库、查询与哈希查重。"""

from datetime import datetime, timezone

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from reading_assistant.storage.models import Document, HitlTask, QaCacheEntry


def get_document(session: Session, document_id: int) -> Document | None:
    """按 id 查询文档。"""
    return session.get(Document, document_id)


def get_document_by_file_hash(session: Session, file_hash: str) -> Document | None:
    """按文件字节哈希查询文档。"""
    return session.scalar(select(Document).where(Document.file_hash == file_hash))


def get_document_by_content_hash(session: Session, content_hash: str) -> Document | None:
    """按归一化内容哈希查询文档。"""
    return session.scalar(select(Document).where(Document.content_hash == content_hash))


def list_documents(session: Session) -> list[Document]:
    """列出全部文档（按入库顺序）。"""
    return list(session.scalars(select(Document).order_by(Document.id)))


def insert_document(session: Session, **fields) -> tuple[Document, bool]:
    """插入文档，返回 (document, inserted)。

    并发场景下唯一索引冲突时回滚并返回已存在的记录（inserted=False）。
    """
    try:
        document = Document(**fields)
        session.add(document)
        session.flush()
        return document, True
    except IntegrityError:
        session.rollback()
        existing = get_document_by_file_hash(session, fields['file_hash'])
        if existing is None:
            existing = get_document_by_content_hash(session, fields['content_hash'])
        return existing, False


def create_hitl_task(session: Session, session_id: int | None, question: str) -> HitlTask:
    """创建 HITL 澄清任务（awaiting）。"""
    task = HitlTask(
        session_id=session_id,
        question=question,
        status=HitlTask.STATUS_AWAITING,
    )
    session.add(task)
    session.flush()
    return task


def get_hitl_task(session: Session, task_id: int) -> HitlTask | None:
    """按 id 查询 HITL 任务。"""
    return session.get(HitlTask, task_id)


def submit_hitl_clarification(session: Session, task_id: int, clarification: str) -> HitlTask:
    """提交澄清：awaiting -> approved。"""
    task = _get_awaiting_task(session, task_id)
    task.clarification = clarification
    task.status = HitlTask.STATUS_APPROVED
    return task


def reject_hitl_task(session: Session, task_id: int) -> HitlTask:
    """拒绝澄清：awaiting -> rejected。"""
    task = _get_awaiting_task(session, task_id)
    task.status = HitlTask.STATUS_REJECTED
    return task


def _get_awaiting_task(session: Session, task_id: int) -> HitlTask:
    task = get_hitl_task(session, task_id)
    if task is None:
        raise ValueError(f'HITL 任务不存在: {task_id}')
    if task.status != HitlTask.STATUS_AWAITING:
        raise ValueError(f'任务状态不允许变更: {task.status}')
    return task


def get_qa_cache_entry(
    session: Session, question_hash: str, content_hash: str, document_id: int | None = None
) -> QaCacheEntry | None:
    """精确命中查询:问题hash+文档版本+文档范围（None = 全库）"""
    stmt = select(QaCacheEntry).where(
        QaCacheEntry.question_hash == question_hash,
        QaCacheEntry.content_hash == content_hash,
    )
    if document_id is None:
        stmt = stmt.where(QaCacheEntry.document_id.is_(None))
    else:
        stmt = stmt.where(QaCacheEntry.document_id == document_id)
    return session.scalar(stmt)


def save_qa_cache_entry(
    session: Session,
    *,
    question_raw: str,
    question_normalized: str,
    question_hash: str,
    answer: str | None,
    citations: list | None = None,
    needs_clarification: bool = False,
    question_embedding: list | None = None,
    document_id: int | None = None,
    content_hash: str = '',
) -> QaCacheEntry:
    """写入一条问答缓存；精确键已存在时更新内容并重置命中计数"""
    existing = get_qa_cache_entry(session, question_hash, content_hash, document_id)
    if existing is not None:
        existing.question_raw = (question_raw,)
        existing.question_normalized = (question_normalized,)
        existing.answer = (answer,)
        existing.citations = citations or []
        existing.needs_clarification = needs_clarification
        if question_embedding is not None:
            existing.question_embedding = question_embedding
        existing.hit_count = 0
        existing.last_hit_at = datetime.now(timezone.utc)
        return existing
    entry = QaCacheEntry(
        question_raw=question_raw,
        question_normalized=question_normalized,
        question_hash=question_hash,
        question_embedding=question_embedding or [],
        answer=answer,
        needs_clarification=needs_clarification,
        document_id=document_id,
        content_hash=content_hash,
    )
    session.add(entry)
    return entry


def touch_qa_cache_hit(session: Session, entry: QaCacheEntry) -> None:
    """缓存命中：更新计数与最近命中时间"""
    entry.hit_count = (entry.hit_count or 0) + 1
    entry.last_hit_at = datetime.now(timezone.utc)


def prune_qa_cache(session: Session, max_entries: int) -> None:
    """超出容量上限时，按照last_hit_at 淘汰最久未使用的缓存条目"""
    if max_entries <= 0:
        return
    total = session.scalar(select(func.count()).select_from(QaCacheEntry))
    if total <= max_entries:
        return
    excess = total - max_entries
    stale = list(
        session.scalars(select(QaCacheEntry).order_by(QaCacheEntry.last_hit_at).limit(excess))
    )
    for entry in stale:
        session.delete(entry)
