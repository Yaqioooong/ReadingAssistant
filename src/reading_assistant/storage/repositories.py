"""文档 repository：入库、查询与哈希查重。"""

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from reading_assistant.storage.models import Document


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
