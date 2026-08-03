"""书籍入库服务：分层去重 + 解析 + 落库。"""

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy.orm import Session

from reading_assistant.parsers import ParsedBook, ParseError, parse_book, resolve_book_path
from reading_assistant.storage.models import Document
from reading_assistant.storage.repositories import (
    get_document_by_content_hash,
    get_document_by_file_hash,
    insert_document,
)

_WHITESPACE_RE = re.compile(r'\s+')


def normalize_text(text: str) -> str:
    """归一化文本：折叠空白字符，用于内容哈希。"""
    return _WHITESPACE_RE.sub(' ', text).strip()


def sha256_hex(data: bytes | str) -> str:
    """计算 SHA-256 十六进制摘要。"""
    if isinstance(data, str):
        data = data.encode('utf-8')
    return hashlib.sha256(data).hexdigest()


@dataclass
class AddBookResult:
    """入库结果：文档、是否重复，以及（首次入库时的）解析结果。"""

    document: Document
    duplicate: bool
    parsed: ParsedBook | None = None


class DocumentService:
    """电子书入库服务，按「文件哈希 → 内容哈希 → 插入」分层去重。"""

    def __init__(self, session: Session):
        self._session = session

    def add_book(self, path: str | Path, filename: str | None = None) -> AddBookResult:
        path = resolve_book_path(path)
        try:
            raw = path.read_bytes()
        except OSError as exc:
            raise ParseError(f'文件不存在: {path}') from exc

        # 第 1 层：文件字节哈希
        file_hash = sha256_hex(raw)
        existing = get_document_by_file_hash(self._session, file_hash)
        if existing is not None:
            return AddBookResult(document=existing, duplicate=True)

        # 未命中才解析
        parsed = parse_book(path)

        # 第 2 层：归一化内容哈希（同一本书的不同文件）
        content_hash = sha256_hex(normalize_text(parsed.full_text))
        existing = get_document_by_content_hash(self._session, content_hash)
        if existing is not None:
            return AddBookResult(document=existing, duplicate=True, parsed=parsed)

        # 第 3 层：插入；并发冲突由唯一索引兜底
        document, inserted = insert_document(
            self._session,
            filename=filename or path.name,
            title=parsed.title,
            author=parsed.author,
            file_path=str(path),
            file_hash=file_hash,
            content_hash=content_hash,
        )
        return AddBookResult(document=document, duplicate=not inserted, parsed=parsed)
