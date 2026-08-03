"""P2 存储层测试：模型、repository 与分层去重服务。"""

from pathlib import Path

import pytest
from sqlalchemy.exc import IntegrityError

from reading_assistant.parsers import ParseError
from reading_assistant.storage import (
    Document,
    DocumentService,
    create_db_engine,
    create_session_factory,
    get_document_by_content_hash,
    get_document_by_file_hash,
    init_db,
    insert_document,
    list_documents,
    normalize_text,
    sha256_hex,
)
from reading_assistant.storage.models import ChatMessage, ChatSession, HitlTask


@pytest.fixture
def session_factory():
    engine = create_db_engine('sqlite:///:memory:')
    init_db(engine)
    yield create_session_factory(engine)
    engine.dispose()


@pytest.fixture
def session(session_factory):
    session = session_factory()
    yield session
    session.close()


def _write_book(path: Path, title: str = '第一章 开端', body: str = '正文内容。') -> Path:
    path.write_text(f'{title}\n{body}\n', encoding='utf-8')
    return path


class TestModels:
    def test_document_unique_file_hash(self, session) -> None:
        session.add(Document(filename='a.txt', title='A', file_hash='h1', content_hash='c1'))
        session.flush()

        with pytest.raises(IntegrityError):
            session.add(Document(filename='b.txt', title='B', file_hash='h1', content_hash='c2'))
            session.flush()
        session.rollback()

    def test_session_message_and_hitl_models(self, session) -> None:
        chat = ChatSession(title='测试会话')
        session.add(chat)
        session.flush()
        session.add(ChatMessage(session_id=chat.id, role='user', content='你好'))
        session.add(
            HitlTask(session_id=chat.id, question='信息不足', status=HitlTask.STATUS_AWAITING)
        )
        session.commit()

        assert len(chat.messages) == 1
        assert chat.messages[0].content == '你好'


class TestNormalizeAndHash:
    def test_normalize_text_collapses_whitespace(self) -> None:
        assert normalize_text('第一段 \n\n 第二段\t内容') == '第一段 第二段 内容'

    def test_sha256_hex_accepts_bytes_and_str(self) -> None:
        assert sha256_hex(b'abc') == sha256_hex('abc')


class TestDocumentService:
    def test_add_book_then_same_file_is_duplicate(self, session, tmp_path: Path) -> None:
        book = _write_book(tmp_path / 'book.txt')

        first = DocumentService(session).add_book(book)
        second = DocumentService(session).add_book(book)

        assert first.duplicate is False
        assert first.document.id is not None
        assert first.parsed is not None
        assert second.duplicate is True
        assert second.document.id == first.document.id
        assert len(list_documents(session)) == 1

    def test_same_content_different_filename_is_duplicate(self, session, tmp_path: Path) -> None:
        book_a = _write_book(tmp_path / 'a.txt')
        book_b = _write_book(tmp_path / 'b.txt')

        first = DocumentService(session).add_book(book_a)
        second = DocumentService(session).add_book(book_b)

        assert second.duplicate is True
        assert second.document.id == first.document.id

    def test_different_content_is_not_duplicate(self, session, tmp_path: Path) -> None:
        book_a = _write_book(tmp_path / 'a.txt', body='内容A')
        book_b = _write_book(tmp_path / 'b.txt', body='内容B')

        first = DocumentService(session).add_book(book_a)
        second = DocumentService(session).add_book(book_b)

        assert first.duplicate is False
        assert second.duplicate is False
        assert first.document.id != second.document.id

    def test_missing_file_raises_parse_error(self, session, tmp_path: Path) -> None:
        with pytest.raises(ParseError, match='不存在'):
            DocumentService(session).add_book(tmp_path / 'missing.txt')

    def test_add_book_relative_to_repo_root_from_other_cwd(
        self, session, tmp_path: Path, monkeypatch
    ) -> None:
        monkeypatch.chdir(tmp_path)

        result = DocumentService(session).add_book('data/books/sample_book.txt')

        assert result.duplicate is False
        assert result.parsed is not None
        assert result.document.title == 'sample_book'

    def test_concurrent_conflict_falls_back_to_existing(self, session, tmp_path: Path) -> None:
        book = _write_book(tmp_path / 'book.txt')
        file_hash = sha256_hex(book.read_bytes())

        # 先插入一条相同 file_hash 的记录，模拟并发竞态下另一请求已入库
        first = DocumentService(session).add_book(book)
        session.commit()
        second = DocumentService(session).add_book(book)

        assert second.duplicate is True
        assert second.document.id == first.document.id

        # 直接调用 insert_document 模拟唯一索引冲突兜底
        document, inserted = insert_document(
            session,
            filename='other.txt',
            title='Other',
            file_hash=file_hash,
            content_hash='unused',
        )
        assert inserted is False
        assert document.id == first.document.id


class TestRepository:
    def test_get_and_list(self, session, tmp_path: Path) -> None:
        book = _write_book(tmp_path / 'repo.txt')
        result = DocumentService(session).add_book(book)
        session.commit()

        assert get_document_by_file_hash(session, sha256_hex(book.read_bytes())).id == (
            result.document.id
        )
        assert (
            get_document_by_content_hash(
                session, sha256_hex(normalize_text(result.parsed.full_text))
            ).id
            == result.document.id
        )
        assert [doc.id for doc in list_documents(session)] == [result.document.id]
