"""MCP Server 工具测试：内存库 + Fake 模型，验证工具链路。"""

from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.embeddings import Embeddings

import reading_assistant.mcp.server as server
from reading_assistant.storage import (
    create_db_engine,
    create_session_factory,
    init_db,
)
from reading_assistant.storage.vector_store import InMemoryVectorStore


class FakeEmbeddings(Embeddings):
    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


class FakeLLM:
    def invoke(self, prompt):
        return SimpleNamespace(content='根据原文可以回答，李四喜欢王五。')


def _make_book(name: str, char: str = '李四喜欢王五。') -> Path:
    path = Path(__file__).parent / '_mcp_book_tmp' / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text((char * 60) + '\n', encoding='utf-8')
    return path


@pytest.fixture
def env(tmp_path: Path, monkeypatch) -> InMemoryVectorStore:
    engine = create_db_engine('sqlite:///:memory:')
    init_db(engine)
    factory = create_session_factory(engine)
    store = InMemoryVectorStore()
    upload_dir = tmp_path / 'uploads'
    upload_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(server, 'get_session_factory', lambda: factory)
    monkeypatch.setattr(server, 'get_vector_store', lambda: store)
    monkeypatch.setattr(server, 'get_llm', lambda: FakeLLM())
    monkeypatch.setattr(server, 'get_embedding_model', lambda: FakeEmbeddings())
    monkeypatch.setattr(server, 'get_upload_dir', lambda: upload_dir)
    return store


class TestMCP:
    def test_list_books_empty(self, env) -> None:
        assert server.list_books() == []

    def test_upload_then_list_and_ask_single(self, env, tmp_path: Path) -> None:
        book = _make_book('甲.txt')
        result = server.upload_book(str(book))
        assert result['id'] > 0
        assert result['chunk_count'] >= 1
        assert result['index_status'] == 'indexed'
        assert result['duplicate'] is False

        books = server.list_books()
        assert len(books) == 1 and books[0]['filename'] == '甲.txt'

        ask = server.ask_book('谁喜欢王五？', document_ids=[result['id']])
        assert ask['answer']
        assert ask['citations'], '单书问答应有引用'
        assert ask['needs_clarification'] is False

    def test_upload_duplicate_returns_existing(self, env, tmp_path: Path) -> None:
        book = _make_book('乙.txt')
        first = server.upload_book(str(book))
        second = server.upload_book(str(book))
        assert second['id'] == first['id']
        assert second['duplicate'] is True

    def test_multi_doc_ask_covers_both(self, env, tmp_path: Path) -> None:
        book_a = server.upload_book(str(_make_book('多A.txt', '甲喜欢乙。')))
        book_b = server.upload_book(str(_make_book('多B.txt', '丙在一九四九年建国。')))
        ask = server.ask_book('两本书分别说了什么？', document_ids=[book_a['id'], book_b['id']])
        assert ask['answer']
        cited_docs = {
            c.get('document')
            for c in ask['citations']
            if c.get('document') is not None
        }
        assert '多A.txt' in cited_docs and '多B.txt' in cited_docs

    def test_reindex_book(self, env, tmp_path: Path) -> None:
        book = server.upload_book(str(_make_book('丙.txt')))
        refreshed = server.reindex_book(book['id'])
        assert refreshed['id'] == book['id']
        assert refreshed['index_status'] == 'indexed'

    def test_upload_missing_path_raises(self, env) -> None:
        with pytest.raises(ValueError, match='文件不存在'):
            server.upload_book('/no/such/file.txt')

    def test_upload_unsupported_format_raises(self, env, tmp_path: Path) -> None:
        bad = tmp_path / 'x.xyz'
        bad.write_text('abc')
        with pytest.raises(ValueError, match='不支持的书籍格式'):
            server.upload_book(str(bad))

    def test_reindex_unknown_id_raises(self, env) -> None:
        with pytest.raises(ValueError, match='文档不存在'):
            server.reindex_book(999)
