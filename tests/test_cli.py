"""P6 CLI 测试。"""

from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.embeddings import Embeddings
from typer.testing import CliRunner

from reading_assistant.cli import create_cli
from reading_assistant.storage import (
    ChatMessage,
    ChatSession,
    Document,
    create_db_engine,
    create_session_factory,
    init_db,
)
from reading_assistant.storage.vector_store import InMemoryVectorStore, StoredChunk


class FakeEmbeddings(Embeddings):
    """固定向量的假 Embedding。"""

    def __init__(self, vector: tuple[float, float] = (1.0, 0.0)) -> None:
        self._vector = list(vector)

    def embed_query(self, text: str) -> list[float]:
        return list(self._vector)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [list(self._vector) for _ in texts]


class FakeLLM:
    """返回固定内容的假 LLM。"""

    def __init__(self, content: str = '这是基于原文的测试回答。') -> None:
        self._content = content

    def invoke(self, prompt: str):
        return SimpleNamespace(content=self._content)


@pytest.fixture
def env():
    engine = create_db_engine('sqlite:///:memory:')
    init_db(engine)
    return {
        'session_factory': create_session_factory(engine),
        'vector_store': InMemoryVectorStore(),
        'llm': FakeLLM(),
        'embedding_model': FakeEmbeddings([1.0, 0.0]),
        'engine': engine,
    }


def _write_book(tmp_path: Path) -> Path:
    path = tmp_path / 'book.txt'
    path.write_text('第一章 开端\n正文一。\n第二章 发展\n正文二。\n', encoding='utf-8')
    return path


def _cli_app(env):
    return create_cli(
        session_factory=env['session_factory'],
        vector_store=env['vector_store'],
        llm=env['llm'],
        embedding_model=env['embedding_model'],
    )


class TestIngestCommand:
    def test_ingest_success(self, env, tmp_path: Path) -> None:
        runner = CliRunner()

        result = runner.invoke(_cli_app(env), ['ingest', str(_write_book(tmp_path))])

        assert result.exit_code == 0
        assert '入库成功' in result.output
        assert 'document_id=' in result.output
        assert env['vector_store'].count() > 0

    def test_ingest_duplicate(self, env, tmp_path: Path) -> None:
        runner = CliRunner()
        book = _write_book(tmp_path)
        runner.invoke(_cli_app(env), ['ingest', str(book)])

        second = runner.invoke(_cli_app(env), ['ingest', str(book)])

        assert second.exit_code == 0
        assert '重复上传' in second.output

    def test_ingest_missing_file(self, env, tmp_path: Path) -> None:
        runner = CliRunner()

        result = runner.invoke(_cli_app(env), ['ingest', str(tmp_path / 'missing.txt')])

        assert result.exit_code == 1
        assert '解析失败' in result.output


def _seed_indexed_doc(env, document_id: int = 1) -> None:
    """登记一条 index_status=='indexed' 的文档（检索白名单前置）。"""
    with env['session_factory']() as session:
        session.add(
            Document(
                filename='book.txt',
                title='book',
                file_hash=f'h{document_id}',
                content_hash=f'c{document_id}',
                index_status='indexed',
            )
        )
        session.commit()


class TestAskCommand:
    def test_ask_answers_and_creates_session(self, env) -> None:
        _seed_indexed_doc(env)
        env['vector_store'].add(
            [
                StoredChunk(
                    id='doc1-0',
                    text='张三在第一章出场。',
                    metadata={'document_id': 1, 'chapter': '第一章', 'chapter_index': 0},
                    embedding=[1.0, 0.0],
                )
            ]
        )
        runner = CliRunner()

        result = runner.invoke(_cli_app(env), ['ask', '张三是谁'])

        assert result.exit_code == 0
        assert '回答：' in result.output
        assert '引用 [第一章]' in result.output

    def test_ask_no_chunks_creates_hitl_task(self, env) -> None:
        runner = CliRunner()

        result = runner.invoke(_cli_app(env), ['ask', '完全没有检索结果的提问'])

        assert result.exit_code == 0
        assert '信息不足' in result.output
        assert '澄清任务' in result.output

    def test_ask_reuses_session(self, env) -> None:
        session = env['session_factory']()
        chat = ChatSession()
        session.add(chat)
        session.commit()
        session_id = chat.id
        session.close()
        _seed_indexed_doc(env)
        env['vector_store'].add(
            [
                StoredChunk(
                    id='doc1-0',
                    text='张三出场。',
                    metadata={'document_id': 1, 'chapter': '第一章', 'chapter_index': 0},
                    embedding=[1.0, 0.0],
                )
            ]
        )
        runner = CliRunner()

        result = runner.invoke(_cli_app(env), ['ask', '张三是谁', '--session-id', str(session_id)])

        assert result.exit_code == 0
        session = env['session_factory']()
        messages = session.query(ChatMessage).filter_by(session_id=session_id).all()
        assert len(messages) == 2
        session.close()
