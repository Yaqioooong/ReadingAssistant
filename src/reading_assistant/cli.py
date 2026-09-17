"""命令行入口：``python -m reading_assistant.cli ingest <book>`` / ``ask <问题>``。"""

from functools import lru_cache
from pathlib import Path
from uuid import uuid4

import typer

from reading_assistant.graph import (
    build_ingest_graph,
    build_qa_graph,
    interrupt_task_id,
)
from reading_assistant.model.factory import get_chat_model, get_embedding_model
from reading_assistant.parsers import ParseError
from reading_assistant.runtime import get_checkpointer
from reading_assistant.storage import (
    ChatSession,
    create_db_engine,
    create_session_factory,
    session_scope,
)
from reading_assistant.storage.vector_store import create_vector_store


@lru_cache
def _default_session_factory():
    return create_session_factory(create_db_engine())


def create_cli(session_factory=None, vector_store=None, llm=None, embedding_model=None):
    """创建 CLI 应用；测试可注入 mock 依赖。"""
    app = typer.Typer(help='ReadingAssistant 命令行工具', no_args_is_help=True)

    @app.command('ingest')
    def ingest(book: Path = typer.Argument(..., help='电子书路径（txt/epub/pdf/docx）')):
        """解析并入库一本电子书。"""
        try:
            graph = build_ingest_graph(
                session_factory or _default_session_factory(),
                vector_store or create_vector_store(),
                embedding_model=embedding_model or get_embedding_model(),
            )
            result = graph.invoke(
                {'book_path': str(book)},
                config={'configurable': {'thread_id': f'ingest-cli-{uuid4().hex}'}},
            )
        except ParseError as exc:
            typer.echo(f'解析失败：{exc}', err=True)
            raise typer.Exit(code=1) from exc

        if result['duplicate']:
            typer.echo(f'重复上传，已复用文档 #{result["document_id"]}')
        else:
            typer.echo(
                f'入库成功：document_id={result["document_id"]} '
                f'分块数={result.get("chunk_count", 0)}'
            )

    @app.command('ask')
    def ask(
        question: str = typer.Argument(..., help='自然语言问题'),
        document_id: int | None = typer.Option(None, '--document-id', help='限定检索的文档 id'),
        session_id: int | None = typer.Option(None, '--session-id', help='复用已有会话'),
    ):
        """提问并返回回答与引用。"""
        session_factory_local = session_factory or _default_session_factory()
        graph = build_qa_graph(
            session_factory_local,
            vector_store or create_vector_store(),
            llm=llm or get_chat_model(),
            embedding_model=embedding_model or get_embedding_model(),
            checkpointer=get_checkpointer(),
        )
        current_session_id = session_id
        if current_session_id is None:
            with session_scope(session_factory_local) as session:
                chat = ChatSession()
                session.add(chat)
                session.flush()
                current_session_id = chat.id

        result = graph.invoke(
            {
                'question': question,
                'session_id': current_session_id,
                'document_id': document_id,
            },
            config={'configurable': {'thread_id': f'qa-cli-{uuid4().hex}'}},
        )
        task_id = interrupt_task_id(result)
        if task_id is not None:
            typer.echo(
                f'信息不足，已创建澄清任务 #{task_id}（awaiting），'
                '请补充细节后重新提问。'
            )
            return
        if result.get('needs_clarification'):
            typer.echo('信息不足，已转入澄清流程（未能定位任务 id）。')
            return
        typer.echo(f'回答：{result.get("answer")}')
        for citation in result.get('citations') or []:
            chapter = citation.get('chapter') or '未知章节'
            typer.echo(f'  引用 [{chapter}]: {citation.get("excerpt")}')

    return app


app = create_cli()


if __name__ == '__main__':
    app()
