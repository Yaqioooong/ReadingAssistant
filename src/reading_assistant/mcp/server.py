"""MCP Server 实现：读书库问答能力对外暴露。

设计要点：
- 工具薄封装，业务逻辑复用入库图/问答图（与 HTTP 路由同一套流水线），
  避免出现“API 一个行为、MCP 另一个行为”的分叉。
- 依赖通过 runtime 的惰性 getter 获取（不依赖 HTTP 层，保持适配器对等），
  测试可用 monkeypatch 整体替换（内存库 + Fake 模型），与 create_app 的注入哲学一致。
- upload_book 接收的是本机文件路径（MCP 运行在宿主机上）；文件会复制到
  应用 uploads 目录后再走标准入库链路，保证后续 reindex 有持久源文件。

用法：
    uv run readingassistant-mcp            # stdio 传输（默认）
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path

from fastmcp import FastMCP

from reading_assistant.graph import build_ingest_graph, build_qa_graph
from reading_assistant.parsers import get_parser
from reading_assistant.runtime import (
    get_embedding_model,
    get_llm,
    get_session_factory,
    get_upload_dir,
    get_vector_store,
)
from reading_assistant.storage import (
    get_document,
)
from reading_assistant.storage.repositories import list_documents
from reading_assistant.utils.logger_handler import get_logger

logger = get_logger('mcp')

mcp = FastMCP('reading-assistant')

_EXTENSIONS = ('.txt', '.epub', '.pdf', '.docx')


def _document_out(document) -> dict:
    return {
        'id': document.id,
        'filename': document.filename,
        'title': document.title,
        'author': document.author,
        'chunk_count': document.chunk_count,
        'index_status': document.index_status,
        'created_at': (
            document.created_at.isoformat() if document.created_at else None
        ),
    }


@mcp.tool()
def list_books() -> list[dict]:
    """列出书库中已入库的全部书籍（元数据与索引状态）。"""
    session_factory = get_session_factory()
    with session_factory() as session:
        return [_document_out(doc) for doc in list_documents(session)]


@mcp.tool()
def upload_book(book_path: str) -> dict:
    """上传并入库一本电子书（txt/epub/pdf/docx）。

    Args:
        book_path: 本机书籍文件的绝对路径。重复上传同一本书会返回已有记录。
    """
    source = Path(book_path)
    if not source.is_file():
        raise ValueError(f'文件不存在: {book_path}')
    try:
        get_parser(source.name)
    except Exception as exc:  # noqa: BLE001 —— 统一转成可读的 ValueError
        raise ValueError(f'不支持的书籍格式: {source.name}') from exc

    upload_dir = get_upload_dir()
    upload_dir.mkdir(parents=True, exist_ok=True)
    target = upload_dir / f'{uuid.uuid4().hex}_{source.name}'
    target.write_bytes(source.read_bytes())
    logger.info('MCP[upload] %s -> %s', source.name, target)

    session_factory = get_session_factory()
    graph = build_ingest_graph(
        session_factory,
        get_vector_store(),
        embedding_model=get_embedding_model(),
    )
    result = graph.invoke(
        {'book_path': str(target), 'filename': source.name},
        config={'configurable': {'thread_id': f'mcp-ingest-{uuid.uuid4().hex}'}},
    )
    with session_factory() as session:
        document = get_document(session, result['document_id'])
        return {**_document_out(document), 'duplicate': result['duplicate']}


@mcp.tool()
def ask_book(
    question: str,
    document_ids: list[int] | None = None,
    clarification: str | None = None,
) -> dict:
    """向书库提问（RAG 问答，带引用）。

    Args:
        question: 用户问题。
        document_ids: 可选。限定检索范围：空/缺省 = 全部书籍；
            传多个 = 跨书对比（并行检索合流）；传一个 = 单书问答。
        clarification: 可选。HITL 澄清补充说明（配合 needs_clarification）。
    """
    session_factory = get_session_factory()
    doc_ids = document_ids or []
    graph = build_qa_graph(
        session_factory,
        get_vector_store(),
        llm=get_llm(),
        embedding_model=get_embedding_model(),
    )
    state: dict = {
        'question': question,
        'session_id': None,
        'document_id': doc_ids[0] if len(doc_ids) == 1 else None,
        'clarification': clarification,
    }
    if len(doc_ids) > 1:
        state['document_ids'] = doc_ids
    start = time.perf_counter()
    result = graph.invoke(
        state,
        config={'configurable': {'thread_id': f'mcp-qa-{uuid.uuid4().hex}'}},
    )
    cost_ms = (time.perf_counter() - start) * 1000
    logger.info('MCP[ask] q=%.30s docs=%s (%.0fms)', question, doc_ids, cost_ms)
    return {
        'answer': result.get('answer'),
        'citations': result.get('citations') or [],
        'needs_clarification': result.get('needs_clarification', False),
        'hitl_task_id': result.get('hitl_task_id'),
    }


@mcp.tool()
def reindex_book(document_id: int) -> dict:
    """重建指定书籍的向量索引（重新解析 + 分块 + 向量化）。

    Args:
        document_id: 书库中的书籍 id（见 list_books）。
    """
    session_factory = get_session_factory()
    with session_factory() as session:
        document = get_document(session, document_id)
        if document is None:
            raise ValueError(f'文档不存在: document_id={document_id}')
        if not document.file_path or not Path(document.file_path).exists():
            raise ValueError(f'源文件缺失，无法重建: {document.file_path}')

    graph = build_ingest_graph(
        session_factory,
        get_vector_store(),
        embedding_model=get_embedding_model(),
    )
    result = graph.invoke(
        {'book_path': document.file_path, 'force': True},
        config={'configurable': {'thread_id': f'mcp-reindex-{document_id}-{uuid.uuid4().hex}'}},
    )
    with session_factory() as session:
        refreshed = get_document(session, document_id)
        return {**_document_out(refreshed), 'duplicate': result['duplicate']}


def main() -> None:
    """启动 MCP Server（stdio）。"""
    logger.info('MCP Server 启动 transport=stdio')
    mcp.run()


if __name__ == '__main__':
    main()
