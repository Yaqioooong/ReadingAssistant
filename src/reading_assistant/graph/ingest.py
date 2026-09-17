"""入库流水线：解析 → 分块 → 向量化 → 入库（含去重短路）。"""

from datetime import datetime, timezone
from typing import TypedDict

from langchain_core.embeddings import Embeddings
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from sqlalchemy.orm import Session, sessionmaker

from reading_assistant.model.factory import get_embedding_model
from reading_assistant.parsers import Chapter, ParsedBook, parse_book
from reading_assistant.rag import chunk_book, make_chunk_id
from reading_assistant.storage import (
    DocumentNotFoundError,
    DocumentService,
    get_document,
    session_scope,
    update_document_index_status,
)
from reading_assistant.storage.vector_store import StoredChunk, VectorStore
from reading_assistant.utils.logger_handler import get_logger

logger = get_logger('ingest')


class IngestState(TypedDict, total=False):
    """入库图的共享状态。"""

    book_path: str
    filename: str | None
    document_id: int | None
    duplicate: bool
    title: str
    author: str | None
    chapters: list[dict]
    chunk_count: int
    force: bool


def build_ingest_graph(
    session_factory: sessionmaker[Session],
    vector_store: VectorStore,
    embedding_model: Embeddings | None = None,
    checkpointer=None,
):
    """构建入库图。

    ``checkpointer`` 缺省为内存检查点；生产环境可传入
    :func:`reading_assistant.graph.checkpointer.create_postgres_checkpointer`。
    """
    embeddings = embedding_model or get_embedding_model()

    def add_book(state: IngestState) -> dict:
        with session_scope(session_factory) as session:
            result = DocumentService(session).add_book(
                state['book_path'], filename=state.get('filename')
            )
            if result.parsed is None and state.get('force'):
                # reindex：file_hash 命中时 add_book 不重新解析，手动补上章节内容
                result.parsed = parse_book(state['book_path'])
            result.document.index_status = 'indexing'
            # 租约开始时间：崩溃残留的 indexing 记录靠它与启动对账识别
            result.document.index_started_at = datetime.now(timezone.utc)
            logger.info('入库[add_book] doc=%s file=%s dup=%s force=%s',
                        result.document.id, state['book_path'], result.duplicate,
                        bool(state.get('force')))
            return {
                'document_id': result.document.id,
                'duplicate': result.duplicate,
                'title': result.document.title,
                'author': result.document.author,
                'chapters': (
                    [
                        {'title': chapter.title, 'content': chapter.content}
                        for chapter in result.parsed.chapters
                    ]
                    if result.parsed is not None
                    else []
                ),
            }

    def chunk_and_index(state: IngestState) -> dict:
        book = ParsedBook(
            title=state.get('title', ''),
            author=state.get('author'),
            chapters=[
                Chapter(chapter['title'], chapter['content'])
                for chapter in state.get('chapters', [])
            ],
        )
        document_id = state['document_id']
        try:
            with session_scope(session_factory) as session:
                document = get_document(session, document_id)
                if document is None:
                    # 判空：记录不存在时若继续，会在 None 上取 .chunk_count 抛
                    # AttributeError，被外层 except 吞掉并在「不存在的记录」上标 failed。
                    logger.error('入库[chunk_and_index] 文档记录不存在 doc=%s，中止入库',
                                 document_id)
                    raise DocumentNotFoundError(f'文档记录不存在: {document_id}')
                content_hash = document.content_hash
                version_tag = (content_hash or '')[:8]

            chunks = chunk_book(book)
            vectors = embeddings.embed_documents([chunk.text for chunk in chunks])
            # 内容寻址 + 版本化 id：doc{document_id}-{content_hash[:8]}-{index}
            new_ids = [
                make_chunk_id(document_id, content_hash, chunk.index) for chunk in chunks
            ]
            # 顺序契约：先 upsert 新版本 → 再 delete_stale 旧版本。
            # 崩溃在两步之间只会「同内容重复」（无害），绝不出现错误内容、绝不全丢。
            vector_store.add(
                [
                    StoredChunk(
                        id=new_ids[index],
                        text=chunk.text,
                        metadata={
                            'document_id': document_id,
                            'content_hash': content_hash,
                            **chunk.metadata,
                        },
                        embedding=vectors[index],
                    )
                    for index, chunk in enumerate(chunks)
                ]
            )
            removed = vector_store.delete_stale(
                document_id, content_hash, keep_ids=new_ids
            )
            with session_scope(session_factory) as session:
                document = get_document(session, document_id)
                if document is None:
                    logger.error('入库[chunk_and_index] 文档记录在写入后消失 doc=%s',
                                 document_id)
                    raise DocumentNotFoundError(f'文档记录不存在: {document_id}')
                document.chunk_count = len(chunks)
                document.index_status = 'indexed'
        except DocumentNotFoundError:
            # 记录本就不存在：不得再对不存在的记录写 failed
            raise
        except Exception:
            logger.exception('入库[chunk_and_index] 失败 doc=%s', document_id)
            with session_scope(session_factory) as session:
                update_document_index_status(session, document_id, 'failed')
            raise
        logger.info('入库[chunk_and_index] doc=%s version=%s chunk=%d stale_removed=%d '
                    'status=indexed', document_id, version_tag, len(chunks), removed)
        return {'chunk_count': len(chunks)}

    def route_after_add(state: IngestState) -> str:
        return END if state.get('duplicate') and not state.get('force') else 'chunk_and_index'

    graph = StateGraph(IngestState)
    graph.add_node('add_book', add_book)
    graph.add_node('chunk_and_index', chunk_and_index)
    graph.add_edge(START, 'add_book')
    graph.add_conditional_edges('add_book', route_after_add)
    graph.add_edge('chunk_and_index', END)
    return graph.compile(checkpointer=checkpointer or InMemorySaver())
