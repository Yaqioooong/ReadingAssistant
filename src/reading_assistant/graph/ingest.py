"""入库流水线：解析 → 分块 → 向量化 → 入库（含去重短路）。"""

from typing import TypedDict

from langchain_core.embeddings import Embeddings
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from sqlalchemy.orm import Session, sessionmaker

from reading_assistant.model.factory import get_embedding_model
from reading_assistant.parsers import Chapter, ParsedBook
from reading_assistant.rag import chunk_book
from reading_assistant.storage import (
    DocumentService,
    get_document,
    session_scope,
)
from reading_assistant.storage.vector_store import StoredChunk, VectorStore


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
        chunks = chunk_book(book)
        vectors = embeddings.embed_documents([chunk.text for chunk in chunks])
        document_id = state['document_id']
        vector_store.add(
            [
                StoredChunk(
                    id=f'doc{document_id}-{chunk.index}',
                    text=chunk.text,
                    metadata={'document_id': document_id, **chunk.metadata},
                    embedding=vectors[index],
                )
                for index, chunk in enumerate(chunks)
            ]
        )
        with session_scope(session_factory) as session:
            document = get_document(session, document_id)
            document.chunk_count = len(chunks)
        return {'chunk_count': len(chunks)}

    def route_after_add(state: IngestState) -> str:
        return END if state.get('duplicate') else 'chunk_and_index'

    graph = StateGraph(IngestState)
    graph.add_node('add_book', add_book)
    graph.add_node('chunk_and_index', chunk_and_index)
    graph.add_edge(START, 'add_book')
    graph.add_conditional_edges('add_book', route_after_add)
    graph.add_edge('chunk_and_index', END)
    return graph.compile(checkpointer=checkpointer or InMemorySaver())
