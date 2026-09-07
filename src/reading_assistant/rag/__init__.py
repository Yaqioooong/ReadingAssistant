"""RAG 层：文本分块与向量检索。"""

from reading_assistant.rag.chunking import TextChunk, chunk_book, chunk_text
from reading_assistant.rag.retriever import (
    HybridRetriever,
    RetrievedChunk,
    Retriever,
    create_retriever,
    fuse_rrf,
)

__all__ = [
    'HybridRetriever',
    'RetrievedChunk',
    'Retriever',
    'TextChunk',
    'chunk_book',
    'chunk_text',
    'create_retriever',
    'fuse_rrf',
]
