"""RAG 层：文本分块与向量检索。"""

from reading_assistant.rag.chunking import TextChunk, chunk_book, chunk_text
from reading_assistant.rag.retriever import RetrievedChunk, Retriever

__all__ = ['RetrievedChunk', 'Retriever', 'TextChunk', 'chunk_book', 'chunk_text']
