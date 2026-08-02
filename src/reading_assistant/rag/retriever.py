"""检索器：Embedding + 向量库 top-k 检索。"""

from dataclasses import dataclass

from langchain_core.embeddings import Embeddings

from reading_assistant.config import get_settings
from reading_assistant.model.factory import get_embedding_model
from reading_assistant.storage.vector_store import VectorStore


@dataclass
class RetrievedChunk:
    """带引用信息的检索片段。"""

    chunk_id: str
    score: float
    text: str
    document_id: int | None = None
    chapter: str | None = None
    chapter_index: int | None = None

    @property
    def citation(self) -> str:
        """格式化为引用文本，如「文档 3｜第一章 开端」。"""
        parts = [f'文档 {self.document_id}'] if self.document_id is not None else []
        if self.chapter:
            parts.append(self.chapter)
        return '｜'.join(parts)


class Retriever:
    """基于向量相似度的检索器。"""

    def __init__(self, vector_store: VectorStore, embedding_model: Embeddings | None = None):
        self._vector_store = vector_store
        self._embedding_model = embedding_model or get_embedding_model()

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        document_id: int | None = None,
    ) -> list[RetrievedChunk]:
        """检索与 query 最相关的 top-k 片段。"""
        settings = get_settings()
        k = top_k or settings.top_k
        embedding = self._embedding_model.embed_query(query)
        where = {'document_id': document_id} if document_id is not None else None
        hits = self._vector_store.query(embedding, top_k=k, where=where)
        return [
            RetrievedChunk(
                chunk_id=hit.id,
                score=hit.score,
                text=hit.text,
                document_id=hit.metadata.get('document_id'),
                chapter=hit.metadata.get('chapter'),
                chapter_index=hit.metadata.get('chapter_index'),
            )
            for hit in hits
        ]
