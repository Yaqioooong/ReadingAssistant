"""向量库适配器：隔离具体实现（ChromaDB / 内存）。"""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from reading_assistant.config import get_settings


@dataclass
class StoredChunk:
    """待入库的分块。"""

    id: str
    text: str
    metadata: dict = field(default_factory=dict)
    embedding: list[float] | None = None


@dataclass
class SearchHit:
    """检索结果。"""

    id: str
    score: float
    metadata: dict = field(default_factory=dict)
    text: str = ''


class VectorStore(ABC):
    """向量库统一接口。"""

    @abstractmethod
    def add(self, chunks: list[StoredChunk]) -> None:
        """批量写入分块。"""

    @abstractmethod
    def query(
        self, embedding: list[float], top_k: int = 6, where: dict | None = None
    ) -> list[SearchHit]:
        """按向量检索 top-k 分块。"""

    @abstractmethod
    def count(self) -> int:
        """返回分块总数。"""

    @abstractmethod
    def delete(self, document_id: int) -> None:
        """删除指定文档的全部向量分块"""

    @abstractmethod
    def all_chunks(self) -> list[StoredChunk]:
        """返回库内全部分块（含文本与元数据），供 BM25 索引构建。"""


class ChromaVectorStore(VectorStore):
    """基于 chromadb 的本地持久化实现。"""

    def __init__(self, persist_dir: str | None = None, collection_name: str | None = None):
        import chromadb

        settings = get_settings()
        self._client = chromadb.PersistentClient(
            path=persist_dir or str(settings.chroma_persist_path)
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name or settings.chroma_collection_name
        )
        self._content_version = 0  # 写入版本号：add/delete 自增，供 BM25 索引失效检测

    def add(self, chunks: list[StoredChunk]) -> None:
        has_embeddings = any(chunk.embedding is not None for chunk in chunks)
        self._collection.upsert(
            ids=[chunk.id for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            metadatas=[chunk.metadata for chunk in chunks],
            embeddings=([chunk.embedding for chunk in chunks] if has_embeddings else None),
        )
        self._content_version += 1

    def query(
        self, embedding: list[float], top_k: int = 6, where: dict | None = None
    ) -> list[SearchHit]:
        result = self._collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            where=where,
        )
        ids = result.get('ids', [[]])[0]
        distances = result.get('distances', [[]])[0]
        metadatas = result.get('metadatas', [[]])[0]
        documents = result.get('documents', [[]])[0]
        return [
            SearchHit(
                id=chunk_id,
                # Chroma 返回的是距离（越小越相关），统一转成 0~1 相似度（越大越相关）
                score=1.0 / (1.0 + distances[index]) if index < len(distances) else 0.0,
                metadata=metadatas[index] or {},
                text=documents[index] or '',
            )
            for index, chunk_id in enumerate(ids)
        ]

    def count(self) -> int:
        return self._collection.count()

    def delete(self, document_id: int) -> None:
        self._collection.delete(where={'document_id': document_id})
        self._content_version += 1

    def all_chunks(self) -> list[StoredChunk]:
        """Chroma 全量拉取（含文本/元数据/向量），供 BM25 索引构建。"""
        result = self._collection.get(include=['documents', 'metadatas', 'embeddings'])
        ids = result.get('ids') or []
        documents = result.get('documents') or []
        metadatas = result.get('metadatas') or []
        embeddings = result.get('embeddings')
        chunks = []
        for index, chunk_id in enumerate(ids):
            embedding = None
            if embeddings is not None and index < len(embeddings) and embeddings[index] is not None:
                embedding = list(embeddings[index])
            chunks.append(
                StoredChunk(
                    id=str(chunk_id),
                    text=str(documents[index] or '') if index < len(documents) else '',
                    metadata=(
                        dict(metadatas[index])
                        if index < len(metadatas) and metadatas[index]
                        else {}
                    ),
                    embedding=embedding,
                )
            )
        return chunks


class InMemoryVectorStore(VectorStore):
    """内存实现：用于测试与本地降级。"""

    def __init__(self) -> None:
        self._chunks: dict[str, StoredChunk] = {}
        self._content_version = 0

    def add(self, chunks: list[StoredChunk]) -> None:
        for chunk in chunks:
            self._chunks[chunk.id] = chunk
        self._content_version += 1

    def query(
        self, embedding: list[float], top_k: int = 6, where: dict | None = None
    ) -> list[SearchHit]:
        scored = []
        for chunk in self._chunks.values():
            if where and not all(chunk.metadata.get(key) == value for key, value in where.items()):
                continue
            score = (
                _cosine_similarity(embedding, chunk.embedding)
                if chunk.embedding is not None
                else 0.0
            )
            scored.append(
                SearchHit(id=chunk.id, score=score, metadata=chunk.metadata, text=chunk.text)
            )
        scored.sort(key=lambda hit: hit.score, reverse=True)
        return scored[:top_k]

    def count(self) -> int:
        return len(self._chunks)

    def delete(self, document_id: int) -> None:
        for chunk_id in list(self._chunks):
            if self._chunks[chunk_id].metadata.get('document_id') == document_id:
                del self._chunks[chunk_id]
        self._content_version += 1

    def all_chunks(self) -> list[StoredChunk]:
        """内存实现：直接返回全部分块。"""
        return list(self._chunks.values())


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


def create_vector_store(backend: str | None = None) -> VectorStore:
    """按配置创建向量库实例；backend 可选 'chroma' 或 'memory'。"""
    backend = (backend or get_settings().vector_store_backend).lower()
    if backend == 'memory':
        return InMemoryVectorStore()
    if backend == 'chroma':
        return ChromaVectorStore()
    raise ValueError(f'未知的向量库后端: {backend}')
