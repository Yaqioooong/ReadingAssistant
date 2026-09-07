"""检索器：Embedding + 向量库 top-k 检索。"""

import math
import threading
from collections import OrderedDict
from dataclasses import dataclass

from langchain_core.embeddings import Embeddings

from reading_assistant.config import get_settings
from reading_assistant.model.factory import get_embedding_model
from reading_assistant.storage import normalize_question, sha256_hex
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
        settings = get_settings()
        self._embed_cache: OrderedDict[str, list[float]] = OrderedDict()
        self._embed_cache_enabled = settings.cache_enabled
        self._embed_cache_max = settings.cache_max_entries
        self._embed_lock = threading.Lock()  # 多文档 fan-out 并发读取保护

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        document_id: int | None = None,
    ) -> list[RetrievedChunk]:
        """检索与 query 最相关的 top-k 片段。"""
        settings = get_settings()
        k = top_k or settings.top_k
        embedding = self._embed(query)
        where = {'document_id': document_id} if document_id is not None else None
        hits = self._vector_store.query(embedding, top_k=k, where=where)
        min_score = settings.retrieval_min_score
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
            if hit.score >= min_score
        ]

    def embed(self, query: str) -> list[float]:
        """公开的embedding接口，带L1缓存，供问答图语义缓存使用"""
        return self._embed(query)

    def _embed(self, query: str) -> list[float]:
        """L1 embedding缓存：归一化query命中则复用向量，跳过付费API
        对归一化文本做embedding, 保证缓存键与向量式中一致
        """

        if not self._embed_cache_enabled:
            return self._embedding_model.embed_query(query)
        key = sha256_hex(normalize_question(query))
        with self._embed_lock:
            if key in self._embed_cache:
                self._embed_cache.move_to_end(key)  # LRU语义
                return self._embed_cache[key]
        # 模型调用放在锁外，避免 fan-out 线程被串行化
        vector = self._embedding_model.embed_query(normalize_question(query))
        with self._embed_lock:
            self._embed_cache[key] = vector
            if len(self._embed_cache) > self._embed_cache_max:
                self._embed_cache.popitem(last=False)  # 淘汰最久未使用的key
        return vector


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """余弦相似度；任一向量缺失/零向量时返回 0。"""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


def fuse_rrf(rankings: list[list[str]], k: int = 60) -> list[str]:
    """RRF 融合：对多路有序 id 列表按 Σ 1/(k+rank) 打分后降序。

    纯函数、与具体检索实现解耦，便于单测。
    """
    scores: dict[str, float] = {}
    for ranked in rankings:
        for rank, chunk_id in enumerate(ranked, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=scores.get, reverse=True)


class HybridRetriever(Retriever):
    """稠密向量 + BM25 双路召回，RRF 融合后按稠密质量分把关。

    - 稠密路：向量库 top ``pool_size``（不过 min_score，交由最终闸门）
    - 稀疏路：BM25 索引 top ``pool_size``（进程内、随向量库版本失效重建）
    - 融合：RRF(k=rrf_k) 取 top_k；最终结果仍要求稠密余弦 ≥ min_score
      （不在稠密候选池内的结果用存储向量现算余弦，避免稀疏路引入低质噪音）
    """

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_model: Embeddings | None = None,
        pool_size: int | None = None,
        rrf_k: int | None = None,
        bm25_tokenizer: str | None = None,
    ) -> None:
        super().__init__(vector_store, embedding_model)
        settings = get_settings()
        self._pool_size = pool_size or settings.hybrid_pool_size
        self._rrf_k = rrf_k or settings.rrf_k
        from reading_assistant.rag.bm25_index import BM25Index

        self._bm25 = BM25Index.get_for(
            vector_store, tokenizer=bm25_tokenizer or settings.bm25_tokenizer
        )

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        document_id: int | None = None,
    ) -> list[RetrievedChunk]:
        settings = get_settings()
        k = top_k or settings.top_k
        min_score = settings.retrieval_min_score
        embedding = self._embed(query)
        where = {'document_id': document_id} if document_id is not None else None

        dense = self._vector_store.query(embedding, top_k=self._pool_size, where=where)
        dense_ids = [hit.id for hit in dense]
        dense_by_id = {hit.id: hit for hit in dense}

        bm25_ids = [
            chunk_id for chunk_id, _score in self._bm25.query(
                query, top_k=self._pool_size, document_id=document_id
            )
        ]

        fused = fuse_rrf([dense_ids, bm25_ids], self._rrf_k)[:k]
        results: list[RetrievedChunk] = []
        for chunk_id in fused:
            hit = dense_by_id.get(chunk_id)
            if hit is not None:
                score = hit.score
            else:
                # 稀疏路捞到但稠密 top-pool 之外：用存储向量现算余弦做质量闸门
                info = self._bm25.chunk_info(chunk_id)
                if info is None or not info.embedding:
                    continue
                score = _cosine_similarity(embedding, info.embedding)
            if score < min_score:
                continue
            meta = hit.metadata if hit is not None else {}
            text = hit.text if hit is not None else ''
            if hit is None:
                info = self._bm25.chunk_info(chunk_id)
                if info is not None:
                    meta = info.metadata
                    text = info.text
            results.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    score=score,
                    text=text,
                    document_id=meta.get('document_id'),
                    chapter=meta.get('chapter'),
                    chapter_index=meta.get('chapter_index'),
                )
            )
        return results


def create_retriever(
    vector_store: VectorStore,
    embedding_model: Embeddings | None = None,
) -> Retriever:
    """按配置创建检索器：'hybrid' 走混合检索，其余保持纯稠密（默认）。"""
    mode = get_settings().retrieval_mode
    if mode == 'hybrid':
        return HybridRetriever(vector_store, embedding_model)
    return Retriever(vector_store, embedding_model)
