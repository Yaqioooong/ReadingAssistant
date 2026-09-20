"""检索器：Embedding + 向量库 top-k 检索。"""

import math
import re
import threading
from collections import OrderedDict
from dataclasses import dataclass

from langchain_core.embeddings import Embeddings

from reading_assistant.config import get_settings
from reading_assistant.model.factory import get_embedding_model
from reading_assistant.storage import normalize_question, sha256_hex
from reading_assistant.storage.vector_store import VectorStore
from reading_assistant.utils.logger_handler import get_logger

logger = get_logger('retriever')


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
        """检索与 query 最相关的 top-k 片段。

        ``min_score`` 是**余弦相似度**阈值：向量库返回的 ``hit.score`` 已由
        :class:`~reading_assistant.storage.vector_store.VectorStore` 统一换算为余弦。
        """
        settings = get_settings()
        k = top_k or settings.top_k
        embedding = self._embed(query)
        where = {'document_id': document_id} if document_id is not None else None
        hits = self._vector_store.query(embedding, top_k=k, where=where)
        min_score = settings.retrieval_min_score
        results = [
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
        if not results:
            # 精确标识符兜底：P002/P-002 这类编号查询在语义通道全灭时，
            # 用字面等价 token 把所在 chunk 捞回来（编号命中是强信号，不受语义阈值拦截）
            results = self._identifier_fallback(query, document_id, k, embedding, min_score)
        return results

    def embed(self, query: str) -> list[float]:
        """公开的embedding接口，带L1缓存，供问答图语义缓存使用"""
        return self._embed(query)

    def _identifier_fallback(
        self,
        query: str,
        document_id: int | None,
        top_k: int,
        embedding: list[float],
        min_score: float,
    ) -> list[RetrievedChunk]:
        """字面标识符兜底：语义无果时按规范化标识符匹配 chunk 文本。"""
        wanted = _identifier_tokens(query)
        if not wanted:
            return []
        try:
            chunks = self._vector_store.all_chunks()
        except Exception:  # noqa: BLE001 —— 向量库不可用时放弃兜底
            return []
        matched: list[RetrievedChunk] = []
        for chunk in chunks:
            if document_id is not None and chunk.metadata.get('document_id') != document_id:
                continue
            tokens = _chunk_ascii_tokens(chunk.text)
            if not (tokens & wanted):
                continue
            cosine = (
                _cosine_similarity(embedding, chunk.embedding) if chunk.embedding else 0.0
            )
            score = max(cosine, min_score)  # 兜底命中按质量下限进入候选，供 judge 决定
            matched.append(
                RetrievedChunk(
                    chunk_id=chunk.id,
                    score=score,
                    text=chunk.text,
                    document_id=chunk.metadata.get('document_id'),
                    chapter=chunk.metadata.get('chapter'),
                    chapter_index=chunk.metadata.get('chapter_index'),
                )
            )
        matched.sort(key=lambda item: item.score, reverse=True)
        return matched[:top_k]

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
        vector = self._embedding_model.embed_query(normalize_question(query))
        with self._embed_lock:
            self._embed_cache[key] = vector
            if len(self._embed_cache) > self._embed_cache_max:
                self._embed_cache.popitem(last=False)  # 淘汰最久未使用的key
        return vector


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """余弦相似度；任一向量缺失/零向量时返回 0。

    ⚠️ 返回值必须显式收敛为内建 ``float``：入参可能来自 Chroma（numpy 标量序列），
    ``sum(x * y ...)`` 会把 numpy 类型透传出去，进而让 QA state 无法被
    LangGraph checkpoint 用 msgpack 序列化（TypeError → 接口 500）。
    """
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return float(dot / (norm_a * norm_b)) if norm_a and norm_b else 0.0


def fuse_rrf(rankings: list[list[str]], k: int = 60) -> list[str]:
    """RRF 融合：对多路有序 id 列表按 Σ 1/(k+rank) 打分后降序。

    纯函数、与具体检索实现解耦，便于单测。

    **隐式不变量**：``fuse_rrf`` 与候选池大小 ``pool``（``hybrid_pool_size``）
    通过 ``rrf_k`` 强耦合。设两路各召回 ``pool`` 条、每路内 id 互不相同，
    则对某一 id 而言：

    - **双路命中**（两路都进池）：最低分出现在 rank=pool 且同 id 恰好是两路
      最后一名时，为 ``2/(k+pool)``。
    - **单路命中**（只在一路进池）：最高分是 rank=1 时的 ``1/(k+1)``。

    故「双路必然碾压单路」⟺ ``2/(k+pool) > 1/(k+1)`` ⟺ ``k > pool - 2``。

    - ``k > pool - 2``（默认 60 > 48）：RRF 正常语义——**两路共现的片段优先**，
      单路强命中排在双路弱命中之后。
    - ``k <= pool - 2``：语义**静默翻转**——单路 rank1 的分数反而 ≥ 双路最低分，
      融合退化为「谁先谁上」，双路一致性不再被奖励。

    例如把 ``hybrid_pool_size`` 调到 70（阈值 68），保持 ``k=60`` 时
    ``60 < 68`` 即触发翻转。创建 :class:`HybridRetriever` 时会对此校验告警。
    """
    scores: dict[str, float] = {}
    for ranked in rankings:
        for rank, chunk_id in enumerate(ranked, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=scores.get, reverse=True)


class HybridRetriever(Retriever):
    """稠密向量 + BM25 双路召回，RRF 融合后按**余弦**质量分把关。

    - 稠密路：向量库 top ``pool_size``（不过 min_score，交由最终闸门）
    - 稀疏路：BM25 索引 top ``pool_size``（进程内、随向量库版本失效重建）
    - 融合：RRF(k=rrf_k) 取 top_k；最终结果仍要求**余弦相似度** ≥ min_score
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
        self._check_rrf_invariant()
        from reading_assistant.rag.bm25_index import BM25Index

        self._bm25 = BM25Index.get_for(
            vector_store, tokenizer=bm25_tokenizer or settings.bm25_tokenizer
        )

    def _check_rrf_invariant(self) -> None:
        """校验 RRF 不变量 ``rrf_k > pool_size - 2``，不满足则明确告警。

        不满足时融合语义会**静默翻转**：单路 rank1 命中不再必然输给双路共现，
        「双路一致性」这一 RRF 的核心收益失效（详见 :func:`fuse_rrf`）。
        """
        if self._rrf_k <= self._pool_size - 2:
            logger.warning(
                '混合检索[配置] RRF 不变量被破坏：rrf_k=%d <= pool_size-2=%d。'
                '后果：单路强命中可能压过双路共现，RRF「双路优先」语义静默翻转；'
                '请调大 rrf_k 或调小 hybrid_pool_size。',
                self._rrf_k,
                self._pool_size - 2,
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
        if not results:
            results = self._identifier_fallback(
                query, document_id, k, embedding, min_score
            )
        return results



_ASCII_RUN = re.compile(r'[A-Za-z0-9]+(?:[ \t\-_][A-Za-z0-9]+)*')


def _canonical_ascii(token: str) -> str:
    """ASCII 标识符规范化：去分隔符/折叠大小写（P-002/P002/p 002 → p002）。"""
    return ''.join(ch for ch in (token or '').lower() if ch.isalnum())


def _identifier_tokens(text: str) -> set[str]:
    """提取查询中的“字母+数字混合”标识符 token（产品编号/型号类）。"""
    found: set[str] = set()
    for raw in _ASCII_RUN.findall(text or ''):
        canon = _canonical_ascii(raw)
        if (
            len(canon) >= 3
            and any(ch.isdigit() for ch in canon)
            and any(ch.isalpha() for ch in canon)
        ):
            found.add(canon)
    return found


def _chunk_ascii_tokens(text: str) -> set[str]:
    """分块文本的规范化 ASCII token 集合（供字面标识符命中判定）。"""
    return {_canonical_ascii(raw) for raw in _ASCII_RUN.findall(text or '')}


def create_retriever(
    vector_store: VectorStore,
    embedding_model: Embeddings | None = None,
) -> Retriever:
    """按配置创建检索器：'hybrid' 走混合检索，其余保持纯稠密（默认）。"""
    mode = get_settings().retrieval_mode
    if mode == 'hybrid':
        return HybridRetriever(vector_store, embedding_model)
    return Retriever(vector_store, embedding_model)
