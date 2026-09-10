"""BM25 稀疏检索索引。

基于 rank-bm25，从向量库的全量 chunk 构建进程内索引，用于与稠密向量
检索做 RRF 融合。chunk 文本只存在于向量库中（PG 不落 chunk 文本），
因此索引数据源是 ``VectorStore.all_chunks()``。

索引缓存挂在 vector_store 实例上（跨请求复用，避免每问重建）；
靠 store 的 ``_content_version``（add/delete 自增）与 count 指纹做失效检测，
新增/删除文档后下次查询自动重建。分块内容变更但数量不变的重建
（reindex force 覆盖同 id）不在此自动范围——由显式调用 refresh() 兜底。
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from reading_assistant.storage.vector_store import StoredChunk, VectorStore

# 挂载到 vector_store 实例上的缓存属性名
_CACHE_ATTR = '_bm25_index'


_ASCII_RUN = re.compile(r'[A-Za-z0-9]+(?:[ \t\-_][A-Za-z0-9]+)*')


def _canonical_ascii(token: str) -> str:
    """ASCII 标识符规范化：P-002 / P002 / p 002 → p002（与检索层兜底一致）。"""
    return ''.join(ch for ch in (token or '').lower() if ch.isalnum())


def _default_tokenizer() -> callable:
    """延迟导入 jieba（首用加载词典较慢，避免拖慢进程启动）。

    返回 jieba token + ASCII 标识符规范化 token（P-002/P002 词面等价）。
    """

    def tokenize(text: str) -> list[str]:
        import jieba

        tokens = [token for token in jieba.cut(text or '') if token and token.strip()]
        for raw in _ASCII_RUN.findall(text or ''):
            canon = _canonical_ascii(raw)
            if len(canon) >= 3 and canon not in tokens:
                tokens.append(canon)
        return tokens

    return tokenize


@dataclass
class _IndexData:
    """一次构建的不可变快照。"""

    bm25: BM25Okapi
    chunk_ids: list[str]
    # chunk_id -> StoredChunk（含文本/元数据/向量），供融合后取详情
    chunks_by_id: dict[str, StoredChunk]
    count: int
    version: int


class BM25Index:
    """进程内 BM25 索引（jieba 分词），随向量库内容增量失效重建。"""

    def __init__(self, vector_store: VectorStore, tokenizer: str = 'jieba') -> None:
        self._vector_store = vector_store
        self._tokenizer = tokenize if (tokenize := _default_tokenizer()) else None
        self._data: _IndexData | None = None
        self._lock = threading.Lock()  # 懒重建互斥

    @classmethod
    def get_for(cls, vector_store: VectorStore, tokenizer: str = 'jieba') -> 'BM25Index':
        """获取挂载在 vector_store 上的共享索引（跨请求复用）。"""
        existing = getattr(vector_store, _CACHE_ATTR, None)
        if existing is None:
            existing = cls(vector_store, tokenizer=tokenizer)
            setattr(vector_store, _CACHE_ATTR, existing)
        return existing

    # -- 内部 -----------------------------------------------------------------

    def _signature(self) -> tuple[int, int]:
        version = int(getattr(self._vector_store, '_content_version', 0))
        return version, self._vector_store.count()

    def _rebuild(self) -> None:
        chunks = self._vector_store.all_chunks()
        chunk_ids: list[str] = []
        tokenized_docs: list[list[str]] = []
        chunks_by_id: dict[str, StoredChunk] = {}
        for chunk in chunks:
            chunk_ids.append(chunk.id)
            tokenized_docs.append(self._tokenize(chunk.text or ''))
            chunks_by_id[chunk.id] = chunk
        bm25 = BM25Okapi(tokenized_docs) if tokenized_docs else None
        version = int(getattr(self._vector_store, '_content_version', 0))
        self._data = _IndexData(
            bm25=bm25,
            chunk_ids=chunk_ids,
            chunks_by_id=chunks_by_id,
            count=len(chunk_ids),
            version=version,
        )

    def _ensure_fresh(self) -> None:
        version, count = self._signature()
        if (
            self._data is not None
            and self._data.version == version
            and self._data.count == count
        ):
            return
        with self._lock:
            version, count = self._signature()
            if (
                self._data is not None
                and self._data.version == version
                and self._data.count == count
            ):
                return
            self._rebuild()

    def refresh(self) -> None:
        """强制重建（内容级变更但 chunk 数量不变的场景由调用方主动触发）。"""
        self._rebuild()

    def _tokenize(self, text: str) -> list[str]:
        return self._tokenizer(text)

    # -- 查询 -----------------------------------------------------------------

    def query(
        self,
        text: str,
        top_k: int = 10,
        document_id: int | None = None,
    ) -> list[tuple[str, float]]:
        """BM25 检索，返回 (chunk_id, score) 按相关度降序。"""
        self._ensure_fresh()
        data = self._data
        if data is None or data.bm25 is None or not data.chunk_ids:
            return []
        tokens = self._tokenize(text)
        if not tokens:
            return []
        scores = data.bm25.get_scores(tokens)
        ranked: list[tuple[str, float]] = []
        for index, chunk_id in enumerate(data.chunk_ids):
            if document_id is not None:
                meta_doc_id = data.chunks_by_id[chunk_id].metadata.get('document_id')
                if meta_doc_id != document_id:
                    continue
            ranked.append((chunk_id, float(scores[index])))
        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked[:top_k]

    def chunk_info(self, chunk_id: str) -> StoredChunk | None:
        """按 id 取分块详情（文本/元数据/向量），供融合结果补全。"""
        self._ensure_fresh()
        if self._data is None:
            return None
        return self._data.chunks_by_id.get(chunk_id)

    def size(self) -> int:
        """当前索引覆盖的 chunk 数。"""
        self._ensure_fresh()
        return self._data.count if self._data is not None else 0
