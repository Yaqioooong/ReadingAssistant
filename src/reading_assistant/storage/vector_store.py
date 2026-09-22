"""向量库适配器：隔离具体实现（ChromaDB / 内存）。"""

import math
import re
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
    """检索结果。

    ``score`` 统一为**余弦相似度**（越大越相关，范围约 [-1, 1]），
    便于稠密 / 稀疏两路共用同一个 ``min_score`` 阈值。
    """

    id: str
    score: float
    metadata: dict = field(default_factory=dict)
    text: str = ''


_CHUNK_POSITION_RE = re.compile(r'-(\d+)$')


def _chunk_position(chunk_id: str) -> int:
    """从 chunk id 尾号取书内位置（``doc7-<hash8>-198`` → 198）。

    取不到时返回 -1（排最前），保证排序结果稳定而不抛错 ——
    排序只用于「章内顺序」，退化不该让整次问答失败。
    """
    match = _CHUNK_POSITION_RE.search(chunk_id or '')
    return int(match.group(1)) if match else -1


def _and_where(**conditions) -> dict | None:
    """把多个等值条件组装成 Chroma 合法的 ``where``。

    ⚠️ Chroma 的 ``get``/``query`` 要求 ``where`` 里**只能有一个操作符**：
    直接写 ``{'document_id': 7, 'chapter_index': 18}`` 会被拒
    （``ValueError: Expected where to have exactly one operator``），
    必须写成 ``{'$and': [{'document_id': 7}, {'chapter_index': 18}]}``。

    实测（2026-09-21）：这条只在**真实 Chroma** 上炸 —— InMemoryVectorStore 自己实现
    线性过滤、不校验 where 形状，所以单测全绿而线上 read_chapter 每次都失败。
    凡是「按两条元数据定位」的新代码，都必须走这个 helper。
    """
    items = [{key: value} for key, value in conditions.items() if value is not None]
    if not items:
        return None
    if len(items) == 1:
        return items[0]
    return {'$and': items}


def _scope_where(document_ids: list[int] | None) -> dict | None:
    """把文档范围转成 Chroma 的 where 子句；None 表示全库。"""
    if not document_ids:
        return None
    if len(document_ids) == 1:
        return {'document_id': document_ids[0]}
    return {'document_id': {'$in': list(document_ids)}}


class VectorStore(ABC):
    """向量库统一接口。"""

    @abstractmethod
    def add(self, chunks: list[StoredChunk]) -> None:
        """批量写入分块。"""

    @abstractmethod
    def query(
        self, embedding: list[float], top_k: int = 6, where: dict | None = None
    ) -> list[SearchHit]:
        """按向量检索 top-k 分块；``score`` 为余弦相似度。"""

    @abstractmethod
    def count(self) -> int:
        """返回分块总数。"""

    @abstractmethod
    def delete(self, document_id: int) -> None:
        """删除指定文档的全部向量分块"""

    @abstractmethod
    def delete_stale(
        self,
        document_id: int,
        keep_content_hash: str,
        keep_ids: list[str] | None = None,
    ) -> int:
        """删除 ``document_id`` 下不属于「当前版本」的分块，返回删除条数。

        - ``keep_ids`` 为本次刚 upsert 的新版本 id 显式列表（首选判据）；
        - 未提供 ``keep_ids`` 时，退化为「``metadata.content_hash ==
          keep_content_hash`` 视为当前版本」（迁移 / 兼容场景）。
        - 历史 chunk 可能缺失 ``content_hash`` 键，故在 **Python 侧** 过滤、
          再按显式 id 列表删除；不使用 Chroma ``where={'content_hash': {'$ne': ...}}``
          （``$ne`` 对缺失键的行为跨 Chroma 版本不可靠）。

        顺序契约：调用方必须**先 upsert 新版本、再调用本方法**。若在两步之间
        崩溃，只会留下「同内容重复」，不会出现错误内容、不会整篇丢失。
        """

    @abstractmethod
    def all_chunks(self) -> list[StoredChunk]:
        """返回库内全部分块（含文本与元数据），供 BM25 索引构建。"""

    # ---- 结构化导航能力（供 agent 工具使用，见 graph/agent.py）----

    @abstractmethod
    def search_text(
        self,
        term: str,
        document_ids: list[int] | None = None,
        limit: int = 100,
    ) -> list[SearchHit]:
        """**字面包含**检索（grep）：返回正文含 ``term`` 的分块。

        与 ``query``（向量相似度）是两条完全不同的通道，用途也不同：

        - 序数类问题（「第一个妖怪是什么」）的答案段落**从不包含「第一个」**，
          所以向量通道永远找不到它；而字面通道能确定性地定位实体所在章。
          实测（2026-09-21）：`grep('寅将军')` 恰好命中 1 章，
          而向量通道在 min_score=0、top_k=200 下都捞不到 —— 存在性判定必须走这条路。
        - ``score`` 固定为 ``0.0``：字面命中没有相似度语义，不要拿它参与排序。
        - 实现应走底层原生全文过滤（Chroma ``where_document``），
          **不要**退化成 ``all_chunks()`` 全扫 —— 那是 O(全库) 的，
          书量上去以后会成为第一个崩的地方。
        """

    @abstractmethod
    def list_chapters(self, document_id: int) -> list[tuple[int, str]]:
        """返回 ``document_id`` 的章节目录 ``[(chapter_index, chapter), ...]``，按书内顺序。

        章顺序 = ``chapter_index`` 升序，这是**序数推理的唯一可靠依据**
        （embedding 排序在章级只有 0.037 的 margin，不可用）。
        """

    @abstractmethod
    def get_chapter(self, document_id: int, chapter_index: int) -> list[StoredChunk]:
        """返回某一章的**全部**分块，按书内顺序（chunk id 尾号升序）。

        刻意**不做相似度过滤**：读整章的意义就是绕开 ``retrieval_min_score`` ——
        实测该阈值会把「描述式提问 → 诗体描写段落」这类低余弦的正确段落直接杀掉
        （2026-09-21：答案 chunk 余弦 0.4296 < 0.45）。
        """


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
        # 距离空间：Chroma 默认 l2（现有集合 metadata 为 None → l2）。
        # 用于把 distance 换算成统一的余弦相似度。
        metadata = self._collection.metadata or {}
        self._space = str(metadata.get('hnsw:space') or metadata.get('space') or 'l2').lower()
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
                score=self._distance_to_cosine(
                    distances[index] if index < len(distances) else 2.0
                ),
                metadata=metadatas[index] or {},
                text=documents[index] or '',
            )
            for index, chunk_id in enumerate(ids)
        ]

    def search_text(
        self,
        term: str,
        document_ids: list[int] | None = None,
        limit: int = 100,
    ) -> list[SearchHit]:
        """字面包含检索：走 Chroma 原生 ``where_document``（非全扫）。"""
        if not term:
            return []
        where = _scope_where(document_ids)
        # 注意：where_document 的 $contains 是子串匹配，不做分词。
        # 中文没有词边界，短词（如「妖」）会大量误命中 —— 由上层截断与聚合处理。
        result = self._collection.get(
            where=where,
            where_document={'$contains': term},
            limit=limit,
            include=['metadatas', 'documents'],
        )
        ids = result.get('ids') or []
        metadatas = result.get('metadatas') or []
        documents = result.get('documents') or []
        return [
            SearchHit(
                id=chunk_id,
                score=0.0,  # 字面命中无相似度语义
                metadata=metadatas[index] or {},
                text=documents[index] or '',
            )
            for index, chunk_id in enumerate(ids)
        ]

    def list_chapters(self, document_id: int) -> list[tuple[int, str]]:
        """章节目录：只取 metadatas，不取正文（避免把整本书拉进内存）。"""
        result = self._collection.get(
            where={'document_id': document_id}, include=['metadatas']
        )
        seen: dict[int, str] = {}
        for metadata in result.get('metadatas') or []:
            metadata = metadata or {}
            index = metadata.get('chapter_index')
            if index is None:
                continue
            seen.setdefault(int(index), str(metadata.get('chapter') or ''))
        return sorted(seen.items())

    def get_chapter(self, document_id: int, chapter_index: int) -> list[StoredChunk]:
        """按章取全部正文；顺序由 chunk id 尾号决定（= 书内位置）。"""
        result = self._collection.get(
            where=_and_where(document_id=document_id, chapter_index=chapter_index),
            include=['metadatas', 'documents'],
        )
        chunks = [
            StoredChunk(
                id=chunk_id,
                text=(result.get('documents') or [])[index] or '',
                metadata=(result.get('metadatas') or [])[index] or {},
            )
            for index, chunk_id in enumerate(result.get('ids') or [])
        ]
        chunks.sort(key=lambda c: _chunk_position(c.id))
        return chunks

    def _distance_to_cosine(self, distance: float) -> float:
        """把 Chroma 距离换算成余弦相似度（统一量纲）。

        本项目 embedding 已归一化（DashScope），因此：
        - cosine 空间：Chroma 距离 = 1 - cos → ``cos = 1 - dist``
        - l2 空间（默认）：Chroma 返回**平方**欧氏距离 = 2 - 2cos → ``cos = 1 - dist/2``

        ⚠️ 必须显式收敛为内建 ``float``：Chroma 返回的 distance 是 **numpy 标量**，
        经 ``1 - dist/2`` 与 ``min/max`` 运算后仍是 numpy 标量（float32/float64）。
        它会一路混进 QA state 的 ``chunks[*]['score']``，而 LangGraph checkpoint
        用 ormsgpack 序列化 state 时不认 numpy 类型，直接抛
        ``TypeError: Type is not msgpack serializable: numpy.float64``
        —— 表现为「同一条问题偶尔 500」。类型收敛放在源头，别指望下游各自防御。
        注意：n_results 较小时 Chroma 可能返回内建 float，所以此坑是**间歇性**的，
        单元测试要给 numpy 输入才能稳定复现。
        """
        distance = float(distance)
        cosine = 1.0 - distance if self._space == 'cosine' else 1.0 - distance / 2.0
        return float(max(-1.0, min(1.0, cosine)))

    def count(self) -> int:
        return self._collection.count()

    def delete(self, document_id: int) -> None:
        self._collection.delete(where={'document_id': document_id})
        self._content_version += 1

    def delete_stale(
        self,
        document_id: int,
        keep_content_hash: str,
        keep_ids: list[str] | None = None,
    ) -> int:
        keep = set(keep_ids) if keep_ids is not None else None
        stale: list[str] = []
        for chunk in self.all_chunks():
            metadata = chunk.metadata or {}
            if metadata.get('document_id') != document_id:
                continue
            if keep is not None:
                if chunk.id not in keep:
                    stale.append(chunk.id)
            elif metadata.get('content_hash') != keep_content_hash:
                stale.append(chunk.id)
        if stale:
            self._collection.delete(ids=stale)
            self._content_version += 1
        return len(stale)

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
                # 收敛为内建 float：Chroma 返回的是 numpy 标量序列，
                # 一旦泄漏到下游（BM25 现算余弦 → score → QA state）就会让
                # LangGraph checkpoint 的 ormsgpack 序列化直接抛 TypeError。
                embedding = [float(value) for value in embeddings[index]]
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

    def search_text(
        self,
        term: str,
        document_ids: list[int] | None = None,
        limit: int = 100,
    ) -> list[SearchHit]:
        if not term:
            return []
        hits = [
            SearchHit(id=chunk.id, score=0.0, metadata=chunk.metadata, text=chunk.text)
            for chunk in self._chunks.values()
            if term in (chunk.text or '')
            and (not document_ids
                 or chunk.metadata.get('document_id') in set(document_ids))
        ]
        hits.sort(key=lambda hit: _chunk_position(hit.id))
        return hits[:limit]

    def list_chapters(self, document_id: int) -> list[tuple[int, str]]:
        seen: dict[int, str] = {}
        for chunk in self._chunks.values():
            if chunk.metadata.get('document_id') != document_id:
                continue
            index = chunk.metadata.get('chapter_index')
            if index is None:
                continue
            seen.setdefault(int(index), str(chunk.metadata.get('chapter') or ''))
        return sorted(seen.items())

    def get_chapter(self, document_id: int, chapter_index: int) -> list[StoredChunk]:
        chunks = [
            chunk for chunk in self._chunks.values()
            if chunk.metadata.get('document_id') == document_id
            and chunk.metadata.get('chapter_index') == chapter_index
        ]
        chunks.sort(key=lambda chunk: _chunk_position(chunk.id))
        return chunks

    def count(self) -> int:
        return len(self._chunks)

    def delete(self, document_id: int) -> None:
        for chunk_id in list(self._chunks):
            if self._chunks[chunk_id].metadata.get('document_id') == document_id:
                del self._chunks[chunk_id]
        self._content_version += 1

    def delete_stale(
        self,
        document_id: int,
        keep_content_hash: str,
        keep_ids: list[str] | None = None,
    ) -> int:
        keep = set(keep_ids) if keep_ids is not None else None
        stale: list[str] = []
        for chunk_id, chunk in self._chunks.items():
            if chunk.metadata.get('document_id') != document_id:
                continue
            if keep is not None:
                if chunk_id not in keep:
                    stale.append(chunk_id)
            elif chunk.metadata.get('content_hash') != keep_content_hash:
                stale.append(chunk_id)
        for chunk_id in stale:
            del self._chunks[chunk_id]
        if stale:
            self._content_version += 1
        return len(stale)

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
