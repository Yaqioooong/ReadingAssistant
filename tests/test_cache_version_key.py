"""缓存键的「文档版本」维度回归：content_hash 到底有没有在执行。

历史状态（2026-09-18 取证）：缓存键概念上是 (问题, 文档版本, 文档范围)，
但两条路径都没真正执行这个维度：

1. **单书**：`question_hash` 已含 document_id，而 `Document.content_hash` 从不更新
   → content_hash 永远命中同一行，是纯冗余维度；
2. **全库**：`_document_content_hash` 直接返回空串 → **没有任何版本维度**，
   上传/删除/重索引任何一本书之后旧答案照样命中。

更糟的是约束与代码不一致：代码按 (question_hash, content_hash) 查，
DB 却只许 question_hash 单列唯一 → 版本真的变了就插不进第二行，
`IntegrityError` 冒到 API，整轮问答失败且永久复发。

本文件把这三件事钉住。
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.embeddings import Embeddings
from sqlalchemy import inspect as sa_inspect
from sqlalchemy import select, text

from reading_assistant.graph.qa import build_qa_graph
from reading_assistant.storage import (
    create_db_engine,
    create_session_factory,
    init_db,
    list_qa_cache_entries,
)
from reading_assistant.storage.database import QA_CACHE_LEGACY_INDEX, QA_CACHE_VERSION_INDEX
from reading_assistant.storage.models import Document, QaCacheEntry
from reading_assistant.storage.vector_store import InMemoryVectorStore, StoredChunk

QUESTION = '第一个角色住在哪里？'
ANSWER = '答案来自当时那一版原文。'


class _Emb(Embeddings):
    """所有文本同向量：缓存走精确通道即可，不需要语义区分。"""

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


class _LLM:
    """只回 book 意图与固定答案；计数便于确认「真的重新生成了」。"""

    def __init__(self) -> None:
        self.answer_calls = 0

    def invoke(self, prompt: str) -> SimpleNamespace:
        if '意图分类任务' in prompt:
            return SimpleNamespace(content='book')
        self.answer_calls += 1
        return SimpleNamespace(content=ANSWER)


class _Env:
    def __init__(self) -> None:
        self.engine = create_db_engine('sqlite:///:memory:')
        init_db(self.engine)
        self.factory = create_session_factory(self.engine)
        self.store = InMemoryVectorStore()
        self.llm = _LLM()

    def add_document(self, doc_id: int, content_hash: str, indexed: bool = True) -> None:
        with self.factory() as session:
            session.add(
                Document(
                    id=doc_id,
                    filename=f'{content_hash}.docx',
                    title=content_hash,
                    file_hash=f'f{doc_id}',
                    content_hash=content_hash,
                    index_status='indexed' if indexed else 'indexing',
                )
            )
            session.commit()
        self.store.add([
            StoredChunk(
                id=f'c{doc_id}',
                text=f'{content_hash} 的正文。',
                embedding=[1.0, 0.0],
                metadata={
                    'document_id': doc_id,
                    'chapter': '1',
                    'chapter_index': 0,
                    'content_hash': content_hash,
                },
            )
        ])

    def set_document_version(self, doc_id: int, content_hash: str) -> None:
        """模拟「同一本书内容变了」——这是 content_hash 唯一有意义的场景。"""
        with self.factory() as session:
            session.get(Document, doc_id).content_hash = content_hash
            session.commit()

    def graph(self):
        return build_qa_graph(
            self.factory, self.store, llm=self.llm, embedding_model=_Emb()
        )

    def ask(self, graph, *, document_id: int | None = None, turn: str = 'a') -> dict:
        payload: dict = {'question': QUESTION}
        if document_id is None:
            payload['document_ids'] = []
        else:
            payload['document_id'] = document_id
        return graph.invoke(
            payload, config={'configurable': {'thread_id': f'{QUESTION}-{turn}'}}
        )

    def cache_rows(self) -> list[QaCacheEntry]:
        with self.factory() as session:
            return list(session.scalars(select(QaCacheEntry)))


# --------------------------------------------------------------- 单书路径


def test_single_doc_revision_invalidates_cache() -> None:
    """文档改版后重问：必须重新生成，且**不得抛异常**（旧行为在这里撞唯一约束）。"""
    env = _Env()
    env.add_document(1, 'v1')
    graph = env.graph()

    first = env.ask(graph, document_id=1, turn='1')
    assert first['answer'] == ANSWER
    assert first['cache_hit'] is False
    calls_after_first = env.llm.answer_calls

    env.set_document_version(1, 'v2')          # 书的内容变了
    second = env.ask(graph, document_id=1, turn='2')   # 旧代码：IntegrityError

    assert second['cache_hit'] is False, '版本变了必须重算，不能发旧答案'
    assert env.llm.answer_calls > calls_after_first, '应重新生成回答'


def test_two_versions_coexist_as_separate_rows() -> None:
    """复合键生效的证据：同一问题可以有两行，只差 content_hash。"""
    env = _Env()
    env.add_document(1, 'v1')
    graph = env.graph()
    env.ask(graph, document_id=1, turn='1')
    env.set_document_version(1, 'v2')
    env.ask(graph, document_id=1, turn='2')

    rows = env.cache_rows()
    hashes = {row.question_hash for row in rows}
    assert len(hashes) == 1, '同一个问题应当只有一个 question_hash'
    assert len(rows) == 2, '两个版本应各占一行，而不是撞唯一约束'
    assert {row.content_hash for row in rows} == {'v1', 'v2'}


def test_single_doc_same_version_still_hits() -> None:
    """防回归：版本没变时缓存照常命中（别把维度做成了「永远不命中」）。"""
    env = _Env()
    env.add_document(1, 'v1')
    graph = env.graph()
    env.ask(graph, document_id=1, turn='1')
    calls = env.llm.answer_calls

    again = env.ask(graph, document_id=1, turn='2')
    assert again['cache_hit'] is True
    assert env.llm.answer_calls == calls, '命中缓存不应重新生成'


# --------------------------------------------------------------- 全库路径


def test_full_library_corpus_change_invalidates_cache() -> None:
    """全库问答：语料变了（新增一本书）旧答案必须失效。"""
    env = _Env()
    env.add_document(1, 'a1')
    graph = env.graph()

    first = env.ask(graph, turn='1')
    assert first['answer'] == ANSWER and first['cache_hit'] is False

    env.add_document(2, 'b1')                  # 语料变了
    second = env.ask(graph, turn='2')
    assert second['cache_hit'] is False, '语料变了不能继续发旧答案'
    assert second['answer'] == ANSWER


def test_full_library_corpus_change_invalidates_cached_refusal() -> None:
    """最坏形态：全库拒答被长期缓存 —— 书入库后重问不能还答「没提到」。"""
    env = _Env()
    env.add_document(1, 'a1')
    graph = env.graph()
    with env.factory() as session:
        session.add(
            QaCacheEntry(
                question_raw=QUESTION,
                question_normalized=QUESTION,
                question_hash='refusal-hash',
                question_embedding=[],
                answer=None,
                citations=[],
                needs_clarification=True,
                document_id=None,
                content_hash='stale-corpus',
            )
        )
        session.commit()

    env.add_document(2, 'b1')                  # 那本书终于入库了
    result = env.ask(graph, turn='after-ingest')
    assert result.get('needs_clarification') is False, '语料已变，不应复用旧的拒答判定'


def test_full_library_same_corpus_still_hits() -> None:
    """防回归：语料没变时全库缓存照常命中。"""
    env = _Env()
    env.add_document(1, 'a1')
    graph = env.graph()
    env.ask(graph, turn='1')
    calls = env.llm.answer_calls

    again = env.ask(graph, turn='2')
    assert again['cache_hit'] is True
    assert env.llm.answer_calls == calls


def test_corpus_fingerprint_ignores_unindexed_documents() -> None:
    """尚未入库完成的文档不该改变指纹 —— 否则与检索层的白名单漂移。"""
    from reading_assistant.graph.qa import _corpus_fingerprint

    env = _Env()
    env.add_document(1, 'a1')
    with env.factory() as session:
        before = _corpus_fingerprint(session)

    env.add_document(2, 'b1', indexed=False)   # 只是上传，还没索引完
    with env.factory() as session:
        after = _corpus_fingerprint(session)

    assert before == after, 'indexing 状态的文档不参与指纹（检索层也检索不到它）'


# --------------------------------------------------------------- 语义层


def test_semantic_candidates_exclude_other_versions() -> None:
    """语义层候选本就按 content_hash 筛 —— 复合键下这个筛选才真正有意义。"""
    env = _Env()
    with env.factory() as session:
        session.add(
            QaCacheEntry(
                question_raw=QUESTION,
                question_normalized=QUESTION,
                question_hash='h1',
                question_embedding=[1.0, 0.0],
                answer=ANSWER,
                citations=[],
                needs_clarification=False,
                document_id=None,
                content_hash='v1',
            )
        )
        session.commit()

    with env.factory() as session:
        assert list_qa_cache_entries(session, 'v1', None) != [], '同版本应可见'
        assert list_qa_cache_entries(session, 'v2', None) == [], '异版本不应进入候选'


# --------------------------------------------------------------- 迁移


_LEGACY_QA_CACHE_DDL = """
CREATE TABLE qa_cache (
  id INTEGER NOT NULL PRIMARY KEY,
  question_raw TEXT NOT NULL,
  question_normalized TEXT NOT NULL,
  question_hash VARCHAR(64) NOT NULL,
  question_embedding JSON NOT NULL,
  answer TEXT,
  citations JSON NOT NULL,
  needs_clarification BOOLEAN NOT NULL,
  document_id INTEGER,
  content_hash VARCHAR(64) NOT NULL,
  cached_chunk_count INTEGER NOT NULL DEFAULT 0,
  hit_count INTEGER NOT NULL DEFAULT 0,
  created_at DATETIME NOT NULL,
  last_hit_at DATETIME NOT NULL
)
"""


def _make_legacy_qa_cache(engine) -> None:
    """忠实复刻迁移前的 qa_cache：三个索引，question_hash 那个是**唯一**的。

    三个索引都建 —— 老库是 ``create_all`` 出来的，content_hash / document_id
    的普通索引本来就存在，少建会让「迁移库 vs 全新库」对比失真。
    """
    with engine.begin() as conn:
        conn.execute(text(_LEGACY_QA_CACHE_DDL))
        conn.execute(text(
            'CREATE UNIQUE INDEX ix_qa_cache_question_hash ON qa_cache (question_hash)'
        ))
        conn.execute(text('CREATE INDEX ix_qa_cache_content_hash ON qa_cache (content_hash)'))
        conn.execute(text('CREATE INDEX ix_qa_cache_document_id ON qa_cache (document_id)'))


def _insert_row(conn, question_hash: str, content_hash: str) -> None:
    conn.execute(
        text(
            'INSERT INTO qa_cache (question_raw, question_normalized, question_hash, '
            'question_embedding, answer, citations, needs_clarification, document_id, '
            'content_hash, cached_chunk_count, hit_count, created_at, last_hit_at) VALUES '
            '(:q, :q, :h, :emb, :a, :cit, 0, NULL, :ch, 0, 0, CURRENT_TIMESTAMP, '
            'CURRENT_TIMESTAMP)'
        ),
        {'q': QUESTION, 'h': question_hash, 'emb': json.dumps([]),
         'a': ANSWER, 'cit': json.dumps([]), 'ch': content_hash},
    )


def test_migration_replaces_legacy_unique_index(tmp_path: Path) -> None:
    """老库迁移：单列唯一索引必须被换成复合唯一，否则第二个版本写不进去。"""
    engine = create_db_engine(f'sqlite:///{tmp_path / "legacy.db"}')
    _make_legacy_qa_cache(engine)
    with engine.begin() as conn:
        _insert_row(conn, 'h1', 'v1')
        # 迁移前：同一 question_hash 的第二个版本必然被拒
        with pytest.raises(Exception):
            _insert_row(conn, 'h1', 'v2')

    init_db(engine)

    inspector = sa_inspect(engine)
    indexes = {idx['name']: idx for idx in inspector.get_indexes('qa_cache')}
    assert bool(indexes[QA_CACHE_VERSION_INDEX]['unique']) is True
    assert indexes[QA_CACHE_VERSION_INDEX]['column_names'] == ['question_hash', 'content_hash']
    assert not indexes[QA_CACHE_LEGACY_INDEX]['unique'], '旧单列唯一索引必须降级为普通索引'

    with engine.begin() as conn:            # 迁移后：第二个版本可以共存
        _insert_row(conn, 'h1', 'v2')
        assert conn.execute(
            text('SELECT count(*) FROM qa_cache WHERE question_hash = :h'), {'h': 'h1'}
        ).scalar() == 2


def test_migrated_schema_matches_fresh_schema(tmp_path: Path) -> None:
    """迁移库与全新库的索引集合必须一致 —— 否则两条路径的 schema 会分叉。"""
    legacy = create_db_engine(f'sqlite:///{tmp_path / "legacy2.db"}')
    _make_legacy_qa_cache(legacy)
    init_db(legacy)

    fresh = create_db_engine(f'sqlite:///{tmp_path / "fresh2.db"}')
    init_db(fresh)

    def shape(engine):
        return {
            idx['name']: (tuple(idx['column_names']), bool(idx['unique']))
            for idx in sa_inspect(engine).get_indexes('qa_cache')
        }

    assert shape(legacy) == shape(fresh)


def test_migration_repairs_partial_migration_state(tmp_path: Path) -> None:
    """迁移必须**收敛**：中间态（旧唯一索引已删、普通索引没补上）也要被修好。

    这不是假想——2026-09-18 实测发生过：开发服务器带 --reload，
    每次改 .py 都重启并跑一遍 init_db，一个「只在首次迁移时正确」的条件式实现
    就在真实库上留下了缺索引的中间态。迁移每次启动都执行，所以它必须把库
    收敛到目标形态，而不是依赖「这是第一次」。
    """
    engine = create_db_engine(f'sqlite:///{tmp_path / "partial.db"}')
    _make_legacy_qa_cache(engine)
    with engine.begin() as conn:                 # 制造中间态：删掉旧唯一索引，不补普通索引
        conn.execute(text('DROP INDEX ix_qa_cache_question_hash'))

    init_db(engine)

    indexes = {idx['name']: idx for idx in sa_inspect(engine).get_indexes('qa_cache')}
    assert QA_CACHE_LEGACY_INDEX in indexes, '普通索引应被补齐'
    assert not indexes[QA_CACHE_LEGACY_INDEX]['unique']
    assert indexes[QA_CACHE_VERSION_INDEX]['unique']


def test_migration_is_idempotent(tmp_path: Path) -> None:
    """迁移可重复执行（每次启动都会跑）。"""
    engine = create_db_engine(f'sqlite:///{tmp_path / "fresh.db"}')
    init_db(engine)
    marker = sa_inspect(engine).get_indexes('qa_cache')
    init_db(engine)
    init_db(engine)
    assert sa_inspect(engine).get_indexes('qa_cache') == marker
