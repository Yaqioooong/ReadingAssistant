"""index_status 对账（P0-3）：租约过期的 ``indexing`` 记录按向量库实际内容修正。

故障背景（实测）：库里曾出现 **3/6 文档卡在 ``indexing``** 且无人察觉 ——
``index_status`` 全项目零消费方，且进程崩溃不走 ``except`` 分支，连 ``failed``
都写不上。启动对账用「租约 + 与向量库核对」把状态拉回真实值。

对账必须满足两条：**幂等**（可重复运行）、**只碰过期的 indexing 记录**。
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from reading_assistant.storage import (
    create_db_engine,
    create_session_factory,
    init_db,
)
from reading_assistant.storage.models import Document
from reading_assistant.storage.reconcile import reconcile_index_status
from reading_assistant.storage.vector_store import InMemoryVectorStore, StoredChunk

LEASE = 600


@pytest.fixture
def env():
    engine = create_db_engine('sqlite:///:memory:')
    init_db(engine)
    return create_session_factory(engine), InMemoryVectorStore()


def _seed_document(
    factory,
    *,
    seq: int,
    index_status: str,
    chunk_count: int,
    started_at: datetime | None,
    content_hash: str | None = None,
) -> int:
    """插入一条文档记录；file_hash / content_hash 必须唯一，故按 seq 派生。"""
    content_hash = content_hash or f'{seq:064d}'
    with factory() as session:
        doc = Document(
            filename=f'book{seq}.txt',
            title=f'book{seq}',
            file_path=f'/tmp/book{seq}.txt',
            file_hash=f'{seq + 1000:064d}',
            content_hash=content_hash,
            chunk_count=chunk_count,
            index_status=index_status,
            index_started_at=started_at,
        )
        session.add(doc)
        session.commit()
        return doc.id


def _seed_vectors(store: InMemoryVectorStore, doc_id: int, content_hash: str, n: int) -> None:
    store.add(
        [
            StoredChunk(
                id=f'doc{doc_id}-{content_hash[:8]}-{i}',
                text=f'片段{i}',
                metadata={'document_id': doc_id, 'content_hash': content_hash},
                embedding=[1.0, 0.0],
            )
            for i in range(n)
        ]
    )


def _status_of(factory, doc_id: int) -> str:
    with factory() as session:
        return session.get(Document, doc_id).index_status


def _ago(minutes: int) -> datetime:
    return datetime.now(timezone.utc) - timedelta(minutes=minutes)


class TestReconcileIndexStatus:
    def test_expired_lease_with_complete_vectors_repaired_to_indexed(self, env) -> None:
        """向量完整 → 说明入库其实已完成，只是状态没来得及翻转。"""
        factory, store = env
        doc_id = _seed_document(
            factory, seq=1, index_status='indexing', chunk_count=3, started_at=_ago(30)
        )
        with factory() as session:
            content_hash = session.get(Document, doc_id).content_hash
        _seed_vectors(store, doc_id, content_hash, 3)

        actions = reconcile_index_status(factory, store, LEASE, apply=True)

        assert [a.document_id for a in actions] == [doc_id]
        assert actions[0].new_status == 'indexed'
        assert _status_of(factory, doc_id) == 'indexed'

    def test_expired_lease_with_missing_vectors_marks_failed(self, env) -> None:
        """向量不完整 → 确实中断了，标 failed。"""
        factory, store = env
        doc_id = _seed_document(
            factory, seq=2, index_status='indexing', chunk_count=5, started_at=_ago(30)
        )
        with factory() as session:
            content_hash = session.get(Document, doc_id).content_hash
        _seed_vectors(store, doc_id, content_hash, 2)  # 只写了一半

        actions = reconcile_index_status(factory, store, LEASE, apply=True)

        assert actions[0].new_status == 'failed'
        assert _status_of(factory, doc_id) == 'failed'

    def test_empty_vectors_marks_failed(self, env) -> None:
        factory, store = env
        doc_id = _seed_document(
            factory, seq=3, index_status='indexing', chunk_count=4, started_at=_ago(30)
        )

        actions = reconcile_index_status(factory, store, LEASE, apply=True)

        assert actions[0].new_status == 'failed'
        assert _status_of(factory, doc_id) == 'failed'

    def test_stale_vectors_from_other_version_count_as_incomplete(self, env) -> None:
        """向量数量对得上、但版本指纹不同 → 仍是不完整（内容已过期）。"""
        factory, store = env
        doc_id = _seed_document(
            factory, seq=4, index_status='indexing', chunk_count=3, started_at=_ago(30)
        )
        _seed_vectors(store, doc_id, 'b' * 64, 3)  # 旧版本残留

        actions = reconcile_index_status(factory, store, LEASE, apply=True)

        assert actions[0].new_status == 'failed'

    def test_fresh_lease_is_skipped(self, env) -> None:
        """租约未过期 → 可能仍在正常入库，绝不能动。"""
        factory, store = env
        doc_id = _seed_document(
            factory, seq=5, index_status='indexing', chunk_count=3, started_at=_ago(1)
        )

        actions = reconcile_index_status(factory, store, LEASE, apply=True)

        assert actions == []
        assert _status_of(factory, doc_id) == 'indexing'

    def test_missing_started_at_is_treated_as_expired(self, env) -> None:
        """历史记录没有 index_started_at（加列前的遗留）→ 视为已过期。"""
        factory, store = env
        doc_id = _seed_document(
            factory, seq=6, index_status='indexing', chunk_count=1, started_at=None
        )
        with factory() as session:
            content_hash = session.get(Document, doc_id).content_hash
        _seed_vectors(store, doc_id, content_hash, 1)

        actions = reconcile_index_status(factory, store, LEASE, apply=True)

        assert [a.document_id for a in actions] == [doc_id]
        assert _status_of(factory, doc_id) == 'indexed'

    def test_dry_run_reports_but_does_not_write(self, env) -> None:
        factory, store = env
        doc_id = _seed_document(
            factory, seq=7, index_status='indexing', chunk_count=2, started_at=_ago(30)
        )

        actions = reconcile_index_status(factory, store, LEASE, apply=False)

        assert [a.document_id for a in actions] == [doc_id]
        assert _status_of(factory, doc_id) == 'indexing', 'dry-run 不得写入'

    def test_indexed_documents_untouched(self, env) -> None:
        factory, store = env
        doc_id = _seed_document(
            factory, seq=8, index_status='indexed', chunk_count=1, started_at=_ago(30)
        )

        actions = reconcile_index_status(factory, store, LEASE, apply=True)

        assert actions == []
        assert _status_of(factory, doc_id) == 'indexed'

    def test_failed_documents_untouched(self, env) -> None:
        """failed 是明确终态，不参与对账。"""
        factory, store = env
        doc_id = _seed_document(
            factory, seq=9, index_status='failed', chunk_count=0, started_at=_ago(30)
        )

        actions = reconcile_index_status(factory, store, LEASE, apply=True)

        assert actions == []
        assert _status_of(factory, doc_id) == 'failed'

    def test_idempotent(self, env) -> None:
        """重复运行：第二次不应再产生任何动作。"""
        factory, store = env
        doc_id = _seed_document(
            factory, seq=10, index_status='indexing', chunk_count=2, started_at=_ago(30)
        )
        with factory() as session:
            content_hash = session.get(Document, doc_id).content_hash
        _seed_vectors(store, doc_id, content_hash, 2)

        first = reconcile_index_status(factory, store, LEASE, apply=True)
        second = reconcile_index_status(factory, store, LEASE, apply=True)

        assert len(first) == 1
        assert second == [], '对账必须幂等：已修正的记录不应再次出现'
        assert _status_of(factory, doc_id) == 'indexed'

    def test_multiple_candidates_reconciled_independently(self, env) -> None:
        """复现原始场景：同一批里既有可修复的，也有真失败的。"""
        factory, store = env
        healthy = _seed_document(
            factory, seq=11, index_status='indexing', chunk_count=2, started_at=_ago(30)
        )
        broken = _seed_document(
            factory, seq=12, index_status='indexing', chunk_count=2, started_at=_ago(30)
        )
        with factory() as session:
            hash_ok = session.get(Document, healthy).content_hash
            hash_bad = session.get(Document, broken).content_hash
        _seed_vectors(store, healthy, hash_ok, 2)   # 完整
        _seed_vectors(store, broken, hash_bad, 0)   # 空

        reconcile_index_status(factory, store, LEASE, apply=True)

        assert _status_of(factory, healthy) == 'indexed'
        assert _status_of(factory, broken) == 'failed'
