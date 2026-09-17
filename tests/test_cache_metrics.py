"""缓存指标体系测试：三通道归因、请求级埋点、统计接口与用户反馈。

三通道用受控向量角度触发（cos(0,14°)=0.970 走语义，cos(0,20°)=0.940 落在标识符区间），
并用不同 document_id 隔离，避免候选互相干扰。

注意：``Retriever._embed`` 对**归一化后**的 query 做 embedding 并以此为 L1 缓存键，
因此 ``embed_query`` 收到的是 normalize_question 之后、去掉标点且小写的文本 ——
用例里的关键词判断必须按归一化形态写，否则会静默落到默认向量。
"""

import math
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient
from langchain_core.embeddings import Embeddings
from sqlalchemy import select

from reading_assistant.api import create_app
from reading_assistant.storage import create_db_engine, create_session_factory, init_db
from reading_assistant.storage.models import (
    Document,
    QaCacheEntry,
    QaFeedback,
    QaRequestEvent,
)
from reading_assistant.storage.vector_store import InMemoryVectorStore, StoredChunk

ANSWER = '镇元子住在万寿山五庄观。'
SEMANTIC_Q1, SEMANTIC_Q2 = '乙喜欢谁？', '乙喜欢的是谁？'
# 注意：'P002产品的…' 归一化后与 Q1 完全相同（会走 exact），故 Q2 需去掉「产品」二字
ID_Q1 = 'P-002产品的上市时间是什么时候？'
ID_Q2 = 'P002上市时间是什么时候？'


def _vec(deg: float) -> list[float]:
    rad = math.radians(deg)
    return [math.cos(rad), math.sin(rad)]


class ChannelEmb(Embeddings):
    """按关键词返回受控角度向量，精确命中三类缓存通道的触发条件。

    判定词均按归一化形态书写（小写、无标点）——查询文本在到达这里之前
    已被 normalize_question 处理过。
    """

    def embed_query(self, text: str) -> list[float]:
        raw = text or ''
        if '喜欢的是谁' in raw:  # 归一化后的 SEMANTIC_Q2
            return _vec(14)  # 与 _vec(0) 余弦 0.970 ≥ 0.95 → 语义通道
        if 'p002上市' in raw:  # 归一化后的 ID_Q2（ID_Q1 是 'p002产品的上市…'，不命中）
            return _vec(20)  # 余弦 0.940 落在 [0.90, 0.95) → 标识符通道
        return _vec(0)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [_vec(0) for _ in texts]


class FakeLLM:
    def __init__(self, answer: str = ANSWER) -> None:
        self.answer = answer

    def invoke(self, prompt: str) -> SimpleNamespace:
        if '意图分类任务' in prompt:
            part = prompt.split('【当前提问】', 1)[-1]
            question = next((ln.strip() for ln in part.splitlines() if ln.strip()), '')
            if question in ('你好', 'hello'):
                return SimpleNamespace(content='chat')
            return SimpleNamespace(content='book')
        if '不需要检索书籍内容' in prompt:
            return SimpleNamespace(content='你好呀，想从书里了解点什么？')
        return SimpleNamespace(content=self.answer)


def _build(tmp_path: Path, answer: str = ANSWER) -> tuple[TestClient, object]:
    """三本各一块的语料，分别用于 exact / semantic / identifier 三通道隔离测试。"""
    engine = create_db_engine('sqlite:///:memory:')
    init_db(engine)
    factory = create_session_factory(engine)
    with factory() as session:
        session.add_all(
            [
                Document(filename=f'd{i}.txt', title=f'd{i}', file_hash=f'f{i}',
                         content_hash=f'c{i}', index_status='indexed')
                for i in (1, 2, 3)
            ]
        )
        session.commit()
    store = InMemoryVectorStore()
    store.add(
        [
            StoredChunk(
                id=f'doc{i}-0',
                text=f'文档{i}正文片段',
                metadata={'document_id': i, 'chapter': '正文', 'chapter_index': 0},
                embedding=_vec(0),
            )
            for i in (1, 2, 3)
        ]
    )
    app = create_app(
        session_factory=factory,
        vector_store=store,
        llm=FakeLLM(answer),
        embedding_model=ChannelEmb(),
        upload_dir=tmp_path / 'uploads',
    )
    return TestClient(app), factory


def _ask(client: TestClient, sid: str, question: str, doc: int) -> dict:
    return client.post(
        f'/api/sessions/{sid}/messages',
        json={'question': question, 'document_ids': [doc]},
    ).json()


class TestChannelAttribution:
    def test_exact_hit_is_attributed(self, tmp_path: Path) -> None:
        client, _ = _build(tmp_path)
        with client:
            sid = client.post('/api/sessions').json()['session_id']
            first = _ask(client, sid, '甲喜欢谁？', 1)
            assert first['cache_channel'] == 'miss', '首次提问应记为 miss'
            assert first['cache_hit'] is False
            second = _ask(client, sid, '甲喜欢谁？', 1)
            assert second['cache_channel'] == 'exact'
            assert second['cache_hit'] is True

    def test_semantic_hit_is_attributed(self, tmp_path: Path) -> None:
        client, _ = _build(tmp_path)
        with client:
            sid = client.post('/api/sessions').json()['session_id']
            _ask(client, sid, SEMANTIC_Q1, 2)
            hit = _ask(client, sid, SEMANTIC_Q2, 2)
            assert hit['cache_channel'] == 'semantic', hit
            assert hit['cache_similarity'] is not None
            assert 0.95 <= hit['cache_similarity'] < 1.0

    def test_identifier_hit_is_attributed(self, tmp_path: Path) -> None:
        client, _ = _build(tmp_path)
        with client:
            sid = client.post('/api/sessions').json()['session_id']
            _ask(client, sid, ID_Q1, 3)
            hit = _ask(client, sid, ID_Q2, 3)
            assert hit['cache_channel'] == 'identifier', hit
            assert 0.90 <= hit['cache_similarity'] < 0.95

    def test_multi_document_is_disabled(self, tmp_path: Path) -> None:
        client, _ = _build(tmp_path)
        with client:
            sid = client.post('/api/sessions').json()['session_id']
            resp = client.post(
                f'/api/sessions/{sid}/messages',
                json={'question': '甲喜欢谁？', 'document_ids': [1, 2]},
            ).json()
            assert resp['cache_channel'] == 'disabled', '多文档问答跳过缓存'


class TestRequestEvents:
    def test_events_are_persisted_with_channel(self, tmp_path: Path) -> None:
        client, factory = _build(tmp_path)
        with client:
            sid = client.post('/api/sessions').json()['session_id']
            _ask(client, sid, '甲喜欢谁？', 1)
            _ask(client, sid, '甲喜欢谁？', 1)
            with factory() as session:
                events = list(session.scalars(select(QaRequestEvent).order_by(QaRequestEvent.id)))
            assert [e.cache_channel for e in events] == ['miss', 'exact']
            assert all(e.latency_ms is not None for e in events)
            assert all(e.question for e in events)
            assert events[1].cache_hit is True

    def test_stale_refusal_cache_marks_invalidated(self, tmp_path: Path) -> None:
        """带澄清补充重问时，旧拒答缓存应被标记失效并重新回答。

        判据已从「回答文本像不像拒答」（``content[:80]`` 词表匹配）换成结构性判断：
        拒答行 + 本轮带澄清 → 该判定不再适用。

        这不只是为了消掉字符串匹配的双向误判 —— 拒答缓存行的 ``answer`` 是 ``None``，
        旧实现**结构上就够不到这个场景**：用户补充信息后重问，语义通道会命中旧拒答行，
        再次转 HITL，把刚提供的信息原样丢掉。
        """
        client, factory = _build(tmp_path)
        with client:
            sid = client.post('/api/sessions').json()['session_id']
            _ask(client, sid, '甲喜欢谁？', 1)
            # 模拟一条「信息不足」判定（拒答行的真实形态：answer 为空）
            with factory() as session:
                entry = session.scalars(select(QaCacheEntry)).first()
                assert entry is not None
                entry.answer = None
                entry.citations = []
                entry.needs_clarification = True
                entry.cached_chunk_count = 0
                session.commit()
            again = client.post(
                f'/api/sessions/{sid}/messages',
                json={
                    'question': '甲喜欢谁？',
                    'document_ids': [1],
                    'clarification': '我问的是文档一里那个甲',
                },
            ).json()
            assert again['cache_hit'] is False
            with factory() as session:
                events = list(session.scalars(select(QaRequestEvent).order_by(QaRequestEvent.id)))
            assert events[-1].cache_invalidated is True

class TestSkippedTurns:
    """闲聊/历史类轮次不经 cache_check，应归为 skipped 而非混入命中率分母。"""

    def test_chat_turn_is_skipped(self, tmp_path: Path) -> None:
        client, factory = _build(tmp_path)
        with client:
            sid = client.post('/api/sessions').json()['session_id']
            body = _ask(client, sid, '你好', 1)
            assert body['intent'] == 'chat'
            with factory() as session:
                events = list(session.scalars(select(QaRequestEvent)))
            assert len(events) == 1
            assert events[0].cache_channel == 'skipped', '闲聊轮次应标记为 skipped'

    def test_cacheable_rate_excludes_skipped(self, tmp_path: Path) -> None:
        client, _ = _build(tmp_path)
        with client:
            sid = client.post('/api/sessions').json()['session_id']
            _ask(client, sid, '甲喜欢谁？', 1)   # miss（可缓存）
            _ask(client, sid, '甲喜欢谁？', 1)   # exact
            _ask(client, sid, '你好', 1)         # skipped
            body = client.get('/api/stats/cache').json()
            s = body['summary']
            assert s['total_requests'] == 3
            assert s['cacheable_requests'] == 2
            assert s['hit_rate'] == round(1 / 3, 6) or abs(s['hit_rate'] - 1 / 3) < 1e-6
            assert abs(s['hit_rate_cacheable'] - 0.5) < 1e-6, '可缓存口径应排除闲聊轮次'
            assert body['by_channel']['skipped'] == 1


class TestStatsEndpoint:
    def test_empty_store_returns_structure_without_error(self, tmp_path: Path) -> None:
        client, _ = _build(tmp_path)
        with client:
            resp = client.get('/api/stats/cache')
            assert resp.status_code == 200
            body = resp.json()
            for key in ('summary', 'by_channel', 'semantic', 'invalidated', 'feedback',
                        'entries', 'daily'):
                assert key in body, f'缺少字段 {key}'
            assert body['summary']['total_requests'] == 0
            assert body['summary']['hit_rate'] is None or body['summary']['hit_rate'] == 0
            assert body['semantic']['p50'] is None
            assert body['feedback']['mis_hit_rate'] is None
            assert len(body['daily']) == 7

    def test_counts_are_aggregated_by_channel(self, tmp_path: Path) -> None:
        client, _ = _build(tmp_path)
        with client:
            sid = client.post('/api/sessions').json()['session_id']
            _ask(client, sid, '甲喜欢谁？', 1)
            _ask(client, sid, '甲喜欢谁？', 1)          # exact
            _ask(client, sid, SEMANTIC_Q1, 2)
            _ask(client, sid, SEMANTIC_Q2, 2)          # semantic
            _ask(client, sid, ID_Q1, 3)
            _ask(client, sid, ID_Q2, 3)                # identifier
            body = client.get('/api/stats/cache').json()
            assert body['summary']['total_requests'] == 6
            assert body['summary']['hit_total'] == 3
            assert body['by_channel']['exact'] == 1
            assert body['by_channel']['semantic'] == 1
            assert body['by_channel']['identifier'] == 1
            assert body['by_channel']['miss'] == 3
            assert body['semantic']['count'] == 1, '仅语义通道写入相似度样本'
            assert body['semantic']['threshold'] == 0.95
            assert body['entries']['total'] == 3

    def test_window_size_is_validated(self, tmp_path: Path) -> None:
        client, _ = _build(tmp_path)
        with client:
            assert client.get('/api/stats/cache?days=0').status_code == 422
            assert client.get('/api/stats/cache?days=91').status_code == 422
            assert client.get('/api/stats/cache?days=30').json()['window_days'] == 30


class TestFeedback:
    def test_feedback_is_recorded(self, tmp_path: Path) -> None:
        client, factory = _build(tmp_path)
        with client:
            resp = client.post('/api/feedback', json={
                'session_id': 1, 'question': '甲喜欢谁？', 'vote': 'up',
                'cache_hit': False, 'cache_channel': 'miss',
            })
            assert resp.status_code == 201
            assert resp.json()['vote'] == 'up'
            with factory() as session:
                rows = list(session.scalars(select(QaFeedback)))
            assert len(rows) == 1 and rows[0].cache_hit is False

    def test_invalid_vote_is_rejected(self, tmp_path: Path) -> None:
        client, _ = _build(tmp_path)
        with client:
            assert client.post('/api/feedback', json={
                'question': 'q', 'vote': 'maybe',
            }).status_code == 422
            assert client.post('/api/feedback', json={
                'question': '', 'vote': 'up',
            }).status_code == 422

    def test_mis_hit_rate_tracks_cached_downvotes(self, tmp_path: Path) -> None:
        client, _ = _build(tmp_path)
        with client:
            # 缓存命中且被点踩 → 误命中
            client.post('/api/feedback', json={
                'question': 'q1', 'vote': 'down', 'cache_hit': True, 'cache_channel': 'semantic',
            })
            # 缓存命中且被点赞 → 正常
            client.post('/api/feedback', json={
                'question': 'q2', 'vote': 'up', 'cache_hit': True, 'cache_channel': 'exact',
            })
            # 非缓存点踩 → 计入非缓存错误率，不污染误命中率
            client.post('/api/feedback', json={
                'question': 'q3', 'vote': 'down', 'cache_hit': False,
            })
            fb = client.get('/api/stats/cache').json()['feedback']
            assert fb['cache_hit_feedback'] == 2
            assert fb['cache_hit_down'] == 1
            assert fb['mis_hit_rate'] == 0.5
            assert fb['non_cache_feedback'] == 1
            assert fb['non_cache_error_rate'] == 1.0

    def test_mis_hit_rate_is_null_without_feedback(self, tmp_path: Path) -> None:
        client, _ = _build(tmp_path)
        with client:
            fb = client.get('/api/stats/cache').json()['feedback']
            assert fb['mis_hit_rate'] is None, '无反馈时应为 null，前端显示「—」'
