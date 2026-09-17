"""引用列表与答案 [n] 标记的一致性测试：只展示答案真正引用过的片段。"""

from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient
from langchain_core.embeddings import Embeddings

from reading_assistant.api import create_app
from reading_assistant.graph.qa import _renumber_citations, _used_citation_indexes
from reading_assistant.storage import create_db_engine, create_session_factory, init_db
from reading_assistant.storage.models import Document
from reading_assistant.storage.vector_store import InMemoryVectorStore, StoredChunk

CHUNK_COUNT = 3


class CiteLLM:
    """按预设答案作答，用于断言引用列表与答案标记是否一致。"""

    def __init__(self, answer: str) -> None:
        self.answer = answer

    def invoke(self, prompt: str) -> SimpleNamespace:
        return SimpleNamespace(content=self.answer)


class FlatEmb(Embeddings):
    """查询与文档块同向 → 全部片段都过检索阈值，保证召回恒为 CHUNK_COUNT 条。"""

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


def _client_with(answer: str, tmp_path: Path) -> TestClient:
    engine = create_db_engine('sqlite:///:memory:')
    init_db(engine)
    factory = create_session_factory(engine)
    with factory() as session:
        session.add(
            Document(
                filename='d.txt',
                title='d',
                file_hash='f1',
                content_hash='c1',
                index_status='indexed',
            )
        )
        session.commit()
    store = InMemoryVectorStore()
    store.add(
        [
            StoredChunk(
                id=f'doc1-{i}',
                text=f'片段{i}正文',
                metadata={'document_id': 1, 'chapter': '正文', 'chapter_index': 0},
                embedding=[1.0, 0.0],
            )
            for i in range(CHUNK_COUNT)
        ]
    )
    return TestClient(
        create_app(
            session_factory=factory,
            vector_store=store,
            llm=CiteLLM(answer),
            embedding_model=FlatEmb(),
            upload_dir=tmp_path / 'uploads',
        )
    )


def _ask(client: TestClient, sid: str) -> dict:
    return client.post(
        f'/api/sessions/{sid}/messages',
        json={'question': '正文讲了什么？', 'document_ids': [1]},
    ).json()


class TestUsedCitationIndexes:
    def test_parses_in_order_and_dedupes(self) -> None:
        assert _used_citation_indexes('依据[1]，另有[3]，再看[1]。', 6) == [1, 3]

    def test_ignores_out_of_range_and_empty(self) -> None:
        assert _used_citation_indexes('见[9]且[0]', 3) == []
        assert _used_citation_indexes('没有标记', 3) == []
        assert _used_citation_indexes(None, 3) == []
        assert _used_citation_indexes('见[1]', 0) == []


class TestRenumberCitations:
    def test_renumbers_and_rewrites_markers(self) -> None:
        citations = [{'index': 2}, {'index': 5}]
        content, out = _renumber_citations('先[2]后[5]', citations)
        assert content == '先[1]后[2]'
        assert [c['index'] for c in out] == [1, 2]

    def test_swap_does_not_chain(self) -> None:
        """[1]→[2] 与 [2]→[1] 互换时必须一次替换完成，不能串联污染。"""
        citations = [{'index': 2}, {'index': 1}]
        content, out = _renumber_citations('A[1] B[2]', citations)
        assert content == 'A[2] B[1]'
        assert [c['index'] for c in out] == [1, 2]

    def test_untouched_when_already_sequential(self) -> None:
        citations = [{'index': 1}, {'index': 2}]
        content, out = _renumber_citations('见[1][2]', citations)
        assert content == '见[1][2]'
        assert out == citations

    def test_unknown_markers_are_left_alone(self) -> None:
        citations = [{'index': 3}]
        content, out = _renumber_citations('见[3]，另见[9]', citations)
        assert content == '见[1]，另见[9]'
        assert [c['index'] for c in out] == [1]

    def test_empty_inputs(self) -> None:
        assert _renumber_citations(None, [{'index': 3}]) == (None, [{'index': 3}])
        assert _renumber_citations('见[1]', []) == ('见[1]', [])


class TestCitationFiltering:
    def test_only_cited_chunks_are_returned(self, tmp_path: Path) -> None:
        client = _client_with('镇元子住在万寿山五庄观[1]。', tmp_path)
        with client:
            sid = client.post('/api/sessions').json()['session_id']
            body = _ask(client, sid)
            assert len(body['citations']) == 1, body['citations']
            assert body['citations'][0]['index'] == 1
            assert body['citations'][0]['chunk_id'] == 'doc1-0'

    def test_sparse_markers_are_renumbered(self, tmp_path: Path) -> None:
        """答案引用 [1][3] → 列表重排为 [1][2]，答案标记同步改写。"""
        client = _client_with('前半见[1]，后半见[3]。', tmp_path)
        with client:
            sid = client.post('/api/sessions').json()['session_id']
            body = _ask(client, sid)
            assert [c['index'] for c in body['citations']] == [1, 2]
            assert [c['chunk_id'] for c in body['citations']] == ['doc1-0', 'doc1-2']
            assert '[2]' in body['answer'] and '[3]' not in body['answer'], body['answer']

    def test_single_remote_citation_becomes_first(self, tmp_path: Path) -> None:
        """只引用第 3 条时，列表应显示 [1]（而不是让人以为丢了前两条）。"""
        client = _client_with('通天河拦住去路[3]。', tmp_path)
        with client:
            sid = client.post('/api/sessions').json()['session_id']
            body = _ask(client, sid)
            assert len(body['citations']) == 1
            assert body['citations'][0]['index'] == 1
            assert body['citations'][0]['chunk_id'] == 'doc1-2'
            assert '[1]' in body['answer'] and '[3]' not in body['answer']

    def test_answer_without_markers_keeps_all(self, tmp_path: Path) -> None:
        client = _client_with('这是一段没有任何引用标记的回答。', tmp_path)
        with client:
            sid = client.post('/api/sessions').json()['session_id']
            body = _ask(client, sid)
            assert len(body['citations']) == CHUNK_COUNT, '未标注引用时保留全部，避免丢失溯源线索'

    def test_out_of_range_marker_falls_back_to_all(self, tmp_path: Path) -> None:
        client = _client_with('引用了不存在的片段[9]。', tmp_path)
        with client:
            sid = client.post('/api/sessions').json()['session_id']
            body = _ask(client, sid)
            assert len(body['citations']) == CHUNK_COUNT
