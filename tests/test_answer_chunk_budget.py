"""答案 prompt 的检索片段预算：整条累积 + 首条保留 + 与 citations 编号同源。

片段侧先前**一个 token 都不数**：历史那侧有 `_select_window`（4000 token 预算），
而注入答案的检索片段只有 `top_k` / `max_chunks` 这种按条数的限制 ——
语料一长，prompt 总量就没有上界（实测 top_k=16 时最坏 18.3k token）。

这里每条断言钉住一个具体约束，其中「prompt 的 [n] 与 citations 的 index 同源」
是本次改动的核心不变式，用真实图跑，避免有人只改一处就以为改好了。
"""
from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient
from langchain_core.embeddings import Embeddings

from reading_assistant.api import create_app
from reading_assistant.config import get_settings
from reading_assistant.graph.qa import (
    _CHUNK_LABEL_OVERHEAD,
    _build_answer_prompt,
    _count_tokens,
    _select_answer_chunks,
)
from reading_assistant.storage import create_db_engine, create_session_factory, init_db
from reading_assistant.storage.models import Document
from reading_assistant.storage.vector_store import InMemoryVectorStore, StoredChunk


def _chunk(text: str | None, index: int = 0, document_id: int = 1) -> dict:
    return {
        'chunk_id': f'c{index}',
        'text': text,
        'document_id': document_id,
        'chapter': '正文',
        'chapter_index': 0,
    }


def _fit_budget(*texts: str) -> int:
    """刚好装得下这些片段的预算（含片段前缀开销）。"""
    return sum(_count_tokens(t) + _CHUNK_LABEL_OVERHEAD for t in texts)


class TestSelectAnswerChunks:
    """片段选择：从最相关的一端整条收，超预算即停。"""

    def test_keeps_everything_within_budget(self) -> None:
        chunks = [_chunk('甲' * 20, i) for i in range(3)]
        assert len(_select_answer_chunks(chunks, _fit_budget('甲' * 20) * 3)) == 3

    def test_preserves_input_order(self) -> None:
        """顺序即 citation 编号，必须原样返回（不排序、不重排）。"""
        chunks = [_chunk(f'第{i}段内容', i) for i in range(4)]
        picked = _select_answer_chunks(chunks, 10_000)
        assert [c['chunk_id'] for c in picked] == ['c0', 'c1', 'c2', 'c3']

    def test_drops_from_the_tail_not_the_head(self) -> None:
        """丢的是末尾最弱的几条 —— chunk[0] 是检索排序第一名。"""
        chunks = [_chunk('甲' * 20, i) for i in range(5)]
        one = _fit_budget('甲' * 20)
        # 预算恰好等于 3 条的开销：第 3 条进得去，第 4 条「再加一条就超预算」
        picked = _select_answer_chunks(chunks, one * 3)
        assert [c['chunk_id'] for c in picked] == ['c0', 'c1', 'c2']

    def test_always_keeps_the_first_chunk(self) -> None:
        """首条无条件保留：否则预算偏小时片段会被清空，
        而 judge 刚按「有片段」把路由判进 answer，立刻自相矛盾。"""
        chunks = [_chunk('甲' * 500, 0), _chunk('乙' * 500, 1)]
        picked = _select_answer_chunks(chunks, 10)
        assert len(picked) == 1

    def test_never_truncates_a_chunk_in_half(self) -> None:
        """留下的都是完整片段（与 _select_window 同形），不做半块截断。"""
        texts = ['甲' * 30, '乙' * 40, '丙' * 50]
        chunks = [_chunk(t, i) for i, t in enumerate(texts)]
        picked = _select_answer_chunks(chunks, _fit_budget(texts[0], texts[1]))
        assert [c['text'] for c in picked] == texts[:2]

    def test_zero_budget_means_no_limit(self) -> None:
        """与 _clip_tokens 的 cap<=0 同义：预算未配置视为不限。"""
        chunks = [_chunk('甲' * 300, i) for i in range(4)]
        assert len(_select_answer_chunks(chunks, 0)) == 4

    def test_empty_input(self) -> None:
        assert _select_answer_chunks([], 1000) == []

    def test_none_text_does_not_crash(self) -> None:
        chunks = [_chunk(None, 0), _chunk('乙' * 10, 1)]
        assert len(_select_answer_chunks(chunks, 1000)) == 2

    def test_does_not_mutate_input(self) -> None:
        chunks = [_chunk('甲' * 300, i) for i in range(3)]
        snapshot = list(chunks)
        picked = _select_answer_chunks(chunks, 50)
        assert chunks == snapshot, '入参列表不得被修改'
        assert picked is not chunks, '应返回新列表'


class TestDefaultBudgetCoversMeasuredWorstCase:
    """护栏：默认预算不得小到「对当前语料主动裁掉检索证据」。

    实测（1520 块真实语料）chunk token p50=882 / p90=1051 / max=1174；
    top_k=16 时合计 均值 13.1k / 最坏 18.3k。默认值必须覆盖最坏值 ——
    这个预算是护栏而不是压缩器，本项目的教训是静默降质最贵。
    """

    def test_default_budget_holds_sixteen_max_size_chunks(self) -> None:
        # 1520 块语料实测单块 max=1174 token，这里按这个量级造最坏形状
        text = '西游记' * 400
        per = _count_tokens(text) + _CHUNK_LABEL_OVERHEAD
        assert per >= 1174, f'这条测试的前提是单块达到实测上限量级，实测 {per}'
        chunks = [_chunk(text, i) for i in range(16)]
        budget = get_settings().answer_chunk_token_budget
        assert len(_select_answer_chunks(chunks, budget)) == 16, (
            f'默认预算 {budget} 装不下 16×{per} token，会主动裁掉检索证据'
        )


class TestPromptNumberingFollowsChunks:
    """prompt 里的 [n] 编号必须落在**传进去的那份列表**上。"""

    def test_labels_are_one_based_and_contiguous(self) -> None:
        chunks = [_chunk('甲' * 10, 0), _chunk('乙' * 10, 1)]
        prompt = _build_answer_prompt('问题', chunks)
        assert '[1]' in prompt and '[2]' in prompt
        assert '[3]' not in prompt

    def test_pruned_list_does_not_leak_dropped_labels(self) -> None:
        chunks = [_chunk('甲' * 40, i) for i in range(5)]
        kept = _select_answer_chunks(chunks, _fit_budget('甲' * 40) * 2)
        prompt = _build_answer_prompt('问题', kept)
        labels = sorted(int(n) for n in re.findall(r'^\[(\d+)\]$', prompt, re.M))
        assert labels == [1, 2], '裁剪后的编号必须是 1..N，不能保留被丢掉的编号'


class _FakeLLM:
    """按 prompt 内容路由的最小假模型（无 bind_tools → 走纯文本兜底）。"""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def invoke(self, prompt: str) -> SimpleNamespace:
        self.prompts.append(prompt)
        if '意图分类任务' in prompt:
            return SimpleNamespace(content='book')
        return SimpleNamespace(content='这是回答，不带引用标注。')


class _FlatEmb(Embeddings):
    def embed_query(self, text: str) -> list[float]:
        return [0.99, 0.1]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


def _make_client(tmp_path: Path, llm: _FakeLLM, chunks: int) -> TestClient:
    engine = create_db_engine('sqlite:///:memory:')
    init_db(engine)
    factory = create_session_factory(engine)
    with factory() as session:
        session.add(
            Document(
                filename='d1.docx', title='d1', file_hash='f1',
                content_hash='c1', index_status='indexed',
            )
        )
        session.commit()
    store = InMemoryVectorStore()
    store.add(
        [
            StoredChunk(
                id=f'd1-{i}',
                text='西游记' * 30,
                metadata={'document_id': 1, 'chapter': '正文', 'chapter_index': 0},
                embedding=[1.0, 0.0],
            )
            for i in range(chunks)
        ]
    )
    return TestClient(
        create_app(
            session_factory=factory,
            vector_store=store,
            llm=llm,
            embedding_model=_FlatEmb(),
            upload_dir=tmp_path / 'uploads',
        )
    )


class TestCitationIndexesMatchPromptLabels:
    """核心不变式（真图跑）：answer 节点里 prompt 与 citations 必须同源。

    回归形状：只把 `_build_answer_prompt` 的入参换成裁剪后的列表、
    却忘了 citations 循环还在用 `state['chunks']` —— 编号就会错位，
    用户看到的引用编号与答案正文的 [n] 对不上。
    """

    def test_pruned_chunks_shrink_citations_too(self, tmp_path: Path, monkeypatch) -> None:
        total = 5
        per = _count_tokens('西游记' * 30) + _CHUNK_LABEL_OVERHEAD
        budget = per * 3  # 恰好只装得下 3 条

        base = get_settings()
        monkeypatch.setattr(
            'reading_assistant.graph.qa.get_settings',
            lambda: base.model_copy(update={'answer_chunk_token_budget': budget}),
        )
        llm = _FakeLLM()
        client = _make_client(tmp_path, llm, total)
        with client:
            sid = client.post('/api/sessions').json()['session_id']
            resp = client.post(
                f'/api/sessions/{sid}/messages', json={'question': '西游记讲了什么？'}
            ).json()

        book_prompts = [p for p in llm.prompts if '原文片段：' in p]
        assert book_prompts, 'book 分支应产生带片段的作答 prompt'
        labels = sorted(int(n) for n in re.findall(r'^\[(\d+)\]$', book_prompts[-1], re.M))
        assert labels == [1, 2, 3], f'预算 {budget}（per={per}）应恰好保留 3 条，实际 {labels}'

        citations = resp['citations']
        assert len(citations) == 3, f'citations 必须与 prompt 片段数一致，实际 {len(citations)}'
        assert [c['index'] for c in citations] == [1, 2, 3]
        assert total > len(labels), '前提：入库片段数要多于被保留的条数，否则测不到裁剪'
