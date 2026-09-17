"""eval.hard_negative 纯辅助函数的单元测试。

不需要 embedding / 向量库，只验证 chunk 定位与顺序解析这类易错逻辑。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from eval.hard_negative import _chunk_ordinal, _gold_ids_of


@dataclass
class FakeChunk:
    id: str
    text: str = ''
    metadata: dict = field(default_factory=dict)


def test_chunk_ordinal_from_id_suffix():
    """本项目 chunk_id 形如 doc2-17，末位数字即文档内顺序。"""
    assert _chunk_ordinal(FakeChunk('doc2-17')) == 17
    assert _chunk_ordinal(FakeChunk('doc7-1331')) == 1331
    assert _chunk_ordinal(FakeChunk('doc2-0')) == 0


def test_chunk_ordinal_prefers_metadata():
    chunk = FakeChunk('doc2-17', metadata={'chunk_index': 3})
    assert _chunk_ordinal(chunk) == 3
    chunk2 = FakeChunk('doc2-17', metadata={'index': 5})
    assert _chunk_ordinal(chunk2) == 5


def test_chunk_ordinal_returns_none_when_unparseable():
    assert _chunk_ordinal(FakeChunk('no-digits-here')) is None
    assert _chunk_ordinal(FakeChunk('')) is None


def test_chunk_ordinal_ignores_non_int_metadata():
    """metadata 里的顺序号不是 int 时，应退化到解析 id。"""
    chunk = FakeChunk('doc2-17', metadata={'chunk_index': '17'})
    assert _chunk_ordinal(chunk) == 17


def test_gold_ids_of_accepts_both_keys():
    assert _gold_ids_of({'chunk_ids': ['a', 'b']}) == ['a', 'b']
    assert _gold_ids_of({'gold': ['c']}) == ['c']
    assert _gold_ids_of({}) == []


def test_gold_ids_of_coerces_to_str():
    assert _gold_ids_of({'chunk_ids': [1, 2]}) == ['1', '2']
