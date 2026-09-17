"""eval.metrics 纯函数指标库的单元测试。

指标是评测机制的度量衡，算错了整套回归都是假的，所以这里覆盖：
边界（空 gold / 空 retrieved / k 越界）、重复 id、中文分词、
拒答与幻觉两个方向、聚合口径。
"""

from __future__ import annotations

import math

import pytest

from eval.judge import cohen_kappa
from eval.metrics import (
    abstention_metrics,
    aggregate_retrieval_metrics,
    answer_relevancy,
    average_precision_at_k,
    case_retrieval_metrics,
    char_bigrams,
    citation_presence,
    citation_validity,
    compute_case_metrics,
    context_precision,
    context_recall,
    detect_question_type,
    faithfulness,
    hit_at_k,
    is_abstention,
    mrr_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    split_sentences,
    strip_citations,
    summarize_generation,
    token_overlap,
)

# --------------------------------------------------------------------------
# 检索层指标
# --------------------------------------------------------------------------


def test_recall_at_k_basic():
    assert recall_at_k(['a', 'b', 'c'], ['a', 'c'], 3) == 1.0
    assert recall_at_k(['a', 'b', 'c'], ['a', 'c'], 1) == 0.5
    assert recall_at_k(['x', 'y'], ['a'], 2) == 0.0


def test_recall_at_k_k_larger_than_list():
    assert recall_at_k(['a'], ['a'], 10) == 1.0


def test_recall_at_k_empty_inputs():
    assert recall_at_k([], ['a'], 5) == 0.0
    assert recall_at_k(['a'], [], 5) == 0.0
    assert recall_at_k([], [], 5) == 0.0


def test_precision_at_k():
    assert precision_at_k(['a', 'b', 'c', 'd'], ['a', 'c'], 2) == 0.5
    assert precision_at_k(['a', 'b'], ['a', 'b'], 2) == 1.0
    assert precision_at_k(['a'], [], 1) == 0.0
    assert precision_at_k(['a'], ['a'], 0) == 0.0


def test_hit_at_k():
    assert hit_at_k(['x', 'a'], ['a'], 2) == 1.0
    assert hit_at_k(['x', 'y'], ['a'], 2) == 0.0
    assert hit_at_k(['a'], [], 1) == 0.0


def test_mrr_at_k():
    assert mrr_at_k(['a'], ['a'], 3) == 1.0
    assert mrr_at_k(['x', 'a', 'b'], ['a'], 3) == pytest.approx(0.5)
    assert mrr_at_k(['x', 'y'], ['a'], 3) == 0.0
    # 命中位置在 k 之外不算
    assert mrr_at_k(['x', 'y', 'a'], ['a'], 2) == 0.0


def test_ndcg_at_k_perfect_and_partial():
    assert ndcg_at_k(['a'], ['a'], 1) == pytest.approx(1.0)
    # 唯一相关项排在第 2 位，k=2：DCG=1/log2(3)，IDCG=1
    assert ndcg_at_k(['x', 'a'], ['a'], 2) == pytest.approx(1 / math.log2(3), abs=1e-6)
    assert ndcg_at_k(['x', 'y'], ['a'], 2) == 0.0
    assert ndcg_at_k(['a', 'b'], [], 2) == 0.0


def test_ndcg_at_k_capped_at_one():
    # 重复 id 不应让 NDCG 超过 1
    assert ndcg_at_k(['a', 'a'], ['a'], 2) <= 1.0


def test_average_precision_at_k():
    # 命中在第 1 和第 3 位：AP = (1/1 + 2/3) / min(2,3)
    expected = pytest.approx(1.6667 / 2, abs=1e-4)
    assert average_precision_at_k(['a', 'x', 'b'], ['a', 'b'], 3) == expected
    assert average_precision_at_k(['x', 'y'], ['a'], 2) == 0.0
    assert average_precision_at_k(['a'], [], 1) == 0.0


def test_duplicate_ids_do_not_crash():
    for fn in (recall_at_k, precision_at_k, hit_at_k):
        assert 0.0 <= fn(['a', 'a', 'b'], ['a'], 3) <= 1.0


def test_case_retrieval_metrics_keys():
    metrics = case_retrieval_metrics(['a', 'b'], ['a'], ks=(1, 3))
    for k in (1, 3):
        for name in ('recall', 'precision', 'hit', 'mrr', 'ndcg', 'map'):
            assert f'{name}@{k}' in metrics, f'缺少 {name}@{k}'


def test_aggregate_retrieval_metrics_mean():
    rows = [
        case_retrieval_metrics(['a'], ['a'], ks=(5,)),
        case_retrieval_metrics(['x'], ['a'], ks=(5,)),
    ]
    agg = aggregate_retrieval_metrics(rows, ks=(5,))
    assert agg['recall@5'] == pytest.approx(0.5)
    assert agg['hit@5'] == pytest.approx(0.5)


def test_aggregate_retrieval_metrics_empty():
    agg = aggregate_retrieval_metrics([], ks=(5,))
    assert agg['recall@5'] == 0.0
    assert agg['ndcg@5'] == 0.0


# --------------------------------------------------------------------------
# 中文文本工具
# --------------------------------------------------------------------------


def test_char_bigrams_chinese_and_ascii():
    assert '李四' in char_bigrams('李四')
    grams = char_bigrams('hello world')
    assert 'hello' in grams and 'world' in grams


def test_token_overlap_partial_and_full():
    assert token_overlap('张三喜欢李四', '张三喜欢李四') == pytest.approx(1.0)
    assert token_overlap('李四', '张三喜欢李四') == pytest.approx(1.0)
    assert token_overlap('天空是绿色的', '张三喜欢李四') == pytest.approx(0.0)
    assert token_overlap('', '任何文本') == 0.0


def test_token_overlap_short_string_falls_back_to_substring():
    assert token_overlap('李', '张三李四') == 1.0
    assert token_overlap('王', '张三李四') == 0.0


def test_split_sentences():
    assert split_sentences('张三喜欢李四。王五不喜欢。') == ['张三喜欢李四', '王五不喜欢']
    assert split_sentences('') == []


# --------------------------------------------------------------------------
# 生成层指标
# --------------------------------------------------------------------------


def test_context_recall_and_precision():
    assert context_recall(['a', 'b'], ['a', 'x']) == pytest.approx(0.5)
    assert context_precision(['a'], ['a', 'x']) == pytest.approx(0.5)
    assert context_recall([], ['a']) == 0.0
    assert context_precision(['a'], []) == 0.0


def test_context_recall_respects_k():
    assert context_recall(['a'], ['x', 'a'], k=1) == 0.0
    assert context_recall(['a'], ['x', 'a'], k=2) == 1.0


def test_faithfulness_fully_grounded():
    assert faithfulness('张三喜欢李四。', ['张三喜欢李四。']) == pytest.approx(1.0)


def test_faithfulness_half_hallucinated():
    answer = '张三喜欢李四。天空是绿色的。'
    assert faithfulness(answer, ['张三喜欢李四。']) == pytest.approx(0.5)


def test_faithfulness_empty_inputs():
    assert faithfulness('', ['上下文']) == 0.0
    assert faithfulness('答案', []) == 0.0
    assert faithfulness('答案', ['']) == 0.0


def test_citation_validity():
    assert citation_validity(['c1', 'c2'], ['c1']) == pytest.approx(0.5)
    assert citation_validity(['c1'], ['c1']) == pytest.approx(1.0)
    assert citation_validity([], ['c1']) == 0.0


def test_citation_presence():
    assert citation_presence('张三喜欢李四[1]。') == 1.0
    assert citation_presence('张三喜欢李四【2】。') == 1.0
    assert citation_presence('张三喜欢李四。') == 0.0
    assert citation_presence('') == 0.0


def test_detect_question_type():
    assert detect_question_type('张三喜欢谁？') == 'who'
    assert detect_question_type('为什么张三喜欢李四？') == 'why'
    assert detect_question_type('王五喜欢李四吗？') == 'yesno'
    assert detect_question_type('李四几岁？') == 'howmany'


def test_is_abstention():
    assert is_abstention('无法回答该问题') is True
    assert is_abstention('文中没有提到这一点') is True
    assert is_abstention('') is True
    assert is_abstention('张三喜欢李四') is False


def test_answer_relevancy():
    assert answer_relevancy('张三喜欢谁？', '张三喜欢李四。') == pytest.approx(1.0)
    # 拒答不算相关
    assert answer_relevancy('张三喜欢谁？', '无法回答') == 0.0
    assert answer_relevancy('张三喜欢谁？', '') == 0.0


def test_abstention_metrics_both_directions():
    records = [
        {'expect_unanswerable': True, 'answer': '无法回答'},
        {'expect_unanswerable': True, 'answer': '张三喜欢李四'},
        {'expect_unanswerable': False, 'answer': '张三喜欢李四'},
        {'expect_unanswerable': False, 'answer': '信息不足'},
    ]
    result = abstention_metrics(records)
    assert result['abstention_rate'] == pytest.approx(0.5)
    assert result['hallucination_rate'] == pytest.approx(0.5)
    assert result['false_refusal_rate'] == pytest.approx(0.5)
    assert result['n_unanswerable'] == 2
    assert result['n_answerable'] == 2


def test_abstention_metrics_empty():
    result = abstention_metrics([])
    assert result['abstention_rate'] is None
    assert result['false_refusal_rate'] is None
    assert result['hallucination_rate'] is None


# --------------------------------------------------------------------------
# 组合入口
# --------------------------------------------------------------------------


def test_compute_case_metrics_without_answer():
    case = {'id': 'c1', 'question': '张三喜欢谁？', 'chunk_ids': ['a']}
    metrics = compute_case_metrics(case, ['a', 'x'])
    assert metrics['recall@5'] == pytest.approx(1.0)
    assert metrics['context_recall'] == pytest.approx(1.0)
    # 没有答案时不产出答案侧指标
    assert 'faithfulness' not in metrics


def test_compute_case_metrics_with_answer():
    case = {'id': 'c1', 'question': '张三喜欢谁？', 'chunk_ids': ['a']}
    metrics = compute_case_metrics(
        case,
        ['a'],
        answer='张三喜欢李四[1]。',
        contexts=['张三喜欢李四。'],
        cited_ids=['a'],
    )
    assert metrics['faithfulness'] == pytest.approx(1.0)
    assert metrics['citation_presence'] == 1.0
    assert metrics['citation_validity'] == pytest.approx(1.0)
    assert metrics['answer_relevancy'] > 0


def test_compute_case_metrics_accepts_gold_alias():
    case = {'id': 'c1', 'question': 'q', 'gold': ['a']}
    metrics = compute_case_metrics(case, ['a'])
    assert metrics['context_recall'] == pytest.approx(1.0)


def test_summarize_generation_skips_missing():
    rows = [
        {'context_recall': 1.0, 'faithfulness': 1.0},
        {'context_recall': 0.0},
    ]
    summary = summarize_generation(rows)
    assert summary['context_recall'] == pytest.approx(0.5)
    assert summary['faithfulness'] == pytest.approx(1.0)
    assert summary['citation_validity'] is None


def test_summarize_generation_empty():
    summary = summarize_generation([])
    assert summary['context_recall'] is None


def test_strip_citations():
    assert strip_citations('张三喜欢李四[1]。') == '张三喜欢李四。'
    assert strip_citations('张三喜欢李四【2】') == '张三喜欢李四'
    assert strip_citations('无引用的答案') == '无引用的答案'


def test_faithfulness_ignores_citation_markers():
    """回归用例：引用标记不得稀释忠忠实度（曾把 1.0 算成 0.0）。"""
    assert faithfulness('张三喜欢李四[1]。', ['张三喜欢李四。']) == pytest.approx(1.0)
    assert faithfulness('张三喜欢李四【1】【2】。', ['张三喜欢李四。']) == pytest.approx(1.0)


def test_detect_question_type_howmany_measure_words():
    """回归用例：几+量词 应识别为数量提问。"""
    for question in ('李四几岁？', '书里几位主角？', '一共几章？', '来过几次？'):
        assert detect_question_type(question) == 'howmany', question


def test_char_bigrams_drops_punctuation():
    """标点不应产生假 token。"""
    assert char_bigrams('李四。') == char_bigrams('李四')


# --------------------------------------------------------------------------
# 判官校准
# --------------------------------------------------------------------------


def test_cohen_kappa_perfect_agreement():
    assert cohen_kappa([1, 0, 1, 0], [1, 0, 1, 0]) == pytest.approx(1.0)


def test_cohen_kappa_total_disagreement():
    assert cohen_kappa([1, 1, 1, 1], [0, 0, 0, 0]) == 0.0


def test_cohen_kappa_invalid_inputs():
    assert cohen_kappa([], []) == 0.0
    assert cohen_kappa([1, 0], [1]) == 0.0


def test_cohen_kappa_between_zero_and_one():
    kappa = cohen_kappa([1, 1, 0, 0, 1, 0], [1, 0, 0, 1, 1, 0])
    assert 0.0 <= kappa <= 1.0
