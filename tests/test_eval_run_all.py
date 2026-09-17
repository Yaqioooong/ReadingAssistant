"""eval.run_all 汇总编排的单元测试。

这里锁住两个真实踩过的坑：
1. 各 runner 返回值形态不一致 —— ``run_intent_eval.run`` 返回 ``(summary, cases)`` 元组，
   直接当 dict 用会 AttributeError。
2. 落盘与 compare 的顺序 —— build_report 要读最新的报告文件，报告没落盘就对比会
   FileNotFoundError。
"""

from __future__ import annotations

from eval.run_all import (
    _demote_headings,
    _normalize_suite_result,
    _overall,
    render_summary_markdown,
)

# --------------------------------------------------------------------------
# 返回值归一化
# --------------------------------------------------------------------------


def test_normalize_dict_passthrough():
    payload = {'summary': {'mode': 'fake'}, 'per_case': []}
    assert _normalize_suite_result('retrieval', payload) is payload


def test_normalize_intent_tuple():
    """回归用例：intent runner 返回元组，必须被归一成报告 dict。"""
    summary = {'mode': 'fake', 'route_accuracy': 1.0}
    cases = [{'id': 'mt-01', 'pass': True}]
    result = _normalize_suite_result('intent', (summary, cases))
    assert result['summary'] == summary
    assert result['cases'] == cases


def test_normalize_tuple_with_non_list_cases():
    result = _normalize_suite_result('intent', ({'mode': 'fake'}, None))
    assert result['cases'] == []


def test_normalize_tuple_with_non_dict_summary():
    result = _normalize_suite_result('weird', ('not-a-dict', []))
    assert 'error' in result


def test_normalize_garbage_returns_error_not_raises():
    for garbage in ('字符串', 42, None, [], object()):
        result = _normalize_suite_result('x', garbage)
        assert isinstance(result, dict)
        assert 'error' in result


# --------------------------------------------------------------------------
# 汇总渲染
# --------------------------------------------------------------------------


def _suite_payload() -> dict:
    """构造一份三层齐全的 suite payload（intent 部分来自元组归一）。"""
    return {
        'generated_at': '2026-09-16 15:00:00',
        'mode': 'fake',
        'duration_s': 1.0,
        'requested': ['retrieval', 'e2e', 'intent'],
        'suites': {
            'retrieval': {
                'summary': {'mode': 'fake', 'hybrid': {'recall@5': 0.5}},
                'per_case': [],
            },
            'e2e': {
                'summary': {'mode': 'fake', 'pass_rate': 1.0},
                'results': [],
            },
            'intent': _normalize_suite_result(
                'intent', ({'mode': 'fake', 'route_accuracy': 1.0}, [])
            ),
        },
        'overall': {'retrieval.hybrid.recall@5': 0.5},
    }


def test_render_summary_handles_all_suite_shapes():
    """回归用例：三层形态不一致时汇总渲染不得抛异常。"""
    markdown = render_summary_markdown(_suite_payload())
    assert '检索层' in markdown
    assert '生成层' in markdown
    assert '路由层' in markdown
    assert '✅ 完成' in markdown


def test_render_summary_marks_failed_suite():
    payload = _suite_payload()
    payload['suites']['e2e'] = {'error': 'RuntimeError: boom'}
    markdown = render_summary_markdown(payload)
    assert '❌ 失败' in markdown
    assert 'boom' in markdown


def test_render_summary_marks_not_run_suite():
    payload = _suite_payload()
    del payload['suites']['e2e']
    markdown = render_summary_markdown(payload)
    assert '未运行' in markdown


def test_render_summary_includes_compare_section():
    payload = _suite_payload()
    payload['compare'] = {
        'type': 'suite',
        'system': None,
        'threshold': 0.02,
        'baseline': 'a.json',
        'latest': 'b.json',
        'generated_at': '2026-09-16 15:00:00',
        'metrics': [],
        'cases': {
            'regressed': [],
            'fixed': [],
            'new': [],
            'removed': [],
            'still_failing': [],
        },
        'has_regression': False,
        'regressed_metrics': [],
    }
    markdown = render_summary_markdown(payload)
    assert '相对基线的变化' in markdown


# --------------------------------------------------------------------------
# 总览抽取
# --------------------------------------------------------------------------


def test_overall_skips_failed_suites():
    suites = {
        'retrieval': {'summary': {'mode': 'x', 'hybrid': {'recall@5': 0.5}}, 'per_case': []},
        'e2e': {'error': 'boom'},
    }
    overall = _overall(suites)
    assert 'retrieval.hybrid.recall@5' in overall
    assert not any(key.startswith('e2e.') for key in overall)


def test_overall_empty():
    assert _overall({}) == {}


def test_demote_headings():
    source = '# 标题\n\n## 子标题\n\n| 表格 | 行 |'
    demoted = _demote_headings(source, levels=2)
    assert '### 标题' in demoted
    assert '#### 子标题' in demoted
    assert '| 表格 | 行 |' in demoted
