"""黄金评测集文件本身的结构守卫测试。

背景：``eval/README.md`` 把「加题」写成纯手工编辑 JSON，而 ``tests/`` 里原有的
只有「代码」测试（metrics / 编排 / 负样本辅助函数），**没有任何测试盯数据集文件**。
踩过的坑都是静默的：

1. ``document`` 写错文件名 —— 上传后 ``doc_map`` 取不到 ``document_id``，该题在
   filtered 口径下退化成全库检索，指标含义悄悄变了，报告上看不出来。
2. ``contains`` 写错一个字 —— 短语在原文 0 命中，gold 集为空，
   ``run_retrieval_eval`` 直接 ``[skip]`` 掉：脚本正常退出、报告正常落盘，
   只是这条题**连分母都不算**。
3. 可答题漏写 ``expect_keywords`` —— ``run_eval`` 退化成「回答非空即通过」，恒真。

另外锁住两条例集设计契约（它们是刻意的，不是巧合）：

* 生成层与检索层共用同一批 id（``love-01`` 等 14 条），**同 id 必同题干** ——
   同一知识点在两层各测一次，靠 id 对齐；题干漂移会让两层的回归对比失去可比性。
* 跨文档用例的 ``documents`` 顺序必须被打乱 —— 顺序是这道题唯一的自变量。

只读 JSON 与语料目录，不连库、不调模型。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

EVAL_DIR = Path(__file__).resolve().parents[1] / 'eval'
CORPUS_DIR = EVAL_DIR / 'uploads'

E2E_GOLD = EVAL_DIR / 'golden_set.json'
MULTI_DOC_GOLD = EVAL_DIR / 'golden_multi_doc.json'
RETRIEVAL_GOLD = EVAL_DIR / 'retrieval_gold.json'
ADVERSARIAL_GOLD = EVAL_DIR / 'retrieval_adversarial_gold.json'

ALL_GOLD_FILES = [E2E_GOLD, MULTI_DOC_GOLD, RETRIEVAL_GOLD, ADVERSARIAL_GOLD]

# README §五 规定的五类：缺哪类，评测就只会告诉你「它能答对」，不会告诉你「它什么时候答错」。
CATEGORY_HINTS = {
    '正向基础': ('正向',),
    '陷阱': ('陷阱',),
    '多跳推断': ('多跳', '推断'),
    '信息不足': ('信息不足', '不可答'),
    '跨文档': ('全库', '跨文档'),
}

ADVERSARIAL_CATEGORIES = frozenset({'专名', '态度', '引语', '时间', '人物', '结构'})


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def _cases(path: Path) -> list[dict]:
    return _load(path)['cases']


def _corpus_files() -> set[str]:
    return {p.name for p in CORPUS_DIR.iterdir() if p.is_file()}


def _scope_names(case: dict) -> list[str]:
    names = list(case.get('documents') or [])
    if case.get('document'):
        names.append(case['document'])
    return names


# --------------------------------------------------------------------------
# 文件形态
# --------------------------------------------------------------------------


@pytest.mark.parametrize('path', ALL_GOLD_FILES, ids=lambda p: p.name)
def test_gold_file_parses_and_has_cases(path: Path):
    data = _load(path)
    assert isinstance(data.get('cases'), list), f'{path.name} 缺少 cases 列表'
    assert data['cases'], f'{path.name} 的 cases 为空'


def test_retrieval_gold_total_cases_matches_len():
    data = _load(RETRIEVAL_GOLD)
    assert data['total_cases'] == len(data['cases'])


# --------------------------------------------------------------------------
# 唯一性：id 是报告里的 per-case 键，同文件内重名会让回归对比张冠李戴
# --------------------------------------------------------------------------


@pytest.mark.parametrize('path', ALL_GOLD_FILES, ids=lambda p: p.name)
def test_case_ids_unique_within_file(path: Path):
    ids = [case['id'] for case in _cases(path)]
    dupes = {i for i in ids if ids.count(i) > 1}
    assert not dupes, f'{path.name} 内重复的 case id: {sorted(dupes)}'


@pytest.mark.parametrize('path', ALL_GOLD_FILES, ids=lambda p: p.name)
def test_questions_unique_within_file(path: Path):
    questions = [case['question'] for case in _cases(path)]
    dupes = {q for q in questions if questions.count(q) > 1}
    assert not dupes, f'{path.name} 内重复的 question: {sorted(dupes)}'


def test_shared_layer_ids_keep_identical_questions():
    """生成层与检索层共用 id 是刻意设计（同一知识点两层各测一次）。

    一旦同 id 的题干漂移，两层报告就无法横向对比 —— 而 ``build_retrieval_gold.py``
    是从端到端结果沉淀 gold 的，这种漂移很容易顺手发生。
    """
    e2e = {c['id']: c['question'] for c in _cases(E2E_GOLD)}
    ret = {c['id']: c['question'] for c in _cases(RETRIEVAL_GOLD)}
    mismatched = [i for i in sorted(set(e2e) & set(ret)) if e2e[i] != ret[i]]
    assert not mismatched, f'两层共用 id 但题干不一致: {mismatched}'


# --------------------------------------------------------------------------
# document 解析：拼错即静默降级，这里必须硬失败
# --------------------------------------------------------------------------


@pytest.mark.parametrize('path', ALL_GOLD_FILES, ids=lambda p: p.name)
def test_referenced_documents_exist_in_corpus(path: Path):
    corpus = _corpus_files()
    missing = [
        f'{case["id"]} -> {doc}'
        for case in _cases(path)
        for doc in _scope_names(case)
        if doc not in corpus
    ]
    assert not missing, f'{path.name} 引用了语料目录里不存在的文件: {missing}'


@pytest.mark.parametrize('path', ALL_GOLD_FILES, ids=lambda p: p.name)
def test_no_case_mixes_document_and_documents(path: Path):
    bad = [c['id'] for c in _cases(path) if c.get('document') and c.get('documents')]
    assert not bad, f'{path.name} 同时设置了 document 与 documents: {bad}'


def _epub_fixtures() -> list[Path]:
    return sorted(CORPUS_DIR.glob('*.epub'))


def test_corpus_still_contains_the_epub_regression_subject():
    """《西游记》是 2026-09-17 扩集引入的大部头语料（1332 chunks / 全库 1520）。

    删掉它，整套检索指标会立刻回到天花板效应（原先 14 条全 1.0）。这里把它钉住。
    """
    fixtures = _epub_fixtures()
    assert len(fixtures) == 1, f'语料目录里应有且仅有 1 个 epub，实际 {[f.name for f in fixtures]}'
    assert fixtures[0].stat().st_size > 1024, 'epub 语料疑似被截断/占位'


def test_epub_fixture_filename_matches_the_referenced_document():
    """文件名本身就是「文档主键」，改名不报错但会让整批用例静默失效。

    ``_upload_books`` 用 ``path.name`` 作为 ``doc_map`` 的键，gold 里的 ``document``
    靠它换成 ``document_id``。所以：

    * 语料改名 → ``document`` 解析不到 → e2e 题退化成全库检索、检索题 gold 为空被
      ``[skip]``（连分母都不算）；
    * gold 里写了语料目录里不存在的名字 → 同上。

    这条断言把两个方向都堵上：真实 fixture 名必须出现在 gold 的引用集合里。
    """
    name = _epub_fixtures()[0].name
    referenced = {
        doc for path in ALL_GOLD_FILES for case in _cases(path) for doc in _scope_names(case)
    }
    assert name in referenced, (
        f'语料里的 epub 名为 {name!r}，但没有任何 gold 用例引用它；'
        f'gold 引用的文档为 {sorted(referenced)}'
    )


# --------------------------------------------------------------------------
# 生成层（golden_set / golden_multi_doc）判定策略必须明确
# --------------------------------------------------------------------------


def _e2e_cases() -> list[dict]:
    return _cases(E2E_GOLD) + _cases(MULTI_DOC_GOLD)


def test_e2e_cases_declare_a_judging_strategy():
    bad = []
    for case in _e2e_cases():
        keywords = case.get('expect_keywords')
        if case.get('expect_unanswerable'):
            # 不可答题靠「诚实拒答标记」判定，给关键词说明写题人把两种口径搞混了
            if keywords:
                bad.append(f'{case["id"]}: 不可答题不应带 expect_keywords')
        elif not keywords:
            # 无关键词时 run_eval 只检查「回答非空」——恒真，等于没判
            bad.append(f'{case["id"]}: 可答题缺少 expect_keywords（会退化成「回答非空即通过」）')
    assert not bad, bad


def test_e2e_category_coverage():
    notes = [case.get('note', '') for case in _cases(E2E_GOLD)]
    missing = [
        label
        for label, hints in CATEGORY_HINTS.items()
        if not any(any(h in note for h in hints) for note in notes)
    ]
    assert not missing, f'黄金集缺少这些类别: {missing}'


def test_multi_doc_cases_shuffle_document_order():
    """documents 顺序是这道题唯一的自变量，全部同序会让「不淹没正确文档」失去意义。"""
    orders = [tuple(case['documents']) for case in _cases(MULTI_DOC_GOLD)]
    assert len(set(orders)) > 1, f'所有跨文档用例的 documents 顺序完全相同: {orders[0]}'


# --------------------------------------------------------------------------
# 检索层：gold 定位方式必须唯一且成立
# --------------------------------------------------------------------------


@pytest.mark.parametrize('path', [RETRIEVAL_GOLD, ADVERSARIAL_GOLD], ids=lambda p: p.name)
def test_retrieval_cases_locator_is_unambiguous(path: Path):
    bad = []
    for case in _cases(path):
        locators = [bool(case.get('contains')), bool(case.get('chunk_ids'))]
        if sum(locators) != 1:
            bad.append(f'{case["id"]}: contains/chunk_ids 应恰好有一个，实际 {locators}')
    assert not bad, bad


@pytest.mark.parametrize('path', [RETRIEVAL_GOLD, ADVERSARIAL_GOLD], ids=lambda p: p.name)
def test_contains_cases_are_scoped_to_a_document(path: Path):
    """``contains`` 是「在库内扫短语」定位 gold —— 不限定文档就等于扫全库。

    这样一条普通短语可能命中几十个 chunk，gold 集膨胀后 recall 恒为 1，指标失去判别力。
    所以短语定位题必须写明 document/document_id。
    """
    bad = [c['id'] for c in _cases(path) if c.get('contains') and not _scope_names(c)]
    assert not bad, f'{path.name} 里 contains 题未限定文档: {bad}'


@pytest.mark.parametrize('path', [RETRIEVAL_GOLD, ADVERSARIAL_GOLD], ids=lambda p: p.name)
def test_documentless_cases_carry_explicit_chunk_ids(path: Path):
    """不指定文档的题在 filtered 与 cross 两个口径下跑的是同一次全库检索。

    允许存在（``cross-01`` 就是刻意的全库题），但必须自带 ``chunk_ids``：
    否则它既无法用短语定位，又会让两个口径的分母重复计数同一个结果。
    """
    scopes = [c['id'] for c in _cases(path) if not _scope_names(c) and not c.get('chunk_ids')]
    assert not scopes, f'{path.name} 里无文档且无 chunk_ids 的用例: {scopes}'


def test_adversarial_categories_are_known():
    unknown = sorted(
        {
            case['category']
            for case in _cases(ADVERSARIAL_GOLD)
            if case.get('category') not in ADVERSARIAL_CATEGORIES
        }
    )
    assert not unknown, f'未知的对抗题类别: {unknown}'
