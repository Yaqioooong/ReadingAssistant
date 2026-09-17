"""统一评测指标库：检索层 IR 指标 + 生成层 RAG 指标。

设计原则
--------
- 全部为**纯函数**（输入 list/dict → 输出 float/dict），无 IO、无全局状态，可独立单测。
- 检索层对齐经典 IR 定义（binary relevance）：
  Recall@k / Precision@k / Hit@k / MRR@k / NDCG@k / MAP@k。
- 生成层对齐 RAGAS 的语义（context recall / context precision / faithfulness /
  answer relevancy / abstention），但用**确定性规则**实现，
  保证离线评测可重复、零成本、无外部依赖。
- 中文按字符 bigram 处理，ASCII 按空白分词。

术语
----
- ``gold``      标注的应当被召回的 chunk_id 列表
- ``retrieved`` 检索系统实际返回的 chunk_id 列表（已按相关性降序）
- ``answer``    模型生成的答案文本
- ``contexts``  喂给模型的上下文文本列表（通常来自 citations 对应的 chunk）
- ``cited_ids`` 答案中显式引用的 chunk_id 列表
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence

DEFAULT_KS: tuple[int, ...] = (5, 10, 20)

# 拒答信号词：命中任一即认为模型选择了「不回答」
_ABSTENTION_PATTERNS = (
    '无法回答',
    '无法确定',
    '不能回答',
    '没有提到',
    '没有提及',
    '未提及',
    '文中未',
    '书中未',
    '文中没有',
    '信息不足',
    '资料不足',
    '不足以回答',
    '不知道',
    '不清楚',
    '找不到',
    '未能找到',
    '没有相关',
    '不支持该问题',
    '抱歉',
    '无从得知',
    '没有足够',
)

# 疑问类型 → 答案形态的粗粒度特征
_HOWMANY_RE = re.compile(r'几[个次岁位条年天本页章遍件种名]')

_QUESTION_TYPES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ('who', ('谁', '哪个人', '哪位', '谁人')),
    ('when', ('什么时候', '何时', '哪一年', '几年')),
    ('where', ('哪里', '哪儿', '何地', '在什么地方')),
    ('why', ('为什么', '为何', '原因', '怎么会')),
    ('how', ('怎么', '如何', '怎样', '怎么样')),
    ('howmany', ('多少', '几个', '几次', '多久')),
    ('yesno', ('吗', '是不是', '是否', '对不对', '有没有')),
)

_ASCII_WORD_RE = re.compile(r'[a-zA-Z0-9]+')
_SENTENCE_SPLIT_RE = re.compile(r'[。！？!?\n;；]+')
# 引用标记（[1] / 【2】 / (3)），属于噪声，不参与内容 token 统计
_CITATION_RE = re.compile(r'[\[【(（]\s*\d+\s*[\]】)）]')
# 标点/符号：构 bigram 前剔除，避免「四。」「四[」这类假 token 稀释重叠度
_PUNCT_RE = re.compile(
    '[\\s。，、；：！？…—·「」『』（）《》【】\\[\\]{}()<>,.;:!?\'"“”‘’\\-_/\\\\|~`@#$%^&*+=]'
)


# --------------------------------------------------------------------------
# 基础工具
# --------------------------------------------------------------------------


def _dedup(items: Sequence[str]) -> list[str]:
    """按出现顺序去重（保持排名语义）。"""
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _topk(items: Sequence[str], k: int) -> list[str]:
    if k <= 0:
        return []
    return list(items)[:k]


def strip_citations(text: str) -> str:
    """去掉答案里的引用标记，避免 ``[1]`` 被当成内容 token 稀释指标。"""
    return _CITATION_RE.sub('', text or '')


def char_bigrams(text: str) -> set[str]:
    """中文字符 bigram；英文/数字按词切分后保留整词。"""
    cleaned = _PUNCT_RE.sub('', text or '')
    grams: set[str] = {cleaned[i : i + 2] for i in range(len(cleaned) - 1)}
    grams.update(word.lower() for word in _ASCII_WORD_RE.findall(text or ''))
    return grams


def token_overlap(needle: str, haystack: str) -> float:
    """needle 的 token 有多少比例出现在 haystack 中，返回 0~1。

    needle 为空时返回 0.0；对极短串（无法构成 bigram）退化为子串包含判断。
    """
    needle_grams = char_bigrams(needle)
    if not needle_grams:
        stripped = (needle or '').strip()
        if not stripped:
            return 0.0
        return 1.0 if stripped in (haystack or '') else 0.0
    hay_grams = char_bigrams(haystack)
    if not hay_grams:
        return 0.0
    return len(needle_grams & hay_grams) / len(needle_grams)


def split_sentences(text: str) -> list[str]:
    """按中英文句末标点切句，过滤空白句。"""
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text or '') if s.strip()]


# --------------------------------------------------------------------------
# 检索层：单点指标
# --------------------------------------------------------------------------


def recall_at_k(retrieved: Sequence[str], gold: Sequence[str], k: int) -> float:
    """Recall@k = |top-k ∩ gold| / |gold|。gold 为空返回 0.0。"""
    gold_set = set(gold)
    if not gold_set:
        return 0.0
    hit = len(set(_topk(retrieved, k)) & gold_set)
    return hit / len(gold_set)


def precision_at_k(retrieved: Sequence[str], gold: Sequence[str], k: int) -> float:
    """Precision@k = |top-k ∩ gold| / k。k<=0 或 retrieved 为空返回 0.0。"""
    if k <= 0:
        return 0.0
    top = _topk(retrieved, k)
    if not top:
        return 0.0
    gold_set = set(gold)
    return len([cid for cid in top if cid in gold_set]) / k


def hit_at_k(retrieved: Sequence[str], gold: Sequence[str], k: int) -> float:
    """Hit@k：top-k 里只要有一个相关就为 1.0，否则 0.0。"""
    gold_set = set(gold)
    if not gold_set:
        return 0.0
    return 1.0 if set(_topk(retrieved, k)) & gold_set else 0.0


def mrr_at_k(retrieved: Sequence[str], gold: Sequence[str], k: int) -> float:
    """MRR@k = 1 / 第一个相关结果的排名；未命中返回 0.0。"""
    gold_set = set(gold)
    if not gold_set:
        return 0.0
    for rank, cid in enumerate(_topk(retrieved, k), start=1):
        if cid in gold_set:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], gold: Sequence[str], k: int) -> float:
    """NDCG@k（binary gain）= DCG@k / IDCG@k。

    DCG@k  = Σ_{i=1..k} rel_i / log2(i+1)
    IDCG@k = Σ_{i=1..min(|gold|,k)} 1 / log2(i+1)
    """
    gold_set = set(gold)
    if not gold_set or k <= 0:
        return 0.0
    top = _topk(retrieved, k)
    dcg = 0.0
    for index, cid in enumerate(top, start=1):
        if cid in gold_set:
            dcg += 1.0 / math.log2(index + 1)
    ideal_hits = min(len(gold_set), k, len(top)) if top else 0
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    if idcg == 0:
        return 0.0
    return min(dcg / idcg, 1.0)


def average_precision_at_k(retrieved: Sequence[str], gold: Sequence[str], k: int) -> float:
    """AP@k = (Σ_{命中位 i} Precision@i) / min(|gold|, k)。"""
    gold_set = set(gold)
    if not gold_set or k <= 0:
        return 0.0
    top = _topk(retrieved, k)
    if not top:
        return 0.0
    hits = 0
    total = 0.0
    for index, cid in enumerate(top, start=1):
        if cid in gold_set:
            hits += 1
            total += hits / index
    denominator = min(len(gold_set), k)
    if denominator == 0:
        return 0.0
    return min(total / denominator, 1.0)


# 兼容别名：MAP@k 单条即 AP@k
map_at_k = average_precision_at_k


def case_retrieval_metrics(
    retrieved: Sequence[str], gold: Sequence[str], ks: Sequence[int] = DEFAULT_KS
) -> dict:
    """单条 case 的全部检索指标，key 形如 ``recall@5`` / ``ndcg@10``。"""
    result: dict[str, float] = {}
    for k in ks:
        result[f'recall@{k}'] = recall_at_k(retrieved, gold, k)
        result[f'precision@{k}'] = precision_at_k(retrieved, gold, k)
        result[f'hit@{k}'] = hit_at_k(retrieved, gold, k)
        result[f'mrr@{k}'] = mrr_at_k(retrieved, gold, k)
        result[f'ndcg@{k}'] = ndcg_at_k(retrieved, gold, k)
        result[f'map@{k}'] = average_precision_at_k(retrieved, gold, k)
    return result


def aggregate_retrieval_metrics(per_case: Sequence[dict], ks: Sequence[int] = DEFAULT_KS) -> dict:
    """对每-case 指标 dict 求算术平均。空列表返回全 0。"""
    keys: list[str] = []
    for k in ks:
        keys.extend(
            [
                f'recall@{k}',
                f'precision@{k}',
                f'hit@{k}',
                f'mrr@{k}',
                f'ndcg@{k}',
                f'map@{k}',
            ]
        )
    if not per_case:
        return dict.fromkeys(keys, 0.0)
    out: dict[str, float] = {}
    for key in keys:
        values = [float(row.get(key, 0.0)) for row in per_case]
        out[key] = round(sum(values) / len(values), 4) if values else 0.0
    return out


# --------------------------------------------------------------------------
# 生成层：上下文质量
# --------------------------------------------------------------------------


def context_recall(
    gold_ids: Sequence[str], retrieved_ids: Sequence[str], k: int | None = None
) -> float:
    """上下文召回率 = |gold ∩ retrieved| / |gold|，语义等价于检索层 Recall@k。"""
    gold_set = set(gold_ids)
    if not gold_set:
        return 0.0
    pool = _dedup(retrieved_ids) if k is None else _topk(_dedup(retrieved_ids), k)
    return len(set(pool) & gold_set) / len(gold_set)


def context_precision(
    gold_ids: Sequence[str], retrieved_ids: Sequence[str], k: int | None = None
) -> float:
    """上下文精确率 = top-k 中属于 gold 的 chunk 占比。"""
    gold_set = set(gold_ids)
    pool = _dedup(retrieved_ids) if k is None else _topk(_dedup(retrieved_ids), k)
    if not pool:
        return 0.0
    return len([cid for cid in pool if cid in gold_set]) / len(pool)


# --------------------------------------------------------------------------
# 生成层：答案质量
# --------------------------------------------------------------------------


def faithfulness(answer: str, contexts: Sequence[str], min_overlap: float = 0.6) -> float:
    """答案忠实度（规则代理版 RAGAS faithfulness）。

    把答案切句，逐句与**全部上下文拼接文本**做 token 重叠；
    重叠比例 >= ``min_overlap`` 视为该句被上下文支撑。
    返回被支撑句子的占比。答案为空返回 0.0。

    注意：这是代理指标，衡量「答案是否能在上下文中找到依据」，
    不等价于 LLM 判官对事实一致性的判断，但零成本且完全可重复。
    """
    sentences = split_sentences(strip_citations(answer))
    if not sentences:
        return 0.0
    haystack = '\n'.join(contexts or [])
    if not haystack.strip():
        return 0.0
    supported = sum(1 for s in sentences if token_overlap(s, haystack) >= min_overlap)
    return supported / len(sentences)


def citation_validity(cited_ids: Sequence[str], valid_ids: Sequence[str]) -> float:
    """引用有效率 = 引用的 chunk_id 中真实存在的比例（防幻觉引用）。

    没有任何引用时返回 0.0（表示未提供引用）。
    """
    cited = _dedup(cited_ids)
    if not cited:
        return 0.0
    valid = set(valid_ids)
    return len([cid for cid in cited if cid in valid]) / len(cited)


def citation_presence(answer: str) -> float:
    """答案是否带引用标记（``[1]`` / ``【1】`` / ``(doc:1)`` 形态），1.0 或 0.0。"""
    if not answer:
        return 0.0
    return 1.0 if _CITATION_RE.search(answer or '') else 0.0


def detect_question_type(question: str) -> str:
    """粗粒度疑问类型识别，返回 who/when/where/why/how/howmany/yesno/unknown。"""
    text = question or ''
    for name, keywords in _QUESTION_TYPES:
        if any(kw in text for kw in keywords):
            return name
        # 「几」+ 量词（几岁/几位/几条…）也属于数量提问
        if name == 'howmany' and _HOWMANY_RE.search(text):
            return name
    return 'unknown'


def is_abstention(answer: str) -> bool:
    """判断答案是否为「拒答/无法回答」。空答案也算拒答。"""
    text = (answer or '').strip()
    if not text:
        return True
    return any(pattern in text for pattern in _ABSTENTION_PATTERNS)


def answer_relevancy(question: str, answer: str) -> float:
    """答案相关性（规则代理版）。

    打分构成：
    - 答案非空且不是拒答：0.4
    - 疑问类型与答案形态匹配：0.4
      （who→答案含 2~4 字人名样式；when→含数字/年/月/日；
        howmany→含数字；yesno→含是/否/不/没有/对）
    - 与问题存在 token 重叠（答案确实在谈问题里的实体）：0.2

    返回值 0~1，越高越相关。
    """
    text = (answer or '').strip()
    if not text or is_abstention(text):
        return 0.0

    score = 0.4
    qtype = detect_question_type(question)
    if qtype == 'who':
        name_like = re.search(r'[\u4e00-\u9fa5]{2,4}(是|喜欢|讨厌|想要)', text)
        matched = bool(name_like) or len(text) <= 30
    elif qtype == 'when':
        matched = bool(re.search(r'\d|[零一二三四五六七八九十]年|月|日|岁', text))
    elif qtype == 'howmany':
        matched = bool(re.search(r'\d', text)) or any(
            c in text for c in '一二三四五六七八九十两半'
        )
    elif qtype == 'yesno':
        matched = any(kw in text for kw in ('是', '否', '不', '没', '对', '错'))
    elif qtype in ('why', 'how', 'where'):
        matched = len(text) >= 6
    else:
        matched = len(text) >= 2
    if matched:
        score += 0.4

    if token_overlap(question or '', text) > 0.1:
        score += 0.2
    return round(min(score, 1.0), 4)


def abstention_metrics(records: Sequence[dict]) -> dict:
    """拒答质量汇总。

    ``records`` 每项需含：``expect_unanswerable``(bool) 与 ``answer``(str)。

    返回：
    - ``abstention_rate``        不可答题中正确拒答的比例（越高越好）
    - ``false_refusal_rate``     可答题中被误拒答的比例（越低越好）
    - ``hallucination_rate``     不可答题中**未拒答**（即凭空作答）的比例（越低越好）

    三者分母各自独立，避免互相稀释。
    """
    unanswerable = [r for r in records if r.get('expect_unanswerable')]
    answerable = [r for r in records if not r.get('expect_unanswerable')]

    def _rate(items: Sequence[dict], predicate) -> float | None:
        if not items:
            return None
        return round(sum(1 for r in items if predicate(r)) / len(items), 4)

    return {
        'abstention_rate': _rate(unanswerable, lambda r: is_abstention(r.get('answer', ''))),
        'false_refusal_rate': _rate(answerable, lambda r: is_abstention(r.get('answer', ''))),
        'hallucination_rate': _rate(
            unanswerable, lambda r: not is_abstention(r.get('answer', ''))
        ),
        'n_unanswerable': len(unanswerable),
        'n_answerable': len(answerable),
    }


def compute_case_metrics(
    case: dict,
    retrieved: Sequence[str],
    answer: str = '',
    contexts: Sequence[str] = (),
    cited_ids: Sequence[str] = (),
    ks: Sequence[int] = DEFAULT_KS,
) -> dict:
    """单条完整 case 的检索 + 生成指标合体，供各评测脚本统一调用。"""
    gold_ids = case.get('gold') or case.get('chunk_ids') or []
    valid_ids = list(retrieved)
    out: dict = {}
    out.update(case_retrieval_metrics(retrieved, gold_ids, ks=ks))
    out['context_recall'] = round(context_recall(gold_ids, retrieved), 4)
    out['context_precision'] = round(context_precision(gold_ids, retrieved), 4)
    if answer or contexts:
        out['faithfulness'] = round(faithfulness(answer, contexts), 4)
        out['answer_relevancy'] = answer_relevancy(case.get('question', ''), answer)
        out['citation_presence'] = citation_presence(answer)
        out['citation_validity'] = round(citation_validity(cited_ids, valid_ids), 4)
    return out


def summarize_generation(case_metrics: Sequence[dict]) -> dict:
    """把逐-case 生成指标聚合成汇总。缺失字段自动跳过。"""
    keys = (
        'context_recall',
        'context_precision',
        'faithfulness',
        'answer_relevancy',
        'citation_presence',
        'citation_validity',
    )
    out: dict = {}
    for key in keys:
        values = [float(row[key]) for row in case_metrics if row.get(key) is not None]
        out[key] = round(sum(values) / len(values), 4) if values else None
    return out
