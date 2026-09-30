"""确定性的证据和引用校验。

这里不试图替代 LLM entailment judge，而是提供一层便宜、可回放的护栏：
检查子问题是否有证据、引用编号是否存在、引用是否至少与回答句共享关键文本。
弱支持只进入观测状态，不擅自删除可能正确的答案；明显无效的引用会被移除。
"""

from __future__ import annotations

import re
from typing import TypedDict


class EvidenceJudgement(TypedDict):
    status: str
    evidence_count: int
    lexical_overlap: float
    missing_info: str | None


class AnswerVerification(TypedDict):
    status: str
    invalid_citations: list[int]
    weak_claims: list[str]
    unsupported_claims: list[str]


_CITATION = re.compile(r'\[(\d+)\]')
_WORD = re.compile(r'[A-Za-z0-9_-]+|[\u4e00-\u9fff]')
_NON_CLAIM = ('原文未找到依据', '无法从当前检索证据确认', '抱歉', '请稍后重试')


def _terms(text: str) -> set[str]:
    raw = _WORD.findall(text or '')
    terms = set(raw)
    # 中文单字召回太宽，只在已有连续文本时增加二字片段。
    han = ''.join(c for c in raw if len(c) == 1 and '\u4e00' <= c <= '\u9fff')
    terms.update(han[index : index + 2] for index in range(max(0, len(han) - 1)))
    return {term.lower() for term in terms if len(term) > 1 or term.isalnum()}


def judge_evidence(question: str, chunks: list[dict]) -> EvidenceJudgement:
    if not chunks:
        return {
            'status': 'insufficient',
            'evidence_count': 0,
            'lexical_overlap': 0.0,
            'missing_info': '没有检索到可用原文片段',
        }
    question_terms = _terms(question)
    evidence_terms = _terms(' '.join(str(chunk.get('text') or '') for chunk in chunks))
    overlap = len(question_terms & evidence_terms) / max(1, len(question_terms))
    return {
        # 向量检索可能命中语义同义表达，不能仅凭字面 overlap 否定它。
        'status': 'evidence_ready',
        'evidence_count': len(chunks),
        'lexical_overlap': round(overlap, 4),
        'missing_info': None,
    }


def citation_indexes(answer: str) -> list[int]:
    return [int(value) for value in _CITATION.findall(answer or '')]


def remove_invalid_citations(answer: str, citations: list[dict]) -> tuple[str, list[int]]:
    valid = {int(item.get('index')) for item in citations if item.get('index') is not None}
    invalid = sorted({index for index in citation_indexes(answer) if index not in valid})
    if not invalid:
        return answer, []
    cleaned = _CITATION.sub(
        lambda match: '' if int(match.group(1)) in invalid else match.group(0),
        answer or '',
    )
    return cleaned, invalid


def verify_answer_claims(answer: str, citations: list[dict]) -> AnswerVerification:
    valid = {int(item.get('index')): str(item.get('excerpt') or '') for item in citations}
    invalid = sorted({index for index in citation_indexes(answer) if index not in valid})
    weak: list[str] = []
    unsupported: list[str] = []
    # 引用通常紧跟句号之后（``结论。[1]``），不能先把引用拆成单独句子。
    for sentence in re.split(r'(?<=[。！？!?])(?!\s*\[\d+\])|\n+', answer or ''):
        sentence = sentence.strip()
        if (
            not sentence
            or sentence.startswith('【')
            or any(mark in sentence for mark in _NON_CLAIM)
        ):
            continue
        refs = citation_indexes(sentence)
        if not refs:
            unsupported.append(sentence)
            continue
        if any(ref not in valid for ref in refs):
            continue
        claim_terms = _terms(_CITATION.sub('', sentence))
        support_terms = _terms(' '.join(valid[ref] for ref in refs))
        overlap = len(claim_terms & support_terms) / max(1, len(claim_terms))
        if overlap < 0.08:
            weak.append(sentence)
    status = 'pass'
    if invalid:
        status = 'repaired'
    if weak or unsupported:
        status = 'needs_review' if status == 'pass' else 'repaired_needs_review'
    return {
        'status': status,
        'invalid_citations': invalid,
        'weak_claims': weak,
        'unsupported_claims': unsupported,
    }
