"""问题分析与多问题拆解。

拆解保持确定性、零 LLM 成本：显式编号、分号和多个问号才触发多问题分支。
保守拆分比把一个复杂问题误拆成互不相关的问题更安全；后续可在此模块接入
结构化 LLM planner，而不改变 QA 图的 fan-out/fan-in 契约。
"""

from __future__ import annotations

import re
from typing import TypedDict


class SubQuestion(TypedDict, total=False):
    id: str
    question: str
    status: str
    chunks: list[dict]
    answer: str | None
    citations: list[dict]
    error: str | None


_NUMBERED = re.compile(r'(?:^|\n|\s)(?:\(?\d+[.)]|[一二三四五六七八九十]+、)\s*')


def _clean_part(part: str) -> str:
    return re.sub(r'^[\s,，、:：]+|[\s,，、:：]+$', '', part).strip()


def split_question(question: str) -> list[str]:
    """拆分明确的并列问题；无法确定时返回原问题。"""
    text = (question or '').strip()
    if not text:
        return []

    marks = list(_NUMBERED.finditer(text))
    if len(marks) >= 2:
        parts = []
        for index, match in enumerate(marks):
            end = marks[index + 1].start() if index + 1 < len(marks) else len(text)
            part = _clean_part(text[match.end() : end])
            if part:
                parts.append(part)
        if len(parts) >= 2:
            return parts

    question_parts = [_clean_part(p) for p in re.split(r'(?<=[?？])\s*', text)]
    question_parts = [p for p in question_parts if p]
    if len(question_parts) >= 2:
        return question_parts

    semi_parts = [_clean_part(p) for p in re.split(r'[;；]\s*', text)]
    semi_parts = [p for p in semi_parts if p]
    if len(semi_parts) >= 2 and all(len(p) >= 4 for p in semi_parts):
        return semi_parts

    return [text]


def analyze_question(question: str) -> tuple[bool, list[SubQuestion]]:
    parts = split_question(question)
    is_multi = len(parts) > 1
    return is_multi, [
        {
            'id': f'q{index}',
            'question': part,
            'status': 'pending',
            'chunks': [],
            'answer': None,
            'citations': [],
            'error': None,
        }
        for index, part in enumerate(parts, start=1)
    ]
