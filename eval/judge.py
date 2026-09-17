"""LLM-as-judge：可选的语义级判官，带人工校准门禁。

定位
----
``metrics.py`` 里的指标是**规则代理**：便宜、可重复、零成本，但只能衡量
「答案能不能在上下文里找到依据」，没法判断语义等价、事实矛盾、答非所问的细微差别。

本模块补上语义判官，但**默认关闭**：

- 规则指标永远是主口径（确定性、可复现、能做 CI 门禁）
- LLM judge 是**辅助口径**，只在 ``--judge llm`` 时启用
- judge 输出必须先跟人工标注对齐（Cohen's kappa），kappa 低于阈值就判定该 judge
  结果不可信 —— 这是 RAGAS / LLM-judge 类方案的标配护栏

用法
----
    uv run python -m eval.judge --calibrate          # 用 human_labels.json 校准
    uv run python -m eval.judge --demo "问题" "答案" "上下文"
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
HUMAN_LABELS = EVAL_DIR / 'human_labels.json'

# kappa 低于该值判定 judge 不可用
KAPPA_THRESHOLD = 0.6

_JUDGE_PROMPT = """你是严格的 RAG 评测员。请只依据给定上下文，判断答案质量。

【问题】
{question}

【检索到的上下文】
{contexts}

【模型答案】
{answer}

请从两个维度打分，输出**纯 JSON**（不要 markdown 代码块、不要多余文字）：
{{
  "faithfulness": 0 或 1,   // 答案的每个事实断言是否都能在上下文中找到依据？有杜撰则为 0
  "relevancy": 0 或 1,      // 答案是否直接回应了问题？答非所问、只复述上下文则为 0
  "reason": "一句话说明判据"
}}
"""

_JSON_RE = re.compile(r'\{.*\}', re.DOTALL)


def _extract_json(text: str) -> dict:
    """从模型输出里抠出 JSON，容忍 markdown 代码块和前后废话。"""
    if not text:
        raise ValueError('judge 返回为空')
    match = _JSON_RE.search(text)
    if not match:
        raise ValueError(f'judge 输出中没有 JSON：{text[:200]}')
    return json.loads(match.group(0))


def _normalize_score(value) -> int:
    """把 judge 的各种输出形态归一到 0/1。"""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return 1 if value >= 0.5 else 0
    text = str(value).strip().lower()
    return 1 if text in ('1', 'true', 'yes', '是', 'pass', 'ok', '高') else 0


def judge_case(
    question: str,
    answer: str,
    contexts: list[str] | tuple[str, ...],
    model=None,
    max_retries: int = 2,
) -> dict:
    """调用 LLM 判官，返回 ``{faithfulness, relevancy, reason, error}``。

    模型构建失败或解析失败时**不抛异常**，返回带 ``error`` 的结果，
    让调用方可以降级回规则指标。
    """
    prompt = _JUDGE_PROMPT.format(
        question=question or '',
        contexts='\n---\n'.join(contexts or []) or '（无上下文）',
        answer=answer or '（空答案）',
    )
    last_error = ''
    for attempt in range(max_retries + 1):
        try:
            if model is None:
                from reading_assistant.model.factory import get_chat_model

                model = get_chat_model()
            response = model.invoke(prompt)
            raw = getattr(response, 'content', response)
            if isinstance(raw, list):  # 部分实现返回 content blocks
                raw = ''.join(
                    block.get('text', '') if isinstance(block, dict) else str(block)
                    for block in raw
                )
            parsed = _extract_json(str(raw))
            return {
                'faithfulness': _normalize_score(parsed.get('faithfulness')),
                'relevancy': _normalize_score(parsed.get('relevancy')),
                'reason': str(parsed.get('reason', ''))[:300],
                'error': '',
            }
        except Exception as exc:  # noqa: BLE001 - 判官失败必须可降级
            last_error = f'{type(exc).__name__}: {exc}'
            if attempt == max_retries:
                break
    return {'faithfulness': None, 'relevancy': None, 'reason': '', 'error': last_error}


def cohen_kappa(labels_a: list[int], labels_b: list[int]) -> float:
    """Cohen's kappa：衡量两个标注者（judge vs 人工）的一致性，扣除随机巧合。

    kappa = (Po - Pe) / (1 - Pe)；样本为空或完全随机一致时返回 0.0。
    """
    if not labels_a or len(labels_a) != len(labels_b):
        return 0.0
    n = len(labels_a)
    po = sum(1 for a, b in zip(labels_a, labels_b, strict=True) if a == b) / n
    pa1 = sum(labels_a) / n
    pb1 = sum(labels_b) / n
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    if pe >= 1.0:
        return 1.0 if po >= 1.0 else 0.0
    return round((po - pe) / (1 - pe), 4)


def calibrate(
    human_path: Path | None = None,
    cases: list[dict] | None = None,
    model=None,
) -> dict:
    """用人工标注校准 judge。

    ``human_labels.json`` 格式::

        {"love-01": {"faithfulness": 1, "relevancy": 1}, ...}

    ``cases`` 需含 ``id`` / ``question`` / ``answer`` / ``contexts``。
    返回各维度的一致率与 kappa，并给出是否可用的结论。
    """
    human_path = Path(human_path or HUMAN_LABELS)
    if not human_path.exists():
        return {
            'available': False,
            'reason': f'缺少人工标注文件 {human_path}，无法校准',
        }
    human = json.loads(human_path.read_text(encoding='utf-8'))
    cases = cases or []

    judged_f: list[int] = []
    human_f: list[int] = []
    judged_r: list[int] = []
    human_r: list[int] = []
    errors = 0

    for case in cases:
        cid = str(case.get('id', ''))
        label = human.get(cid)
        if not label:
            continue
        result = judge_case(
            case.get('question', ''),
            case.get('answer', ''),
            case.get('contexts', []),
            model=model,
        )
        if result['error']:
            errors += 1
            continue
        if label.get('faithfulness') is not None:
            judged_f.append(int(result['faithfulness']))
            human_f.append(int(label['faithfulness']))
        if label.get('relevancy') is not None:
            judged_r.append(int(result['relevancy']))
            human_r.append(int(label['relevancy']))

    def _agreement(a: list[int], b: list[int]) -> float | None:
        if not a:
            return None
        return round(sum(1 for x, y in zip(a, b, strict=True) if x == y) / len(a), 4)

    kappa_f = cohen_kappa(judged_f, human_f)
    kappa_r = cohen_kappa(judged_r, human_r)
    usable = bool(judged_f or judged_r) and max(kappa_f, kappa_r) >= KAPPA_THRESHOLD

    report = {
        'available': True,
        'n_labeled': len(human),
        'n_compared': len(judged_f),
        'errors': errors,
        'faithfulness': {
            'agreement': _agreement(judged_f, human_f),
            'kappa': kappa_f,
        },
        'relevancy': {
            'agreement': _agreement(judged_r, human_r),
            'kappa': kappa_r,
        },
        'kappa_threshold': KAPPA_THRESHOLD,
        'judge_usable': usable,
    }

    if not usable:
        report['warning'] = (
            f'judge 与人工一致性不足（kappa<{KAPPA_THRESHOLD}），'
            '该 judge 结果不可作为验收依据，请以规则指标为准'
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description='LLM 判官与校准')
    parser.add_argument('--calibrate', action='store_true', help='用 human_labels.json 校准')
    parser.add_argument('--labels', help='人工标注文件路径')
    parser.add_argument('--cases', help='待判 case 的 json 路径（含 id/question/answer/contexts）')
    parser.add_argument(
        '--demo', nargs=3, metavar=('QUESTION', 'ANSWER', 'CONTEXT'), help='单条试跑'
    )
    args = parser.parse_args()

    if args.demo:
        question, answer, context = args.demo
        print(json.dumps(judge_case(question, answer, [context]), ensure_ascii=False, indent=2))
        return

    cases = []
    if args.cases:
        payload = json.loads(Path(args.cases).read_text(encoding='utf-8'))
        cases = payload.get('cases') or payload.get('results') or []
    result = calibrate(Path(args.labels) if args.labels else None, cases)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result.get('warning'):
        print(f"\n⚠️  {result['warning']}")


if __name__ == '__main__':
    main()
