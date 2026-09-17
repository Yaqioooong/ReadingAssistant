"""一键跑全套评测，产出统一 suite 报告 + 回归对比。

三层评测一次跑完
----------------
- ``retrieval`` 检索层：vector vs hybrid 召回质量（Recall/Precision/NDCG/MRR/MAP）
- ``e2e``       生成层：端到端问答质量（pass_rate / 关键词命中 / 拒答识别）
- ``intent``    路由层 + 多轮：该不该走 RAG、能不能记住上文（多轮集已并入）

用法
----
    uv run python -m eval.run_all --mode real                  # 跑全套
    uv run python -m eval.run_all --mode fake --only e2e,intent # 只跑部分（冒烟）
    uv run python -m eval.run_all --mode real --judge llm       # 附加 LLM 判官
    uv run python -m eval.run_all --mode real --no-compare      # 跳过基线对比

产出
----
- ``eval/reports/suite_<ts>.json``  机器可读统一报告
- ``eval/reports/suite_<ts>.md``    人类可读汇总
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

from eval import compare_reports
from eval import judge as judge_mod

EVAL_DIR = Path(__file__).resolve().parent
REPORT_DIR = EVAL_DIR / 'reports'

SUITE_LABELS = {
    'retrieval': '检索层',
    'e2e': '生成层',
    'intent': '路由层 + 多轮',
}
ALL_SUITES = tuple(SUITE_LABELS)


def _run_retrieval(mode: str, limit: int) -> dict:
    from eval import run_retrieval_eval

    if limit:
        # run_retrieval_eval 不支持限制 case 数，明确告知而不是假装生效
        print('[suite] 提示：检索层不支持 --limit，本次将跑完整检索集')
    return run_retrieval_eval.run(mode=mode)


def _run_e2e(mode: str, limit: int) -> dict:
    from eval import run_eval

    return run_eval.run(mode=mode, limit=limit)


def _run_intent(mode: str, limit: int) -> dict:
    from eval import run_intent_eval

    return run_intent_eval.run(mode=mode, limit=limit)


def _normalize_suite_result(name: str, result) -> dict:
    """把各 runner 的返回值统一成报告 dict。

    runner 的返回形态并不一致：``run_intent_eval.run`` 返回 ``(summary, cases)`` 元组
    （它自己的 ``main()`` 就是这么解包的），而 retrieval / e2e 返回完整 dict。
    这里统一归一，避免上游差异炸掉整个汇总报告。

    无法识别时返回带 ``error`` 的 dict，而不是抛异常 —— 单层失败不该中断全套。
    """
    if isinstance(result, dict):
        return result
    if isinstance(result, tuple) and result:
        summary = result[0]
        cases = result[1] if len(result) > 1 else []
        if isinstance(summary, dict):
            # intent 的报告形态是 {'summary': ..., 'cases': ...}
            return {
                'summary': summary,
                'cases': cases if isinstance(cases, list) else [],
            }
    return {
        'error': f'{name} runner 返回了无法识别的结果类型：{type(result).__name__}',
    }


RUNNERS = {
    'retrieval': _run_retrieval,
    'e2e': _run_e2e,
    'intent': _run_intent,
}


def _apply_judge(suite: str, payload: dict, model=None) -> dict | None:
    """对 e2e 结果附加 LLM 判官评分，并返回校准结论。"""
    rows = payload.get('results') or payload.get('cases') or []
    if not rows:
        return None
    scored = 0
    for row in rows:
        contexts = row.get('contexts') or row.get('context') or []
        if isinstance(contexts, str):
            contexts = [contexts]
        verdict = judge_mod.judge_case(
            row.get('question', ''), row.get('answer', ''), contexts, model=model
        )
        row['judge'] = verdict
        if not verdict['error']:
            scored += 1
    return {'suite': suite, 'scored': scored, 'total': len(rows)}


def _demote_headings(markdown: str, levels: int = 2) -> str:
    """把嵌入的 markdown 标题整体降级，避免文档大纲层级错乱。"""
    prefix = '#' * levels
    return '\n'.join(
        f'{prefix}{line}' if line.startswith('#') else line
        for line in markdown.split('\n')
    )


def _overall(suites: dict) -> dict:
    """从各 suite 报告里抽关键指标拼一个总览。"""
    overall: dict = {}
    for name, payload in suites.items():
        if not isinstance(payload, dict) or payload.get('error'):
            continue
        try:
            rtype = compare_reports.detect_type(payload)
        except ValueError:
            continue
        for key, value in compare_reports.extract_metrics(payload, rtype).items():
            overall[f'{name}.{key}'] = value
    return overall


def render_summary_markdown(payload: dict) -> str:
    lines = ['# RAG 评测汇总（suite）', '']
    lines.append(f"- 模式：`{payload['mode']}`")
    lines.append(f"- 生成时间：{payload['generated_at']}")
    lines.append(f"- 总耗时：{payload['duration_s']}s")
    lines.append('')

    lines.append('## 各层概览')
    lines.append('')
    for name in ALL_SUITES:
        payload_suite = payload['suites'].get(name)
        label = SUITE_LABELS[name]
        if payload_suite is None:
            lines.append(f'- **{label}** (`{name}`)：未运行')
        elif payload_suite.get('error'):
            lines.append(f"- **{label}** (`{name}`)：❌ 失败 — {payload_suite['error']}")
        else:
            lines.append(f'- **{label}** (`{name}`)：✅ 完成')
    lines.append('')

    lines.append('## 关键指标')
    lines.append('')
    lines.append('| 指标 | 值 |')
    lines.append('| --- | ---: |')
    for key, value in payload['overall'].items():
        lines.append(f'| `{key}` | {value:.4f} |')
    lines.append('')

    judge_info = payload.get('judge')
    if judge_info:
        lines.append('## LLM 判官')
        lines.append('')
        lines.append(f'- 已评分：{judge_info.get("scored")}/{judge_info.get("total")}')
        calibration = judge_info.get('calibration') or {}
        if calibration.get('available'):
            lines.append(
                f"- 校准：faithfulness kappa={calibration['faithfulness']['kappa']} "
                f"relevancy kappa={calibration['relevancy']['kappa']}"
            )
            if calibration.get('warning'):
                lines.append(f"- ⚠️ {calibration['warning']}")
        else:
            lines.append(f"- 校准：{calibration.get('reason', '未进行')}")
        lines.append('')

    compare = payload.get('compare')
    if compare:
        lines.append('## 相对基线的变化')
        lines.append('')
        lines.append(_demote_headings(compare_reports.render_markdown(compare)))
        lines.append('')

    return '\n'.join(lines)


def run(
    mode: str = 'real',
    only: str | None = None,
    limit: int = 0,
    use_judge: bool = False,
    compare: bool = True,
    pin: bool = False,
) -> dict:
    selected = (
        [s.strip() for s in only.split(',') if s.strip()] if only else list(ALL_SUITES)
    )
    unknown = [s for s in selected if s not in RUNNERS]
    if unknown:
        raise SystemExit(f'未知的 suite：{unknown}，可选 {list(RUNNERS)}')

    started = time.time()
    print(f'[suite] mode={mode} suites={selected} 开始')
    suites: dict = {}

    for name in selected:
        print(f'\n{"=" * 60}\n[suite] ▶ {SUITE_LABELS[name]}（{name}）\n{"=" * 60}')
        try:
            raw = RUNNERS[name](mode, limit)
            suites[name] = _normalize_suite_result(name, raw)
        except Exception as exc:  # noqa: BLE001 - 单个 suite 失败不应中断其余
            print(f'[suite] ✗ {name} 失败：{exc}')
            traceback.print_exc()
            suites[name] = {'error': f'{type(exc).__name__}: {exc}'}

    judge_info = None
    if use_judge and 'e2e' in suites and not suites['e2e'].get('error'):
        print('\n[suite] ▶ LLM 判官评分中…')
        judge_info = _apply_judge('e2e', suites['e2e'])
        if judge_info:
            cases = [
                {
                    'id': row.get('id'),
                    'question': row.get('question'),
                    'answer': row.get('answer'),
                    'contexts': row.get('contexts') or [],
                }
                for row in (suites['e2e'].get('results') or [])
            ]
            judge_info['calibration'] = judge_mod.calibrate(cases=cases)
            if judge_info['calibration'].get('warning'):
                print(f"⚠️  {judge_info['calibration']['warning']}")

    payload = {
        'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'mode': mode,
        'duration_s': round(time.time() - started, 1),
        'requested': selected,
        'suites': suites,
    }
    payload['overall'] = _overall({k: v for k, v in suites.items()})

    REPORT_DIR.mkdir(exist_ok=True)
    stamp = time.strftime('%Y%m%d_%H%M%S')
    json_path = REPORT_DIR / f'suite_{stamp}.json'
    md_path = REPORT_DIR / f'suite_{stamp}.md'

    if judge_info:
        payload['judge'] = judge_info

    # 先把本次结果落盘 —— build_report / pin 都要读到这份文件，
    # 顺序反了会报 FileNotFoundError（曾踩过）。
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8'
    )

    pinned_now = False
    if compare:
        pinned = compare_reports.baseline_path('suite')
        if pin or not pinned.exists():
            compare_reports.pin(latest=str(json_path))
            payload['compare'] = None
            pinned_now = True
        else:
            try:
                payload['compare'] = compare_reports.build_report(pinned, json_path)
            except Exception as exc:  # noqa: BLE001 - 对比失败不影响本次结果落盘
                print(f'[suite] 基线对比失败：{exc}')
                payload['compare'] = None

        # compare 结论回写进报告
        json_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8'
        )

    md_path.write_text(render_summary_markdown(payload), encoding='utf-8')

    print('\n' + '=' * 60)
    print('[suite] 关键指标')
    for key, value in payload['overall'].items():
        print(f'  {key:<38} {value}')
    print(f'\n[suite] JSON 报告：{json_path}')
    print(f'[suite] 汇总文档：{md_path}')
    if payload.get('compare'):
        verdict = '⚠️ 有回归' if payload['compare']['has_regression'] else '✅ 无回归'
        print(f'[suite] 基线对比：{verdict}')
    elif compare and pinned_now:
        print('[suite] 已把本次结果钉为基线，下次运行即可看到 delta')
    elif compare:
        print('[suite] ⚠️ 未取得基线对比结果（详见上方日志），本次仍已落盘')

    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description='一键跑全套 RAG 评测')
    parser.add_argument('--mode', choices=['fake', 'real'], default='real')
    parser.add_argument('--only', help='只跑部分 suite，逗号分隔，如 retrieval,e2e')
    parser.add_argument('--limit', type=int, default=0, help='每个 suite 限制 case 数')
    parser.add_argument('--judge', choices=['off', 'llm'], default='off')
    parser.add_argument('--no-compare', action='store_true', help='跳过基线对比')
    parser.add_argument('--pin', action='store_true', help='把本次结果钉为基线')
    args = parser.parse_args()
    run(
        mode=args.mode,
        only=args.only,
        limit=args.limit,
        use_judge=args.judge == 'llm',
        compare=not args.no_compare,
        pin=args.pin,
    )


if __name__ == '__main__':
    main()
