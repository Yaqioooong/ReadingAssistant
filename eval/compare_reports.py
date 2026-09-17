"""评测报告基线对比：看清指标涨跌与 per-case 回归。

解决的问题
----------
``eval/reports/`` 会不断堆积报告，但单看一份报告无法判断「这次改动是涨了还是跌了」。
本脚本把两份同类型报告做结构化 diff：

1. **汇总指标 delta 表**：绝对值 / 相对变化 / 方向 / 是否越过回归阈值
2. **per-case 状态迁移**：PASS→FAIL（回归）、FAIL→PASS（修复）、NEW、REMOVED
3. **回归门禁**：``--fail-on-regression`` 时有 PASS→FAIL 则 exit 1，可直接接 CI

用法
----
    uv run python -m eval.compare_reports --list                    # 列出可用报告
    uv run python -m eval.compare_reports                           # 最新 vs 基线
    uv run python -m eval.compare_reports --baseline eval/reports/xxx.json
    uv run python -m eval.compare_reports --pin                     # 把最新报告钉成基线
    uv run python -m eval.compare_reports --out report.md --fail-on-regression
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
REPORT_DIR = EVAL_DIR / 'reports'

# 单份报告类型的判定顺序（从上往下匹配）
_TYPE_ORDER = ('suite', 'retrieval', 'intent', 'e2e')

# 越低越好的指标名（其余默认越高越好）
LOWER_IS_BETTER = frozenset(
    {
        'avg_latency_ms',
        'latency_ms',
        'false_refusal_rate',
        'hallucination_rate',
    }
)

# 这些 summary 字段是计数/元信息，不参与 delta 比较
_NON_METRIC_KEYS = frozenset({'mode', 'cases', 'case_count', 'turn_total', 'total', 'n'})

# 噪声指标族的显著性策略（键为指标名末段，可同时匹配 ``avg_latency_ms`` 与
# ``e2e.avg_latency_ms``）。
#
# 为什么需要：全局 ``--threshold`` 是**绝对值**（默认 0.02），只适配 0~1 的比率型
# 指标。对量纲不同的指标（延迟单位是毫秒），``abs(delta) >= 0.02`` 恒成立 ——
# 实测历史 6 次 real 运行中，**5/5 次相邻对比都被标记**，而同期质量指标
# （pass_rate）恒为 1.0 从未变化。假警报率 100%，会训练人忽略回归信号。
#
# 策略语义：**相对变化与绝对变化必须同时超阈**。
#   rel     —— 相对变化下限（None 表示不要求）
#   min_abs —— 绝对变化下限
#
# 对 ``avg_latency_ms`` 取 rel=0.5 / min_abs=200ms：实测 581~1421ms 的自然波动
# 区间内，相邻对比的相对变化最大 45%（1421→774），不会误报；而真正的 2 倍
# 劣化仍会被拦下。
_THRESHOLD_OVERRIDES: dict[str, dict] = {
    'avg_latency_ms': {'rel': 0.5, 'min_abs': 200.0},
}


# --------------------------------------------------------------------------
# 加载与识别
# --------------------------------------------------------------------------


def load_report(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding='utf-8'))


def detect_type(report: dict) -> str:
    """按顶层 key 判定报告类型。"""
    if 'suites' in report:
        return 'suite'
    if 'per_case' in report:
        return 'retrieval'
    if 'results' in report:
        return 'e2e'
    if 'cases' in report:
        return 'intent'
    raise ValueError(f'无法识别的报告类型，顶层 keys={list(report.keys())}')


def _rows_of(report: dict, rtype: str) -> list[dict]:
    if rtype == 'retrieval':
        return report.get('per_case', []) or []
    if rtype == 'e2e':
        return report.get('results', []) or []
    if rtype == 'intent':
        return report.get('cases', []) or []
    return []


def _case_id(row: dict) -> str:
    return str(row.get('id') or row.get('case_id') or '?')


def list_reports(report_dir: Path | None = None) -> list[Path]:
    """按修改时间倒序列出所有报告。"""
    directory = Path(report_dir or REPORT_DIR)
    if not directory.exists():
        return []
    files = [p.resolve() for p in directory.glob('*.json')]
    return sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)


def baseline_path(rtype: str, report_dir: Path | None = None) -> Path:
    return Path(report_dir or REPORT_DIR) / f'baseline_{rtype}.json'


def pick_latest(rtype: str | None = None, report_dir: Path | None = None) -> Path | None:
    """选最新报告；给了 rtype 则只在该类型里找，并跳过 baseline_* 文件。"""
    for path in list_reports(report_dir):
        if path.name.startswith('baseline_'):
            continue
        try:
            if rtype and detect_type(load_report(path)) != rtype:
                continue
        except (ValueError, json.JSONDecodeError):
            continue
        return path
    return None


def resolve_pair(
    baseline: str | None, latest: str | None, report_dir: Path | None = None
) -> tuple[Path, Path]:
    """决定要比对的两份报告，返回 (基线, 最新)，两者均为绝对路径。

    基线优先级：``--baseline`` 显式指定 > ``baseline_<type>.json`` > 同类型次新报告。
    """
    latest_path = Path(latest).resolve() if latest else pick_latest(report_dir=report_dir)
    if latest_path is None:
        raise SystemExit('没有找到任何报告，先跑一次评测吧。')
    rtype = detect_type(load_report(latest_path))

    if baseline:
        return Path(baseline).resolve(), latest_path

    pinned = baseline_path(rtype, report_dir)
    if pinned.exists():
        return pinned.resolve(), latest_path

    for candidate in list_reports(report_dir):
        if candidate == latest_path or candidate.name.startswith('baseline_'):
            continue
        try:
            if detect_type(load_report(candidate)) == rtype:
                return candidate, latest_path
        except (ValueError, json.JSONDecodeError):
            continue

    raise SystemExit(f'只找到一份 {rtype} 报告，没有可对比的基线。用 --pin 先钉住基线。')


def extract_metrics(report: dict, rtype: str) -> dict[str, float]:
    """把 summary 展平成 ``{metric: value}``；retrieval 类型加 system 前缀。"""
    summary = report.get('summary', {}) or {}
    out: dict[str, float] = {}

    if rtype == 'suite':
        for suite_name, suite_report in (report.get('suites') or {}).items():
            if not isinstance(suite_report, dict):
                continue
            for key, value in extract_metrics(suite_report, detect_type(suite_report)).items():
                out[f'{suite_name}.{key}'] = value
        return out

    if rtype == 'retrieval':
        for key, value in summary.items():
            if not isinstance(value, dict):
                continue
            for metric, val in value.items():
                if isinstance(val, (int, float)):
                    out[f'{key}.{metric}'] = float(val)
        return out

    for key, value in summary.items():
        if key in _NON_METRIC_KEYS or isinstance(value, (dict, list)):
            continue
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            out[key] = float(value)
    return out


def _is_lower_better(metric: str) -> bool:
    return metric.rsplit('.', 1)[-1] in LOWER_IS_BETTER


def _threshold_override(metric: str) -> dict | None:
    return _THRESHOLD_OVERRIDES.get(metric.rsplit('.', 1)[-1])


def _exceeds_threshold(metric: str, base: float, delta: float, threshold: float) -> bool:
    """变化是否达到「显著」——按指标族的量纲选口径。

    - 无 override：沿用全局绝对阈值（``abs(delta) >= threshold``），适配比率型指标
    - 有 override：相对与绝对变化**同时**超阈才算显著（见 ``_THRESHOLD_OVERRIDES``）
    """
    policy = _threshold_override(metric)
    if policy is None:
        return abs(delta) >= threshold

    if abs(delta) < policy.get('min_abs', 0.0):
        return False

    rel_threshold = policy.get('rel')
    if rel_threshold is None or not base:
        return True
    return abs(delta) / abs(base) > rel_threshold


def extract_case_status(report: dict, rtype: str, system: str | None = None) -> dict[str, dict]:
    """返回 ``{case_id: {'pass': bool, 'detail': str}}``。

    retrieval 报告按 ``<system>_mrr > 0`` 判 PASS；e2e/intent 直接读 ``pass`` 字段。
    """
    out: dict[str, dict] = {}
    for row in _rows_of(report, rtype):
        cid = _case_id(row)
        if rtype == 'retrieval':
            chosen = system or _pick_system(report)
            mrr = row.get(f'{chosen}_mrr')
            passed = bool(mrr and float(mrr) > 0)
            detail = f'{chosen}_mrr={mrr}'
        else:
            passed = bool(row.get('pass'))
            detail = str(row.get('fail_reason') or '')
        out[cid] = {'pass': passed, 'detail': detail}
    return out


def _pick_system(report: dict, preferred: str | None = None) -> str:
    """挑一个默认系统：优先 hybrid，其次 summary 里第一个 dict 型 system。"""
    summary = report.get('summary', {}) or {}
    systems = [k for k, v in summary.items() if isinstance(v, dict)]
    if preferred and preferred in systems:
        return preferred
    if 'hybrid' in systems:
        return 'hybrid'
    if systems:
        return systems[0]
    return 'vector'


# --------------------------------------------------------------------------
# 对比
# --------------------------------------------------------------------------


def diff_metrics(
    base: dict[str, float], new: dict[str, float], threshold: float = 0.02
) -> list[dict]:
    """逐指标算 delta。

    返回 ``[{metric, base, new, delta, rel, direction, is_regression, is_improvement}]``。
    """
    rows: list[dict] = []
    for metric in sorted(set(base) | set(new)):
        b = base.get(metric)
        n = new.get(metric)
        if b is None or n is None:
            rows.append(
                {
                    'metric': metric,
                    'base': b,
                    'new': n,
                    'delta': None,
                    'rel': None,
                    'direction': 'NEW' if b is None else 'REMOVED',
                    'is_regression': False,
                    'is_improvement': False,
                }
            )
            continue
        delta = n - b
        rel = (delta / abs(b)) if b else None
        lower_better = _is_lower_better(metric)
        significant = _exceeds_threshold(metric, b, delta, threshold)
        if not significant:
            direction = '='
        elif (delta > 0) != lower_better:
            direction = 'UP'
        else:
            direction = 'DOWN'
        is_regression = significant and (
            (delta < 0) if not lower_better else (delta > 0)
        )
        is_improvement = significant and not is_regression
        rows.append(
            {
                'metric': metric,
                'base': round(b, 4),
                'new': round(n, 4),
                'delta': round(delta, 4),
                'rel': round(rel, 4) if rel is not None else None,
                'direction': direction,
                'is_regression': is_regression,
                'is_improvement': is_improvement,
            }
        )
    return rows


def diff_cases(base: dict[str, dict], new: dict[str, dict]) -> dict[str, list[dict]]:
    """per-case 状态迁移分类。"""
    result: dict[str, list[dict]] = {
        'regressed': [],
        'fixed': [],
        'new': [],
        'removed': [],
        'still_failing': [],
    }
    for cid in sorted(set(base) | set(new)):
        b, n = base.get(cid), new.get(cid)
        if b is None and n is not None:
            result['new'].append({'id': cid, 'pass': n['pass'], 'detail': n['detail']})
        elif n is None and b is not None:
            result['removed'].append({'id': cid, 'pass': b['pass'], 'detail': b['detail']})
        elif b is not None and n is not None:
            if b['pass'] and not n['pass']:
                result['regressed'].append({'id': cid, 'detail': n['detail']})
            elif not b['pass'] and n['pass']:
                result['fixed'].append({'id': cid, 'detail': n['detail']})
            elif not n['pass']:
                result['still_failing'].append({'id': cid, 'detail': n['detail']})
    return result


def build_report(
    base_path: Path, new_path: Path, threshold: float = 0.02, system: str | None = None
) -> dict:
    base_report = load_report(base_path)
    new_report = load_report(new_path)
    rtype = detect_type(new_report)
    base_type = detect_type(base_report)
    if rtype != base_type:
        raise SystemExit(f'报告类型不一致：基线是 {base_type}，最新是 {rtype}')

    chosen_system = _pick_system(new_report, system) if rtype == 'retrieval' else None
    metric_rows = diff_metrics(
        extract_metrics(base_report, base_type), extract_metrics(new_report, rtype), threshold
    )
    case_diff = diff_cases(
        extract_case_status(base_report, base_type, chosen_system),
        extract_case_status(new_report, rtype, chosen_system),
    )
    return {
        'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'type': rtype,
        'system': chosen_system,
        'threshold': threshold,
        'baseline': str(base_path),
        'latest': str(new_path),
        'metrics': metric_rows,
        'cases': case_diff,
        'has_regression': bool(case_diff['regressed'])
        or any(r['is_regression'] for r in metric_rows),
        'regressed_metrics': [r['metric'] for r in metric_rows if r['is_regression']],
    }


# --------------------------------------------------------------------------
# 输出
# --------------------------------------------------------------------------


def render_markdown(result: dict) -> str:
    lines: list[str] = []
    lines.append('# 评测基线对比')
    lines.append('')
    type_label = f'`{result["type"]}`'
    if result['system']:
        type_label += f'（系统 `{result["system"]}`）'
    lines.append(f'- 类型：{type_label}')
    lines.append(f'- 基线：`{result["baseline"]}`')
    lines.append(f'- 最新：`{result["latest"]}`')
    lines.append(f'- 回归阈值：{result["threshold"]}')
    lines.append(f'- 时间：{result["generated_at"]}')
    lines.append('')

    lines.append('## 汇总指标 delta')
    lines.append('')
    lines.append('| 指标 | 基线 | 最新 | Δ | 相对 | 判定 |')
    lines.append('| --- | ---: | ---: | ---: | ---: | :---: |')
    for row in result['metrics']:
        if row['delta'] is None:
            mark = f"`{row['direction']}`"
            base_s = '-' if row['base'] is None else f"{row['base']:.4f}"
            new_s = '-' if row['new'] is None else f"{row['new']:.4f}"
            lines.append(f"| `{row['metric']}` | {base_s} | {new_s} | - | - | {mark} |")
            continue
        mark = '🔴 回归' if row['is_regression'] else ('🟢 提升' if row['is_improvement'] else '—')
        rel = f"{row['rel']:+.1%}" if row['rel'] is not None else '-'
        sign = '+' if row['delta'] >= 0 else ''
        lines.append(
            f"| `{row['metric']}` | {row['base']:.4f} | {row['new']:.4f} | "
            f"{sign}{row['delta']:.4f} | {rel} | {mark} |"
        )
    lines.append('')

    cases = result['cases']
    lines.append('## per-case 状态迁移')
    lines.append('')
    lines.append(
        f'- 🔴 回归 PASS→FAIL：**{len(cases["regressed"])}**'
        f'  🟢 修复 FAIL→PASS：**{len(cases["fixed"])}**'
        f'  仍失败：{len(cases["still_failing"])}'
        f'  新增：{len(cases["new"])}  移除：{len(cases["removed"])}'
    )
    lines.append('')
    if cases['regressed']:
        lines.append('### 🔴 回归用例（优先排查）')
        lines.append('')
        for item in cases['regressed']:
            lines.append(f"- `{item['id']}` — {item['detail'] or '无失败原因'}")
        lines.append('')
    if cases['fixed']:
        lines.append('### 🟢 修复用例')
        lines.append('')
        for item in cases['fixed']:
            lines.append(f"- `{item['id']}`")
        lines.append('')
    if cases['new']:
        lines.append('### 新增用例')
        lines.append('')
        for item in cases['new']:
            flag = 'PASS' if item['pass'] else 'FAIL'
            lines.append(f"- `{item['id']}` [{flag}]")
        lines.append('')
    if cases['removed']:
        lines.append('### 移除用例')
        lines.append('')
        for item in cases['removed']:
            lines.append(f"- `{item['id']}`")
        lines.append('')

    verdict = '⚠️ 检测到回归' if result['has_regression'] else '✅ 无回归'
    lines.append(f'## 结论：{verdict}')
    lines.append('')
    if result['regressed_metrics']:
        lines.append('回归指标：' + ', '.join(f'`{m}`' for m in result['regressed_metrics']))
        lines.append('')
    return '\n'.join(lines)


def render_console(result: dict) -> None:
    print(f"[compare] type={result['type']} system={result['system'] or '-'}")
    print(f"  baseline: {result['baseline']}")
    print(f"  latest  : {result['latest']}")
    print()
    print(f"  {'指标':<26} {'基线':>9} {'最新':>9} {'Δ':>9} {'相对':>8}  判定")
    print('  ' + '-' * 74)
    for row in result['metrics']:
        if row['delta'] is None:
            print(f"  {row['metric']:<26} {'-':>9} {'-':>9} {'-':>9} {'-':>8}  {row['direction']}")
            continue
        if row['is_regression']:
            mark = '🔴 回归'
        elif row['is_improvement']:
            mark = '🟢 提升'
        else:
            mark = '—'
        rel = f"{row['rel']:+.1%}" if row['rel'] is not None else '-'
        print(
            f"  {row['metric']:<26} {row['base']:>9.4f} {row['new']:>9.4f} "
            f"{row['delta']:>+9.4f} {rel:>8}  {mark}"
        )
    print()
    cases = result['cases']
    print(
        f"  per-case: 🔴回归 {len(cases['regressed'])} | 🟢修复 {len(cases['fixed'])} | "
        f"仍失败 {len(cases['still_failing'])} | "
        f"新增 {len(cases['new'])} | 移除 {len(cases['removed'])}"
    )
    if cases['regressed']:
        print('\n  🔴 回归用例：')
        for item in cases['regressed']:
            print(f"    - {item['id']}: {item['detail'] or '无失败原因'}")
    print()
    print('  ' + ('⚠️  检测到回归' if result['has_regression'] else '✅ 无回归'))


def pin(report_dir: Path | None = None, latest: str | None = None) -> Path:
    """把某份报告钉成该类型的基线。"""
    source = Path(latest) if latest else pick_latest(report_dir=report_dir)
    if source is None:
        raise SystemExit('没有可钉的报告。')
    rtype = detect_type(load_report(source))
    target = baseline_path(rtype, report_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    print(f'[pin] {rtype} 基线已固定：{source} → {target}')
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description='评测报告基线对比')
    parser.add_argument('--baseline', help='基线报告路径')
    parser.add_argument('--latest', help='最新报告路径（默认为最新一份）')
    parser.add_argument('--threshold', type=float, default=0.02, help='回归阈值，默认 0.02')
    parser.add_argument('--system', help='retrieval 报告对比的系统（默认 hybrid）')
    parser.add_argument('--out', help='把 markdown 报告写到该路径')
    parser.add_argument('--json', action='store_true', help='打印机器可读 JSON')
    parser.add_argument('--fail-on-regression', action='store_true', help='有回归则 exit 1')
    parser.add_argument('--pin', action='store_true', help='把最新报告钉为基线后退出')
    parser.add_argument('--list', action='store_true', help='列出可用报告后退出')
    args = parser.parse_args()

    if args.list:
        for path in list_reports():
            try:
                rtype = detect_type(load_report(path))
            except (ValueError, json.JSONDecodeError):
                rtype = 'unknown'
            stamp = time.strftime('%Y-%m-%d %H:%M', time.localtime(path.stat().st_mtime))
            pinned = ' [baseline]' if path.name.startswith('baseline_') else ''
            print(f'  {stamp}  {rtype:<10} {path.name}{pinned}')
        return

    if args.pin:
        pin(latest=args.latest)
        return

    base_path, new_path = resolve_pair(args.baseline, args.latest)
    result = build_report(base_path, new_path, threshold=args.threshold, system=args.system)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        render_console(result)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(render_markdown(result), encoding='utf-8')
        print(f'\n[compare] markdown 已写入 {out_path}')

    if args.fail_on_regression and result['has_regression']:
        sys.exit(1)


if __name__ == '__main__':
    main()
