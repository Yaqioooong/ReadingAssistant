"""CQR 端到端 A/B：召回改进（cap=4 时 4/8 → 6/8）到底有没有变成「答对」。

背景（reports/cqr-referent-selection-20260923.md §8 待办第 1 条）：
09-23 修好了候选**召回**，但那个 4/8 → 6/8 是**确定性中间指标** ——
正确指称进了查询，不等于端到端答对。当时卡在 API 余额（402），A/B 一直没跑。

本探针复用官方 runner（``eval/run_agent_eval.py``）的环境与提问路径，
只筛出 CQR 真正影响的用例类（代词 + 定指回指），用 ``CQR_ENABLED`` 做单变量 A/B。

⚠️ 为什么要重复跑：本集**没有重复运行机制，单次结果不可信**（噪声底约 1–2 题，
见 09-22 的经历）。所以默认 3 次；结论只看**分离与方向**，不看单次的绝对数字。

跑法（两组由脚本自己切，不用手动设环境变量）：
    uv run python scripts/probe_cqr_e2e.py                 # 默认 repeats=3
    uv run python scripts/probe_cqr_e2e.py --repeats 5
    uv run python scripts/probe_cqr_e2e.py --classes pronoun

副作用：会在临时 sqlite 里建会话；不碰生产库。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'eval'))

from run_agent_eval import _ask, _build_env, _load_golden  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent.parent / 'eval' / 'reports'


def _run_arm(cases: list[dict], repeats: int) -> dict[str, list[bool]]:
    """跑一组（CQR 开关由调用方通过环境变量设定），返回 {case_id: [每次是否通过]}。"""
    factory, vector_store = _build_env()
    outcomes: dict[str, list[bool]] = defaultdict(list)
    for _ in range(repeats):
        for case in cases:
            try:
                state = _ask(
                    factory, vector_store, case['question'],
                    case.get('history'), thread=f"cqrab-{case['id']}-{len(outcomes[case['id']])}",
                )
                answer = str(state.get('answer') or '')
                ok = bool(answer) and (case.get('expect_contains') or '') in answer
            except Exception:  # noqa: BLE001 单题失败不该中断整轮
                ok = False
            outcomes[case['id']].append(ok)
    return outcomes


def main() -> None:
    parser = argparse.ArgumentParser(description='CQR 端到端 A/B')
    parser.add_argument('--repeats', type=int, default=3, help='每组的重复次数（默认 3）')
    parser.add_argument('--classes', default='pronoun,anaphora',
                        help='逗号分隔的用例类；默认只跑 CQR 真正影响的类')
    parser.add_argument('--out', default=None, help='结果 JSON 路径')
    args = parser.parse_args()
    wanted = {c.strip() for c in args.classes.split(',') if c.strip()}

    cases = [c for c in _load_golden() if c.get('class') in wanted]
    if not cases:
        print(f'没有匹配 {wanted} 的用例', file=sys.stderr)
        sys.exit(2)

    print(f'用例 {len(cases)} 条（类={sorted({c["class"] for c in cases})}）· 每组重复 {args.repeats} 次')
    print('每组先跑 OFF 再跑 ON，两次之间不共享任何状态\n')

    # ⚠️ 必须禁缓存，否则**重复跑是假的**。
    # ``_build_env()`` 每个臂只调一次（sqlite:///:memory:），于是臂内 N 次重复共享同一个库：
    # 第 0 次算出答案写进 qa_cache，第 1..N-1 次直接命中缓存回放同一个答案。
    # 症状是「检索片段 0 条却给出带 [1][2] 引用的完整答案」，而重复看起来「完全一致」——
    # 那不是稳定，是回放。实测踩到：一度把某题的 3/3 vs 0/3 当成干净回归，
    # 换成独立样本后才发现它是不稳定的（两次真样本 1 过 1 败）。
    # 本仓既有的测量纪律一直是「禁缓存跑 N 次」（见 cqr-*.md 的 A/B 小节）。
    os.environ['CACHE_ENABLED'] = 'false'
    from reading_assistant.config import get_settings

    arms: dict[str, dict[str, list[bool]]] = {}
    for label, enabled in (('OFF', 'false'), ('ON', 'true')):
        os.environ['CQR_ENABLED'] = enabled
        os.environ['CQR_ANAPHORA_ENABLED'] = enabled
        # get_settings 是 lru_cache：换组必须清掉，否则第二组读到的还是第一组的配置
        get_settings.cache_clear()
        if get_settings().cache_enabled:  # 护栏：静默开着缓存会让整轮 A/B 失效
            raise SystemExit('缓存没关掉，重复跑会变成回放 —— 拒绝出结果')
        print(f'--- CQR={label}（缓存已禁）---')
        arms[label] = _run_arm(cases, args.repeats)

    print()
    header = f'{"用例":<34}{"类":<10}{"OFF":>8}{"ON":>8}'
    print(header)
    print('-' * len(header))
    total = {'OFF': 0, 'ON': 0}
    n_runs = len(cases) * args.repeats
    for case in cases:
        cid = case['id']
        cells = {}
        for label in ('OFF', 'ON'):
            hits = sum(arms[label].get(cid, []))
            cells[label] = hits
            total[label] += hits
        mark = '  ← 变好' if cells['ON'] > cells['OFF'] else ('  ← 变差' if cells['ON'] < cells['OFF'] else '')
        print(f'{cid:<34}{case["class"]:<10}{cells["OFF"]:>4}/{args.repeats}{cells["ON"]:>6}/{args.repeats}{mark}')

    print('-' * len(header))
    print(f'{"合计":<34}{"":<10}{total["OFF"]:>4}/{n_runs}{total["ON"]:>6}/{n_runs}')
    print(f'\n方向：CQR 开时 {total["ON"]} 通过，关时 {total["OFF"]} 通过 '
          f'（差 {total["ON"] - total["OFF"]:+d}）')
    print('判读：只看分离与方向。单次数字不可信（本集噪声底约 1–2 题）。')

    out = Path(args.out) if args.out else OUT_DIR / 'cqr_ab.json'
    out.write_text(
        json.dumps(
            {'repeats': args.repeats, 'cases': [c['id'] for c in cases],
             'off': {k: v for k, v in arms['OFF'].items()},
             'on': {k: v for k, v in arms['ON'].items()}},
            ensure_ascii=False, indent=2,
        ),
        encoding='utf-8',
    )
    print(f'结果：{out}')


if __name__ == '__main__':
    main()
