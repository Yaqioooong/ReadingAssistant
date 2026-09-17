"""人工抽检工具：给自动生成的 gold 做抽样复核，堵住「自己出题自己判」的偏差。

背景
----
``build_retrieval_gold.py`` 从端到端跑通的 case 里沉淀出 gold chunk，
本质上是**用模型自己的输出当标准答案**。一旦检索退步，这些 case 直接跌出 PASS，
gold 集反而缩小 —— 误差不会被发现，只会被掩盖。所以必须有人工抽检闭环：

1. 分层抽样（按题型/note 分组），固定随机种子，保证可复现
2. 产出人读的 markdown + 人填的 csv
3. ``--apply`` 把人工判定回写 gold，剔除被标错的 gold chunk，并留审计记录

用法
----
    uv run python -m eval.sample_review --gold eval/retrieval_gold.json --ratio 0.2
    uv run python -m eval.sample_review --apply eval/reports/review_20260916_120000.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import time
from collections import defaultdict
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
REPORT_DIR = EVAL_DIR / 'reports'
DEFAULT_GOLD = EVAL_DIR / 'retrieval_gold.json'
AUDIT_FILE = REPORT_DIR / 'review_audit.jsonl'

# 人工判定词的归一化映射
VERDICT_ALIASES = {
    '对': 'ok',
    '正确': 'ok',
    'ok': 'ok',
    'pass': 'ok',
    'y': 'ok',
    'yes': 'ok',
    '1': 'ok',
    '错': 'bad',
    '错误': 'bad',
    'bad': 'bad',
    'fail': 'bad',
    'n': 'bad',
    'no': 'bad',
    '0': 'bad',
    '不确定': 'unsure',
    'unsure': 'unsure',
    '?': 'unsure',
}

CSV_FIELDS = (
    'case_id',
    'question',
    'unit',
    'gold_chunk_id',
    'chunk_preview',
    'auto_status',
    'verdict',
    'suggestion',
    'reviewer',
    'reviewed_at',
)

_PREFIX_RE = re.compile(r'^([a-zA-Z]+)')


def load_gold(path: Path) -> tuple[dict, Path]:
    """读 gold 文件，返回 (完整 payload, 路径)。"""
    path = Path(path)
    return json.loads(path.read_text(encoding='utf-8')), path


def stratify_key(case: dict) -> str:
    """分层依据：优先 note，其次 id 的字母前缀（如 adv-/love-），最后 all。"""
    note = (case.get('note') or '').strip()
    if note:
        return note
    match = _PREFIX_RE.match(str(case.get('id', '')))
    return match.group(1) if match else 'all'


def review_units(case: dict) -> list[dict]:
    """把一条 case 拆成若干待复核单元。

    - 检索 gold（有 chunk_ids）：每个 gold chunk 一个单元
    - 端到端 gold（有 expect_keywords）：整题一个单元
    """
    gold_ids = case.get('chunk_ids') or case.get('gold') or []
    if gold_ids:
        return [{'unit': 'chunk', 'gold_chunk_id': str(cid)} for cid in gold_ids]
    return [{'unit': 'case', 'gold_chunk_id': ''}]


def sample(cases: list[dict], ratio: float, seed: int) -> list[dict]:
    """按分层键做比例抽样，固定种子保证可复现。"""
    buckets: dict[str, list[dict]] = defaultdict(list)
    for case in cases:
        buckets[stratify_key(case)].append(case)

    rng = random.Random(seed)
    picked: list[dict] = []
    for key in sorted(buckets):
        group = sorted(buckets[key], key=lambda c: str(c.get('id', '')))
        n = max(1, round(len(group) * ratio)) if ratio < 1 else len(group)
        picked.extend(rng.sample(group, min(n, len(group))))
    return sorted(picked, key=lambda c: str(c.get('id', '')))


def fetch_previews(chunk_ids: list[str], mode: str) -> dict[str, str]:
    """可选：从语料里取 chunk 原文前 120 字，供人工判断用。"""
    if not chunk_ids:
        return {}
    try:
        from eval.run_retrieval_eval import _make_fake_env, _make_real_env

        _doc_map, store, _embedding = _make_real_env() if mode == 'real' else _make_fake_env()
        wanted = set(chunk_ids)
        return {
            chunk.id: (chunk.text or '')[:120]
            for chunk in store.all_chunks()
            if chunk.id in wanted
        }
    except Exception as exc:  # 预览是锦上添花，失败不该阻断抽检
        print(f'[review] 预览获取失败（不影响抽检）：{exc}')
        return {}


def build_sheet(
    cases: list[dict],
    seed: int,
    ratio: float,
    previews: dict[str, str],
) -> list[dict]:
    rows: list[dict] = []
    stamp = time.strftime('%Y-%m-%d %H:%M:%S')
    for case in cases:
        for unit in review_units(case):
            rows.append(
                {
                    'case_id': case.get('id', ''),
                    'question': case.get('question', ''),
                    'unit': unit['unit'],
                    'gold_chunk_id': unit['gold_chunk_id'],
                    'chunk_preview': previews.get(unit['gold_chunk_id'], ''),
                    'auto_status': 'auto',
                    'verdict': '',
                    'suggestion': '',
                    'reviewer': '',
                    'reviewed_at': stamp,
                }
            )
    return rows


def render_markdown(rows: list[dict], meta: dict) -> str:
    lines = ['# 人工抽检清单', '']
    lines.append(f"- 来源 gold：`{meta['gold']}`")
    lines.append(f"- 抽样比例：{meta['ratio']}（seed={meta['seed']}）")
    lines.append(f"- 抽中 case：{meta['n_cases']}  待复核单元：{len(rows)}")
    lines.append(f"- 生成时间：{meta['generated_at']}")
    lines.append('')
    lines.append('> 判定请在对应的 csv 里填 `verdict` 列：`对` / `错` / `不确定`，')
    lines.append('> 然后跑 `python -m eval.sample_review --apply <csv>` 回写 gold。')
    lines.append('')
    lines.append('| case_id | 问题 | 单元 | gold chunk | 原文预览 |')
    lines.append('| --- | --- | --- | --- | --- |')
    for row in rows:
        preview = (row['chunk_preview'] or '').replace('|', '\\|')[:60]
        question = (row['question'] or '').replace('|', '\\|')[:40]
        lines.append(
            f"| `{row['case_id']}` | {question} | {row['unit']} | "
            f"`{row['gold_chunk_id'] or '-'}` | {preview} |"
        )
    lines.append('')
    return '\n'.join(lines)


def normalize_verdict(value: str) -> str:
    return VERDICT_ALIASES.get((value or '').strip().lower(), '')


def apply_verdicts(csv_path: Path, gold_path: Path) -> dict:
    """把人工判定回写 gold：标错的 gold chunk 被剔除，并写审计记录。"""
    gold, path = load_gold(gold_path)
    cases = {str(c.get('id')): c for c in gold.get('cases', [])}

    with Path(csv_path).open(encoding='utf-8-sig', newline='') as fh:
        rows = list(csv.DictReader(fh))

    audit: list[dict] = []
    stats = {'rows': len(rows), 'reviewed': 0, 'bad': 0, 'unsure': 0, 'ok': 0, 'removed_chunks': 0,
             'deprecated_cases': 0}
    stamp = time.strftime('%Y-%m-%d %H:%M:%S')

    for row in rows:
        verdict = normalize_verdict(row.get('verdict', ''))
        if not verdict:
            continue
        stats['reviewed'] += 1
        stats[verdict] += 1
        case_id = str(row.get('case_id', ''))
        case = cases.get(case_id)
        if case is None:
            continue
        if verdict == 'bad' and row.get('unit') == 'chunk':
            chunk_id = str(row.get('gold_chunk_id', ''))
            gold_ids = case.get('chunk_ids') or []
            if chunk_id in gold_ids:
                case['chunk_ids'] = [cid for cid in gold_ids if cid != chunk_id]
                stats['removed_chunks'] += 1
                audit.append(
                    {
                        'at': stamp,
                        'case_id': case_id,
                        'action': 'remove_gold_chunk',
                        'chunk_id': chunk_id,
                        'reviewer': row.get('reviewer', ''),
                        'suggestion': row.get('suggestion', ''),
                    }
                )
                if not case['chunk_ids']:
                    case['deprecated'] = True
                    case['deprecated_reason'] = '人工复核后无有效 gold chunk'
                    stats['deprecated_cases'] += 1
                    audit.append(
                        {
                            'at': stamp,
                            'case_id': case_id,
                            'action': 'deprecate_case',
                            'reason': '人工复核后无有效 gold chunk',
                            'reviewer': row.get('reviewer', ''),
                        }
                    )
        elif verdict == 'bad' and row.get('unit') == 'case':
            case['deprecated'] = True
            case['deprecated_reason'] = '人工复核判定为错误用例'
            stats['deprecated_cases'] += 1
            audit.append(
                {
                    'at': stamp,
                    'case_id': case_id,
                    'action': 'deprecate_case',
                    'reason': row.get('suggestion', ''),
                    'reviewer': row.get('reviewer', ''),
                }
            )

    gold['reviewed_at'] = stamp
    gold['review_stats'] = stats
    path.write_text(json.dumps(gold, ensure_ascii=False, indent=2), encoding='utf-8')

    REPORT_DIR.mkdir(exist_ok=True)
    with AUDIT_FILE.open('a', encoding='utf-8') as fh:
        for entry in audit:
            fh.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print('[review] 回写完成')
    detail = f"对 {stats['ok']} / 错 {stats['bad']} / 不确定 {stats['unsure']}"
    print(f'  复核行数：{stats["reviewed"]}（{detail}）')
    print(f"  剔除 gold chunk：{stats['removed_chunks']}  作废 case：{stats['deprecated_cases']}")
    print(f"  gold 已更新：{path}")
    print(f"  审计记录：{AUDIT_FILE}")
    return stats


def run(
    gold_path: Path | None = None,
    ratio: float = 0.2,
    seed: int = 20260916,
    with_preview: bool = False,
    mode: str = 'real',
) -> dict:
    gold, path = load_gold(gold_path or DEFAULT_GOLD)
    cases = gold.get('cases', [])
    picked = sample(cases, ratio, seed)

    previews: dict[str, str] = {}
    if with_preview:
        all_ids = [str(cid) for case in picked for cid in (case.get('chunk_ids') or [])]
        previews = fetch_previews(all_ids, mode)

    rows = build_sheet(picked, seed, ratio, previews)
    meta = {
        'gold': str(path),
        'ratio': ratio,
        'seed': seed,
        'n_cases': len(picked),
        'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
    }

    REPORT_DIR.mkdir(exist_ok=True)
    stamp = time.strftime('%Y%m%d_%H%M%S')
    csv_path = REPORT_DIR / f'review_{stamp}.csv'
    md_path = REPORT_DIR / f'review_{stamp}.md'

    with csv_path.open('w', encoding='utf-8-sig', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    md_path.write_text(render_markdown(rows, meta), encoding='utf-8')

    print(f'[review] 抽样完成：{len(picked)}/{len(cases)} case，{len(rows)} 个复核单元')
    print(f'  待填 csv：{csv_path}')
    print(f'  阅读清单：{md_path}')
    print(f"  下一句：填完 csv 后跑 python -m eval.sample_review --apply {csv_path}")
    return {'csv': str(csv_path), 'md': str(md_path), 'rows': len(rows), 'cases': len(picked)}


def agreement(csv_path: Path) -> dict:
    """人工判定 vs 自动判定的方向一致率（人工只标了「错」而自动 PASS 的即为分歧）。"""
    with Path(csv_path).open(encoding='utf-8-sig', newline='') as fh:
        rows = list(csv.DictReader(fh))
    labeled = [
        (normalize_verdict(r.get('verdict', '')), r.get('auto_status', 'auto')) for r in rows
    ]
    judged = [(v, a) for v, a in labeled if v in ('ok', 'bad')]
    if not judged:
        return {'n': 0, 'agreement': None}
    agree = sum(1 for v, a in judged if (v == 'ok') == (a in ('auto', 'ok', 'pass')))
    return {'n': len(judged), 'agreement': round(agree / len(judged), 4)}


def main() -> None:
    parser = argparse.ArgumentParser(description='RAG gold 人工抽检')
    parser.add_argument('--gold', help='gold 文件（默认 retrieval_gold.json）')
    parser.add_argument('--ratio', type=float, default=0.2, help='抽样比例，默认 0.2')
    parser.add_argument('--seed', type=int, default=20260916, help='随机种子（保证可复现）')
    parser.add_argument(
        '--with-preview', action='store_true', help='拉取 chunk 原文预览（需构建环境）'
    )
    parser.add_argument('--mode', choices=['fake', 'real'], default='real')
    parser.add_argument('--apply', help='回写某份填好的 csv 到 gold')
    parser.add_argument('--agreement', help='统计某份 csv 的人工/自动一致率')
    args = parser.parse_args()

    if args.apply:
        apply_verdicts(Path(args.apply), Path(args.gold) if args.gold else DEFAULT_GOLD)
        return
    if args.agreement:
        print(json.dumps(agreement(Path(args.agreement)), ensure_ascii=False))
        return
    run(
        gold_path=Path(args.gold) if args.gold else None,
        ratio=args.ratio,
        seed=args.seed,
        with_preview=args.with_preview,
        mode=args.mode,
    )


if __name__ == '__main__':
    main()
