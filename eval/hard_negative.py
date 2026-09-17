"""硬负样本挖掘：给检索评测补齐「语义相近但答案无关」的干扰项。

为什么需要
----------
现有的 ``retrieval_adversarial_gold.json`` 只覆盖了人工设计的陷阱题，
真正能暴露混合检索短板的，是**向量空间里离 gold chunk 很近、但内容不相关**的 chunk。
这类负样本无法凭空合成，必须从真实语料里挖。

挖掘两类负样本
--------------
1. ``hard_negatives``   语义负样本：用 gold chunk 自己的 embedding 去检索，
   取相似度高但**不是 gold** 的 chunk。相似度越高越「硬」。
2. ``neighbor_negatives`` 邻接负样本：同一文档里与 gold chunk 位置相邻
   （±``window``）的 chunk。这类最容易在切分/召回时混进来。

输出：``eval/retrieval_hard_neg.json``，可直接被后续检索评测消费。

用法
----
    uv run python -m eval.hard_negative --mode real
    uv run python -m eval.hard_negative --mode real --min-sim 0.75 --top-n 30
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
REPORT_DIR = EVAL_DIR / 'reports'
DEFAULT_GOLD = EVAL_DIR / 'retrieval_gold.json'
OUT_FILE = EVAL_DIR / 'retrieval_hard_neg.json'

_ORDINAL_RE = re.compile(r'(\d+)\s*$')


def _load_cases(path: Path) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    return data['cases']


def _chunk_ordinal(chunk) -> int | None:
    """拿到 chunk 在文档中的顺序号。

    优先读 metadata（``chunk_index`` / ``index``），
    退化到从 chunk_id 尾部的数字解析（本项目 id 形如 ``doc2-17``）。
    """
    meta = getattr(chunk, 'metadata', None) or {}
    for key in ('chunk_index', 'index', 'chunk'):
        value = meta.get(key)
        if isinstance(value, int):
            return value
    match = _ORDINAL_RE.search(str(getattr(chunk, 'id', '')))
    return int(match.group(1)) if match else None


def _gold_ids_of(case: dict) -> list[str]:
    ids = case.get('chunk_ids') or case.get('gold') or []
    return [str(cid) for cid in ids]


def mine(
    mode: str = 'real',
    gold_file: Path | None = None,
    top_n: int = 20,
    min_sim: float | None = None,
    max_per_case: int = 5,
    window: int = 2,
    limit: int = 0,
) -> dict:
    """执行挖掘，返回并落盘结果。

    ``min_sim`` 默认为 ``None``：硬负样本按**相对相似度**选取，即每个 gold chunk
    最近的非 gold 邻居。这一点很关键 —— 绝对阈值极易失效，而且**失败是静默的**
    （正常退出、正常落盘，只是结果是空的）。

    实测（1520 个 chunk / 24 个 gold）：
    - 所有「非自身」pair 的相似度上限 0.7441，但其中 8 条 ≥0.6 的
      **全部是同一 case 内部 gold 之间的相似**，本就会被 ``gold_set`` 过滤；
    - 排除整个 gold 集后，真实候选池上限只有 **0.5792**。

    所以 ``--min-sim 0.6`` 在本题库上什么也挖不到。除非你确知所用模型的打分尺度，
    否则不要设它 —— 先跑一次看打印出的分布区间再决定。
    """
    from eval.run_retrieval_eval import _make_fake_env, _make_real_env

    gold_file = Path(gold_file or DEFAULT_GOLD)
    cases = _load_cases(gold_file)
    if limit:
        cases = cases[:limit]

    print(f'[hard-neg] mode={mode} cases={len(cases)} 构建环境…')
    doc_map, store, embedding = _make_real_env() if mode == 'real' else _make_fake_env()

    chunks_all = list(store.all_chunks())
    by_id = {chunk.id: chunk for chunk in chunks_all}
    by_doc: dict = {}
    for chunk in chunks_all:
        doc_id = (chunk.metadata or {}).get('document_id')
        by_doc.setdefault(doc_id, []).append(chunk)
    for doc_id in by_doc:
        by_doc[doc_id].sort(key=lambda c: (_chunk_ordinal(c) is None, _chunk_ordinal(c)))

    results: list[dict] = []
    sim_pool: list[float] = []
    counts: Counter = Counter()

    for case in cases:
        gold_ids = _gold_ids_of(case)
        if not gold_ids:
            continue
        gold_set = set(gold_ids)
        document_id = case.get('document_id') or doc_map.get(case.get('document', ''))

        sem: dict[str, dict] = {}
        for gid in gold_ids:
            chunk = by_id.get(gid)
            if chunk is None or not (chunk.text or '').strip():
                continue
            vector = embedding.embed_query(chunk.text)
            hits = store.query(vector, top_k=top_n)
            for hit in hits:
                if hit.id in gold_set or hit.id not in by_id:
                    continue
                sim = float(getattr(hit, 'score', 0.0) or 0.0)
                # 无论是否过滤都记录，这样统计里能看到完整分布
                sim_pool.append(sim)
                if min_sim is not None and sim < min_sim:
                    continue
                if hit.id not in sem or sim > sem[hit.id]['similarity']:
                    sem[hit.id] = {
                        'chunk_id': hit.id,
                        'similarity': round(sim, 4),
                        'source_gold': gid,
                        'preview': (by_id[hit.id].text or '')[:120],
                    }
        hard = sorted(sem.values(), key=lambda d: -d['similarity'])[:max_per_case]
        for rank, item in enumerate(hard, start=1):
            item['rank'] = rank

        neigh: list[dict] = []
        for gid in gold_ids:
            chunk = by_id.get(gid)
            if chunk is None:
                continue
            ordinal = _chunk_ordinal(chunk)
            doc_id = (chunk.metadata or {}).get('document_id', document_id)
            if ordinal is None or doc_id is None:
                continue
            for other in by_doc.get(doc_id, []):
                if other.id in gold_set:
                    continue
                other_ordinal = _chunk_ordinal(other)
                if other_ordinal is None:
                    continue
                distance = abs(other_ordinal - ordinal)
                if 0 < distance <= window:
                    neigh.append(
                        {
                            'chunk_id': other.id,
                            'distance': distance,
                            'near_gold': gid,
                            'preview': (other.text or '')[:120],
                        }
                    )
        dedup_neigh: dict[str, dict] = {}
        for item in neigh:
            existing = dedup_neigh.get(item['chunk_id'])
            if existing is None or item['distance'] < existing['distance']:
                dedup_neigh[item['chunk_id']] = item
        neighbor_list = sorted(dedup_neigh.values(), key=lambda d: d['distance'])[:max_per_case]

        counts['cases'] += 1
        if hard:
            counts['cases_with_hard'] += 1
        if neighbor_list:
            counts['cases_with_neighbor'] += 1
        counts['hard_total'] += len(hard)
        counts['neighbor_total'] += len(neighbor_list)

        results.append(
            {
                'case_id': case.get('id'),
                'question': case.get('question'),
                'document': case.get('document'),
                'document_id': document_id,
                'gold': gold_ids,
                'hard_negatives': hard,
                'neighbor_negatives': neighbor_list,
            }
        )
        tag = 'OK ' if hard else '---'
        print(
            f"  [{tag}] {case.get('id', '?'):<16} gold={len(gold_ids)} "
            f"hard={len(hard)} neighbor={len(neighbor_list)}"
        )

    stats = {
        'cases': counts['cases'],
        'cases_with_hard': counts['cases_with_hard'],
        'cases_with_neighbor': counts['cases_with_neighbor'],
        'hard_total': counts['hard_total'],
        'neighbor_total': counts['neighbor_total'],
        'avg_hard_per_case': round(counts['hard_total'] / counts['cases'], 2)
        if counts['cases']
        else 0.0,
        'avg_neighbor_per_case': round(counts['neighbor_total'] / counts['cases'], 2)
        if counts['cases']
        else 0.0,
        'similarity_min': round(min(sim_pool), 4) if sim_pool else None,
        'similarity_max': round(max(sim_pool), 4) if sim_pool else None,
        'similarity_mean': round(sum(sim_pool) / len(sim_pool), 4) if sim_pool else None,
    }

    payload = {
        'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'mode': mode,
        'params': {
            'top_n': top_n,
            'min_sim': min_sim,
            'max_per_case': max_per_case,
            'window': window,
            'source_gold': str(gold_file),
        },
        'stats': stats,
        'cases': results,
    }
    OUT_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')

    print('\n=== 挖掘统计 ===')
    print(f"  case 数：{stats['cases']}（有语义负样本 {stats['cases_with_hard']}）")
    print(f"  语义负样本：{stats['hard_total']} 条，平均 {stats['avg_hard_per_case']}/题")
    print(f"  邻接负样本：{stats['neighbor_total']} 条，平均 {stats['avg_neighbor_per_case']}/题")
    if sim_pool:
        print(
            f"  相似度分布：min={stats['similarity_min']} "
            f"mean={stats['similarity_mean']} max={stats['similarity_max']}"
        )
    if stats['hard_total'] == 0:
        print()
        print('  ⚠️  未挖到任何语义负样本。')
        if sim_pool:
            print(
                f"      观测到的最高非 gold 相似度为 {stats['similarity_max']}"
                f"（均值 {stats['similarity_mean']}）。"
            )
            print(
                '      若你设置了 --min-sim，它很可能高于该模型的实际打分区间；'
                '本项目 score=1/(1+距离) 会把区间压扁，建议不设或设得很低。'
            )
        else:
            print('      语料里没有可用的非 gold chunk，或 gold chunk 全部缺失。')
    if counts['neighbor_total'] == 0:
        print()
        print('  ⚠️  未挖到邻接负样本：gold chunk 可能已覆盖其所在文档的全部内容。')
        print('      （小样本文档常见；把语料换成有富余段落的大文档即可。）')
    if mode == 'fake':
        print('  提示：fake 模式向量退化，负样本无语义意义，仅供链路冒烟。')
    print(f'\n已保存：{OUT_FILE}')
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description='RAG 硬负样本挖掘')
    parser.add_argument('--mode', choices=['fake', 'real'], default='real')
    parser.add_argument('--gold', help='gold 文件路径（默认 retrieval_gold.json）')
    parser.add_argument('--top-n', type=int, default=20, help='每个 gold chunk 检索候选数')
    parser.add_argument(
        '--min-sim',
        type=float,
        default=None,
        help='语义负样本相似度下限（默认不设，按相对相似度取最近的非 gold 邻居）',
    )
    parser.add_argument('--max-per-case', type=int, default=5, help='每题最多保留负样本数')
    parser.add_argument('--window', type=int, default=2, help='邻接窗口大小（±N 个 chunk）')
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    mine(
        mode=args.mode,
        gold_file=Path(args.gold) if args.gold else None,
        top_n=args.top_n,
        min_sim=args.min_sim,
        max_per_case=args.max_per_case,
        window=args.window,
        limit=args.limit,
    )


if __name__ == '__main__':
    main()
