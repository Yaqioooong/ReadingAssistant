"""构建检索评测黄金集（gold）。

从端到端黄金集的“可答且 PASS”用例中，把回答引用（citations）的 chunk_id
沉淀为该用例的弱标注相关块——正确回答引用的块即检索应当召回的块。

用法：
    uv run python -m eval.build_retrieval_gold --mode real    # 生产链路，产出有效 gold
    uv run python -m eval.build_retrieval_gold --mode fake    # 冒烟：链路可跑通（gold 语义弱）

输出：eval/retrieval_gold.json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from eval.run_eval import (
    _answer_hits,
    _load_golden,
    _make_client,
    _upload_books,
)

EVAL_DIR = Path(__file__).resolve().parent
GOLD_OUT = EVAL_DIR / 'retrieval_gold.json'


def _collect_case(client, case: dict, doc_map: dict) -> dict | None:
    """复刻端到端评测的判定：可答且 PASS → 沉淀引用 chunk_id。"""
    session_id = client.post('/api/sessions').json()['session_id']
    doc_name = case.get('document')
    doc_ids = [doc_map[doc_name]] if doc_name and doc_name in doc_map else []
    resp = client.post(
        f'/api/sessions/{session_id}/messages',
        json={'question': case['question'], 'document_ids': doc_ids},
    )
    body = resp.json()
    answer = body.get('answer') or ''
    citations = body.get('citations') or []
    if case.get('expect_unanswerable'):
        return None  # 信息不足类没有“应当召回的块”，不进检索 gold

    if not _answer_hits(answer, case.get('expect_keywords') or []):
        return None  # 未答对：引用不可信

    chunk_ids: list[str] = []
    for citation in citations:
        chunk_id = citation.get('chunk_id')
        if chunk_id and chunk_id not in chunk_ids:
            chunk_ids.append(chunk_id)
    if not chunk_ids:
        return None
    return {
        'id': case['id'],
        'question': case['question'],
        'document': doc_name,
        'document_id': doc_ids[0] if doc_ids else None,
        'chunk_ids': chunk_ids,
    }


def run(mode: str = 'real', limit: int = 0) -> dict:
    cases = _load_golden()
    if limit:
        cases = cases[:limit]
    client, engine = _make_client(mode)
    gold: list[dict] = []
    with client:
        doc_map = _upload_books(client)
        print(f'[gold] mode={mode} docs={len(doc_map)} cases={len(cases)}')
        for case in cases:
            entry = _collect_case(client, case, doc_map)
            if entry is not None:
                gold.append(entry)
                short_q = case['question'][:24]
                n_chunks = len(entry['chunk_ids'])
                print(f"  [OK ] {case['id']:12s} chunks={n_chunks} {short_q}")
            else:
                flag = 'SKIP(unanswerable)' if case.get('expect_unanswerable') else 'SKIP(fail)'
                print(f"  [---] {case['id']:12s} {flag}")
    payload = {
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'mode': mode,
        'total_cases': len(gold),
        'cases': gold,
    }
    GOLD_OUT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    print(f'\n检索 gold 已保存: {GOLD_OUT}（{len(gold)} 条可答 PASS 用例）')
    if mode == 'fake':
        print('提示: fake 模式向量退化，gold 仅用于链路冒烟；有效指标请用 --mode real。')
    if engine is not None:
        engine.dispose()
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description='构建检索评测黄金集')
    parser.add_argument('--mode', choices=['fake', 'real'], default='real')
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    run(mode=args.mode, limit=args.limit)


if __name__ == '__main__':
    main()
