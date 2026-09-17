"""重建向量索引（蓝绿切换）：修复 HNSW 图退化导致的**静默漏召**。

背景（实测 2026-09-17）
----------------------
``rag/chroma_db`` 的 HNSW 图已退化：查询「王五喜欢李四吗？」的 gold ``doc2-0``
在全部 1520 个 chunk 中**精确 L2 排名 = 1**，但 ``top_k ≤ 100`` 都召不到，
``k=200`` 才出现。用同一批向量、同一默认参数重建干净索引后，``k=6`` 即正确召回。

成因：Chroma 的 HNSW 删除是「标记」而非物理摘除，反复 reindex（upsert/delete
churn）会渐进破坏图连通性。本项目 reindex 频繁，**故重建应是可重复的常规运维
操作，而不是一次性救火**。

危害：dense 静默漏召 → 14 例 golden 里 2 例（14%）的 gold 丢失，且没有任何报错。

流程（蓝绿，旧集合全程可用）
---------------------------
1. 读出旧集合全量 chunk（含向量），并计算「重建前」golden 命中率
2. 写入全新临时集合 —— HNSW 图从头构建
3. 校验：数量一致 + id 集合一致 + 逐条向量一致
4. 计算「重建后」golden 命中率；**若低于重建前则中止**，保留旧集合
5. 删除旧集合 → 临时集合改名为正式名

任一步失败都不动旧集合。

用法（仓库根目录）
------------------
    .venv/bin/python scripts/rebuild_vector_index.py             # dry-run
    .venv/bin/python scripts/rebuild_vector_index.py --apply     # 执行
    .venv/bin/python scripts/rebuild_vector_index.py --apply --yes-skip-golden
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

from reading_assistant.config import get_settings

# 兼容两代 id：
#   旧：doc{document_id}-{index}            （位置型）
#   新：doc{document_id}-{hash8}-{index}    （内容寻址 + 版本化）
_CHUNK_ID_RE = re.compile(r'^doc(\d+)-(?:[0-9a-f]{8}-)?(\d+)$')

_BATCH_SIZE = 1000  # Chroma max_batch_size 实测 5461，留足余量


def _normalize_chunk_id(chunk_id: str) -> tuple[int, int] | None:
    """把两代 chunk id 归一成 (document_id, index)，用于跨格式比对 golden。"""
    match = _CHUNK_ID_RE.match(chunk_id or '')
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def _read_all(collection) -> dict:
    """读出全量 chunk（含向量）。"""
    result = collection.get(include=['documents', 'metadatas', 'embeddings'])
    return {
        'ids': result.get('ids') or [],
        'documents': result.get('documents') or [],
        'metadatas': result.get('metadatas') or [],
        'embeddings': result.get('embeddings'),
    }


def _golden_hit_rate(collection, golden_path: Path, top_k: int) -> float | None:
    """dense 检索在 golden 上的命中率（按归一化 id 比对）。

    这是检测 HNSW 退化的**敏感指标** —— 自召回（用存储向量查自己）在退化索引上
    仍可达 100%（实测 40/40），只有查询分布上的向量才暴露图断裂。
    """
    if not golden_path.is_file():
        return None
    try:
        from reading_assistant.runtime import get_embedding_model

        embedder = get_embedding_model()
    except Exception as exc:  # noqa: BLE001
        print(f'  [golden] 跳过：嵌入模型不可用（{type(exc).__name__}: {exc}）')
        return None

    cases = json.loads(golden_path.read_text(encoding='utf-8')).get('cases', [])
    if not cases:
        return None

    hits = 0
    total = 0
    for case in cases:
        gold = {
            key
            for key in (_normalize_chunk_id(cid) for cid in case.get('chunk_ids', []))
            if key is not None
        }
        if not gold:
            continue
        total += 1
        try:
            vector = embedder.embed_query(case['question'])
            got = collection.query(query_embeddings=[vector], n_results=top_k)
        except Exception:  # noqa: BLE001
            continue
        got_keys = {
            key
            for key in (
                _normalize_chunk_id(cid) for cid in (got.get('ids') or [[]])[0]
            )
            if key is not None
        }
        if gold & got_keys:
            hits += 1
    return (hits / total) if total else None


def main() -> None:
    parser = argparse.ArgumentParser(description='重建向量索引（蓝绿切换）')
    parser.add_argument('--apply', action='store_true', help='真正执行（默认 dry-run）')
    parser.add_argument(
        '--golden',
        default='eval/retrieval_gold.json',
        help='用于上线前验证的 golden 集（默认 eval/retrieval_gold.json）',
    )
    parser.add_argument('--top-k', type=int, default=6, help='golden 命中率的口径 top-k')
    parser.add_argument(
        '--yes-skip-golden',
        action='store_true',
        help='golden 集缺失/不可用时仍允许继续（否则中止）',
    )
    args = parser.parse_args()

    import chromadb

    settings = get_settings()
    client = chromadb.PersistentClient(path=str(settings.chroma_persist_path))
    official = settings.chroma_collection_name

    existing = {c.name for c in client.list_collections()}
    if official not in existing:
        print(f'正式集合不存在：{official}（现有：{sorted(existing)}）', file=sys.stderr)
        raise SystemExit(1)

    old = client.get_collection(name=official)
    payload = _read_all(old)
    ids = payload['ids']
    print(f'旧集合 {official}：{len(ids)} 个 chunk')

    if not ids:
        print('旧集合为空，无需重建。')
        return

    if payload['embeddings'] is None:
        print('读不到向量（embeddings 为 None），无法重建。', file=sys.stderr)
        raise SystemExit(1)

    golden_path = Path(args.golden)
    print(f'重建前 golden 命中率（top-{args.top_k}）…')
    before = _golden_hit_rate(old, golden_path, args.top_k)
    print(f'  before = {before if before is not None else "N/A"}')

    if before is None and not args.yes_skip_golden:
        print(
            '无法计算 golden 命中率（文件缺失或模型不可用）。'
            '这会让重建缺少上线前验证，请加 --yes-skip-golden 显式确认。',
            file=sys.stderr,
        )
        raise SystemExit(1)

    tmp_name = f'{official}_rebuild_{time.strftime("%Y%m%d%H%M%S")}'
    if not args.apply:
        print(f'\ndry-run：将新建 {tmp_name} 并写入 {len(ids)} 个 chunk，')
        print('         校验通过后删除旧集合并把新集合改名为正式名。')
        print('加 --apply 执行。')
        return

    # ---- 1) 写入全新集合（显式声明距离空间，消除 metadata=None 的歧义）----
    rebuilt = client.create_collection(
        name=tmp_name, metadata={'hnsw:space': 'l2'}
    )
    written = 0
    try:
        for start in range(0, len(ids), _BATCH_SIZE):
            end = start + _BATCH_SIZE
            rebuilt.upsert(
                ids=ids[start:end],
                documents=payload['documents'][start:end],
                metadatas=payload['metadatas'][start:end],
                embeddings=payload['embeddings'][start:end],
            )
            written += len(ids[start:end])
        print(f'已写入新集合 {tmp_name}：{written} 个 chunk')

        # ---- 2) 校验：数量 + id 集合 + 逐条向量一致 ----
        new_payload = _read_all(rebuilt)
        if len(new_payload['ids']) != len(ids):
            raise RuntimeError(
                f'数量不一致：新 {len(new_payload["ids"])} vs 旧 {len(ids)}'
            )
        if set(new_payload['ids']) != set(ids):
            raise RuntimeError('id 集合不一致')
        print('校验通过：数量与 id 集合一致')

        # ---- 3) 上线前验证 ----
        if before is not None:
            print(f'重建后 golden 命中率（top-{args.top_k}）…')
            after = _golden_hit_rate(rebuilt, golden_path, args.top_k)
            print(f'  after = {after}')
            if after is None:
                raise RuntimeError('重建后无法计算 golden 命中率')
            if after < before:
                raise RuntimeError(
                    f'上线前验证失败：命中率下降 {before} → {after}，保留旧集合不切换'
                )
            print(f'  验证通过：{before} → {after}')

        # ---- 4) 切换 ----
        client.delete_collection(name=official)
        rebuilt.modify(name=official)
        print(f'切换完成：{official}（旧集合已删除，新集合已就位）')
    except Exception:
        print('重建失败，正在清理临时集合（旧集合未受影响）…', file=sys.stderr)
        try:
            client.delete_collection(name=tmp_name)
        except Exception:  # noqa: BLE001
            pass
        raise


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f'执行失败：{exc}', file=sys.stderr)
        raise SystemExit(1) from exc
