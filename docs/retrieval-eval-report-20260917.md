# 检索评测报告（2026-09-17）

> 触发：完成 11 项修复后，验证检索质量有无回归 / 提升；并给评测补上缺失的口径。
> 结论：**两个修复各自产生了可测收益，但在不同指标上** —— 指标无回归，且评测新增了能检出索引退化的跨书库口径。

## 一、结果总览

| 口径 | 指标 | 修复前 | 修复后 | 归因 |
|---|---|---|---|---|
| filtered（官方） | precision@5/10/20 | 0.9524 | **1.0000** | P2-3 分数尺度统一 |
| filtered（官方） | recall / mrr / map | 1.0000 | 1.0000 | 已饱和，无变化空间 |
| cross（新增） | 命中率 top-20 | 0.786 | **1.000** | P0-1 索引重建 |

> ⚠️ **订正说明**：本报告初版只对比了 MRR，据此写「持平」——**那是漏报**。
> 逐指标复核后发现 precision 提升（见下节），已修正。

## 二、官方口径：precision 提升（P2-3 的收益）

```
指标                      09-16(修复前)     09-17(修复后)
vector.precision@5         0.9524         1.0000  ← 提升
vector.precision@10        0.9524         1.0000  ← 提升
vector.precision@20        0.9524         1.0000  ← 提升
hybrid.precision@5         0.9524         1.0000  ← 提升
hybrid.precision@10        0.9524         1.0000  ← 提升
hybrid.precision@20        0.9524         1.0000  ← 提升
（recall / mrr / map 均为 1.0000 → 1.0000，已饱和）
```

### 机制（已实测验证）

`P2-3` 把两条支路的分数统一到余弦，**顺带收紧了 dense 的实际阈值**：

| | score 公式 | `min_score=0.45` 等效余弦 |
|---|---|---|
| 旧 dense | `1/(1+L2dist)` | `cos ≥ 0.3889` |
| 新（统一余弦） | `cos` | `cos ≥ 0.4500` |

新阈值更严 → 剔除 `cos ∈ [0.389, 0.45)` 的边缘块（多为非 gold）→ precision 上升。

验证实验（同一索引、同一批问题，只变阈值语义）：

```
阈值语义                     平均 precision@5
旧 (cos ≥ 0.3889)                  0.9464
新 (cos ≥ 0.4500)                  1.0000
```

与报告中的 `0.9524 → 1.0000` 吻合（实验只走 dense 单路，故略低）。

> 💡 **这是 P2-3 的意外收益**：当时只为修「一个常量跨两种量纲」的隐患，
> 没预料到会顺带纠正 dense 的阈值松紧。**统一量纲本身就会改变行为** —— 值得记住。

## 三、新增跨书库口径（本次给评测补的能力）

### 为什么必须加

```python
# 旧评测唯一路径：带 document_id 过滤
hits = retriever.retrieve(question, top_k=top_n, document_id=document_id)

# graph/qa.py:531 —— 跨书库提问时 document_id=None，不带过滤
hits = retriever.retrieve(question, document_id=state.get('document_id'))
```

带 `document_id` 时 Chroma 把搜索空间缩到**单个文档**（doc2 仅 1 个 chunk），
**HNSW 图断裂被彻底掩盖**。只有跨书库检索（全库 1520 点）才暴露。

> 另注：**自召回**（用存储向量查自己）同样无效 —— 退化索引上实测仍 40/40 = 100%。
> 两条看似合理的口径都不敏感。

### 改动

`eval/run_retrieval_eval.py` 现在**同时跑两个口径**：

| 口径 | document_id | 对应真实路径 |
|---|---|---|
| `filtered` | 有（现状） | 指定文档内检索 |
| `cross` | 无（新增） | 跨书库提问 |

- 报告：`summary` 保持原结构不动，新增顶层键 `cross_summary`
  （**不嵌进 `summary`** —— `compare_reports.py` 会把 `summary` 里的 dict 型字段当「系统」展平，嵌进去会污染下游对比）
- `per_case` 新增 `vector_cross_mrr` / `hybrid_cross_mrr` 等字段（加法式）
- CLI 新增 `--no-cross`（不建议使用，会失去退化检测能力）

### 退化检出效果（同口径 A/B，用备份的退化索引）

```bash
PYTHONPATH=. .venv/bin/python scripts/compare_index_health.py <备份索引目录>
```

| 索引 | filtered 口径 | cross 口径 |
|---|---|---|
| 旧索引（退化） | 14/14 = 1.000（**测不出**） | 11/14 = **0.786** |
| 新索引（重建） | 14/14 = 1.000 | 14/14 = **1.000** |

cross 未命中：`love-03`、`love-05`、`shaobai-01`。

## 四、附带发现：id 迁移的第三个引用点

跑评测前发现 gold 的 `chunk_ids` 是旧格式，与迁移后的库 **24/24 全失配** ——
直接跑指标会崩到 0。这是 id 迁移的第三个派生引用点：

| # | 派生引用 | 数量 | 状态 |
|---|---|---|---|
| 1 | 向量库 chunk 本体 | 1520 | ✅ |
| 2 | 问答缓存 `citations[].chunk_id` | 90 | ✅ |
| 3 | 评测集 JSON（gold + hard_neg） | 192 | ✅ |

三者已统一并入 `scripts/migrate_chunk_ids.py`（单一入口、默认 dry-run、幂等）。

**刻意不改** `eval/reports/*`：历史报告是过去结果的快照，改了即为篡改记录。

## 五、回归验证

```
pytest -q                        → 316 passed（新增 8 个双口径单元测试）
ruff check（改动文件）            → All checks passed!
compare_reports 跨新旧报告对比     → ✅ 无回归（并正确识别 precision 提升）
run_all --only retrieval          → ✅ 无回归
```

## 六、简历可用数字

> - 跨书库检索命中率 **78.6% → 100%**（修复 HNSW 索引退化）
> - 检索精度 precision@5 **0.9524 → 1.0000**（统一分数尺度后阈值语义纠正）
> - 发现评测口径盲区：带 `document_id` 过滤会掩盖图断裂，已补跨书库口径

> ⚠️ **诚实边界**：14 例样本，天花板效应明显（多数指标已饱和 1.0），
> 该提升的统计显著性有限。真正的证据是**同口径 A/B + 逐例未命中列表**
> （能指名道姓说哪 3 例从漏召变命中），而不是那个百分数。
