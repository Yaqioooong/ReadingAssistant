# 检索片段侧完全没有 token 预算 —— 补上上下文管理的另一半

日期：2026-09-24 · 触发：用户问「现在工程中是如何做上下文管理的」，逐层排查时发现片段侧一个 token 都不数。

## 0. 缺口

上下文管理原先只有**一半**：

| 侧 | 机制 | 预算 |
|---|---|---|
| 历史原文 | `_select_window`（从最新往回整条收） | `history_token_budget=4000` token |
| 会话状态 | `ConversationState`（5 个字段） | 无（有界） |
| **注入答案的检索片段** | `_build_answer_prompt` 全文拼接 | **无** |

证据：`_count_tokens` 全仓**只有一个调用点**，就是 `_select_window`（`qa.py:253`）。
片段侧只有 `top_k` / `max_chunks` 这种**按条数**的限制 —— 条数与真实开销脱钩，
语料一长 prompt 总量就没有上界。而 `src/` 里 grep
`context_length|maximum context|too long|compact|truncat|overflow` **一条都没有**：
真的超了模型上下文就是直接抛 API 错误，没有降级路径。

## 1. 实测：片段侧到底吃掉多少 token

`scripts/probe_answer_budget.py`（读真实 `rag/chroma_db`，1520 块）：

```
语料：1520 块（后端 chroma）
单块 token（含 8 前缀开销）：min=23 p50=890 p90=1059 p99=1130 max=1182 mean=829
配置：top_k=16 · 片段预算=24000 token
      （对照：历史窗口预算=4000 token）

中位窗口（16 块）：16 条 / 14231 token → 16 条 / 14231 token（未裁剪）
最坏窗口（最大的 16 块）：16 条 / 18390 token → 16 条 / 18390 token（未裁剪）
理论极值（16 × max）：16 条 / 18912 token → 16 条 / 18912 token（未裁剪）
```

两个要点：

1. **风险在条数，不在单块**。单块已被入库时的 `chunk_size` 限住（实测 max 1182 token），
   所以**不需要**再给 chunk 加单块上限 —— 那是多余复杂度。
2. 片段侧最坏 18.4k，是历史窗口预算（4000）的 **4.6 倍**。加上历史就是 23k+，
   而这部分此前完全不可观测。

## 2. 修法

新增纯函数 `qa._select_answer_chunks(chunks, budget)` + 配置项
`answer_chunk_token_budget`，在 `answer` 节点接线。

### 为什么裁剪只能做在 `answer` 节点里（三个约束）

1. **不能裁到 `judge` 之前**。`judge` 是按「有没有 chunks」判路由的
   （`qa.py:1391`：`return 'answer' if state.get('chunks') else 'create_hitl'`）。
   裁早了正是 **2026-09-24 那个坑的形状**：命中了正确的章却落到错误的块，
   **有片段反而把 agent 挡在门外**（见 `reports/definite-np-anaphora-20260924.md`）。
   所以裁剪要**尽量晚**，晚到构建 prompt 的那一刻。
2. **prompt 与 citations 必须同源**。`_build_answer_prompt` 用
   `enumerate(chunks, start=1)` 编号 `[n]`，而 `answer` 节点随后又用同一份列表生成
   `citations`（其 `index` 注释明写「为 prompt 中该片段的编号」）。
   只改一处 → 编号错位 → 用户看到的引用编号与答案正文的 `[n]` 对不上。
3. **`_build_answer_prompt` 只有一个调用点**（`qa.py:905`），改这里就够了，
   不需要动检索节点或 agent 合并逻辑。

### 判据：与 `_select_window` 刻意同形

- 从**最相关**的一端往回收（`chunk[0]` 是检索排序第一名），丢的是末尾最弱的几条；
- **不做半块截断**：留下的都是完整片段，语义可预测、可断言；
- **第一条无条件保留**（`if picked and used + cost > budget` 里的 `picked and`）——
  否则预算偏小时片段会被清空，而 `judge` 刚刚才按「有片段」把路由判进 `answer`，立刻自相矛盾；
- 不排序、不重排（返回顺序即 citation 编号顺序）；
- `budget <= 0` 视为不限，与 `_clip_tokens` 的 `cap <= 0` 同义。

### 默认值 24000 的推导

覆盖实测「最坏窗口 18,390」与「理论极值 18,912」并留约 30% 余量。
**刻意取「今天不生效」**：这个预算是**护栏不是压缩器**。
本项目的教训是静默降质最贵，主动裁掉检索证据是负收益 ——
历史窗口那次反而把 6 条放大到 4000 token、3.7× 更多原文。
它的职责是拦住语料 / `top_k` 增长后的无界膨胀。

配了一条护栏测试 `test_default_budget_holds_sixteen_max_size_chunks`：
按实测上限量级造 16 条，断言默认预算**一条都不裁**。
若将来有人把默认值调小到会主动裁证据，这条会红。

### 顺带补上的先行指标

`answer` 节点新增一行日志：

```
问答[answer] 片段 %d/%d 条（预算 %d token）· prompt ≈%d token
```

prompt 总量此前**完全不可观测**。先量了这次 `_count_tokens` 的开销：
9.5k token 上中位 **0.94ms**（7 次），全量 prompt 约 2.3ms，相对一次作答的秒级耗时可忽略，
故常开而不加开关。

## 3. 验证

### 反证（逐处关掉，确认对应测试变红）

| 关掉的东西 | 结果 |
|---|---|
| A. citations 循环改回 `state['chunks']`（只改这一处） | 1 failed → 还原 1 passed |
| B. `_select_answer_chunks` 直接返回全部（关掉裁剪） | 1 failed → 还原 1 passed |
| C. 去掉「首条无条件保留」 | 1 failed → 还原 1 passed |

### 测试

- 新增 `tests/test_answer_chunk_budget.py`，13 条：整条保留 / 顺序不变 / 末位淘汰 /
  首条保留 / 不半块截断 / 零预算不限 / 空与 None / 不改入参 / 默认预算护栏 /
  编号 1..N 连续 / **真图级别的「prompt 编号与 citations 同源」不变式**。
- 最后一条用真实图跑（`InMemoryVectorStore` + 假 LLM + 小预算），断言
  prompt 里的 `[n]` 恰好是 `[1,2,3]`、`citations` 恰好 3 条且 index 为 `[1,2,3]`、
  入库片段数（5）多于保留数 —— 这条就是防「只改一处」的。
- 全量：**607 → 620 passed**（+13，账对得上）。
- `uv run ruff check src tests` → All checks passed。
  ⚠️ 但 `uv run ruff check .` 会报 **9 个既有错误**，全在 `experiments/multi_agent.py`（5）
  与 `scripts/deploy_trace.py`（4），与本次改动无关（这两个文件本次未触碰）。
  记忆里的「ruff 干净」应当是指 `src tests` 范围 —— 这一点以前没写明，补记在此。
- 跑全量要用 `uv run python -m pytest`：裸 `uv run pytest` 不把 CWD 加进 `sys.path`，
  会让 6 个 import `tests.*` / `eval.*` 的测试文件在**收集阶段**就报
  `ModuleNotFoundError`（假失败）。

## 4. 未做 / 边界（如实说明）

1. **端到端质量影响未实测。** 默认值今天不生效（实测最坏 18,912 < 24,000），
   所以「零影响」是**推理不是实测**。按本项目规矩标注：没有跑 A/B。
   若将来要调小默认值，必须先跑 A/B。
2. **仍然没有「整段 prompt 超出模型上下文」的总量兜底。** 本次补的是**分段**护栏
   （片段侧）。历史 4000 + 片段 24000 + 系统提示 ≈ 28k 是有界的，
   但没有任何地方拿这个总数去和模型的上下文长度比。真要小模型/超大 `top_k`，
   还得有一个跨段的总量检查 —— 本轮没做。
3. **`cached_chunk_count` 未改。** `qa.py:992` 记的仍是 `len(state['chunks'])`（未裁剪的条数），
   而实际产生答案是裁剪后的列表。刻意不动：`qa.py:391` 已写明该字段不用来判断陈旧性
   （加列前的历史行是 0），静默改变它的语义比这点不一致更糟。留作待定。
4. **没给 API 响应加新字段**（如「本轮裁剪了几条」），前端不动。裁剪只进日志。
5. **`graph/state.py:63` 那段与实现打架的 docstring 仍未修**（写「列表整体替换而非追加」，
   实际是累积去重 + cap 8）—— 本轮刻意不碰，另开一轮。

## 5. 改动清单

| 文件 | 内容 |
|---|---|
| `src/reading_assistant/config.py` | 新增 `answer_chunk_token_budget`（默认 24000，含推导依据注释） |
| `src/reading_assistant/graph/qa.py` | 新增 `_CHUNK_LABEL_OVERHEAD`、`_select_answer_chunks`；`answer` 节点接线（prompt 与 citations 同源）+ 先行指标日志；把节点内重复的 `get_settings()` 提到开头 |
| `tests/test_answer_chunk_budget.py` | **新建**，13 条 |
| `scripts/probe_answer_budget.py` | **新建**，读真实向量库量片段侧 token 分布与预算前后对照 |
| `reports/answer-chunk-budget-20260924.md` | 本文件 |
