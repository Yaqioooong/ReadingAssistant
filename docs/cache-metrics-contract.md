# 缓存指标体系：接口契约（前后端共同依据）

> 2026-09-14 · 目标：让三层缓存（精确 / 语义 / 标识符变体）从"只有精确命中可观测"变为全链路可观测，并补上误命中率。

## 1. 背景：现状缺口

`qa_cache` 表已有 `hit_count` / `last_hit_at` / `created_at` / `needs_clarification` / `citations` / `document_id`，
可算存量指标；但**运行期**指标存在三个缺口：

| 缺口 | 说明 |
|---|---|
| 通道归因缺失 | 代码只在精确命中处打日志，语义命中与标识符复用命中**静默返回**，无法回答"三层各贡献多少" |
| 阈值无据可依 | `cache_similarity_threshold`(0.95) 的命中相似度分布从未采集 |
| 误命中率未知 | 缓存返回错误答案的次数完全不可见 |

## 2. 数据模型（新增两张表）

### `qa_request_events`（请求级埋点，P0/P1）

每轮问答由 API 路由层写一条。放在路由层而非 graph，理由是耗时(latency_ms)在路由层才可得，
且 MCP 等非 HTTP 入口不必写。

| 字段 | 类型 | 说明 |
|---|---|---|
| id | int PK | |
| session_id | int \| null | |
| question | Text | 原始问题（截断到 200 字） |
| intent | String(16) \| null | 检索门分类 book/history/chat |
| cache_hit | Boolean | 是否命中缓存 |
| cache_channel | String(16) \| null | `exact` / `semantic` / `identifier` / `miss` / `disabled`（多文档或缓存关闭，不适用）/ `skipped`（闲聊与历史类轮次，未经缓存检查） |
| cache_similarity | Float \| null | 命中的相似度（精确命中为 null） |
| cache_invalidated | Boolean | 是否因"信息不足式回答"丢弃了脏缓存 |
| latency_ms | Integer \| null | 本轮总耗时 |
| created_at | DateTime(tz) | |

### `qa_feedback`（用户反馈，P2）

| 字段 | 类型 | 说明 |
|---|---|---|
| id | int PK | |
| session_id | int \| null | |
| question | Text | 该轮问题 |
| vote | String(8) | `up` / `down` |
| cache_hit | Boolean | 该答案是否来自缓存 —— 误命中率的分母依据 |
| cache_channel | String(16) \| null | |
| created_at | DateTime(tz) | |

问答接口需要在响应里回传缓存信息，前端据此提交反馈时带上。

## 3. QAState 新增字段（graph/qa.py）

```python
cache_channel: str | None      # exact | semantic | identifier | miss | disabled
cache_similarity: float | None # 语义/标识符命中时的 best_score
cache_invalidated: bool        # 是否丢弃过脏缓存
```

`cache_check` 所有 return 分支都要带上这三项；`AskResponse` 增加 `cache_hit` / `cache_channel` 字段
（`cache_similarity` 可选，便于前端展示）。

`record` 节点写 ChatMessage 时在 meta 里带上 `cache_hit` / `cache_channel`，历史回放也能显示来源。

## 4. 接口契约

### `GET /api/stats/cache?days=7`

`days` 默认 7，范围 1~90。

```json
{
  "window_days": 7,
  "summary": {
    "total_requests": 1987,
    "cacheable_requests": 1902,
    "hit_rate_cacheable": 0.2765,
    "hit_total": 526,
    "miss_total": 1461,
    "hit_rate": 0.2647,
    "avg_latency_hit_ms": 130,
    "avg_latency_miss_ms": 188,
    "saved_ms_total": 30500,
    "saved_calls": 526
  },
  "by_channel": {
    "exact": 220, "semantic": 275, "identifier": 31,
    "miss": 1461, "disabled": 20, "skipped": 65
  },
  "semantic": {
    "count": 275,
    "p50": 0.972, "p95": 0.991, "min": 0.951, "max": 0.999,
    "threshold": 0.95,
    "near_threshold": 12
  },
  "invalidated": { "count": 49, "rate": 0.0247 },
  "feedback": {
    "up": 40, "down": 6, "total": 46,
    "cache_hit_feedback": 20, "cache_hit_down": 4,
    "mis_hit_rate": 0.2,
    "non_cache_feedback": 26, "non_cache_down": 2,
    "non_cache_error_rate": 0.0769
  },
  "entries": {
    "total": 78, "cold": 30, "hitl": 24, "with_citations": 51,
    "expiring_soon": 3, "total_hits": 130,
    "top_hits": [{ "question": "…", "hit_count": 12, "last_hit_at": "…" }]
  },
  "daily": [{ "date": "2026-09-14", "hit": 10, "miss": 30 }]
}
```

- `count` 与 `p50/p95/min/max` 只统计**语义通道命中**样本（口径与 `by_channel.semantic` 一致）；无样本时全部为 null。
- `near_threshold`：**未命中但相似度距阈值不足 0.01** 的次数（「擦肩而过」）——下调阈值能多拿多少命中的直接预估。
- `hit_rate` 用全量请求作分母；`hit_rate_cacheable` 排除 `disabled` 与 `skipped` 后计算。
  闲聊类轮次永远不命中，用全量口径会稀释命中率、失真，**看板应以 cacheable 口径为准**。
- `expiring_soon`：`created_at` 距今超过 `cache_ttl_days - 3` 天的条目数。
- `mis_hit_rate` = `cache_hit_down / cache_hit_feedback`（分母为 0 时 null）。

### `POST /api/feedback`

请求：`{ "session_id": int|null, "question": str, "vote": "up"|"down", "cache_hit": bool, "cache_channel": str|null }`
响应：`{ "id": int, "vote": "up" }`。`vote` 非 up/down → 422；`question` 为空 → 422。

## 5. 前端

### 新 Tab「📊 指标」→ `StatsPanel.vue`
- 顶部 KPI 卡：命中率 / 命中总量 / 平均节省 / **误命中率**（无反馈时显示「—」）
- 通道分布：exact / semantic / identifier 三段式条形（带绝对数与占比）
- 语义命中相似度：P50 / P95 / min / max / 阈值，附「贴近阈值」计数
- 每日趋势：hit vs miss 双色柱
- 热门缓存 Top10 表格（问题 / 命中次数 / 最近命中）
- 存量卡片：总条目、冷条目、HITL 式、临近过期
- 时间窗切换（1 / 7 / 30 天）+ 手动刷新

### ChatPanel：assistant 消息下加 👍 / 👎
- 点击调 `POST /api/feedback`，提交后按钮置为已选状态并禁用
- 历史消息若 meta 里带 `cache_hit`，显示「缓存」小标签（让用户知道答案来源）

## 6. 验收标准

1. `uv run pytest tests/ -q` 全绿（含新增埋点/路由/指标计算用例）。
2. 三通道可分别归因：构造精确 / 语义 / 标识符三类命中，各自落到正确 channel。
3. `GET /api/stats/cache` 返回上述结构，空库不报错（各项为 0 / null）。
4. 反馈写入后，`mis_hit_rate` 随之变化。
5. `uv run ruff check src tests` 通过（100 列限制）。
6. 前端 `npm run build` 通过。
