# QA 问答缓存机制设计方案

> 目标：为 QA 流水线（`src/reading_assistant/graph/qa.py`）引入缓存，减少 DeepSeek / DashScope 的 token 消耗与 API 费用。
> 状态：阶段一（L1 embedding + L2 精确答案缓存）与阶段二（语义相似缓存）均已实现；阶段三（规模化）未开始。

---

## 1. 背景与目标

本项目是 RAG 阅读助手：电子书入库 → 分块向量化 → 用户提问 → 检索原文片段 → LLM 生成回答。

问答流水线为 LangGraph 状态图：

```
START → retrieve → judge → answer / create_hitl → record → END
```

每次提问会触发两类**付费 API 调用**（按 token 计费）：

| 调用点 | 模型 | 说明 |
| --- | --- | --- |
| `Retriever.retrieve()` 内 `embed_query` | DashScope `text-embedding-v4` | 问题向量化 |
| `answer` 节点 `chat_model.invoke` | DeepSeek（`BaseChatOpenAI`） | 生成回答，**大头** |

**设计目标**：对重复 / 语义近似的提问复用历史结果，跳过 LLM 与 embedding 调用，从根源上削减 token。

---

## 2. Token 消耗分析

`answer` 节点的 prompt 由 `_build_answer_prompt` 构造，会把 `top_k`（默认 6）个检索片段拼入，每个片段 `chunk_size`（默认 800 字符）：

```
单次 input ≈ 6 × 800 字符 ≈ 4800 汉字 ≈ 3000~5000 token（中文）
加上输出，单次问答 ≈ 4000~6000 token
```

结论：**answer 是主要消耗点，embedding 是次要但稳定的小额消耗**。缓存应优先覆盖 answer，其次覆盖 embedding。

---

## 3. 缓存分层设计

采用两级缓存，逐层拦截：

```
        ┌─────────────────────────────────────────┐
        │ L2 答案缓存（精确 + 语义）              │
        │  命中 → 整条 retrieve→answer 短路       │
        │  ├─ 精确哈希匹配（相似度=1）            │
        │  └─ 语义相似度匹配（余弦 > 阈值）        │
        └──────────────────┬──────────────────────┘
                           │ miss
        ┌──────────────────▼──────────────────────┐
        │ L1 Embedding 缓存（精确）                │
        │  命中 → 跳过 embed_query               │
        └─────────────────────────────────────────┘
```

### 3.1 L1 — Embedding 缓存（精确命中）

- **作用**：相同（或归一化后相同）的 query 不再重复调用 `embed_query`。
- **Key**：`sha256(归一化问题 + 补充说明)`
- **Value**：向量（JSON 存储，MVP 可接受）。
- **特点**：embedding 模型对相同输入输出确定 → 精确匹配零误判风险，可作为兜底层。

### 3.2 L2 — 答案缓存（精确 + 语义，收益最大）

参考 GPT-Cache 的语义缓存思路，分两档命中：

**① 精确命中（相似度 = 1）**
- Key：`sha256(content_hash | document_id | 归一化问题 | 归一化补充说明)`
- 命中即返回缓存答案 + citations。

**② 语义命中（相似度 > 阈值）**
- 新问题先算向量（走 L1），与「同文档 + 同 content_hash」范围内的历史问题向量计算**余弦相似度**，取 Top-1 候选。
- 相似度 ≥ 阈值（建议起始 `0.95`，稳定后下调到 `0.90~0.92`）即复用答案。
- 例：「第一章讲了啥」与「第一章主要内容是什么」可互相命中。

**命中后行为**：直接从缓存返回 `answer` + `citations`，**跳过 `retrieve` / `judge` / `answer` 全部节点**，直达 `record`。

---

## 4. 缓存键与归一化

### 4.1 归一化规则（问题文本）

- Unicode NFKC 规范化
- 小写化
- 去除空白与标点（保留中英文语义无关的标点差异）
- 可选：繁→简（中文场景）

### 4.2 键组成

所有答案缓存键必须包含 **版本键**，保证文档变更后旧缓存失配：

```
cache_key = sha256(
    content_hash        # Document.content_hash，文档版本
    + "|" + document_id # 文档范围隔离（None 表示全库）
    + "|" + question_normalized
    + "|" + clarification_normalized   # 有补充说明时纳入，None 记为空
)
```

---

## 5. 数据模型（SQLAlchemy）

新增表 `qa_cache`，与现有 `chat_messages` 同库（SQLite / 现有关系型存储）：

```python
class QaCacheEntry(Base):
    __tablename__ = "qa_cache"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    question_raw: Mapped[str] = mapped_column(Text, nullable=False)
    question_normalized: Mapped[str] = mapped_column(Text, nullable=False)
    question_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    question_embedding: Mapped[list] = mapped_column(JSON)   # MVP：JSON 存向量
    answer: Mapped[str | None] = mapped_column(Text)          # needs_clarification 时为空
    citations: Mapped[list] = mapped_column(JSON, default=list)
    needs_clarification: Mapped[bool] = mapped_column(Boolean, default=False)
    document_id: Mapped[int | None] = mapped_column(ForeignKey("documents.id"), index=True)
    content_hash: Mapped[str] = mapped_column(String(64), index=True)  # 失效版本键
    hit_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    last_hit_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
```

> 说明：`needs_clarification` 场景（信息不足）也可缓存，避免重复的「无法回答」继续耗 token。

---

## 6. 失效与淘汰策略

| 策略 | 方案 |
| --- | --- |
| **版本失效** | 以 `content_hash` 为版本键。文档重传 → `content_hash` 变化 → 旧缓存自然失配（查询时按 content_hash 过滤）。可选：ingest 成功后 `DELETE FROM qa_cache WHERE content_hash = old_hash` |
| **TTL** | 配置项 `CACHE_TTL`（建议 30 天），`last_hit_at` / `created_at` 超期删除 |
| **容量上限** | 配置项 `CACHE_MAX_ENTRIES`（建议 10000），超过后按 `last_hit_at` LRU 淘汰 |

---

## 7. 代码落点

### 7.1 L1（最小侵入）

在 `Retriever.retrieve()` 内、`embed_query` 调用前查 L1 缓存；未命中则调用并写回。

### 7.2 L2（推荐短路优化）

在 `build_qa_graph` 中新增 `cache_check` 节点，置于入口：

```
START → cache_check ──hit──→ record ──→ END
                │
               miss
                ↓
        retrieve → judge → answer / create_hitl → record
```

- `cache_check`：归一化问题 → 算向量（走 L1）→ 精确哈希查询 → 语义 Top-1 查询 → 命中则把缓存结果写入 state（`answer` / `citations` / `needs_clarification`），并路由到 `record`。
- 未命中 → 走原有 `retrieve → …` 流程，最终在 `answer`（或 `create_hitl`）完成后**异步/同步写回缓存**。

---

## 8. 收益预估

| 场景 | token 节省 |
| --- | --- |
| 精确重复提问 | 全部归零（连 embedding 都命中 L1） |
| 语义近似提问 | LLM token 归零，仅付一次问题 embedding |
| 首次提问 | 无节省（正常付费，并写入缓存） |

命中率取决于真实提问分布；书籍类 QA 常出现「换说法问同一件事」，语义层预计可带来 20%~50% 命中率提升。

---

## 9. 分阶段落地计划

1. ✅ **阶段一（已完成）**：L1 embedding 缓存 + L2 精确答案缓存（含 qa_cache 表、TTL、LRU 淘汰）。
2. ✅ **阶段二（已完成）**：语义相似度匹配（余弦，阈值 0.95 起步，同文档范围候选）。
3. **阶段三（规模化，可选）**：Redis 后端 + 独立向量集合（复用 Chroma 或新增集合），替换 JSON 向量存储；加入 singleflight 防并发穿透。

---

## 10. 风险与边界

- **语义误判**：阈值过低可能返回不相关答案 → 保守起步（0.95），配合 `hit_count` 统计观察。
- **多轮上下文**：当前 graph 为**单轮无状态**（每次独立 question），缓存按单轮 key 天然安全。若未来引入多轮历史，需把历史摘要纳入 key。
- **补充说明（clarification）**：必须纳入 key，否则澄清前后会串答案。
- **缓存一致性**：书籍内容固定，天然适合缓存；唯一失效源是「重新入库」，已用 `content_hash` 覆盖。
- **并发穿透**：单用户 MVP 可忽略；规模化阶段用 singleflight / 分布式锁。
- **与现有 `@lru_cache` 无关**：`factory.py` 中的 `@lru_cache` 是模型实例缓存（进程内），与本文的响应级缓存正交，二者可共存。

---

## 11. 新增配置项（建议）

| 配置 | 默认 | 说明 |
| --- | --- | --- |
| `CACHE_ENABLED` | `true` | 总开关 |
| `CACHE_SIMILARITY_THRESHOLD` | `0.95` | 语义命中阈值 |
| `CACHE_TTL_DAYS` | `30` | 缓存有效期 |
| `CACHE_MAX_ENTRIES` | `10000` | 容量上限 |
