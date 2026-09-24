# 记忆层替换：滚动摘要 → token 预算原文窗口 + 结构化会话状态

**日期**：2026-09-22
**触发**：用户报告「上下文压缩是超出 6 条后每次都上传给 LLM 进行压缩，耗时耗 token」，
并要求抛弃滚动摘要、换成业界成熟方案，为项目转向智能客服 agent 打基础。
**范围**：仅记忆层（用户明确选择）。**不含**指代消解进 retrieve。

---

## 1. 为什么滚动摘要是错的，而不是「可以优化」

删掉它不是因为实现差，是因为**原语选错了**。四条，逐条成立：

1. **无 schema 的有损 blob。** 它只能当提示词用，无法参与任何确定性判断
   （路由、升级、填槽）。客服场景要的恰恰是后者。
2. **每轮全量重写 = 静默失效生成器。** 一次改坏就永久污染后续所有轮次，
   且**没有任何信号**。这与本项目反复出现的失效形态同构
   （文档打回 indexing、前端未 rebuild、shell 旧 key 盖 .env：「两处都在工作，
   只是没人知道该信哪个」）。
3. **它位于 prompt 前缀且每轮都变 → 主动让 prefix cache 失效。**
   摘要在推高它本想省的那笔成本。
4. **反直觉**：长会话里原文 + 缓存比「每轮一次摘要 LLM 调用」更便宜。

业界现行表述（Anthropic *Effective context engineering for AI agents*）把长期任务
策略归为三条：**compaction / structured note-taking / sub-agent**。
compaction 的铁律是「token 阈值触发 + 压完保留最近 N 条原文」，**不是每轮压**。
本项目原本落在该谱系里最贵的形态上。

---

## 2. 改成了什么

三层正交，各管一件事：

| 摘要原来承担的 | 换成谁 | 性质 |
|---|---|---|
| 记住实体（书名/订单号/产品名） | **结构化状态** `ConversationState` | 精确、代码合并、**零 LLM 成本** |
| 记住近处细节 | **token 预算原文窗口** | 无损、吃 prefix cache |
| 记住「这场对话在谈什么」 | **状态里的 `active_documents`** | 精确书名，承接被窗口挤掉的上下文 |

### 核心机制：模型只提供信号，不负责重写记忆

`merge_state` 是**确定性纯函数**：新值非空才覆盖、未知键忽略、不原地修改入参。
于是可测、可审计、可回放、**无漂移**。模型改口只会产生一条新记录，
不会污染整份记忆 —— 这正是摘要做不到的。

### 零 LLM 成本是硬约束

`ConversationState` 的每个字段都来自流水线**已经算出来的**信号
（本轮检索结果的 `doc_titles`、本轮的 `intent`/`question`）。
`turn_count` 从消息表**数出来**而不是自增 —— `interrupt()` 恢复会让节点重跑，
自增会数重，数出来天然幂等。

> 这条约束是本次改造的全部意义所在：摘要正是死在「每轮一次 LLM」上，
> 换皮重来毫无价值。

---

## 3. 实测证据（禁缓存，`CACHE_ENABLED=false`）

> 项目纪律：凡跑 A/B 或统计，先禁缓存。本次探针第一版**又被语义缓存污染**
> （12 轮里 6 轮命中 semantic，score=0.9713），已按纪律复跑。
> 探针：`scripts/probe_memory_cost.py`、`scripts/probe_window_capacity.py`。

### 3.1 每轮 LLM 调用次数（12 轮，同一份探针跑新旧两版代码）

| 轮次 | 旧（滚动摘要） | 新（原文窗口 + 状态） |
|---|---|---|
| 1–4 | 2 | 2 |
| 5–12 | **3**（gate + 作答 + **summarize**） | **2** |
| **总计** | **32** | **24（−25%）** |
| 摘要调用 | **8** | **0** |

- 旧方案第 5 轮起**每轮恒定多一次 summarize**，总数 `3N−4`；
  新方案持平 `2N`。**差距随轮数线性拉大**：N=50 时旧 146 / 新 100（+46%）。
- 触发点是第 5 轮而非第 4 轮，因为 6 条窗口在第 4 轮时刚好装下前 3 轮的 6 条消息。

### 3.2 窗口容量（真实长度消息：用户 21 字符 / 助手 392 字符）

| | 旧 | 新 |
|---|---|---|
| 消息条数 | 6 | **14** |
| 注入 prompt 的原文 | 827 字符 ≈ 1097 token | **3030 字符 ≈ 3955 token** |
| 助手长答 | **每条被腰斩到 240 字符** | 完整保留 |

→ 原文量 **3.7 倍**，且零额外 LLM 调用。

### 3.3 测试

| | |
|---|---|
| 全量 | **510 → 546 passed**（删 1 条旧摘要用例，新增 37 条） |
| ruff | 干净 |
| 新增 | `tests/test_history_window.py`（14）、`tests/test_conversation_state.py`（20） |
| 改写 | `test_multi_turn.py::TestLongSessionMemory`（3 条），
  并把 `RouterLLM.summarize_calls` 从「应当触发」翻转为**永不该触发的哨兵** |

---

## 4. 一处我先做错、后来改正的设计

初版把 `active_documents` 定义为「**当前话题**涉及的书」，每轮**替换**。
但注入标签写的是【本会话涉及书籍】—— 我把自己骗了：这是两个概念，我混为一谈。

按该字段的**真实目的**（承接被原文窗口挤掉的上下文）判，替换是错的：
换一次书就把上一本抹掉，而它存在的意义恰恰是记住更早的事。
改成**累积 + 去重 + 有上限（8 条）**，合并策略放在
`qa._merge_active_documents`，让 `merge_state` 保持成一个不做领域判断的哑函数。

**教训**：字段名/标签和实际语义对不上时，文档里的那句话会替你做出错误决策。
写下定义时要用**消费方视角**读一遍。

---

## 5. 边界（如实说明，不粉饰）

1. **v1 状态字段有限。** 只有 `active_documents` / `last_question` /
   `last_intent` / `pending_clarification` / `turn_count`。
   **不具备**从对话里抽取具体实体名（如「寅将军」）的能力 ——
   那需要 LLM 抽取，与本轮「零成本」硬约束冲突，留到有明确消费方时再做。
2. **指代消解仍未进 `retrieve`。** `retrieve` 的 query 依旧只有原始问题字符串
   + 澄清补充。这是本项目已知的头号缺口（`ag-pronoun-02`），
   且它是**查询侧**问题，不是记忆长度问题 —— 本次改造不解决它。
   结构化状态已把 `active_documents` 准备好，接进 `retrieve` 是下一步。
3. **摘要输入长度的增长未在本探针中体现。** 旧方案的 summarize prompt
   在探针里恒为 628 字符，因为假模型返回定长摘要，`【既有摘要】` 段不增长。
   生产环境里摘要文本会随轮数增长（且 `_build_summary_prompt` **没有任何长度上限**），
   所以真实的成本差**大于** 3.1 表的调用次数差。这点是推理，不是实测，如实标注。
4. **compaction 兜底未实现。** 目前窗口超预算就丢最早的轮次。
   极长会话（数百轮）仍需要一次 compaction + 保留最近 N 条原文，
   本轮没做，因为现有窗口 3955 token 对客服会话（中位数 5–15 轮）绰绰有余。

---

## 6. 改动的文件

| 文件 | 内容 |
|---|---|
| `src/reading_assistant/graph/state.py` | **新建**。`ConversationState` + `merge_state` + 序列化 + 读写 |
| `src/reading_assistant/graph/qa.py` | 删 summarize 节点与三处摘要函数；`context_load` 改 token 窗口；`_select_window`/`_clip_tokens`/`_count_tokens` 新增；prompt 重排；`_state_section`/`_merge_active_documents`/`_persist_conversation_state` 新增 |
| `src/reading_assistant/config.py` | `history_token_budget`/`history_msg_token_cap`/`history_max_messages` |
| `src/reading_assistant/storage/models.py` | 加 `state` 列；`summary` 标废弃（保留不 DROP） |
| `src/reading_assistant/storage/database.py` | 加 `state` 列迁移 + `_purge_deprecated_summary`（收敛式清空） |
| `tests/test_history_window.py` | 新建，14 条 |
| `tests/test_conversation_state.py` | 新建，20 条 |
| `tests/test_multi_turn.py` | `TestLongSessionSummary` → `TestLongSessionMemory` |
| `scripts/probe_memory_cost.py` | 新建，成本曲线探针 |
| `scripts/probe_window_capacity.py` | 新建，窗口容量探针 |

### prompt 顺序重排（顺带做实的一处收益）

原来 `_build_answer_prompt` 是 `系统提示 → 问题 → 历史 → 原文片段`，
**把每轮都变的「问题」放在历史之前**，等于每轮都从问题那一点起让 prefix cache 失效。
改成 `系统提示 → 历史 → 问题 → 澄清 → 原文片段 → 要求`：
历史是追加式增长的，第 N 轮的历史文本是第 N+1 轮的**字面前缀**，
缓存因此能覆盖「系统提示 + 全部历史」。

---

## 7. 给智能客服方向的扩展点

机制（schema + 确定性合并 + 落库 + 坏数据不炸）已经就位，扩到客服只需两步：

1. 在 `ConversationState` 加字段（`order_id` / `product` / `issue_type` / `escalated` …）；
2. 调用方把新信号塞进 `_persist_conversation_state` 的 `patch`。

`merge_state` 的「未知键忽略」已经为这一步铺好了路：加字段不会让旧记录读不出。

**若新槽位必须从对话里抽取**（如订单号出现在用户自由文本里），
再考虑 LLM 抽取 —— 但应当**搭在已有的作答调用上**（structured output 同一次返回），
而不是新增一次调用。重新引入「每轮一次额外 LLM」是本轮刻意拆掉的东西。
