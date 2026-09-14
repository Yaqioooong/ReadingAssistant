# ReadingAssistant 架构说明

> 面向协作者 / 面试复习的项目架构文档。配合 `docs/interview-concepts.md`（概念弹药库）阅读。
> 文档版本：2026-09-03。代码若与文档不一致，以代码为准（此项目仍在演进）。

---

## 1. 项目概览

基于检索增强生成（RAG）的本地电子书阅读助手：电子书解析入库 → 向量索引 → 多轮问答，带引用溯源与人工澄清（HITL）闭环。

```
Vue3 前端
  │  HTTP (fetch)
FastAPI（routes: documents / sessions / hitl）
  │
  ├─ LangGraph 入库图  : add_book → chunk_and_index
  ├─ LangGraph 问答图  : cache_check → retrieve → judge → answer|create_hitl → record
  │
  ├─ SQLAlchemy ── PostgreSQL(生产) / SQLite(测试/内存)
  └─ VectorStore ── ChromaDB(生产) / InMemory(测试)
```

**核心设计哲学**：一切流程都是显式状态机。节点单责、路由函数决定走向、状态字段（`index_status`、HITL 任务状态）落库可观测。没有散落各处的大型 if/else 业务链。

---

## 2. 目录结构导览

```
src/reading_assistant/
├── api/                  # FastAPI 路由与依赖注入
│   ├── app.py            # create_app（中间件、注入覆盖、静态托管）
│   ├── deps.py           # lru_cache 单例依赖（测试可覆盖）
│   ├── schemas.py        # Pydantic 请求/响应模型
│   └── routes/           # documents / sessions / hitl
├── parsers/              # txt/epub/pdf/docx 统一解析接口
├── rag/                  # chunking（切块）+ retriever（检索）
├── graph/                # LangGraph 流水线
│   ├── ingest.py         # 入库图
│   ├── qa.py             # 问答图（含 QA 缓存、HITL、工具判定）
│   └── checkpointer.py   # 图状态持久化（内存 / PostgreSQL）
├── storage/              # ORM 模型 / repository / service / 向量库适配
└── model/factory.py      # ChatDeepSeek + DashScopeEmbeddings 工厂
eval/                     # 黄金评测集 + 评测脚本
experiments/multi_agent.py# 多 Agent 协作实验
scripts/deploy_trace.py   # 部署验证 + trace 留档
tests/                    # 90 个自动化测试
```

---

## 3. 数据模型（SQLAlchemy）

| 表 | 职责 | 关键字段 |
|---|---|---|
| `documents` | 文档元数据 | file_hash/content_hash(唯一)、chunk_count、file_path、index_status |
| `chat_sessions` | 会话 | title（未用）、created_at、summary(JSON，长会话滚动摘要) |
| `chat_messages` | 消息 | role、content、meta(JSON，存 citations) |
| `hitl_tasks` | 澄清任务 | status(awaiting/approved/rejected)、clarification |
| `qa_cache` | 问答缓存 | question_hash(唯一)、question_embedding、answer、citations、content_hash、hit_count、last_hit_at |
| `qa_request_events` | 缓存埋点（请求级） | cache_hit、cache_channel、cache_similarity、cache_invalidated、latency_ms |
| `qa_feedback` | 用户反馈 | vote(up/down)、cache_hit —— 误命中率的数据来源 |

`documents.index_status`：`pending → indexing → indexed / failed`——双写一致性状态机（见 §7）。

---

## 4. 解析层

- 统一产物 `ParsedBook(title, author, chapters=[Chapter(title, content)])`，上层只认这一个结构
- 工厂 `get_parser(filename)` 按扩展名分发；不支持格式抛 `UnsupportedFormatError`（API → 422）
- txt 解析：自动编码探测 → 正则切章节标题（`第一章 开端` / `第1章` / `Chapter 2` / `序章`…）→ 无标题时兜底 `Chapter('正文')`
- 解析产物同时用于：内容哈希去重、入库切块、reindex 重放

---

## 5. 入库服务与分层去重（storage/service.py）

```
add_book(path):
  1. file_hash = sha256(文件原始字节)         # 同一文件重复上传 → 秒杀
  2. content_hash = sha256(normalize(全文))   # 同书不同文件名 → 识别
  3. insert；唯一索引兜底并发冲突（IntegrityError → rollback → 返回既有记录）
```

- `normalize_text`：空白折叠成单空格再哈希——"排版不同、内容相同"也能去重
- 重复上传返回 `duplicate=True` + 既有 document，不重复解析

---

## 6. 向量库抽象（storage/vector_store.py）

`VectorStore` ABC：`add / query / count / delete`；两种实现：

| 实现 | 用途 | 说明 |
|---|---|---|
| `ChromaVectorStore` | 生产 | PersistentClient 本地持久化 |
| `InMemoryVectorStore` | 测试/降级 | `dict[str, StoredChunk]` |

关键设计：
- **chunk id 确定性**：`doc{document_id}-{chunk.index}` → 向量写入天然幂等，upsert 覆盖
- **score 语义统一**：Chroma 返回距离（小=相关），InMemory 返回余弦相似度（大=相关）——Chroma 侧 `1/(1+d)` 转为相似度，保证下游阈值过滤语义一致
- chunk metadata 带 `document_id`(int) + `chapter` + `chapter_index`，支撑单文档过滤与引用溯源

---

## 7. 入库流水线与双写一致性（graph/ingest.py）

### 图结构
```
START → add_book ──(duplicate && !force)──→ END
                └──(否则)──→ chunk_and_index → END
```
- `force=True`（reindex）跳过 duplicate 短路，重放索引
- `index_status` 贯穿：add_book 置 `indexing` → 成功置 `indexed`，失败落 `failed` 并 **re-raise**

### 双写一致性方案：状态机 + 幂等重放
问题：文档元数据（SQLite/PG）与向量（Chroma）**无跨存储事务**，存在半入库、计数漂移、无补偿的缺口。

解决：
1. `add_book` 写元数据（事务 A）；`chunk_and_index` 向量写入失败 → `index_status='failed'` + 抛异常（失败要大声）
2. `chunk_count` 与 `index_status='indexed'` 在**同一事务**内更新，消除第三处写不一致
3. reindex 端点 `POST /api/documents/{id}/reindex`：`force=True` 重放全图，确定性 chunk id 覆盖旧向量，天然幂等
4. 坑：file_hash 命中时 `add_book` 返回 `parsed=None`（不重复解析）——reindex 需在 force 模式下补 `parse_book`，否则章节为空、索引被清空

---

## 8. 检索器（rag/retriever.py）

- `retrieve(query, document_id=None)`：embed → 向量库 top_k → **`score >= retrieval_min_score` 过滤**
- embedding 带 L1 缓存（OrderedDict），供缓存语义匹配复用
- 阈值校准方法：跑真实场景看分数分布。实测"少白公"相关片段 0.58、无关 0.40~0.42（断层明显）→ 默认阈值 **0.45**

---

## 9. 问答流水线（graph/qa.py）

```
START → cache_check ─(miss)→ retrieve → judge ─(信息不足)→ create_hitl → record → END
        │命中↓                    ↑        └(充足)→ answer ─(工具判定不足)→ create_hitl ↗
        ├─ needs_clarification? ─ create_hitl ────────────────────────────────↗
        └─ 正常命中 → record ─────────────────────────────────────────────────↗
```

### cache_check：三级缓存
- 键：`question_hash = sha256(normalize(question) | normalize(clarification) | document_id)`
  - **document_id 进 hash**：同问题问不同书 = 不同缓存条目（原版漏了，跨文档必撞唯一约束）
- 精确命中：`(hash, content_hash, document_id)` 三元组查表
- 语义命中：遍历同文档候选，问题 embedding 与 `question_embedding` 余弦相似度 ≥ 0.95
- 生命周期：TTL 过期删除、LRU 按 `last_hit_at` 淘汰、命中计数 +1
- **防污染**：命中条目的 answer 若命中"信息不足"式检测 → 删除 + 当未命中（杜绝旧垃圾缓存回流）

### judge
`needs = not chunks and not clarification`——无检索结果且无澄清说明 → HITL

### answer：工具调用判定（主）+ 双兜底
```
bind_tools([final_answer, request_clarification]).invoke(prompt)
 ├─ request_clarification → 转 HITL（不缓存、不落聊天记录）
 ├─ final_answer          → 取 args.answer 为正式回答（正常缓存）
 └─ 无 tool_calls（纯文本）→ _looks_like_no_info 关键词兜底
工具调用异常 → 回退 chat_model.invoke
```
设计意图：判定权交给 LLM 语义理解，让它区分"原文没写（诚实答没有）"与"信息不充分（要澄清）"——关键词方案做不到。澄清后仍判定不足 → 不再重复建任务，把 missing_info 作为说明文本返回。

### 引用列表：与答案 [n] 标记对齐
召回片段在 prompt 中按顺序编号，模型被要求用 [n] 标注依据。回答文本可能只用到其中少数几条，
若把召回结果整段回传，用户会看到"只用了一条却列出六条"的噪声。因此 answer 节点做两步处理：

```
所有召回片段 → 带原始 index
  ├─ 解析答案里的 [n]（_used_citation_indexes，越界/重复自动忽略）
  ├─ 有标记 → 只保留被引用的片段
  │    无标记 → 保留全部（兜底，避免丢失溯源线索）
  └─ 重排为连续 1..n，并同步改写答案里的 [n]（_renumber_citations）
       一次正则替换完成，避免 [1]↔[2] 互换时的串联污染
```

重排的意义：只引用第 3 条时，列表显示 `[3]` 会让人以为丢了前两条；重排后统一从 `[1]` 起连续编号。
写入缓存前完成（answer 与 citations 一并落库），保证命中缓存时展示一致。

### 缓存可观测性（指标看板）
三层缓存的命中判定原先只有精确命中打日志，语义层与标识符层**静默返回**，
"各通道贡献多少"无法回答，缓存阈值（0.95）也缺乏调优依据。现在：

```
cache_check 每个 return 分支 → state.cache_channel
   exact | semantic | identifier | miss | disabled | skipped
   （disabled=多文档或开关关闭；skipped=闲聊/历史类轮次，本就未经缓存检查）
        ↓
API 路由层写 qa_request_events（含 latency_ms，路由层才拿得到耗时）
        ↓
GET /api/stats/cache → 请求级归因 / 相似度分位 / 脏缓存 / 反馈 / 存量健康度
POST /api/feedback  → 缓存答案被点踩即计入误命中率
```

两个关键口径决策：
- **命中率用「可缓存请求」作分母**（排除 disabled 与 skipped）。闲聊轮次永远不命中，
  用全量口径会稀释指标、失真；实测同一批流量：全量口径 50% vs 可缓存口径 71.4%。
- **`near_threshold` 统计「擦肩而过」**（未命中但相似度距阈值 <0.01 的次数），
  这才是下调阈值能多拿多少命中的直接预估 —— 命中样本的相似度必然 ≥ 阈值，按命中统计恒为空。

看板见前端「📊 指标」tab（StatsPanel.vue）；契约细节见 docs/cache-metrics-contract.md。

### HITL 与缓存联动
- `create_hitl` 存 HITL 缓存（answer=None, needs_clarification=True）→ 同问题下次直接命中走 HITL，不再烧 LLM
- 任务状态机 awaiting → approved/rejected；approved 后带 `clarification` 重问，judge 视为已补充
- 前端 `/api/hitl/tasks` 打通：信息不足时渲染补充卡片（提交 → approved 后重跑 / 放弃 → rejected），
  切换会话自动恢复未处理任务；`record` 节点把澄清补充并入用户消息，历史可回放当时实际检索文本

---

## 10. API 层（api/）

- 路由：`/api/documents`（list / upload / reindex）、`/api/sessions`（创建/列表/消息/删除）、`/api/hitl/tasks`（list / submit / reject）、`/api/stats/cache`、`/api/feedback`
- `deps.py` `@lru_cache` 缓存单例；`create_app(...)` 注入参数 → `dependency_overrides`——测试全链路替换 SQLite + Fake 模型 + InMemory 向量库
- 删会话：先删 HITL 任务（外键无级联）→ 再删会话（messages ORM 级联）
- 中间件：每请求记录 method / path / status / 耗时

---

## 11. 日志与可观测性（utils/logger_handler.py）

- 4 个命名 logger：`api` / `ingest` / `qa` / `storage`（+ 实验用 `multi_agent`）
- console INFO + 按天文件 DEBUG，格式含 `文件名:行号`
- 关键节点埋点耗时：解析、add_book、chunk_and_index、LLM 返回、整次提问、缓存命中状态

---

## 12. 测试与评测（双轨）

| 轨道 | 回答的问题 | 内容 |
|---|---|---|
| `tests/` 90 用例 | 对不对 | 单元（解析/存储/向量库）+ 图流水线 + API 集成；覆盖并发去重、双写失败、缓存污染、跨文档缓存冲突、工具两分支、HITL 全流程、会话级联删除 |
| `eval/` 黄金集 15 题 | 好不好 | 陷阱题（防反向误导）、信息不足题（防幻觉）、全库检索题；fake（链路冒烟）/ real（真实模型全链路）双模式 |

真实评测结果（2026-09）：可答准确率 100%、不可答识别率 100%、引用覆盖率 100%、平均延迟 1.28s。

---

## 13. 多 Agent 实验（experiments/multi_agent.py）

Supervisor → 检索 Worker → 总结 Worker 三级协作：
- **Supervisor**：查看馆藏 → 工具调用输出结构化计划（`make_plan`: doc_scope / search_query / note）
- **检索 Worker**：按计划执行向量检索（doc_scope 路由到具体文档或全库）
- **总结 Worker**：基于片段生成带 `[n]` 引用的回答

实测：Supervisor 能把"张三喜欢谁"定向到《张三爱情故事.txt》而非全库扫描。

---

## 14. 踩坑记录（面试前必翻）

1. **双写半入库**：向量写入失败时元数据已提交 → `index_status` 状态机 + reindex 幂等重放解决
2. **reindex 清空索引**：file_hash 命中 parsed=None → force 模式补 parse_book
3. **缓存丢 citations**：`save_qa_cache_entry` 新增分支漏存 citations 字段 → 命中缓存引用区恒空
4. **缓存跨文档冲突**：question_hash 全局唯一但不含文档维度 → hash 拼入 document_id
5. **污染缓存回流**：旧代码把"信息不足"回答当正常回答缓存 → 命中时校验 answer 质量，无效即删
6. **Chroma score 语义相反**：距离 vs 相似度 → `1/(1+d)` 统一
7. **关键词判定片面**：HITL 充分性从关键词启发式升级为 `@tool` 工具调用（LLM 语义表态）
8. **语义命中静默**：三层缓存只有精确命中打日志，语义/标识符层直接 return → 命中无法归因。
   现由 `cache_check` 每个分支写 `state.cache_channel`，路由层落 `qa_request_events`
9. **命中率被稀释**：闲聊轮次不经缓存却计入分母，命中率失真。实测同批流量 **全量 50% vs 可缓存 71.4%**
   → 引入 `disabled`/`skipped` 两类"不适用缓存"并改用可缓存口径
10. **写入后立刻读的竞态**：FastAPI 的 yield 依赖其退出代码在**响应发出之后**才执行，
    `create_session` 只 flush 未 commit → 前端「新建会话 → 立刻提问」偶发 404（会话不存在）。
    改为显式 `session.commit()` 消除（连续 6 轮验证 0 失败）
11. **测试夹具踩到归一化**：`Retriever._embed` 按 `normalize_question` 后的文本做 embedding 并缓存，
    构造语义命中用例时关键词必须按归一化形态写（小写、无标点），否则静默落到默认向量
8. **缓存 tuple 污染**：`existing.answer = (answer,)` 元组包裹 → 更新分支写坏数据
9. **阈值校准依赖实测**：0.45 来自真实分数分布观测，换 embedding 模型需重新校准
