# HITL 真「中断-恢复」实现（2026-09-17）

## 一、改造前是什么状态

「HITL」名义上是人机协同，实际是**客户端重放 + 服务端重算**：

```
提问 → retrieve → judge(信息不足) → create_hitl(建 DB 任务) → record → END
客户端凭 HitlTask 提交澄清 → 重新 POST 一个问题(带 clarification) → 全新 thread_id → 从 START 全量重跑
```

图从未挂起过，因此：

- `create_postgres_checkpointer` 定义了但**全仓库无人调用**；
- 每次 invoke 用一次性 `thread_id`，还挂着 `InMemorySaver` —— 每个节点白写一次 checkpoint（实测一次问答 10 次），却没有任何读取方（`get_state` 零调用）；
- 「恢复」要多付一遍 `gate`(LLM) + `retrieve`(含检索) + `answer`(LLM)，且是两个不同的 thread。

顺带修掉两个从未被执行因而没暴露的缺陷：

| 缺陷 | 症状 | 说明 |
| --- | --- | --- |
| `PostgresSaver.from_conn_string(...)` 返回的是**上下文管理器** | 对它调 `.setup()` 直接 `AttributeError` | 从没被调用过，所以没人发现 |
| `setup()` 每次调用都重建连接 | 若真用起来会重复建表/建连接 | 现改为建池时跑一次 |

## 二、现在怎么工作

```
提问 → context_load → record_question → summarize → gate → cache_check
     → retrieve → judge(不足) → create_hitl(建任务) ──interrupt()──▶ 挂起
                                                                    │
客户端 POST /api/hitl/tasks/{id}/submit(clarification)              │
     → Command(resume=...) ────────────────────────────────────────┘
     → 回到 retrieve（带澄清重检索）→ judge → answer → record → END
```

关键点，每一个都是必要条件（缺了就静默退回「重跑」）：

| # | 机制 | 位置 | 缺了会怎样 |
| --- | --- | --- | --- |
| 1 | `interrupt()` 真挂起 | `graph/qa.py::create_hitl` | 跑到 END，无断点可续 |
| 2 | `Command(resume=...)` 恢复入口 | `api/routes/hitl.py::submit_clarification` | 只能客户端重发 |
| 3 | `HitlTask.thread_id` 落库 | 加列 + `get_or_create_hitl_task` | 换进程/重启即失联 |
| 4 | 持久化 saver（PG + 连接池） | `graph/checkpointer.py` | 多 worker 下恢复请求落到别的进程就找不到 |
| 5 | 图 + saver **单例** | `api/deps.py::get_qa_graph`（挂 `app.state`） | 每请求换 saver = 断点永远查不到 |
| 6 | 任务创建**幂等** | `get_or_create_hitl_task` | `interrupt()` 恢复时节点重跑 → 重复建任务 |
| 7 | 用户消息在**轮次开头**记账 | `record_question` 节点 | 挂起时 `record` 不执行 → 用户的问题丢失 |
| 8 | 澄清用尽后不再挂起 | `route_after_create_hitl` | 再判不足会二次挂起 → 死循环 |
| 9 | 跑完回收 checkpoint | `discard_thread_if_finished` | 单例图 + 内存 saver 每请求堆 ~37KB |

### 几个容易踩的语义

- **挂起时节点返回值不进 state**。所以 `hitl_task_id` 不在 `invoke()` 返回值里，
  只能从挂起载荷取：`interrupt({'hitl_task_id': ...})` → 见 `graph.interrupt_payload()`。
  所有直接调图的调用方（API / CLI / MCP）都必须走这个 helper，否则 `KeyError`。
- **`interrupt()` 恢复时节点从头重跑**（LangGraph 用重放重建入口状态），
  所以中断节点里 `interrupt()` 之前的副作用必须幂等。
- **恢复路径刻意跳过 `cache_check`**：缓存里那条正是上次「答不了」的判定，
  再查一遍只会把它捞回来。
- **`record` 只写回答，用户消息提前到 `record_question`**：否则挂起期间问题丢失，
  用户放弃澄清后历史里查无此问。

## 三、API 契约变化（均为增量字段）

`POST /api/sessions/{id}/messages` → `AskResponse`

| 字段 | 含义 |
| --- | --- |
| `thread_id` | 本次问答的 thread（挂起时凭它恢复） |
| `resumable` | 该澄清是否可断点续答（False = 降级重跑） |
| `needs_clarification` / `hitl_task_id` | 语义不变 |

`POST /api/hitl/tasks/{id}/submit` → 除任务字段外新增

| 字段 | 含义 |
| --- | --- |
| `resumed` | 是否真的从断点恢复（False → 客户端按旧路径重发） |
| `answer` / `citations` / `needs_clarification` | 恢复后的回答 |

**降级是显式的**：`thread_id` 为空（加列前的旧任务）或 checkpoint 已丢失时，
返回 `resumed=false` 而**不假装成功**，客户端回退到重发原问题。

前端 `ChatPanel.vue` 已改为优先使用 `resumed` 的答案，仅在降级时走重发。

## 四、checkpoint 生命周期

挂起中的 checkpoint 是恢复的唯一依据，**必须保留**；跑完的则**必须回收**。

| 时机 | 动作 |
| --- | --- |
| 轮次跑完（无待执行节点） | `discard_thread_if_finished` 立即删 |
| 挂起中 | 保留 |
| 澄清被 reject | 直接删（该分支永不恢复） |
| 会话被删除 | 连带删该会话所有断点 |

实测：50 次正常问答 + 回收 → `saver.storage == 0`；不回收 → 50 条常驻（约 37KB/请求）。
**遗留边界**：用户既不提交也不拒绝的澄清，其 checkpoint 会常驻。PG 每行很小，
但没有 TTL 清理，量大时需要一个定期任务（未实现）。

## 五、配置

```bash
CHECKPOINT_BACKEND=postgres   # 生产：跨进程/跨重启可恢复（多 worker 必须）
CHECKPOINT_BACKEND=memory     # 默认：仅进程内可恢复（测试/单进程）
```

`.env` / `.env.example` 已加。池由 `atexit` 兜底关闭；
**不要在 app shutdown 里关它** —— 默认 saver 是进程级单例，
关掉会让同进程后续的 app 直接 `PoolClosed`（实测会挂掉 40+ 个用例）。

## 六、验收

```bash
uv run pytest tests/test_hitl_resume.py -q          # 15 条契约测试
uv run pytest -q                                     # 全量 399 条
uv run pytest tests/test_hitl_resume.py -q -m postgres   # 跨进程那条（需 PG）
```

`test_cross_process_resume_with_postgres` 是最终验收：**两个独立 saver + 两张图**
（模拟两个 worker）——「进程 A 挂起、进程 B 恢复」。单进程内存 saver 只能证明逻辑对，
只有独立 PG 连接能证明换进程也接得上。

无回归：生成层 `31/38`、`citation_coverage 1.0`、`false_refusal_rate 0.0`，
与改造前逐项一致；`CHECKPOINT_BACKEND=postgres` 下评测零异常。

## 七、一处此前的错误结论（更正）

我先前说「每请求重建图 = 白付 ChromaVectorStore 674ms + BM25 重建 ~1300ms」——
**这是错的**。那组数字是直接调 `create_vector_store()` 测出来的，
而 API 路径用的是 `Depends(get_vector_store)`，它是 `@lru_cache` 的：

```
两次 get_vector_store() 同一个实例? True
```

所以 BM25 索引（挂在 store 实例上）**本来就跨请求复用**。每请求重建图的真实代价是：

- `build_qa_graph()` 约 **11ms**；
- 丢掉 `Retriever._embed_cache`（L1，实例级）→ 检索路径每请求多一次 query embedding 调用。

收益仍然实在（少一次 embedding 往返 + 检索器级缓存跨请求生效），但**不是**秒级。
单例化真正的必要性在**正确性**：恢复要求「同一张图 + 同一个 saver」。
