# 第 05 章 P4 LangGraph 流水线

> 对应阶段：P4 LangGraph —— 入库图、问答图、HITL、检查点。

## 本章目标

用 LangGraph 把「解析 → 分块 → 向量化 → 入库」和「检索 → 判断 → 回答 / 澄清 → 记录」编排成两张**可复用、可检查点**的状态图。

## 1. LangGraph 核心概念（1.x）

| 概念 | 说明 | 关键语法 |
| --- | --- | --- |
| `StateGraph` | 状态机容器，节点共享一个 TypedDict 状态 | `graph = StateGraph(StateType)` |
| 节点 | 一个函数：`(state) -> partial_dict`，返回的键会**覆盖**状态 | `graph.add_node('name', fn)` |
| 边 | 固定流转 | `graph.add_edge(START, 'node')` / `graph.add_edge('node', END)` |
| 条件边 | 按返回值路由 | `graph.add_conditional_edges('node', route_fn, path_map)` |
| 检查点 | 保存每一步状态，支持断点续跑 | `graph.compile(checkpointer=...)` |
| thread | 一次「会话/运行」的 id | `invoke(input, config={'configurable': {'thread_id': 'x'}})` |

**节点返回值**：只返回要更新的键，没提到的键保持不变。**条件边路由函数**返回节点名或 `END`。

## 2. 入库图（`graph/ingest.py`）

**做什么**：`add_book`（去重 + 解析 + 建文档记录）→ 重复则短路结束 → 否则 `chunk_and_index`（分块 + embedding + 写向量库 + 回写 chunk_count）。

**状态定义**：

```python
class IngestState(TypedDict, total=False):
    book_path: str
    filename: str | None
    document_id: int | None
    duplicate: bool
    chapters: list[dict]      # 注意：用 dict 不用 ParsedBook，见「常见坑 1」
    chunk_count: int
```

**节点用闭包注入依赖**：

```python
def build_ingest_graph(session_factory, vector_store, embedding_model=None, checkpointer=None):
    embeddings = embedding_model or get_embedding_model()

    def add_book(state: IngestState) -> dict:
        with session_scope(session_factory) as session:
            result = DocumentService(session).add_book(state['book_path'], filename=state.get('filename'))
            return {'document_id': result.document.id, 'duplicate': result.duplicate, ...}

    def chunk_and_index(state: IngestState) -> dict:
        ...
        return {'chunk_count': len(chunks)}

    def route_after_add(state: IngestState) -> str:
        return END if state.get('duplicate') else 'chunk_and_index'

    graph = StateGraph(IngestState)
    graph.add_node('add_book', add_book)
    graph.add_node('chunk_and_index', chunk_and_index)
    graph.add_edge(START, 'add_book')
    graph.add_conditional_edges('add_book', route_after_add)
    graph.add_edge('chunk_and_index', END)
    return graph.compile(checkpointer=checkpointer or InMemorySaver())
```

**为什么依赖用闭包而不是全局**：测试时可以传入 sqlite 会话工厂、内存向量库、假 embedding，跑完整张图。

## 3. 问答图（`graph/qa.py`）

**做什么**：`retrieve` → `judge` →（`answer` 或 `create_hitl`）→ `record`。

**信息充足性判断**（当前是启发式，可换 LLM 判断）：

```python
def judge(state: QAState) -> dict:
    # 有澄清说明时视为信息已补充；否则无检索结果即为信息不足
    needs = not state.get('chunks') and not state.get('clarification')
    return {'needs_clarification': needs}
```

**条件路由**：

```python
def route_after_judge(state: QAState) -> str:
    return 'create_hitl' if state.get('needs_clarification') else 'answer'

graph.add_conditional_edges(
    'judge', route_after_judge,
    {'answer': 'answer', 'create_hitl': 'create_hitl'},   # 返回值和映射必须一致
)
```

**answer 节点**组装 prompt：问题 +（可选澄清）+ 原文片段列表（带引用编号）。`record` 节点把用户问题与回答（含 citations）写入 `chat_messages`。

**HITL 续答**：用户提交澄清后，重新调用问答图并传入 `clarification`——`retrieve` 把澄清拼进检索问题，`judge` 因为有澄清直接走 answer。状态机不变，只是输入多一个字段。

## 4. LLM 注入与 mock

节点里调用 `chat_model.invoke(prompt)`，测试传一个假对象：

```python
class FakeLLM:
    def invoke(self, prompt: str):
        return SimpleNamespace(content='这是基于原文的测试回答。')
```

**协议约定**：只要返回对象有 `.content` 属性即可，不强制继承 LangChain 类。生产环境传 `get_chat_model()`。

## 5. 检查点（`graph/checkpointer.py`）

```python
def create_inmemory_checkpointer():
    return InMemorySaver()          # 测试/本地开发

def create_postgres_checkpointer(database_url=None):
    from langgraph.checkpoint.postgres import PostgresSaver   # 延迟导入
    saver = PostgresSaver.from_conn_string(url)
    saver.setup()                   # 首次使用前建表
    return saver
```

**注意**：带检查点的图 `invoke` 必须传 `config={'configurable': {'thread_id': ...}}`，否则报 thread 相关错误。

## 关键语法与技巧

- `TypedDict(total=False)`：状态字段全部可选，节点只更新自己关心的键。
- 条件边返回 `END` 常量（字符串 `'__end__'`）表示终止。
- 图构建一次、复用多次：`build_ingest_graph(...)` 返回编译好的图，每次 `invoke` 传不同输入。
- 节点保持「纯函数 + 闭包依赖」，不碰全局单例，测试友好。

## 常见坑

1. **状态必须可序列化**：检查点（尤其 Postgres）要序列化状态。**别把 ORM 对象 / dataclass 塞进 state**，`ParsedBook` 就换成 `chapters: list[dict]`；文档 id 这种简单标量没问题。
2. **节点返回值是「覆盖」不是「追加」**：列表字段每次整体替换；要追加得用 reducer（`Annotated[list, operator.add]`）。
3. **条件边路由函数返回的字符串必须能在 path_map 里找到**，否则运行时 KeyError。
4. **别在节点里共享全局会话**：会话工厂通过闭包注入，每个节点自己开 `session_scope`，提交即关闭。

## 测试与验证

```bash
.venv/bin/pytest tests/test_graph.py -q
```

覆盖：入库建索引、重复短路、问答 + 引用 + 消息落库、无结果进 HITL、澄清续答、HITL 状态流转（awaiting→approved/rejected、非法状态报错）。
