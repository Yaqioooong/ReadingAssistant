# ReadingAssistant — 阅读 Agent

帮助读者在阅读时建立书中**人物与事件脉络**的智能助手。用户上传电子书（EPUB / PDF / DOCX / TXT），Agent 解析全文并建立索引；之后读者可以用自然语言提问，例如「张三在本书中的事件时间线是怎样的？」「李四为什么被抓？」，Agent 结合原文给出带引用的回答。

> **当前状态：P0–P7 已完成。** 解析 → 存储 → RAG → LangGraph → FastAPI → CLI 全链路已实现，74 个 pytest 用例（覆盖率 94%），前端页面与接口约定已对齐。远期能力见文末 Roadmap。

## 目标功能

- **多格式解析**：支持 epub、pdf、docx、txt（旧版 .doc 请先转为 docx）。
- **RAG 问答**：全文分块、向量化，检索原文片段后由 LLM 生成带引用的回答。
- **人物 / 事件时间线**：围绕指定人物抽取事件并按顺序梳理。
- **HITL（人在回路）**：信息不足或问题含糊时，流程暂停并请求用户澄清。
- **聊天记录回溯**：对话持久化到 PostgreSQL，支持多会话与历史回顾。

## 当前进度

**已实现：**

- 前端 UI（Vite + Vue 3）：上传解析页 + 聊天问答页，并已按后端接口约定封装 `frontend/src/api.js`。
- 模型工厂 [`model/factory.py`](src/reading_assistant/model/factory.py)：ChatDeepSeek（`deepseek-v4-flash`）+ DashScope Embedding（`text-embedding-v4`）。
- 基础工具 `utils/`：YAML 配置加载（`config_handler.py`）、日志封装（`logger_handler.py`）、绝对路径工具（`path_tools.py`）。
- 配置清单：`requirements.txt`、`.env.example`、`config/*.yml`。
- `parsers/`：已实现 txt / epub / pdf / docx 四类解析器与工厂分发。
- `rag/`：已实现文本分块（chunking）与向量检索器（retriever），检索结果携带文档/章节元数据与引用。
- `storage/`：已实现 SQLAlchemy 模型、数据库会话管理、分层去重入库服务、向量库适配器（ChromaDB + 内存实现）。
- `graph/`：已实现 LangGraph 入库图与问答图（含 HITL 分支），支持内存/PostgreSQL 检查点。
- `api/`：已实现 FastAPI 应用与路由（文档上传/列表、会话、消息、HITL），对齐前端 `api.js` 调用约定。
- `cli.py`：命令行 `ingest` / `ask`。
- `tests/`：74 个 pytest 用例，覆盖率 94%（smoke / parsers / storage / rag / graph / api / cli）。

## 技术栈

| 层 | 选型 |
| --- | --- |
| 前端 | Vue 3 + Vite（开发端口 5173，`/api` 代理到 8000） |
| 后端 | FastAPI + uvicorn |
| 编排 | LangGraph + langgraph-checkpoint-postgres |
| LLM | DeepSeek `deepseek-v4-flash`（langchain-deepseek） |
| Embedding | 阿里云百炼 DashScope `text-embedding-v4`（langchain-community） |
| 结构化存储 | PostgreSQL（psycopg / SQLAlchemy） |
| 向量存储 | ChromaDB（本地持久化）；内存实现用于测试 |

## 目录结构

```
.
├── src/reading_assistant/
│   ├── api/          # 已实现：main/app + routes（documents/sessions/hitl）+ schemas
│   ├── graph/        # 已实现：ingest（入库）+ qa（问答/HITL）+ checkpointer
│   ├── parsers/      # 已实现：txt / epub / pdf / docx 解析器 + 工厂（*_parser.py 命名）
│   ├── rag/          # 已实现：chunking（分块）+ retriever（检索）
│   ├── storage/      # 已实现：模型 / database / repositories / service（去重）/ vector_store
│   ├── cli.py        # 已实现：ingest / ask 命令
│   ├── config.py             # 已实现：pydantic-settings 统一配置（.env + YAML）
│   ├── model/
│   │   └── factory.py        # 已实现：DeepSeek 对话模型 + DashScope Embedding 工厂
│   ├── utils/
│   │   ├── config_handler.py # 已实现：读取 config/*.yml
│   │   ├── logger_handler.py # 已实现：控制台 + 文件日志
│   │   └── path_tools.py     # 已实现：项目根目录 / 绝对路径
│   └── config/       # agent.yml / chroma.yml / model.yml / prompt.yml
├── tests/            # 74 个 pytest 用例（覆盖各模块与 API/CLI）
├── data/books/       # 测试样例电子书（txt/epub/pdf/docx + 去重副本）
├── frontend/         # Vite + Vue 3 单页应用（上传解析 + 聊天问答）
├── requirements.txt  # 开发安装别名（-e .[dev]）
├── pyproject.toml    # 依赖清单与 ruff / pytest 配置
├── docker-compose.yml # PostgreSQL 16
├── .env.example      # 环境变量模板（复制为 .env 使用）
├── Quick_Start.md    # PostgreSQL 建库步骤
└── README.md
```

> 📚 想了解项目是怎么一步步开发出来的？见 [tutorial/](tutorial/README.md) 分章节开发教程（面向 agent 开发新人）。

> 说明：早期文档与 AGENTS.md 中写的是 `reading_agent`，实际包名为 `reading_assistant`，请以后者为准。

## 快速开始

### 1. 安装 Python 依赖

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

依赖清单以 `pyproject.toml` 为准（dev extras 含 pytest / ruff）；`requirements.txt` 是 `-e .[dev]` 的安装别名。

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填入 DEEPSEEK_API_KEY 与 DASHSCOPE_API_KEY
```

### 3. 启动前端

```bash
cd frontend
npm install
npm run dev
```

打开 http://localhost:5173。前端已把 `/api` 代理到 `http://127.0.0.1:8000`，后端接口见下节。

### 4. 启动后端

先启动 PostgreSQL（可选，仅聊天记录/检查点需要）：

```bash
docker compose up -d
```

然后启动 API：

```bash
uvicorn reading_assistant.api.main:app --reload
```

接口文档见 http://127.0.0.1:8000/docs（Swagger UI）。PostgreSQL 建库步骤见 [Quick_Start.md](Quick_Start.md)。

### 5. 使用命令行（可选）

```bash
python -m reading_assistant.cli ingest data/books/sample_book.txt
python -m reading_assistant.cli ask "罗辑在本书中的事件时间线是怎样的？"
```

安装后也可直接使用 `readingassistant ingest ...` / `readingassistant ask ...`。

## API 接口

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `POST` | `/api/documents/upload` | 上传电子书（multipart `file`），返回 `{id, filename, title, author, chunk_count, duplicate}` |
| `GET` | `/api/documents` | 已入库文档列表 |
| `POST` | `/api/sessions` | 创建会话，返回 `{session_id}` |
| `GET` | `/api/sessions` | 会话列表 |
| `GET` | `/api/sessions/{id}/messages` | 会话聊天记录 |
| `POST` | `/api/sessions/{id}/messages` | 提问，body `{question, document_ids?, clarification?}`，返回 `{answer, citations, needs_clarification, hitl_task_id}` |
| `GET` | `/api/hitl/tasks` | HITL 澄清任务（可按 `session_id` 过滤） |
| `POST` | `/api/hitl/tasks/{id}/submit` | 提交澄清，body `{clarification}` |
| `POST` | `/api/hitl/tasks/{id}/reject` | 拒绝澄清 |

## 配置说明

配置分两层：

- **YAML**（`src/reading_assistant/config/`）：代码实际读取的配置，如 `model.yml` 模型名、`chroma.yml` 向量库参数、`prompt.yml` prompt 路径。
- **环境变量**（`.env`）：密钥与服务参数，模板见 `.env.example`。

| 环境变量 | 说明 | 默认值 |
| --- | --- | --- |
| `APP_NAME` | 应用名 | `ReadingAssistant` |
| `DATABASE_URL` | PostgreSQL 连接串 | `postgresql+psycopg://.../reading_agent` |
| `LLM_MODEL` | DeepSeek 对话模型 | `deepseek-v4-flash` |
| `DEEPSEEK_API_KEY` | DeepSeek API 密钥 | 空 |
| `EMBEDDING_MODEL` | DashScope 向量化模型 | `text-embedding-v4` |
| `DASHSCOPE_API_KEY` | DashScope API 密钥 | 空 |
| `CHROMA_PERSIST_DIR` | 向量库持久化目录 | `rag/chroma_data` |
| `CHROMA_COLLECTION_NAME` | 向量集合名 | `reading_agent_chunks` |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 分块大小与重叠 | `800` / `100` |
| `TOP_K` | 检索返回片段数 | `6` |
| `API_HOST` / `API_PORT` | 服务监听地址 / 端口 | `0.0.0.0` / `8000` |
| `LOG_LEVEL` | 日志级别 | `INFO` |

> 说明：`MD5_HEX_STORE`、`ALLOWED_KNOWEDGE_FILE_TYPE`、`SEPARATORS` 等参数以 `config/chroma.yml` 为准，不通过环境变量覆盖。

## 数据存储（规划中）

| 存储 | 用途 |
| --- | --- |
| `documents` | 已入库的书籍及元数据（含 md5 去重） |
| `chunks`（向量集合） | 分块内容、向量与元数据（文档 id、章节、页码） |
| `chat_sessions` | 会话 |
| `chat_messages` | 聊天记录 |
| `hitl_tasks` | HITL 澄清任务（awaiting / approved / rejected） |

## 前端界面

`frontend/` 为 Vite + Vue 3 单页应用，包含两个页面：

- **上传解析**：拖拽 / 选择电子书上传，调用 `POST /api/documents/upload`，展示解析分块数与文档列表。
- **聊天问答**：新建 / 切换会话，调用 `/api/sessions` 系列接口提问并展示回答；信息不足时展示澄清提示，历史消息可回溯。

开发模式：`cd frontend && npm run dev`（http://localhost:5173，热更新）。
生产模式：`npm run build` 生成 `frontend/dist`，后端检测到后自动托管，直接访问 http://127.0.0.1:8000 即可。

## Roadmap

按依赖顺序排列：

1. ✅ **统一配置层**：新增 `config.py`（pydantic-settings），收敛 `.env` 与 YAML，修复 `utils` 的扁平导入与路径工具问题。
2. ✅ **parsers**：实现 epub / pdf / docx / txt 解析器与工厂注册。
3. ✅ **storage**：SQLAlchemy 模型 + 向量库适配器（ChromaDB / 内存）。
4. ✅ **graph**：LangGraph 入库与问答流水线，含 HITL 状态。
5. ✅ **api**：FastAPI 路由，对齐前端已有调用约定；同步实现 CLI。
6. ✅ **tests**：补齐 pytest 单测与集成测试（74 个用例，覆盖率 94%）。
7. **远期**：引用解析、时间线结构化抽取、本地 Embedding / LLM、更多格式（.doc / mobi / html）、用户认证与多租户隔离。
