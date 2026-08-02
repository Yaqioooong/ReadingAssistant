# ReadingAssistant — 阅读 Agent

帮助读者在阅读时建立书中**人物与事件脉络**的智能助手。用户上传电子书（EPUB / PDF / DOCX / TXT），Agent 解析全文并建立索引；之后读者可以用自然语言提问，例如「张三在本书中的事件时间线是怎样的？」「李四为什么被抓？」，Agent 结合原文给出带引用的回答。

> **当前状态：脚手架阶段。** 仓库已搭好目录结构、依赖清单与前端界面，但后端核心模块（解析、向量检索、LangGraph 流水线、API）尚未实现，均为空包 TODO。下文「目标功能」描述的是产品愿景，并非已上线能力。

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

**尚未实现（TODO）：**

- `parsers/`：无任何电子书解析器。
- `rag/`：无文本分块、无向量检索。
- `storage/`：无 SQLAlchemy 模型、无向量库适配器。
- `graph/`：无 LangGraph 流水线（入库、问答、HITL）。
- `api/`：无 FastAPI 应用与路由，前端调用的 `/api/documents`、`/api/sessions` 等接口尚不存在。
- `tests/`：仅有 P0 冒烟测试（conftest.py / test_smoke.py）。
- CLI 未落地；PostgreSQL 编排已通过 `docker-compose.yml` 提供。

## 技术栈

| 层 | 选型 |
| --- | --- |
| 前端 | Vue 3 + Vite（开发端口 5173，`/api` 代理到 8000） |
| 后端 | FastAPI + uvicorn（规划中） |
| 编排 | LangGraph + langgraph-checkpoint-postgres |
| LLM | DeepSeek `deepseek-v4-flash`（langchain-deepseek） |
| Embedding | 阿里云百炼 DashScope `text-embedding-v4`（langchain-community） |
| 结构化存储 | PostgreSQL（psycopg / SQLAlchemy，规划中） |
| 向量存储 | 配置指向 ChromaDB（`config/chroma.yml`），但依赖清单中是 Milvus 适配器（langchain-milvus / pymilvus），落地前需二选一并补齐对应依赖 |

## 目录结构

```
.
├── src/reading_assistant/
│   ├── api/          # TODO：FastAPI 应用、路由、请求/响应模型（空包）
│   ├── graph/        # TODO：LangGraph 流水线：入库、问答 + HITL（空包）
│   ├── parsers/      # TODO：epub / pdf / docx / txt 解析器（空包）
│   ├── rag/          # TODO：分块、向量检索（空包）
│   ├── storage/      # TODO：SQLAlchemy 模型、向量库适配器、聊天记录（空包）
│   ├── config.py             # 已实现：pydantic-settings 统一配置（.env + YAML）
│   ├── model/
│   │   └── factory.py        # 已实现：DeepSeek 对话模型 + DashScope Embedding 工厂
│   ├── utils/
│   │   ├── config_handler.py # 已实现：读取 config/*.yml
│   │   ├── logger_handler.py # 已实现：控制台 + 文件日志
│   │   └── path_tools.py     # 已实现：项目根目录 / 绝对路径
│   └── config/       # agent.yml / chroma.yml / model.yml / prompt.yml
├── tests/            # P0 冒烟测试：conftest.py + test_smoke.py
├── frontend/         # Vite + Vue 3 单页应用（上传解析 + 聊天问答）
├── requirements.txt  # 开发安装别名（-e .[dev]）
├── pyproject.toml    # 依赖清单与 ruff / pytest 配置
├── docker-compose.yml # PostgreSQL 16
├── .env.example      # 环境变量模板（复制为 .env 使用）
├── Quick_Start.md    # PostgreSQL 建库步骤
└── README.md
```

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

打开 http://localhost:5173。前端已把 `/api` 代理到 `http://127.0.0.1:8000`，但后端 API 尚未实现，页面当前请求会失败——这是预期行为。

### 4. 启动后端（TODO）

后端入口规划为 `reading_assistant.api.main`，对应命令：

```bash
uvicorn reading_assistant.api.main:app --reload
```

该模块目前不存在，待 `api/` 实现后可用。PostgreSQL 建库步骤见 [Quick_Start.md](Quick_Start.md)。

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
生产模式：`npm run build` 生成 `frontend/dist`，规划由后端静态托管（尚未实现）。

## Roadmap

按依赖顺序排列：

1. **统一配置层**：新增 `config.py`（pydantic-settings），收敛 `.env` 与 YAML，修复 `utils` 的扁平导入与路径工具问题。
2. **parsers**：实现 epub / pdf / docx / txt 解析器与工厂注册。
3. **storage**：SQLAlchemy 模型 + 向量库适配器（ChromaDB / Milvus 二选一）。
4. **graph**：LangGraph 入库与问答流水线，含 HITL 状态。
5. **api**：FastAPI 路由，对齐前端已有调用约定；同步实现 CLI。
6. **tests**：补齐 pytest 单测与集成测试。
7. **远期**：引用解析、时间线结构化抽取、本地 Embedding / LLM、更多格式（.doc / mobi / html）、用户认证与多租户隔离。
