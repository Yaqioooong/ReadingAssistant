# 开发计划（Development Plan）

> 项目当前处于脚手架阶段。本计划按依赖顺序拆解开发任务，每个阶段都有明确的交付物与验收标准；**完成一个阶段（含测试）后再进入下一阶段**。

## 阶段总览

| 阶段 | 内容 | 状态 |
| --- | --- | --- |
| P0 工程基础 | 统一配置层、包结构修复、构建与测试基础设施 | ✅ 已完成 |
| P1 解析器 | `parsers/`：epub / pdf / docx / txt | ✅ 已完成 |
| P2 存储层 | `storage/`：SQLAlchemy 模型 + 向量库适配器 + 分层去重 | ✅ 已完成 |
| P3 RAG 层 | `rag/`：分块 + 向量检索 | ✅ 已完成 |
| P4 LangGraph | `graph/`：入库 / 问答 / HITL 流水线 | ⬜ 未开始 |
| P5 FastAPI | `api/`：路由、schema、前端联调 | ⬜ 未开始 |
| P6 CLI | `cli.py`：ingest / ask 命令 | ⬜ 未开始 |
| P7 收尾 | 全量测试、README 同步、首个 commit / PR | ⬜ 未开始 |

## P0 工程基础

- [x] 新增 `src/reading_assistant/config.py`（pydantic-settings），统一读取 `.env` 与 `config/*.yml`，收敛重复配置
- [x] 修复 `model/factory.py` 与 `utils/*` 的扁平导入，改为 `reading_assistant.*` 包内导入
- [x] 修复 `path_tools.get_project_root()`，使其返回仓库根目录（而非包目录）
- [x] 完善 `pyproject.toml`：声明运行时依赖、dev extras、ruff / pytest 配置
- [x] 新增 `docker-compose.yml`（PostgreSQL）并同步 `Quick_Start.md`
- [x] 建立测试基础设施（`conftest.py`、pytest 配置、日志/路径 fixtures）

**验收标准：** `pip install -e ".[dev]"` 可安装；从任意工作目录 `import reading_assistant` 正常；ruff check / format 通过；pytest 可空跑通过。

## P1 解析器（parsers）

- [x] `BookParser` 抽象基类：统一输出纯文本 + 元数据（标题、作者、章节）
- [x] 实现 `TxtParser`、`EpubParser`、`PdfParser`、`DocxParser`
- [x] `parsers/factory.py`：按扩展名注册并分发
- [x] 每个解析器至少一个 happy-path 与一个边界用例测试（空文件、损坏文件、加密 PDF）

**验收标准：** 四类格式解析测试全部通过；`factory` 能识别并分发正确解析器。

## P2 存储层（storage）

- [x] SQLAlchemy 模型：`Document`、`ChatSession`、`ChatMessage`、`HitlTask`
  - `Document` 增加 `file_hash`（unique，文件字节 SHA-256）与 `content_hash`（归一化文本 SHA-256）
  - 去除对 `md5.txt` 的依赖（`chroma.yml` 配置项删除，代码实现不再读取）
- [x] 建表与连接管理（读取 `DATABASE_URL`，不硬编码）
- [x] `vector_store.py` 适配器接口 + 实现（ChromaDB 与 Milvus 二选一，需先统一配置）
- [x] 文档 repository 层：入库 / 查询 / 按哈希查重
- [x] 分层去重入库流程：
  1. 上传后先算文件 SHA-256，查 `file_hash` → 命中直接返回已有 `document_id`（`duplicate=true`，跳过解析）
  2. 未命中才解析 → 文本归一化（去空白/格式）→ 算 `content_hash` → 再查
  3. 命中 `content_hash` 默认复用已有文档（不重复入库）；如需保留两种格式可存独立版本并记录 `duplicate_of`
  4. 插入时捕获唯一索引冲突（`IntegrityError`）→ 重新查询返回已有 id（并发兜底）
  5. 重复上传响应语义：`duplicate=true + document_id`，不报错

**验收标准：**
- 文档入库 / 查询 / 重复上传去重的测试通过
- 去重测试覆盖：同一文件两次上传、同内容不同文件名、并发冲突（唯一索引兜底）
- 向量库适配器接口可 mock，不依赖真实向量库即可单测

## P3 RAG 层（rag）

- [x] `chunking.py`：按 `separators` 分块，`chunk_size=800` / `chunk_overlap=100`
- [x] Embedding（DashScope `text-embedding-v4`）+ top-k 检索（`k=6`）
- [x] 检索结果携带元数据（文档 id、章节、页码），支持引用

**验收标准：** 检索能返回相关片段及排序结果；chunking 边界测试通过。

## P4 LangGraph 流水线（graph）

- [ ] ingest 图：解析 → 分块 → 向量化 → 入库
- [ ] QA 图：检索 → 信息充足性判断 → 回答 / 进入 HITL
- [ ] HITL 澄清状态流转（awaiting / approved / rejected）
- [ ] 通过 checkpoint 将问答记录持久化到 PostgreSQL

**验收标准：** 入库与问答流程端到端可跑通（LLM 可用 mock 代替）；HITL 分支有测试。

## P5 FastAPI（api）

- [ ] `main.py` + 路由：
  - `POST /api/documents/upload`（返回 id、chunk_count）
  - `GET /api/documents`
  - `POST/GET /api/sessions`、`GET /api/sessions/{id}/messages`、`POST /api/sessions/{id}/messages`
  - HITL 澄清相关接口
- [ ] Pydantic 请求 / 响应模型
- [ ] 对齐 `frontend/src/api.js` 的现有调用约定
- [ ] 生产模式静态托管 `frontend/dist`

**验收标准：** 前后端联调通过；接口测试覆盖 happy path 与错误分支。

## P6 CLI

- [ ] `python -m reading_assistant.cli ingest <book>`
- [ ] `python -m reading_assistant.cli ask "<问题>"`

**验收标准：** 命令行完成上传解析与提问两条路径。

## P7 收尾

- [ ] 全量 `pytest` + 覆盖率检查
- [ ] `ruff format` 全仓统一格式
- [ ] 更新 README：移除 TODO 标注，补充接口文档
- [ ] 首个 Conventional Commit / PR（描述变更与测试方式）

## 执行规则

- 严格按阶段顺序推进，**P0 未完成不得进入 P1**，依此类推。
- 每个任务完成后运行对应测试；阶段内所有任务完成并满足验收标准，阶段才算完成。
- 新增第三方依赖需先写入 `requirements.txt` / `pyproject.toml` 并说明理由。
- 未列入本计划的工作，先补充进本计划再实施。
