# 第 06 章 P5 FastAPI 接口

> 对应阶段：P5 FastAPI —— 路由、Schema、前端联调、静态托管。

## 本章目标

把 P0–P4 的成果暴露成 HTTP 接口：上传文档、列文档、建会话、提问、查消息、HITL 提交/拒绝，并让前端（Vue）能直接对接。核心工程点：**依赖注入可测试、契约对齐前端、错误语义清晰**。

## 1. 分层结构

```
api/
├── main.py       # uvicorn 入口：app = create_app()
├── app.py        # create_app() 工厂：装路由、挂依赖覆盖、静态托管
├── schemas.py    # Pydantic 请求/响应模型
├── deps.py       # 依赖：会话工厂 / 向量库 / LLM / Embedding / 上传目录 / 请求级 session
└── routes/       # documents.py / sessions.py / hitl.py
```

## 2. 应用工厂 + 依赖覆盖（为什么可测）

```python
def create_app(session_factory=None, vector_store=None, llm=None,
               embedding_model=None, upload_dir=None) -> FastAPI:
    app = FastAPI(title='ReadingAssistant API', version='0.1.0')
    if session_factory is not None:
        app.dependency_overrides[get_session_factory] = lambda: session_factory
    # ... vector_store / llm / embedding_model / upload_dir 同理
    app.include_router(documents.router)
    app.include_router(sessions.router)
    app.include_router(hitl.router)
    dist = Path(get_abs_path('frontend/dist'))
    if dist.is_dir():
        app.mount('/', StaticFiles(directory=str(dist), html=True), name='static')
    return app
```

**为什么用 `dependency_overrides`**：不传参时是生产组件（真实 PostgreSQL / Chroma / DeepSeek）；测试传入 sqlite + 内存向量库 + FakeLLM，同一份代码两种跑法，不需要 mock 框架。

## 3. 依赖模块（`deps.py`）

```python
@lru_cache
def get_session_factory() -> sessionmaker[Session]:
    return create_session_factory(create_db_engine())

def get_db_session(session_factory=Depends(get_session_factory)):
    session = session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
```

**要点**：依赖项可以依赖另一个依赖（`Depends(get_session_factory)`），FastAPI 会按图解析；测试覆盖 `get_session_factory` 即可让所有用 session 的路由切换到测试库。

## 4. 文档路由（`routes/documents.py`）

**上传三步**：校验格式 → 落盘 → 跑入库图。

```python
@router.post('/upload', response_model=schemas.UploadResponse, status_code=201)
def upload_document(
    file: UploadFile = File(...),
    session_factory=Depends(get_session_factory),
    vector_store=Depends(get_vector_store),
    embedding_model=Depends(get_embedding_model),
    upload_dir: Path = Depends(get_upload_dir),
    session: Session = Depends(get_db_session),
):
    filename = Path(file.filename or 'book').name   # 去掉路径，防目录穿越
    try:
        get_parser(filename)                        # 复用 P1 工厂做格式校验
    except UnsupportedFormatError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    upload_dir.mkdir(parents=True, exist_ok=True)
    target = upload_dir / f'{uuid4().hex}_{filename}'   # uuid 前缀防重名覆盖
    target.write_bytes(file.file.read())
    try:
        result = graph.invoke({'book_path': str(target), 'filename': filename}, ...)
    except ParseError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
```

**技巧**：格式校验直接复用 `get_parser()`——P1 的工厂就是「权威扩展名清单」，不需要再维护一份。

## 5. 会话路由与前端契约（重点！）

前端代码决定了几个**硬性契约**，接口必须照做：

### 5.1 会话 id 必须是字符串

前端有 `s.id.slice(0, 8)`（Vue 模板），int 会直接抛错。pydantic v2 **不会把 int 自动转成 str**（`id: str` + `from_attributes` 遇到 int 会报 `ResponseValidationError`），所以要显式转：

```python
@router.post('', response_model=schemas.SessionCreated, status_code=201)
def create_session(session: Session = Depends(get_db_session)):
    chat = ChatSession()
    session.add(chat)
    session.flush()
    return schemas.SessionCreated(session_id=str(chat.id))
```

### 5.2 提问 payload 是 `document_ids` 数组

```python
class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    document_ids: list[int] = Field(default_factory=list)   # 前端传数组！
    clarification: str | None = None
```

路由里取第一个映射到图的 `document_id`。

### 5.3 引用格式 `{chapter, page, excerpt}`

前端渲染 `citationSource(c)` 用 `c.chapter` / `c.page`，正文展示用 `c.excerpt`。所以 QA 图的 citations 输出也是这个形状：

```python
{'chunk_id': ..., 'chapter': ..., 'page': None, 'excerpt': chunk['text'][:120]}
```

**教训**：接口设计前先读前端代码，契约不符的代价是前后端来回改。

### 5.4 错误语义

- 会话不存在 → `404`
- 空问题 / 不支持的格式 / 解析失败 → `422`（FastAPI 的 `Field(min_length=1)` 自动处理空问题）
- HITL 非法状态变更 → `400`，任务不存在 → `404`

## 6. HITL 路由

```python
@router.post('/{task_id}/submit', response_model=schemas.HitlTaskOut)
def submit_clarification(task_id: int, payload: schemas.HitlClarificationRequest, session=Depends(get_db_session)):
    try:
        return submit_hitl_clarification(session, task_id, payload.clarification)
    except ValueError as exc:
        status = 404 if '不存在' in str(exc) else 400
        raise HTTPException(status_code=status, detail=str(exc)) from exc
```

**技巧**：仓储层抛 `ValueError`（带中文原因），路由层翻译成 HTTP 状态码，业务逻辑不进路由。

## 7. 测试：TestClient + SQLite 内存库的坑

TestClient 的请求运行在**另一个线程**。`sqlite:///:memory:` 每个连接是独立数据库，会出现「no such table」。P2 已在 `create_db_engine` 里对内存 sqlite 加 `StaticPool` 修复：

```python
if url == 'sqlite:///:memory:':
    kwargs['poolclass'] = StaticPool   # 所有连接共享同一个内存库
```

测试夹具：

```python
app = create_app(session_factory=sqlite_factory, vector_store=store,
                 llm=FakeLLM(), embedding_model=FakeEmbeddings(), upload_dir=tmp_path / 'uploads')
with TestClient(app) as client:
    ...
```

## 8. 静态托管

`frontend/dist` 存在时挂载 StaticFiles，`npm run build` 之后直接访问 http://127.0.0.1:8000 就能用前端；API 路由优先于静态挂载匹配。

## 常见坑

1. **pydantic v2 不自动 int→str**：`from_attributes` 只是「从对象取属性」，不做类型强制转换，违反响应模型会抛 `ResponseValidationError`。
2. **UploadFile 读取**：`file.file.read()` 拿到原始字节；文件指针用完 FastAPI 会关。
3. **上传目录**：注入自定义目录时它可能不存在，路由里先 `mkdir(parents=True, exist_ok=True)`。
4. **TestClient 线程**：任何「跨线程共享 SQLite 内存库」的场景都要 `StaticPool`。
5. **别把业务逻辑塞进路由**：路由只做「参数 → 调图/仓储 → 响应模型」，可测性全靠这个边界。

## 测试与验证

```bash
.venv/bin/pytest tests/test_api.py -q
uvicorn reading_assistant.api.main:app --reload
curl http://127.0.0.1:8000/api/documents
```

Swagger 文档在 http://127.0.0.1:8000/docs，可以边看边手动联调。
