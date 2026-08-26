"""FastAPI 应用工厂：测试可注入 mock 依赖。"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from reading_assistant.api.deps import (
    get_embedding_model,
    get_llm,
    get_session_factory,
    get_upload_dir,
    get_vector_store,
)
from reading_assistant.api.routes import documents, hitl, sessions
from reading_assistant.storage import create_db_engine, init_db
from reading_assistant.utils.path_tools import get_abs_path


def create_app(
    session_factory=None,
    vector_store=None,
    llm=None,
    embedding_model=None,
    upload_dir: Path | None = None,
) -> FastAPI:
    """创建 FastAPI 应用；不传参时使用生产组件。"""
    @asynccontextmanager
    async def lifespan(_: FastAPI):
        # 生产模式（未注入 session_factory）时确保表结构存在；
        # 测试注入 sqlite 工厂时不碰生产库
        if session_factory is None:
            init_db(create_db_engine())
        yield
    app = FastAPI(title='ReadingAssistant API', version='0.1.0', lifespan=lifespan)
    if session_factory is not None:
        app.dependency_overrides[get_session_factory] = lambda: session_factory
    if vector_store is not None:
        app.dependency_overrides[get_vector_store] = lambda: vector_store
    if llm is not None:
        app.dependency_overrides[get_llm] = lambda: llm
    if embedding_model is not None:
        app.dependency_overrides[get_embedding_model] = lambda: embedding_model
    if upload_dir is not None:
        app.dependency_overrides[get_upload_dir] = lambda: upload_dir

    app.include_router(documents.router)
    app.include_router(sessions.router)
    app.include_router(hitl.router)

    dist = Path(get_abs_path('frontend/dist'))
    if dist.is_dir():
        app.mount('/', StaticFiles(directory=str(dist), html=True), name='static')
    return app
