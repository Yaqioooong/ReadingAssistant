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
from reading_assistant.utils.logger_handler import get_logger
from reading_assistant.utils.path_tools import get_abs_path

logger = get_logger('api')


def _reconcile_index_status_on_startup() -> None:
    """启动对账：把崩溃残留的 ``indexing`` 记录拉回 indexed / failed。

    仅在生产启动路径调用（不在测试注入依赖时）。任一步失败都不阻断启动，
    只记录日志——对账是「尽力修复」，不能成为新的启动单点。
    """
    from reading_assistant.runtime import get_session_factory, get_vector_store
    from reading_assistant.storage.reconcile import (
        DEFAULT_LEASE_TIMEOUT_SECONDS,
        reconcile_index_status,
    )

    try:
        actions = reconcile_index_status(
            get_session_factory(),
            get_vector_store(),
            lease_timeout_seconds=DEFAULT_LEASE_TIMEOUT_SECONDS,
            apply=True,
        )
        if actions:
            logger.warning('启动对账[索引状态] 修正 %d 条卡死记录', len(actions))
    except Exception:  # noqa: BLE001 对账失败不得阻断启动
        logger.exception('启动对账[索引状态] 执行失败，已跳过')


def create_app(
    session_factory=None,
    vector_store=None,
    llm=None,
    embedding_model=None,
    upload_dir: Path | None = None,
) -> FastAPI:
    """创建 FastAPI 应用；不传参时使用生产组件。"""
    import time

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        # 生产模式（未注入 session_factory）时确保表结构存在；
        # 测试注入 sqlite 工厂时不碰生产库
        if session_factory is None:
            init_db(create_db_engine())
            _reconcile_index_status_on_startup()
        logger.info('ReadingAssistant API 启动')
        yield
        logger.info('ReadingAssistant API 关闭')
    from reading_assistant.api.routes import experiments, feedback, settings, stats

    app = FastAPI(title='ReadingAssistant API', version='0.1.0', lifespan=lifespan)

    @app.middleware('http')
    async def access_log(request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        cost_ms = (time.perf_counter() - start) * 1000
        logger.info('%s %s -> %d (%.0fms)', request.method, request.url.path,
                    response.status_code, cost_ms)
        return response

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
    app.include_router(experiments.router)
    app.include_router(settings.router)
    app.include_router(feedback.router)
    app.include_router(stats.router)

    dist = Path(get_abs_path('frontend/dist'))
    if dist.is_dir():
        app.mount('/', StaticFiles(directory=str(dist), html=True), name='static')
    return app
