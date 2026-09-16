"""运行时单例工厂：数据库会话、向量库、模型与上传目录。

这些 getter 与 HTTP 框架无关，属于「应用运行时」层，供所有适配器复用：
- HTTP 适配器（``api/deps.py``）在此之上再叠加请求级依赖（如 ``get_db_session``）
- MCP 适配器（``mcp/server.py``）直接使用本模块，不反向依赖 HTTP 层

设计约束（由 ``tests/test_layering.py`` 钉住）：
本模块及其下游（``mcp/``）禁止 import fastapi，否则 stdio 进程会被
拖入整个 ASGI 框架，且适配器之间形成横向依赖。
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from sqlalchemy.orm import Session, sessionmaker

from reading_assistant.storage import create_db_engine, create_session_factory
from reading_assistant.storage.vector_store import VectorStore, create_vector_store
from reading_assistant.utils.path_tools import get_abs_path


@lru_cache
def get_session_factory() -> sessionmaker[Session]:
    """全局数据库会话工厂（测试可覆盖）。"""
    return create_session_factory(create_db_engine())


@lru_cache
def get_vector_store() -> VectorStore:
    """全局向量库实例（测试可覆盖）。"""
    return create_vector_store()


@lru_cache
def get_llm():
    """对话模型（测试可覆盖）。"""
    from reading_assistant.model.factory import get_chat_model

    return get_chat_model()


@lru_cache
def get_embedding_model():
    """Embedding 模型（测试可覆盖）。"""
    from reading_assistant.model.factory import get_embedding_model

    return get_embedding_model()


@lru_cache
def get_upload_dir() -> Path:
    """上传文件保存目录。"""
    path = Path(get_abs_path('uploads'))
    path.mkdir(parents=True, exist_ok=True)
    return path
