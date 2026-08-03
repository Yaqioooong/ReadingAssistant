"""API 依赖：数据库会话、向量库、模型与上传目录。"""

from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path

from fastapi import Depends
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


def get_db_session(
    session_factory: sessionmaker[Session] = Depends(get_session_factory),
) -> Iterator[Session]:
    """请求级数据库会话：正常提交，异常回滚。"""
    session = session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
