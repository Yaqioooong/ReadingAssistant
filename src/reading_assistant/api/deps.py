"""API 依赖：请求级数据库会话 + 运行时单例的 re-export。

运行时单例（会话工厂/向量库/模型/上传目录）已下沉到
``reading_assistant.runtime``，以便 MCP 等其它适配器复用而不依赖 HTTP 层。
本模块保留 re-export 维持既有导入路径向后兼容。
"""

from collections.abc import Iterator

from fastapi import Depends
from sqlalchemy.orm import Session, sessionmaker

from reading_assistant.runtime import (  # noqa: F401 —— re-export 保持向后兼容
    get_embedding_model,
    get_llm,
    get_session_factory,
    get_upload_dir,
    get_vector_store,
)

__all__ = [
    'get_db_session',
    'get_embedding_model',
    'get_llm',
    'get_session_factory',
    'get_upload_dir',
    'get_vector_store',
]


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
