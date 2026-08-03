"""数据库连接与会话管理。"""

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from reading_assistant.config import get_settings
from reading_assistant.storage.models import Base


def create_db_engine(database_url: str | None = None) -> Engine:
    """创建数据库引擎；URL 来自配置，不硬编码。"""
    url = database_url or get_settings().database_url
    kwargs = {}
    if url.startswith('sqlite'):
        kwargs['connect_args'] = {'check_same_thread': False}
        if url == 'sqlite:///:memory:':
            # 内存库需要共享同一连接，否则跨线程/多连接各自独立
            kwargs['poolclass'] = StaticPool
    return create_engine(url, pool_pre_ping=True, **kwargs)


def create_session_factory(engine: Engine) -> sessionmaker[Session]:
    """基于引擎创建 session 工厂。"""
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def init_db(engine: Engine) -> None:
    """创建所有表（create_all）。"""
    Base.metadata.create_all(bind=engine)


@contextmanager
def session_scope(session_factory: sessionmaker[Session]) -> Iterator[Session]:
    """事务性 session 上下文：正常提交，异常回滚。"""
    session = session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
