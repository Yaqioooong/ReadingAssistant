"""数据库连接与会话管理。"""

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import Engine, create_engine, text
from sqlalchemy import inspect as sa_inspect
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from reading_assistant.config import get_settings
from reading_assistant.storage.models import Base
from reading_assistant.utils.logger_handler import get_logger

logger = get_logger('storage')


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
    """创建所有表（create_all），并对旧库做轻量加列（SQLite/PG 兼容）。"""
    Base.metadata.create_all(bind=engine)
    _ensure_column(
        engine,
        table='chat_sessions',
        column='summary',
        ddl='ALTER TABLE chat_sessions ADD COLUMN summary TEXT',
    )
    _ensure_column(
        engine,
        table='chat_sessions',
        column='state',
        ddl='ALTER TABLE chat_sessions ADD COLUMN state TEXT',
    )
    _purge_deprecated_summary(engine)
    _ensure_column(
        engine,
        table='documents',
        column='index_started_at',
        ddl='ALTER TABLE documents ADD COLUMN index_started_at TIMESTAMP',
    )
    _ensure_column(
        engine,
        table='qa_cache',
        column='cached_chunk_count',
        ddl='ALTER TABLE qa_cache ADD COLUMN cached_chunk_count INTEGER NOT NULL DEFAULT 0',
    )
    _ensure_column(
        engine,
        table='hitl_tasks',
        column='thread_id',
        ddl='ALTER TABLE hitl_tasks ADD COLUMN thread_id VARCHAR(64)',
    )
    _ensure_qa_cache_version_key(engine)


def _purge_deprecated_summary(engine: Engine) -> None:
    """清掉已废弃的滚动摘要列内容（收敛，不删列）。

    迁移**每次启动都跑**，职责是把库收敛到目标形态 —— 不是「只在首次正确」。
    留着非空的旧摘要是本项目最忌讳的形态：两处都在、没人知道该信哪个。
    （前科：文档打回 indexing、前端未 rebuild、shell 旧 key 盖 .env。）
    列本身不 DROP：用户有活库，破坏性操作要单独走。
    """
    try:
        with engine.begin() as conn:
            conn.execute(text('UPDATE chat_sessions SET summary = NULL WHERE summary IS NOT NULL'))
    except Exception:  # noqa: BLE001 表不存在或方言不支持时忽略
        pass


def _ensure_column(engine: Engine, table: str, column: str, ddl: str) -> None:
    """旧库补列：SQLAlchemy create_all 不修改已存在的表。已存在/表缺失时静默跳过。"""
    try:
        if any(c['name'] == column for c in sa_inspect(engine).get_columns(table)):
            return
        with engine.begin() as conn:
            conn.execute(text(ddl))
    except Exception:  # noqa: BLE001 表不存在或方言不支持时忽略
        pass


QA_CACHE_VERSION_INDEX = 'uq_qa_cache_question_version'
# 历史索引名：unique=True 时代由 create_all 建出的**单列唯一索引**（不是约束）
QA_CACHE_LEGACY_INDEX = 'ix_qa_cache_question_hash'


def _ensure_qa_cache_version_key(engine: Engine) -> None:
    """把问答缓存键从「question_hash 单列唯一」迁到「(question_hash, content_hash) 复合唯一」。

    为什么必须迁：读路径一直按 ``(question_hash, content_hash)`` 当键查
    （``get_qa_cache_entry`` 与语义层 ``list_qa_cache_entries`` 都按 content_hash 筛），
    而 DB 原本只许 ``question_hash`` 唯一 —— 代码声明版本参与身份，DB 却禁止两行只差版本。
    实测后果：文档改版（或全库语料指纹变化）后再问同一问题，读路径正确判 miss，
    写路径插新行却撞唯一约束 → ``IntegrityError`` 冒到 API，**整轮问答失败且永久复发**
    （旧行还在，下次同样撞）。

    幂等：已是复合唯一 → 直接返回；表不存在 → 跳过。SQLite 的 ``DROP INDEX`` 与
    PG 通用，故这里不需要方言分支。
    """
    try:
        inspector = sa_inspect(engine)
        if 'qa_cache' not in inspector.get_table_names():
            return
        indexes = {idx['name']: idx for idx in inspector.get_indexes('qa_cache')}
        unique_constraints = inspector.get_unique_constraints('qa_cache')
    except Exception:  # noqa: BLE001 表缺失或方言不支持时忽略
        return

    def _covers(columns, names) -> bool:
        return sorted(columns) == sorted(names)

    composite = [QA_CACHE_VERSION_INDEX, ('question_hash', 'content_hash')]
    has_composite = (
        any(_covers(idx.get('column_names') or [], composite[1]) and idx.get('unique')
            for idx in indexes.values())
        or any(_covers(uc.get('column_names') or [], composite[1])
               for uc in unique_constraints)
    )

    # 旧库：单列唯一索引/约束必须先让位，否则仍然写不进第二个版本
    stale_unique = [
        name for name, idx in indexes.items()
        if idx.get('unique') and _covers(idx.get('column_names') or [], ['question_hash'])
    ]
    stale_unique += [
        uc['name'] for uc in unique_constraints
        if _covers(uc.get('column_names') or [], ['question_hash'])
    ]

    try:
        with engine.begin() as conn:
            for name in stale_unique:
                conn.execute(text(f'DROP INDEX IF EXISTS {name}'))
            # ⚠️ 收敛式而不是条件式：无论中间历史如何（例如某个中间版本删了旧唯一索引
            # 却没重建），都必须把目标 schema 补齐 —— 这段每次启动都跑，它应当把库
            # **收敛到**目标形态，而不是「只在首次迁移时正确」。
            # 尤其是：历史单列唯一索引**就叫** ix_qa_cache_question_hash，会被上面删掉，
            # 所以不能写「名字已存在就跳过」，否则迁移库会比全新库少一个索引。
            conn.execute(text(
                f'CREATE INDEX IF NOT EXISTS {QA_CACHE_LEGACY_INDEX} '
                'ON qa_cache (question_hash)'
            ))
            if not has_composite:
                conn.execute(text(
                    f'CREATE UNIQUE INDEX IF NOT EXISTS {QA_CACHE_VERSION_INDEX} '
                    'ON qa_cache (question_hash, content_hash)'
                ))
    except Exception:  # noqa: BLE001 权限/方言差异下不阻断启动，但留下可查的痕迹
        logger.warning('存储[迁移] qa_cache 复合唯一键迁移失败，缓存可能仍受旧约束限制',
                       exc_info=True)


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
