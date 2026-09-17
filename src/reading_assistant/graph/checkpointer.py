"""LangGraph 检查点工厂：内存（默认）与 PostgreSQL。

检查点是「中断-恢复」的物理载体：图在 ``interrupt()`` 处挂起时，状态只存在于
checkpoint 里。因此：
- ``memory``：进程内可恢复（够测试/单进程用），进程重启即失；
- ``postgres``：跨进程、跨重启可恢复 —— **多 worker 部署必须用这个**，
  否则恢复请求落到别的进程会找不到 checkpoint。
"""

from __future__ import annotations

from langgraph.checkpoint.memory import InMemorySaver

_POOL_ATTR = '_alma_connection_pool'


def create_inmemory_checkpointer():
    """内存检查点（测试与单进程开发）。"""
    return InMemorySaver()


def _to_psycopg_dsn(url: str) -> str:
    """SQLAlchemy URL → psycopg 连接串（去掉 ``+psycopg`` 方言后缀）。"""
    return url.replace('postgresql+psycopg://', 'postgresql://')


def create_postgres_checkpointer(database_url: str | None = None):
    """PostgreSQL 检查点（跨进程可恢复）。

    ⚠️ 历史实现有两处会直接崩（且因为从未被调用而没暴露）：
    1. ``PostgresSaver.from_conn_string(...)`` 返回的是**上下文管理器/迭代器**，
       对它调 ``.setup()`` 会 AttributeError；
    2. ``setup()`` 每次调用都重建连接 —— 必须只做一次。

    现在改用 ``ConnectionPool``：连接常驻、``setup()`` 只在建池时跑一次，
    池对象挂在 saver 上，由 :func:`close_checkpointer` 统一释放。

    LangGraph 要求连接开 ``autocommit`` 且关闭 prepared statement 缓存
    （``prepare_threshold=0``），否则在 pgbouncer / 连接复用下会踩到
    「prepared statement 已存在」。
    """
    from langgraph.checkpoint.postgres import PostgresSaver
    from psycopg_pool import ConnectionPool

    from reading_assistant.config import get_settings

    url = _to_psycopg_dsn(database_url or get_settings().database_url)
    pool = ConnectionPool(
        conninfo=url,
        min_size=1,
        max_size=8,
        kwargs={'autocommit': True, 'prepare_threshold': 0},
        open=True,
    )
    saver = PostgresSaver(pool)
    saver.setup()
    setattr(saver, _POOL_ATTR, pool)
    # 兜底：脚本/CLI 退出时若没人显式关池，psycopg_pool 会在解释器退出时告警
    # （"couldn't stop thread ..."）。close() 是幂等的，多关一次无害。
    import atexit

    atexit.register(close_checkpointer, saver)
    return saver


def create_checkpointer():
    """按配置创建检查点后端。"""
    from reading_assistant.config import get_settings

    backend = (get_settings().checkpoint_backend or 'memory').lower()
    if backend == 'postgres':
        return create_postgres_checkpointer()
    if backend == 'memory':
        return create_inmemory_checkpointer()
    raise ValueError(f'未知的 checkpoint 后端: {backend}')


def close_checkpointer(saver) -> None:
    """释放 saver 持有的连接池（幂等）。"""
    pool = getattr(saver, _POOL_ATTR, None)
    if pool is not None:
        try:
            pool.close()
        except Exception:  # noqa: BLE001 关池失败不该影响退出流程
            pass


def discard_thread_if_finished(graph, thread_id: str | None) -> bool:
    """本轮若已跑完（没有待执行节点），就删掉该 thread 的 checkpoint。

    为什么必须删：checkpoint 只在「挂起中」才有用途。跑完还留着，就是在用
    每请求 ~37KB 的速度堆内存（实测 200 次请求积 7.2MB，且永不回收）。
    只有真正挂起（``state.next`` 非空）的 thread 才需要保留给恢复用。
    """
    if not thread_id:
        return False
    saver = getattr(graph, 'checkpointer', None)
    if saver is None or not hasattr(saver, 'delete_thread'):
        return False
    try:
        state = graph.get_state({'configurable': {'thread_id': thread_id}})
    except Exception:  # noqa: BLE001 查不到就当没有
        return False
    if state.next:  # 有待执行节点 = 挂起中，保留
        return False
    saver.delete_thread(thread_id)
    return True
