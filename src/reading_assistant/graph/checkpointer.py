"""LangGraph 检查点工厂：内存（默认）与 PostgreSQL。"""

from langgraph.checkpoint.memory import InMemorySaver


def create_inmemory_checkpointer():
    """内存检查点（测试与本地开发）。"""
    return InMemorySaver()


def create_postgres_checkpointer(database_url: str | None = None):
    """PostgreSQL 检查点（生产环境；需先 ``docker compose up -d``）。"""
    from langgraph.checkpoint.postgres import PostgresSaver

    from reading_assistant.config import get_settings

    url = database_url or get_settings().database_url
    saver = PostgresSaver.from_conn_string(url)
    saver.setup()
    return saver
