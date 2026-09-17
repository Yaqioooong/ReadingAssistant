"""API 依赖：请求级数据库会话 + 运行时单例的 re-export。

运行时单例（会话工厂/向量库/模型/上传目录）已下沉到
``reading_assistant.runtime``，以便 MCP 等其它适配器复用而不依赖 HTTP 层。
本模块保留 re-export 维持既有导入路径向后兼容。
"""

from collections.abc import Iterator

from fastapi import Depends, Request
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
    'get_qa_graph',
    'get_session_factory',
    'get_upload_dir',
    'get_vector_store',
]


def get_qa_graph(
    request: Request,
    session_factory: sessionmaker[Session] = Depends(get_session_factory),
    vector_store=Depends(get_vector_store),
    llm=Depends(get_llm),
    embedding_model=Depends(get_embedding_model),
):
    """问答图 —— **每个 app 一份**并缓存。

    两个原因：
    1. **正确性**：恢复要求「同一张图 + 同一个 saver」。每请求重建图 = 每请求换
       saver，挂起时的 checkpoint 再也查不到，「中断-恢复」就成了摆设。
    2. **成本**：重建图会连带重建 retriever —— 实测每请求白付 ChromaVectorStore
       构造 674ms + BM25 全量重建 ~1300ms，且 L1 embedding 缓存跨请求全部失效。

    缓存挂在 ``app.state`` 上（而非模块级 lru_cache），这样测试里每个 app
    拿到自己的图，互不串味。
    """
    graph = getattr(request.app.state, 'qa_graph', None)
    if graph is None:
        from reading_assistant.graph import build_qa_graph
        from reading_assistant.runtime import get_checkpointer

        graph = build_qa_graph(
            session_factory,
            vector_store,
            llm=llm,
            embedding_model=embedding_model,
            checkpointer=getattr(request.app.state, 'checkpointer', None)
            or get_checkpointer(),
        )
        request.app.state.qa_graph = graph
    return graph


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
