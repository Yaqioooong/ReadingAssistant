"""会话路由：创建、列表、消息记录与提问。"""

from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from reading_assistant.api import schemas
from reading_assistant.api.deps import (
    get_db_session,
    get_embedding_model,
    get_llm,
    get_session_factory,
    get_vector_store,
)
from reading_assistant.graph import build_qa_graph
from reading_assistant.storage import ChatMessage, ChatSession, delete_session
from reading_assistant.storage.vector_store import VectorStore
from reading_assistant.utils.logger_handler import get_logger

logger = get_logger('api')

router = APIRouter(prefix='/api/sessions', tags=['sessions'])


def _ensure_session(session: Session, session_id: int) -> None:
    if session.get(ChatSession, session_id) is None:
        raise HTTPException(status_code=404, detail=f'会话不存在: {session_id}')


@router.post('', response_model=schemas.SessionCreated, status_code=201)
def create_session(session: Session = Depends(get_db_session)):
    """创建会话并返回 session_id。"""
    chat = ChatSession()
    session.add(chat)
    session.flush()
    logger.info('创建会话 session_id=%s', chat.id)
    return schemas.SessionCreated(session_id=str(chat.id))


@router.get('', response_model=list[schemas.SessionOut])
def list_sessions(session: Session = Depends(get_db_session)):
    """列出全部会话。"""
    chats = session.scalars(select(ChatSession).order_by(ChatSession.id)).all()
    return [
        schemas.SessionOut(id=str(chat.id), title=chat.title, created_at=chat.created_at)
        for chat in chats
    ]


@router.get('/{session_id}/messages', response_model=list[schemas.MessageOut])
def get_messages(session_id: int, session: Session = Depends(get_db_session)):
    """查看会话的聊天记录。"""
    _ensure_session(session, session_id)
    return list(
        session.scalars(
            select(ChatMessage).where(ChatMessage.session_id == session_id).order_by(ChatMessage.id)
        )
    )


@router.delete('/{session_id}', status_code=204)
def delete_chat_session(
    session_id: int,
    session: Session = Depends(get_db_session),
) -> None:
    """删除会话及其全部消息与 HITL 任务。"""
    if not delete_session(session, session_id):
        logger.warning('删除会话失败，会话不存在 session_id=%s', session_id)
        raise HTTPException(status_code=404, detail=f'会话不存在: {session_id}')
    logger.info('删除会话 session_id=%s', session_id)


@router.post('/{session_id}/messages', response_model=schemas.AskResponse)
def ask_question(
    session_id: int,
    payload: schemas.AskRequest,
    session_factory=Depends(get_session_factory),
    vector_store: VectorStore = Depends(get_vector_store),
    llm=Depends(get_llm),
    embedding_model=Depends(get_embedding_model),
    session: Session = Depends(get_db_session),
):
    """提问：运行问答流水线并记录消息；信息不足时创建 HITL 任务。"""
    import time

    _ensure_session(session, session_id)
    doc_ids = payload.document_ids or []
    # 单文档走原 document_id 路径（缓存友好）；多文档走 fan-out 并行检索
    document_id = doc_ids[0] if len(doc_ids) == 1 else None
    logger.info('提问 session_id=%s docs=%s q=%.40s', session_id, doc_ids, payload.question)
    start = time.perf_counter()
    graph = build_qa_graph(
        session_factory,
        vector_store,
        llm=llm,
        embedding_model=embedding_model,
    )
    invoke_state: dict = {
        'question': payload.question,
        'session_id': session_id,
        'document_id': document_id,
        'clarification': payload.clarification,
    }
    if len(doc_ids) > 1:
        invoke_state['document_ids'] = doc_ids
    result = graph.invoke(
        invoke_state,
        config={'configurable': {'thread_id': f'qa-{uuid4().hex}'}},
    )
    cost_ms = (time.perf_counter() - start) * 1000
    logger.info('提问完成 session_id=%s (%.0fms) hit=%s cit=%d', session_id, cost_ms,
                result.get('cache_hit', False), len(result.get('citations') or []))
    return schemas.AskResponse(
        answer=result.get('answer'),
        citations=result.get('citations') or [],
        needs_clarification=result.get('needs_clarification', False),
        hitl_task_id=result.get('hitl_task_id'),
    )
