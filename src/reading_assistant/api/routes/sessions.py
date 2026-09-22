"""会话路由：创建、列表、消息记录与提问。"""

from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from reading_assistant.api import schemas
from reading_assistant.api.deps import (
    get_db_session,
    get_qa_graph,
    get_session_factory,
)
from reading_assistant.graph import interrupt_payload
from reading_assistant.graph.checkpointer import discard_thread_if_finished
from reading_assistant.storage import (
    ChatMessage,
    ChatSession,
    HitlTask,
    QaRequestEvent,
    delete_session,
    find_answer_for_clarification,
    session_scope,
)
from reading_assistant.utils.logger_handler import get_logger

logger = get_logger('api')

router = APIRouter(prefix='/api/sessions', tags=['sessions'])


def _record_cache_event(
    session_factory,
    *,
    session_id: int,
    question: str,
    intent: str | None,
    cache_hit: bool,
    cache_channel: str | None,
    cache_similarity: float | None,
    cache_invalidated: bool,
    latency_ms: int | None,
) -> None:
    """写一条请求级缓存埋点（供指标看板统计）。

    埋点失败不能影响问答主流程，因此整体吞掉异常并降级为 warning 日志。
    """
    try:
        with session_scope(session_factory) as session:
            session.add(
                QaRequestEvent(
                    session_id=session_id,
                    question=(question or '')[:200],
                    intent=intent,
                    cache_hit=cache_hit,
                    cache_channel=cache_channel,
                    cache_similarity=cache_similarity,
                    cache_invalidated=cache_invalidated,
                    latency_ms=latency_ms,
                )
            )
    except Exception:  # noqa: BLE001 埋点失败不影响问答
        logger.warning('缓存埋点写入失败 session_id=%s', session_id, exc_info=True)


def _ensure_session(session: Session, session_id: int) -> None:
    if session.get(ChatSession, session_id) is None:
        raise HTTPException(status_code=404, detail=f'会话不存在: {session_id}')


@router.post('', response_model=schemas.SessionCreated, status_code=201)
def create_session(session: Session = Depends(get_db_session)):
    """创建会话并返回 session_id。

    这里显式 commit 而非只 flush：FastAPI 的 yield 依赖其退出代码（含 commit）
    在**响应发出之后**才执行，前端「新建会话 → 立刻提问」会读到未提交的会话，
    表现为偶发 404（会话不存在）。显式提交消除该竞态。
    """
    chat = ChatSession()
    session.add(chat)
    session.commit()
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
    graph=Depends(get_qa_graph),
) -> None:
    """删除会话及其全部消息与 HITL 任务（连带回收挂起的断点）。"""
    # 会话没了，它下面挂起的那些断点也永远不会被恢复 → 一并作废
    thread_ids = [
        t for t in session.scalars(
            select(HitlTask.thread_id).where(HitlTask.session_id == session_id)
        ) if t
    ]
    if not delete_session(session, session_id):
        logger.warning('删除会话失败，会话不存在 session_id=%s', session_id)
        raise HTTPException(status_code=404, detail=f'会话不存在: {session_id}')
    saver = getattr(graph, 'checkpointer', None)
    if saver is not None and hasattr(saver, 'delete_thread'):
        for thread_id in thread_ids:
            saver.delete_thread(thread_id)
        if thread_ids:
            logger.info('删除会话 session_id=%s，作废断点 %d 个', session_id, len(thread_ids))
    logger.info('删除会话 session_id=%s', session_id)


@router.post('/{session_id}/messages', response_model=schemas.AskResponse)
def ask_question(
    session_id: int,
    payload: schemas.AskRequest,
    session_factory=Depends(get_session_factory),
    # 图由 app 级单例提供（见 deps.get_qa_graph）—— 恢复必须复用同一张图/saver
    graph=Depends(get_qa_graph),
    session: Session = Depends(get_db_session),
):
    """提问：运行问答流水线并记录消息；信息不足时**挂起**（可断点续答）。"""
    import time

    _ensure_session(session, session_id)

    # 幂等护栏：这次澄清是否已经产出过回答？
    # 客户端契约在 2026-09-17 变更为「提交澄清 = 服务端从断点恢复并生成回答」，
    # 但**重发**的来源很杂：未重建的旧前端（提交后仍重发原问题）、双击、网络重试。
    # 不拦的话就会在恢复生成一次之外再生成一次 —— 用户看到连续两条回答。
    if payload.clarification:
        prior = find_answer_for_clarification(session, session_id, payload.clarification)
        if prior is not None:
            logger.info(
                '提问[幂等] 该澄清已产生过回答，直接复用 session_id=%s message_id=%s',
                session_id, prior.id,
            )
            return schemas.AskResponse(
                answer=prior.content,
                citations=(prior.meta or {}).get('citations') or [],
                needs_clarification=False,
                cache_hit=False,
                cache_channel='clarification_replay',
            )

    doc_ids = payload.document_ids or []
    # 单文档走原 document_id 路径（缓存友好）；多文档走 fan-out 并行检索
    document_id = doc_ids[0] if len(doc_ids) == 1 else None
    logger.info('提问 session_id=%s docs=%s q=%.40s', session_id, doc_ids, payload.question)
    start = time.perf_counter()
    invoke_state: dict = {
        'question': payload.question,
        'session_id': session_id,
        'document_id': document_id,
        'clarification': payload.clarification,
    }
    if len(doc_ids) > 1:
        invoke_state['document_ids'] = doc_ids
    thread_id = f'qa-{uuid4().hex}'
    result = graph.invoke(
        invoke_state,
        config={'configurable': {'thread_id': thread_id}},
    )
    cost_ms = (time.perf_counter() - start) * 1000
    # 图在 create_hitl 处挂起：节点返回值不会进 state（中断时被丢弃），
    # 所以 task_id 只能从**挂起载荷**里取 —— 见 qa.create_hitl 的 interrupt(payload)
    pending = interrupt_payload(result)
    if pending is not None:
        logger.info('提问挂起等待澄清 session_id=%s thread=%s task=%s',
                    session_id, thread_id, pending.get('hitl_task_id'))
        _record_cache_event(
            session_factory,
            session_id=session_id,
            question=payload.question,
            intent=result.get('intent'),
            cache_hit=bool(result.get('cache_hit')),
            cache_channel=result.get('cache_channel'),
            cache_similarity=result.get('cache_similarity'),
            cache_invalidated=bool(result.get('cache_invalidated')),
            latency_ms=int(cost_ms),
        )
        return schemas.AskResponse(
            needs_clarification=True,
            hitl_task_id=pending.get('hitl_task_id'),
            resumable=bool(pending.get('hitl_task_id')),
            thread_id=thread_id,
            intent=result.get('intent'),
            intent_channel=result.get('intent_channel'),
            intent_score=result.get('intent_score'),
            agent_used=bool(result.get('agent_used')),
            agent_steps=result.get('agent_steps') or 0,
            agent_actions=[t.get('action', '') for t in (result.get('agent_trace') or [])],
            cache_hit=bool(result.get('cache_hit')),
            cache_channel=result.get('cache_channel'),
        )
    # 跑完就没必要留着 checkpoint（挂起中的才要留）
    discard_thread_if_finished(graph, thread_id)
    # 闲聊/历史类轮次走 context_answer，不经 cache_check → 归为 skipped
    # （与 disabled 区分：disabled 是多文档或缓存开关关闭，skipped 是本就不适用缓存）
    skipped = result.get('intent') in ('chat', 'history')
    channel = result.get('cache_channel') or ('skipped' if skipped else None)
    logger.info('提问完成 session_id=%s (%.0fms) hit=%s channel=%s cit=%d', session_id,
                cost_ms, result.get('cache_hit', False), channel,
                len(result.get('citations') or []))
    _record_cache_event(
        session_factory,
        session_id=session_id,
        question=payload.question,
        intent=result.get('intent'),
        cache_hit=bool(result.get('cache_hit')),
        cache_channel=channel,
        cache_similarity=result.get('cache_similarity'),
        cache_invalidated=bool(result.get('cache_invalidated')),
        latency_ms=int(cost_ms),
    )
    return schemas.AskResponse(
        answer=result.get('answer'),
        citations=result.get('citations') or [],
        needs_clarification=result.get('needs_clarification', False),
        hitl_task_id=result.get('hitl_task_id'),
        thread_id=thread_id,
        intent=result.get('intent'),
        intent_channel=result.get('intent_channel'),
        intent_score=result.get('intent_score'),
        agent_used=bool(result.get('agent_used')),
        agent_steps=result.get('agent_steps') or 0,
        agent_actions=[t.get('action', '') for t in (result.get('agent_trace') or [])],
        cache_hit=bool(result.get('cache_hit')),
        cache_channel=channel,
        cache_similarity=result.get('cache_similarity'),
    )
