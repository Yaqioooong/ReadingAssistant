"""HITL 路由：澄清任务列表、提交与拒绝。"""

from fastapi import APIRouter, Depends, HTTPException
from langgraph.types import Command
from sqlalchemy import select
from sqlalchemy.orm import Session

from reading_assistant.api import schemas
from reading_assistant.api.deps import get_db_session, get_qa_graph
from reading_assistant.storage import (
    HitlTask,
    find_answer_for_clarification,
    get_hitl_task,
    reject_hitl_task,
    submit_hitl_clarification,
)
from reading_assistant.utils.logger_handler import get_logger

logger = get_logger('api')

router = APIRouter(prefix='/api/hitl/tasks', tags=['hitl'])


@router.get('', response_model=list[schemas.HitlTaskOut])
def list_hitl_tasks(
    session_id: int | None = None,
    session: Session = Depends(get_db_session),
):
    """列出 HITL 澄清任务，可按会话过滤。"""
    stmt = select(HitlTask).order_by(HitlTask.id)
    if session_id is not None:
        stmt = stmt.where(HitlTask.session_id == session_id)
    return list(session.scalars(stmt))


@router.post('/{task_id}/submit', response_model=schemas.HitlTaskOut)
def submit_clarification(
    task_id: int,
    payload: schemas.HitlClarificationRequest,
    session: Session = Depends(get_db_session),
    graph=Depends(get_qa_graph),
):
    """提交澄清：awaiting -> approved，**并从断点继续**原问题的回答。

    真「中断-恢复」与旧「重跑」的区别：
    - 旧：客户端拿到 200 后自己再发一次问题（带 clarification），服务端从 START
      全量重跑 —— gate/context_load/summarize/检索全部重付，且是新的 thread；
    - 新：本端点直接把澄清喂回挂起的图（``Command(resume=...)``），从 create_hitl
      的下一跳继续 —— 本次会话既有的 document_ids / 历史 / 已算向量都在 checkpoint 里，
      不需要客户端重发，也不重复付费调用。

    降级：``task.thread_id`` 为空（加列前的旧任务）或 checkpoint 已丢失时，
    不假装成功 —— ``resumed=False`` 且不带答案，客户端按旧路径重发即可。
    """
    existing = get_hitl_task(session, task_id)
    if existing is None:
        raise HTTPException(status_code=404, detail=f'HITL 任务不存在: {task_id}')

    if existing.status == HitlTask.STATUS_APPROVED:
        # 幂等：重复提交（双击 / 网络重试 / 旧前端）不应再生成一次，
        # 也不该报错 —— 直接返回上次恢复出的那份回答。
        prior = find_answer_for_clarification(
            session, existing.session_id, existing.clarification or payload.clarification
        )
        if prior is not None:
            logger.info('HITL[恢复] 任务已恢复过，幂等返回 task=%s message=%s',
                        task_id, prior.id)
            out = schemas.HitlTaskOut.model_validate(existing)
            return out.model_copy(update={
                'resumed': True,
                'answer': prior.content,
                'citations': [
                    schemas.CitationOut(**c) if isinstance(c, dict) else c
                    for c in ((prior.meta or {}).get('citations') or [])
                ],
            })
        raise HTTPException(status_code=400, detail=f'任务已处理: {task_id}')

    try:
        task = submit_hitl_clarification(session, task_id, payload.clarification)
    except ValueError as exc:
        status = 404 if '不存在' in str(exc) else 400
        raise HTTPException(status_code=status, detail=str(exc)) from exc

    thread_id = task.thread_id
    out = schemas.HitlTaskOut.model_validate(task)
    if not thread_id:
        logger.warning('HITL[恢复] 任务无 thread_id，降级为重跑 task=%s', task_id)
        return out

    try:
        state = graph.get_state({'configurable': {'thread_id': thread_id}})
    except Exception:  # noqa: BLE001 查不到 checkpoint
        state = None
    if state is None or not state.next:
        # 没有待执行节点 = 没有可恢复的断点（进程重启且用内存 saver / 已被回收）
        logger.warning('HITL[恢复] 未找到挂起中的 checkpoint，降级为重跑 task=%s thread=%s',
                       task_id, thread_id)
        return out

    try:
        result = graph.invoke(
            Command(resume=payload.clarification),
            config={'configurable': {'thread_id': thread_id}},
        )
    except Exception:
        logger.exception('HITL[恢复] 恢复执行失败 task=%s thread=%s', task_id, thread_id)
        raise HTTPException(status_code=500, detail='从断点恢复失败，请重试') from None

    # 恢复后图也跑到了 END（本轮不会再挂起：clarification 已存在，create_hitl 直接收尾）
    from reading_assistant.graph.checkpointer import discard_thread_if_finished

    discard_thread_if_finished(graph, thread_id)
    logger.info('HITL[恢复] 完成 task=%s thread=%s cit=%d',
                task_id, thread_id, len(result.get('citations') or []))
    return out.model_copy(update={
        'resumed': True,
        'answer': result.get('answer'),
        'citations': [
            schemas.CitationOut(**c) if isinstance(c, dict) else c
            for c in (result.get('citations') or [])
        ],
        'needs_clarification': bool(result.get('needs_clarification')),
    })


@router.post('/{task_id}/reject', response_model=schemas.HitlTaskOut)
def reject_task(
    task_id: int,
    session: Session = Depends(get_db_session),
    graph=Depends(get_qa_graph),
):
    """拒绝澄清：awaiting -> rejected，并回收该断点的 checkpoint。

    拒绝意味着这条分支永远不会被恢复 —— 留着 checkpoint 只是占地方
    （挂起中的 thread 是**刻意保留**的，所以必须在这里显式作废）。
    """
    try:
        task = reject_hitl_task(session, task_id)
    except ValueError as exc:
        status = 404 if '不存在' in str(exc) else 400
        raise HTTPException(status_code=status, detail=str(exc)) from exc

    thread_id = task.thread_id
    if thread_id:
        # 状态还在挂起（next 非空），discard 对挂起 thread 会返回 False ——
        # 所以这里直接删，不走「跑完了才删」的判断
        saver = getattr(graph, 'checkpointer', None)
        if saver is not None and hasattr(saver, 'delete_thread'):
            saver.delete_thread(thread_id)
            logger.info('HITL[拒绝] 已作废断点 task=%s thread=%s', task_id, thread_id)
    return task
