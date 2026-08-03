"""HITL 路由：澄清任务列表、提交与拒绝。"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from reading_assistant.api import schemas
from reading_assistant.api.deps import get_db_session
from reading_assistant.storage import HitlTask, reject_hitl_task, submit_hitl_clarification

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
):
    """提交澄清：awaiting -> approved。"""
    try:
        return submit_hitl_clarification(session, task_id, payload.clarification)
    except ValueError as exc:
        status = 404 if '不存在' in str(exc) else 400
        raise HTTPException(status_code=status, detail=str(exc)) from exc


@router.post('/{task_id}/reject', response_model=schemas.HitlTaskOut)
def reject_task(task_id: int, session: Session = Depends(get_db_session)):
    """拒绝澄清：awaiting -> rejected。"""
    try:
        return reject_hitl_task(session, task_id)
    except ValueError as exc:
        status = 404 if '不存在' in str(exc) else 400
        raise HTTPException(status_code=status, detail=str(exc)) from exc
