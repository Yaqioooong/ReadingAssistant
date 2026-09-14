"""反馈路由：记录用户对回答的评价，用于计算缓存误命中率。"""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from reading_assistant.api import schemas
from reading_assistant.api.deps import get_db_session
from reading_assistant.storage import QaFeedback
from reading_assistant.utils.logger_handler import get_logger

logger = get_logger('api')

router = APIRouter(prefix='/api/feedback', tags=['feedback'])


@router.post('', response_model=schemas.FeedbackOut, status_code=201)
def create_feedback(
    payload: schemas.FeedbackCreate,
    session: Session = Depends(get_db_session),
):
    """记录一条反馈（up/down）。

    ``cache_hit`` 标记该答案是否来自缓存 —— 误命中率的分母依据，
    因此由前端从响应里回传，而不是事后靠日志推断。
    """
    entry = QaFeedback(
        session_id=payload.session_id,
        question=payload.question[:200],
        vote=payload.vote,
        cache_hit=payload.cache_hit,
        cache_channel=payload.cache_channel,
    )
    session.add(entry)
    session.commit()  # 前端提交后常立刻刷新指标，显式提交避免读到未提交数据
    logger.info(
        '反馈 vote=%s cache_hit=%s channel=%s session_id=%s',
        payload.vote, payload.cache_hit, payload.cache_channel, payload.session_id,
    )
    return schemas.FeedbackOut(id=entry.id, vote=entry.vote)
