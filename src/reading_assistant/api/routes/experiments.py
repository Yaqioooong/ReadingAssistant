"""实验路由：评测运行 / 多 Agent 问答 / 日志浏览。"""

from datetime import date
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from reading_assistant.api.deps import get_llm
from reading_assistant.utils.logger_handler import LOG_ROOT

router = APIRouter(prefix='/api/experiments', tags=['experiments'])

# 日志文件白名单（对应 logger_handler 的命名 logger）
LOG_NAMES = {'api', 'qa', 'ingest', 'storage', 'multi_agent'}


class EvalCase(BaseModel):
    question: str = Field(min_length=1, max_length=300)
    document: str | None = None
    expect_keywords: list[str] = Field(default_factory=list)
    expect_unanswerable: bool = False


class EvalRequest(BaseModel):
    mode: str = Field('fake', pattern='^(fake|real)$')
    limit: int = Field(0, ge=0, le=15)
    cases: list[EvalCase] = Field(default_factory=list)


class MultiAgentRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)


@router.post('/eval')
def run_eval(req: EvalRequest):
    """运行黄金评测集；real 模式调用真实模型，约需 20~30s。"""
    try:
        from eval.run_eval import run
    except ImportError as exc:
        raise HTTPException(status_code=500, detail=f'评测模块不可用: {exc}') from exc
    try:
        custom = [c.model_dump() for c in req.cases] or None
        return run(mode=req.mode, limit=req.limit, cases=custom)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f'评测失败: {exc}') from exc


@router.post('/multi-agent')
def run_multi_agent(req: MultiAgentRequest, llm=Depends(get_llm)):
    """多 Agent 问答：Supervisor 规划 → 检索 → 总结，返回逐步结果。"""
    try:
        from experiments.multi_agent import run
    except ImportError as exc:
        raise HTTPException(status_code=500, detail=f'实验模块不可用: {exc}') from exc
    try:
        return run(req.question, llm=llm)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f'多 Agent 执行失败: {exc}') from exc


@router.get('/logs')
def get_logs(name: str = 'api', lines: int = 150):
    """读取今日指定 logger 的日志尾部。"""
    if name not in LOG_NAMES:
        raise HTTPException(status_code=400, detail=f'未知 logger: {name}')
    path = Path(LOG_ROOT) / f'{name}_{date.today().strftime("%Y%m%d")}.log'
    if not path.exists():
        return {'lines': [], 'path': str(path)}
    try:
        text = path.read_text(encoding='utf-8')
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f'读取日志失败: {exc}') from exc
    return {'lines': text.splitlines()[-lines:], 'path': str(path)}
