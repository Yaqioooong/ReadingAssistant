"""API 请求 / 响应模型。"""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    filename: str
    title: str
    author: str | None = None
    chunk_count: int
    created_at: datetime


class UploadResponse(DocumentOut):
    duplicate: bool = False


class SessionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    title: str | None = None
    created_at: datetime


class SessionCreated(BaseModel):
    session_id: str


class CitationOut(BaseModel):
    index: int | None = None  # prompt 中片段的原始编号，与回答里的 [n] 对应
    chunk_id: str
    chapter: str | None = None
    page: int | None = None
    excerpt: str = ''
    document: str | None = None  # 来源书名（多文档问答时填充）


class MessageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    role: str
    content: str
    meta: dict = Field(default_factory=dict)
    created_at: datetime


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    document_ids: list[int] = Field(default_factory=list)
    clarification: str | None = None


class AskResponse(BaseModel):
    answer: str | None = None
    citations: list[CitationOut] = Field(default_factory=list)
    needs_clarification: bool = False
    hitl_task_id: int | None = None
    intent: str | None = None  # 检索门分类: book | history | chat(供评测/观测)
    # 缓存来源（供指标看板与前端反馈使用）
    cache_hit: bool = False
    cache_channel: str | None = None  # exact | semantic | identifier | miss | disabled
    cache_similarity: float | None = None


class FeedbackCreate(BaseModel):
    """用户对回答的反馈：误命中率统计的输入。"""

    session_id: int | None = None
    question: str = Field(min_length=1, max_length=2000)
    vote: str = Field(pattern='^(up|down)$')
    cache_hit: bool = False
    cache_channel: str | None = None


class FeedbackOut(BaseModel):
    id: int
    vote: str


class HitlTaskOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    session_id: int | None = None
    question: str
    status: str
    clarification: str | None = None
    created_at: datetime
    updated_at: datetime


class HitlClarificationRequest(BaseModel):
    clarification: str = Field(min_length=1)
