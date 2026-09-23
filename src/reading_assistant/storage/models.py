"""SQLAlchemy ORM 模型：文档、会话、消息与 HITL 任务。"""

from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    """所有 ORM 模型的基类。"""


class Document(Base):
    """已入库的电子书及元数据。"""

    __tablename__ = 'documents'

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    title: Mapped[str] = mapped_column(String(512), nullable=False, default='')
    author: Mapped[str | None] = mapped_column(String(255))
    file_path: Mapped[str | None] = mapped_column(String(1024))
    file_hash: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    chunk_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    duplicate_of: Mapped[int | None] = mapped_column(ForeignKey('documents.id'))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    # pending, indexing, indexed, failed
    index_status: Mapped[str] = mapped_column(String(16),nullable=False, default='pending') 
    # 进入 indexing 的时刻：租约凭证；配合启动对账判定「卡死的 indexing」。
    index_started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

class ChatSession(Base):
    """一次对话会话。"""

    __tablename__ = 'chat_sessions'

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    title: Mapped[str | None] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    # 已废弃：2026-09-22 起不再读写。滚动摘要被「token 预算原文窗口 + 结构化状态」取代，
    # 见 graph/state.py。列保留但不再使用（不 DROP —— 用户有活库，破坏性操作要单独走）。
    summary: Mapped[str | None] = mapped_column(Text)
    state: Mapped[str | None] = mapped_column(Text)  # 会话结构化状态 JSON，见 graph/state.py

    messages: Mapped[list['ChatMessage']] = relationship(
        back_populates='session', cascade='all, delete-orphan'
    )


class ChatMessage(Base):
    """会话中的一条消息。"""

    __tablename__ = 'chat_messages'

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    session_id: Mapped[int] = mapped_column(
        ForeignKey('chat_sessions.id'), index=True, nullable=False
    )
    role: Mapped[str] = mapped_column(String(20), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    meta: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    session: Mapped[ChatSession] = relationship(back_populates='messages')


class HitlTask(Base):
    """HITL 澄清任务。"""

    __tablename__ = 'hitl_tasks'

    STATUS_AWAITING = 'awaiting'
    STATUS_APPROVED = 'approved'
    STATUS_REJECTED = 'rejected'

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    session_id: Mapped[int | None] = mapped_column(ForeignKey('chat_sessions.id'))
    question: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default=STATUS_AWAITING)
    clarification: Mapped[str | None] = mapped_column(Text)
    # LangGraph 的 thread_id —— **恢复指针**。图在 create_hitl 里挂起时，
    # 只有这个值能把本次暂停重新接上（进程内靠它查内存 saver，跨进程靠它查 PG）。
    # 没有它就只能「重跑」，那就是今天的行为。空值 = 不可恢复（旧记录 / 降级路径）。
    thread_id: Mapped[str | None] = mapped_column(String(64), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )


class QaCacheEntry(Base):
    """问答结果缓存：命中后跳过LLM/Embedding调用。

    键 = (question_hash, content_hash)，即 **问题 × 文档版本**：

    - ``question_hash`` = 归一化问题 + 澄清补充 + 文档范围(document_id)；
    - ``content_hash`` = 该文档的版本指纹（单书=该文档 content_hash，
      全库=已入库语料指纹），语义是「这条答案是从哪一版原文推出来的」。

    两者都可能变（换版/重传、增删书），所以**必须**是复合唯一：代码的读路径
    （``get_qa_cache_entry`` 与语义层的 ``list_qa_cache_entries``）一直按这两列一起筛选，
    若 DB 只许 ``question_hash`` 单列唯一，就等于「代码声明版本参与身份、DB 却禁止
    两行只差版本」——一旦版本真的变了，写路径插新行必撞唯一约束，整轮问答失败且永久复发。
    """

    __tablename__ = 'qa_cache'
    __table_args__ = (
        Index('uq_qa_cache_question_version', 'question_hash', 'content_hash', unique=True),
    )
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    question_raw: Mapped[str] = mapped_column(Text, nullable=False)
    question_normalized: Mapped[str] = mapped_column(Text, nullable=False)
    # 非唯一索引：单列查询走它，复合唯一索引负责判身份
    question_hash: Mapped[str] = mapped_column(String(64), index=True)
    question_embedding: Mapped[list] = mapped_column(JSON)
    answer: Mapped[str | None] = mapped_column(Text)
    citations: Mapped[list] = mapped_column(JSON, default=list)
    needs_clarification: Mapped[bool] = mapped_column(Boolean, default=False)
    document_id: Mapped[int | None] = mapped_column(ForeignKey('documents.id'), index=True)
    content_hash: Mapped[str] = mapped_column(String(64), index=True)
    # 该回答所依据的片段数，用于区分「环境性失败」（0 个片段）与「书里确实没有」。
    #
    # ⚠️ 加列前的历史行全部是 DEFAULT 0 —— 对这些行来说 0 的含义是「**未知**」，
    # 不是「当时没检索到」。所以**不要**用它去做历史数据清理：
    # 2026-09-17 实测，库里 27 条 needs_clarification 行大多是「拿 A 书的问题问 B 书」
    # 这类**正当**拒答（如向《名词大动词》问「张三喜欢谁？」），按 0 清理会误删它们。
    cached_chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    hit_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    last_hit_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class QaRequestEvent(Base):
    """问答请求级埋点：记录每轮请求的缓存判定结果，供指标看板统计。

    写在 API 路由层而非 graph 内，因为耗时(latency_ms)在路由层才可得，
    且 MCP 等非 HTTP 入口无需埋点。
    """

    __tablename__ = 'qa_request_events'

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    session_id: Mapped[int | None] = mapped_column(Integer, index=True)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    intent: Mapped[str | None] = mapped_column(String(16))
    cache_hit: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    # exact | semantic | identifier | miss | disabled
    cache_channel: Mapped[str | None] = mapped_column(String(16), index=True)
    cache_similarity: Mapped[float | None] = mapped_column(Float)
    cache_invalidated: Mapped[bool] = mapped_column(Boolean, default=False)
    latency_ms: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, index=True
    )


class QaFeedback(Base):
    """用户对回答的反馈：误命中率(cache_hit_down / cache_hit_feedback)的数据来源。"""

    __tablename__ = 'qa_feedback'

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    session_id: Mapped[int | None] = mapped_column(Integer, index=True)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    vote: Mapped[str] = mapped_column(String(8), nullable=False, index=True)  # up | down
    cache_hit: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    cache_channel: Mapped[str | None] = mapped_column(String(16))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, index=True
    )
