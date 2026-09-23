"""会话级结构化状态 —— 滚动摘要的替代品。

## 为什么不用滚动摘要
它的形状是「无 schema 的散文 + 每轮全量重写」，三处硬伤：
1. **无 schema** → 无法参与任何确定性判断（路由、升级、填槽），只能当提示词。
2. **每轮重写** → 一次改坏就永久污染后续所有轮次，且没有任何信号（静默失效）。
3. **位于 prompt 前缀且每轮都变** → 主动让 prefix cache 失效，反而比原文更贵。

## 这里用什么
「结构化状态 + 确定性合并」：字段有 schema，合并是纯函数。
**模型只提供信号，不负责重写记忆** —— 于是可测、可审计、可回放、无漂移。

## 零 LLM 成本（本模块的核心约束）
所有字段都从流水线里**已经算出来的**信号填充（上一轮检索的 doc_titles、
本轮的 intent/question）。本模块**不得**引入任何模型调用 ——
摘要方案正是死在「每轮一次 LLM」上，换皮重来毫无意义。

## 面向客服场景的扩展点
客服需要 order_id / product / issue_type / 升级状态这类槽位。届时只需两步：
①在 ``ConversationState`` 加字段；②调用方把新信号塞进 ``patch``。
机制（schema + 确定性合并 + 落库）完全不变。
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

from sqlalchemy.orm import Session

from reading_assistant.storage import ChatSession
from reading_assistant.utils.logger_handler import get_logger

logger = get_logger('state')


@dataclass
class ConversationState:
    """一个会话的结构化状态。所有字段均可 JSON 序列化。"""

    # 本会话涉及过的书名（取自各轮检索结果的 doc_titles，累积去重、有上限）。
    # 价值：原文窗口把早期轮次挤出去之后，这里仍记得「这场对话谈过哪几本书」，
    # 且是**精确书名**而非散文 —— 正是摘要想干、却会顺手把实体名压没的那件事。
    # 注意：是「谈过的书」不是「当前话题」。换书不该抹掉上一本，
    # 因为这个字段的意义就是承接被窗口挤掉的东西。合并策略见 qa._merge_active_documents。
    active_documents: list[str] = field(default_factory=list)
    last_question: str | None = None
    last_intent: str | None = None
    pending_clarification: str | None = None
    # 派生量：会话内用户消息条数。**从消息表数出来**而不是自增 ——
    # interrupt() 恢复会让节点重跑，自增会数重，数出来天然幂等。
    turn_count: int = 0


def merge_state(
    prev: ConversationState | None, patch: dict[str, Any] | None
) -> ConversationState:
    """把 ``patch`` 确定性地并入 ``prev``。纯函数，无 IO、无模型调用。

    三条语义（都有单测钉死）：
    - **非空才覆盖**：``None`` / 空串 / 空列表一律视为「本轮没提供信号」，保留旧值。
      否则一轮没有检索结果的追问就会把 active_documents 清空 —— 那正是最需要它的场景。
    - **列表整体替换而非追加**：``active_documents`` 表达的是「当前话题」，
      语义上取最新值，不是累积集合。
    - **未知键忽略**：库里可能存着旧版本写的字段，
      读路径不能因为字段增删就炸掉 —— 本项目已多次死在「格式变了没人说」上。
    """
    base = prev or ConversationState()
    data = asdict(base)
    for key, value in (patch or {}).items():
        if key not in data:
            logger.debug('会话状态忽略未知键:%s', key)
            continue
        if value is None or value == '' or value == []:
            continue
        data[key] = value
    return ConversationState(**data)


def to_json(state: ConversationState) -> str:
    """序列化落库。``ensure_ascii=False`` 让库里能直接看懂内容。"""
    return json.dumps(asdict(state), ensure_ascii=False)


def from_json(raw: str | None) -> ConversationState | None:
    """解析落库的 JSON；坏数据返回 None，**绝不抛异常**。

    读路径不能因为一条脏记录把整道题带崩 —— 旧格式、半截写入、
    人工改库都真实发生过（本项目前科累累）。
    """
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except (TypeError, ValueError):
        logger.warning('会话状态 JSON 解析失败，按空状态处理')
        return None
    if not isinstance(data, dict):
        logger.warning('会话状态不是 JSON 对象，按空状态处理')
        return None
    known = set(ConversationState.__dataclass_fields__)
    unknown = set(data) - known
    if unknown:
        logger.info('会话状态含未知字段(已忽略):%s', sorted(unknown))
    return ConversationState(**{k: v for k, v in data.items() if k in known})


def load_conversation_state(session: Session, sid: int) -> ConversationState | None:
    """读回会话状态；无会话/无状态返回 None。"""
    chat = session.get(ChatSession, sid)
    if chat is None:
        return None
    return from_json(getattr(chat, 'state', None))


def save_conversation_state(session: Session, sid: int, state: ConversationState) -> None:
    """写回会话状态；会话不存在则静默跳过（不制造幽灵行）。"""
    chat = session.get(ChatSession, sid)
    if chat is None:
        return
    chat.state = to_json(state)
