"""问答流水线：多轮上下文(滚动摘要) → 检索门 → 检索 → 回答 / HITL 澄清 → 记录。"""

import json
import math
import re
import time
from datetime import datetime, timedelta, timezone
from typing import TypedDict

from langchain_core.embeddings import Embeddings
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from reading_assistant.config import get_settings
from reading_assistant.graph.agent import (
    AgentBudget,
    ToolBox,
    merge_agent_chunks,
    run_agent_loop,
)
from reading_assistant.graph.intent import build_prototype_router, route_fast
from reading_assistant.model.factory import get_chat_model, get_embedding_model
from reading_assistant.rag import create_retriever
from reading_assistant.rag.retriever import _identifier_tokens
from reading_assistant.storage import (
    ChatMessage,
    ChatSession,
    Document,
    create_hitl_task,
    get_document,
    get_or_create_hitl_task,
    get_qa_cache_entry,
    list_documents,
    list_indexed_document_ids,
    list_qa_cache_entries,
    normalize_question,
    prune_qa_cache,
    save_qa_cache_entry,
    session_scope,
    sha256_hex,
    touch_qa_cache_hit,
)
from reading_assistant.storage.vector_store import VectorStore
from reading_assistant.utils.logger_handler import get_logger

logger = get_logger('qa')


class QAState(TypedDict, total=False):
    """问答图的共享状态。"""

    question: str
    session_id: int | None
    document_id: int | None
    document_ids: list[int] | None
    doc_titles: dict[int, str] | None
    clarification: str | None
    chunks: list[dict]
    needs_clarification: bool
    hitl_task_id: int | None
    answer: str | None
    citations: list[dict]
    cache_hit: bool
    question_hash: str | None
    question_embedding: list[float] | None
    history: list[dict] | None  # 最近对话轮次(user/assistant 交错、时间正序,不含本轮)
    intent: str | None  # gate 分类结果: book | history | chat
    summary: str | None  # 长会话滚动摘要文本(较早轮次已压缩)
    cache_channel: str | None  # exact | semantic | identifier | miss | disabled
    cache_similarity: float | None  # 语义/标识符命中时的相似度
    cache_invalidated: bool  # 是否丢弃过"信息不足式"脏缓存
    summary_pending: dict | None  # summarize 增量任务: {upto, anchor, text}
    intent_channel: str | None  # 意图判定通道: rule | prototype | llm(供观测/评测)
    intent_score: float | None  # L2 原型判定时的余弦分(仅观测)
    intent_embedding: list[float] | None  # L2 算出的问题向量, 供 cache_check 复用免重复 embedding
    # --- 检索 agent（有界 ReAct，见 graph/agent.py）---
    agent_used: bool  # 本问题是否已用过 agent（每题至多一次，防循环放大成本）
    agent_trace: list[dict] | None  # 每步 Thought→Action→Observation 摘要（供观测）
    agent_steps: int  # agent 实际执行的步数


def interrupt_payload(result: dict | None) -> dict | None:
    """取图的挂起载荷；没挂起返回 None。

    ⚠️ 图在 ``interrupt()`` 处挂起时，**该节点的返回值不会写进 state**
    （``hitl_task_id`` 因此不在 ``result`` 里），指针只能从挂起载荷里取。
    所有直接 ``invoke`` 图的调用方都必须走这个 helper，否则就会 KeyError
    —— 这正是 2026-09-17 改造打断的那批调用方。
    """
    interrupts = (result or {}).get('__interrupt__') or []
    if not interrupts:
        return None
    value = getattr(interrupts[0], 'value', None)
    return value if isinstance(value, dict) else None


def interrupt_task_id(result: dict | None) -> int | None:
    """挂起中的 HITL 任务 id（未挂起 → None）。"""
    payload = interrupt_payload(result)
    return payload.get('hitl_task_id') if payload else None


def _chunk_to_dict(chunk) -> dict:
    return {
        'chunk_id': chunk.chunk_id,
        'text': chunk.text,
        'score': chunk.score,
        'document_id': chunk.document_id,
        'chapter': chunk.chapter,
        'chapter_index': chunk.chapter_index,
        'citation': chunk.citation,
    }


def _build_answer_prompt(
    question: str,
    chunks: list[dict],
    clarification: str | None = None,
    doc_titles: dict[int, str] | None = None,
    history: list[dict] | None = None,
    summary: str | None = None,
) -> str:
    labeled: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        title = doc_titles.get(chunk.get('document_id')) if doc_titles else None
        prefix = f'《{title}》\n' if title else ''
        labeled.append(f'[{index}]\n{prefix}{chunk["text"]}')
    if labeled:
        context = '\n\n'.join(labeled)
    else:
        context = '（未检索到相关原文片段）'
    lines = [
        '你是阅读助手。请仅根据下方"原文片段"回答问题，不要使用外部知识。',
        f'问题：{question}',
    ]
    if history:
        header = '【最近对话（仅用于理解指代与追问，回答事实以原文片段为准）】\n'
        lines.append(header + _render_history(history))
    if summary:
        lines.append(_summary_section(summary))
    if clarification:
        lines.append(f'用户补充说明：{clarification}')
    lines.extend(
        [
            f'原文片段：\n{context}',
            (
                '回答要求：\n'
                '1. 用简洁自然的中文直接作答，不要复述或粘贴原文片段。\n'
                '2. 需要引用原文时，在对应句子末尾用 [n] 标注（n 为片段编号），'
                '例如：朱六希望张三喜欢王五[1]。\n'
                '   引用要精准：只标注答案实际依据的片段，不要为所有片段都标注。\n'
                '3. 若原文片段足以回答，请调用 final_answer 工具提交回答。\n'
                '4. 若原文片段**不能直接、明确地回答**，必须调用 request_clarification 工具，'
                '并在 missing_info 里说明还缺什么。\n'
                '   ⚠️ 下列情形都属于「不能直接回答」，**不要**用文字绕过、'
                '更不要写「原文片段不足以确定…」之类的说明当作回答：\n'
                '   - 片段里只出现了相关但**不同**的对象（例如问「第一个」却只找到其他同类）；\n'
                '   - 需要更多背景才能定位到问的那一处；\n'
                '   - 问题涉及原文未交代的信息。\n'
                '   漏报「不足」会让系统拿错误的片段作答，比多问一次代价大得多。'
            ),
        ]
    )
    if doc_titles and len(doc_titles) > 1:
        lines.append(
            '注意：本次问题涉及多本书籍。回答时请明确区分不同书籍各自的内容/立场，'
            '原文片段前已标注书名《...》，引用时对应到正确的书籍。'
        )
    return '\n\n'.join(lines)


# ---- 多轮上下文：历史渲染 / 检索门 / 非检索作答 ----
HISTORY_TURNS = 6  # 读回的最近消息条数（user+assistant 合计）
_RENDER_MSG_CAP = 240  # 单条历史渲染上限，控制 prompt 长度


def _render_history(history: list[dict] | None, cap: int = _RENDER_MSG_CAP) -> str:
    """把对话记录渲染成多行文本（时间正序）；供注入 prompt。"""
    if not history:
        return ''
    lines = []
    for msg in history:
        role = '用户' if msg.get('role') == 'user' else '助手'
        content = str(msg.get('content') or '')
        if len(content) > cap:
            content = content[:cap] + '…'
        lines.append(f'{role}：{content}')
    logger.info('rendered history:%s', lines)
    return '\n'.join(lines)


def _load_summary(session: Session, sid: int) -> dict | None:
    """读取会话滚动摘要 JSON：{'upto': 已摘要到消息 id, 'text': 摘要文本}。"""
    chat = session.get(ChatSession, sid)
    if chat is None or not chat.summary:
        return None
    try:
        data = json.loads(chat.summary)
        return data if isinstance(data, dict) else None
    except (ValueError, TypeError):
        return None


def _summary_section(summary: str | None) -> str:
    """「更早对话摘要」注入段：有则返回，无则空串。"""
    if not summary:
        return ''
    return '【更早对话摘要（较早轮次已压缩，仅供参考；细节以最近对话原文为准）】\n' + summary


def _build_summary_prompt(current: str | None, pending_text: str) -> str:
    """滚动摘要 prompt：把新滚出窗口的对话并入既有摘要，只输出新摘要文本。"""
    lines = [
        '对话摘要任务：将“待压缩对话”并入既有摘要，生成一份更新后的对话摘要。',
        '要求：以“用户问过的问题”为主线，保留问题与关键结论；不要编造细节；只输出摘要正文。',
    ]
    if current:
        lines.append(f'【既有摘要】\n{current}')
    lines.extend(
        [
            f'【待压缩对话】\n{pending_text}',
            '更新后的摘要：',
        ]
    )
    return '\n\n'.join(lines)


def _build_gate_prompt(
    question: str,
    history: list[dict] | None,
    summary: str | None = None,
) -> str:
    """检索门 prompt：只让模型输出一个意图词，决定本轮是否检索书籍。"""
    parts = [
        '意图分类任务：判断下面这条用户提问是否需要检索书籍内容，只输出一个词：',
        '- history：用户在查询“我问过/说过什么”这类**会话记录本身**'
        '（如“我上一个问题是什么”“我前面问过哪些问题”），答案就在对话记录里',
        '- chat：问候/寒暄/感谢等不涉及书籍内容的闲聊（如“你好”“hello”“谢谢”）',
        '- book：其他一切，包括承接上文继续追问书籍内容'
        '（如“刚才说的那个计划执行者是谁”“我上一个问题里问的那个人是谁”）——'
        '出现“我上一个问题／上一条／刚才”字样**不等于** history：'
        '问“记录本身”（我上一个问题是什么）才是 history，'
        '问“记录里提到的书内容”（我上一个问题里问的那个人是谁）仍是 book',
        '',
    ]
    if history:
        parts.append(f'【最近对话】\n{_render_history(history)}')
    if summary:
        parts.append(_summary_section(summary))
    parts.append(f'【当前提问】\n{question}')
    parts.append('只输出 book / history / chat 中的一个词：')
    return '\n'.join(parts)


def _build_context_prompt(
    question: str,
    history: list[dict] | None,
    summary: str | None = None,
) -> str:
    """history/chat 分支作答 prompt：不检索书籍，仅按对话记录作答。"""
    return (
        '你是阅读助手。当前用户提问属于对本次对话本身的询问或日常寒暄，'
        '不需要检索书籍内容。\n'
        '回答要求：\n'
        '1. 若能从下方对话记录找到答案（如询问之前问过的问题或说过的话），'
        '必须按记录原文作答，不得编造记录中不存在的内容。\n'
        '2. 若对话记录中没有相关内容，如实说明。\n'
        '3. 若是寒暄（你好/谢谢/你是谁等），礼貌简短回应即可。\n\n'
        + (_summary_section(summary) + '\n\n' if summary else '')
        + '【对话记录】\n'
        f'{_render_history(history)}\n\n'
        '【当前提问】\n'
        f'{question}\n\n'
        '可用 final_answer 工具提交最终回答。'
    )


@tool
def final_answer(answer: str) -> str:
    """检索到的原文片段足以回答用户问题：给出正式回答。

    Args:
        answer: 对用户问题的完整回答
    """
    return answer


@tool
def request_clarification(missing_info: str) -> str:
    """检索到的原文片段不足以回答用户问题：请求澄清。

    Args:
        missing_info: 为回答问题，还需要补充哪方面的书籍内容
    """
    return missing_info


def _retrieval_was_degraded(session, document_id: int | None) -> bool:
    """本次检索是否在退化环境下进行（目标文档尚未入库完成）。

    与检索层共用同一份白名单 ``list_indexed_document_ids()``，保证「检索拿不到片段」
    与「判定这次回答不可信」用的是同一个信号，不会各自漂移。
    """
    indexed = list_indexed_document_ids(session)
    if document_id is not None:
        return document_id not in indexed
    # 全库问答：一个可检索文档都没有，同样属于环境性失败
    return not indexed


def _is_stale_refusal(entry, state) -> bool:
    """缓存里是一条「信息不足」的判定，但本轮带了澄清补充 → 该判定已不适用。

    为什么要这一条：精确通道撞不上车（``question_hash`` 把澄清文本算进去了），
    但**语义通道会** —— 「Q」与「Q\n补充说明：…」的向量相似度通常 ≥
    ``cache_similarity_threshold``，于是用户补充信息后重问，会命中旧的拒答行、
    再次转 HITL，把用户刚提供的信息原样丢掉。

    注意旧实现（按 ``entry.answer[:80]`` 匹配词表）**结构上就够不到这里**：
    拒答缓存行的 ``answer`` 是 None，它连看都不会看一眼。

    这里刻意**不**用 ``cached_chunk_count`` 判断陈旧：对加列前的历史行它是
    DEFAULT 0（含义是「未知」，不是「当时没检索到」），而按 0 清理会误删
    「拿 A 书的问题问 B 书」这类正当拒答。防新增靠 create_hitl 的写入门禁，
    读时只处理「判定对本轮已不适用」这一种确定情形。
    """
    if not (getattr(entry, 'needs_clarification', False) and entry.answer is None):
        return False
    return bool(state.get('clarification'))


def _is_cache_expired(entry, ttl_days: int) -> bool:
    """TTL判断：超过cache_ttl_days天视为过期
    兼容 SQLite（naive）与 Postgres（aware）的时区差异。"""
    if ttl_days <= 0:
        return False
    created = entry.created_at
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc) - created > timedelta(days=ttl_days)


def _corpus_fingerprint(session: Session) -> str:
    """全库问答的「语料版本」指纹：**已入库**文档 (id, content_hash) 集合的哈希。

    为什么需要它：全库路径原先写死空串，于是这些缓存条目**没有语料版本维度** ——
    上传 / 删除 / 重索引任何一本书之后，旧答案照样命中。最坏的形态是
    **全库拒答被长期缓存**：某本书还没入库时问它 → 「原文没提到」写进缓存 →
    书入库完成后重问，仍然答「没提到」（缓存 TTL 30 天）。
    全库问答是主力用法，而语料变化在本项目里很频繁（重复上传、重索引都会改状态）。

    只统计 ``index_status == 'indexed'``：检索层同样只认这批文档
    （``list_indexed_document_ids``），所以「库里有没有这本书」与「这次回答可不可信」
    用的是同一个信号，不会各自漂移。排序保证指纹稳定可复现。
    """
    rows = session.execute(
        select(Document.id, Document.content_hash)
        .where(Document.index_status == 'indexed')
        .order_by(Document.id)
    ).all()
    return sha256_hex('|'.join(f'{doc_id}:{content_hash}' for doc_id, content_hash in rows))


def _document_content_hash(session: Session, document_id: int | None) -> str:
    """取本次问答的「文档版本」维度：单书=该文档 content_hash；全库=语料指纹。

    两者都参与缓存键（见 ``QaCacheEntry`` 的复合唯一索引），语义统一为
    「这条答案是从哪一版原文推出来的」。
    """
    if document_id is None:
        return _corpus_fingerprint(session)
    document = get_document(session, document_id)
    return document.content_hash if document else ''


def _full_question_text(question: str, clarification: str | None) -> str:
    """租种与retrieve节点一致的完整问题文本（含补充说明）"""
    if clarification:
        return f'{question}\n补充说明：{clarification}'
    return question


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """余弦相似度；向量缺失/长度不一致/零向量时返回 0。

    返回值收敛为内建 ``float``，理由同 retriever._cosine_similarity：
    该值会写进 QA state 的 ``cache_similarity``，numpy 标量会让 checkpoint 序列化失败。
    """
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


_CITATION_MARK = re.compile(r'\[(\d{1,2})\]')


def _used_citation_indexes(content: str | None, total: int) -> list[int]:
    """从回答文本解析 [n] 标记，返回按出现顺序去重的有效编号（1..total）。

    回答里没标引用时返回空列表，调用方据此决定回退策略。
    """
    if not content or total <= 0:
        return []
    used: list[int] = []
    for match in _CITATION_MARK.finditer(content):
        num = int(match.group(1))
        if 1 <= num <= total and num not in used:
            used.append(num)
    return used


def _renumber_citations(content: str | None, citations: list[dict]) -> tuple[str, list[dict]]:
    """把引用编号重排为连续的 1..n，并同步改写回答里的 [n] 标记。

    只重写出现在引用列表中的编号，其余标记原样保留；一次正则替换完成，
    避免 [1]→[2]、[2]→[1] 这类互换时的串联污染。
    """
    if not content or not citations:
        return content, citations
    mapping: dict[int, int] = {}
    for new_index, citation in enumerate(citations, start=1):
        old_index = citation.get('index')
        if isinstance(old_index, int):
            mapping[old_index] = new_index
    if all(old == new for old, new in mapping.items()):
        return content, citations  # 已是连续编号，无需改动

    new_content = _CITATION_MARK.sub(
        lambda m: f'[{mapping[int(m.group(1))]}]' if int(m.group(1)) in mapping else m.group(0),
        content,
    )
    new_citations = [{**c, 'index': mapping.get(c.get('index'), c.get('index'))} for c in citations]
    return new_content, new_citations


def build_qa_graph(
    session_factory: sessionmaker[Session],
    vector_store: VectorStore,
    llm=None,
    embedding_model: Embeddings | None = None,
    checkpointer=None,
):
    """构建问答图。

    ``llm`` 与 ``embedding_model``
    """
    chat_model = llm or get_chat_model()
    embedding = embedding_model or get_embedding_model()
    retriever = create_retriever(vector_store, embedding)
    # 意图识别 L2 层：原型向量在图构建时算一次（图/app 是单例，不会每请求重算）
    _embed_documents = getattr(embedding, 'embed_documents', None)
    prototype_router = build_prototype_router(_embed_documents) if _embed_documents else None

    def cache_check(state: QAState) -> dict:
        if len(state.get('document_ids') or []) > 1:
            # 多文档问答缓存键需绑定全部文档版本
            return {'cache_hit': False, 'question_hash': None, 'cache_channel': 'disabled'}
        settings = get_settings()
        if not settings.cache_enabled:
            return {'cache_hit': False, 'question_hash': None, 'cache_channel': 'disabled'}
        question_hash = sha256_hex(
            normalize_question(state['question'])
            + '|'
            + normalize_question(state.get('clarification') or '')
            + '|'
            + str(state.get('document_id') or '')
        )
        with session_scope(session_factory) as session:
            doc_id = state.get('document_id')
            content_hash = _document_content_hash(session, doc_id)
            entry = get_qa_cache_entry(session, question_hash, content_hash, doc_id)
            if entry is None:
                # 精确未命中 -> 语义层：同文档内余弦相似度匹配
                if settings.cache_similarity_threshold > 0:
                    # 复用 gate(L2 原型层) 已经算过的问题向量，省一次 embedding。
                    # ⚠️ 带 clarification 时不能复用：cache_check 要嵌的是
                    # question + 补充说明，与 L2 嵌的裸 question 不是一个向量。
                    embedding = (
                        state.get('intent_embedding') if not state.get('clarification') else None
                    )
                    if embedding is None:
                        embedding = retriever.embed(
                            _full_question_text(state['question'], state.get('clarification')),
                        )
                    best_entry, best_score = None, 0.0
                    id_entry, id_score = None, 0.0
                    for cand in list_qa_cache_entries(session, content_hash, doc_id):
                        score = _cosine_similarity(embedding, cand.question_embedding or [])
                        if score > best_score:
                            best_score, best_entry = score, cand
                        # 标识符变体复用：同一文档内、问题共享产品编号 token（P-002 vs P002）
                        # 且余弦较高时，视为近似复问——仅复用带答案与引用的强缓存
                        if (
                            score >= 0.90
                            and id_score < score
                            and cand.answer
                            and cand.citations
                            and _identifier_tokens(cand.question_raw or '')
                            & _identifier_tokens(state['question'])
                        ):
                            id_score, id_entry = score, cand
                    satisfied = False
                    channel = 'semantic'
                    if best_entry is not None and best_score >= settings.cache_similarity_threshold:
                        satisfied = True
                    elif id_entry is not None and id_score >= 0.90:
                        best_entry, best_score, satisfied = id_entry, id_score, True
                        channel = 'identifier'
                    if satisfied:
                        if _is_cache_expired(best_entry, settings.cache_ttl_days):
                            session.delete(best_entry)
                            logger.info(
                                '问答[缓存] 语义候选已过期丢弃 channel=%s q=%.30s',
                                channel,
                                state['question'],
                            )
                            return {
                                'cache_hit': False,
                                'question_hash': question_hash,
                                'question_embedding': embedding,
                                'cache_channel': 'miss',
                            }
                        if _is_stale_refusal(best_entry, state):
                            # 环境性失败期间写入的「答不了」，环境已恢复 → 删除并重新回答
                            logger.warning(
                                '问答[缓存] 丢弃陈旧拒答缓存 id=%s channel=%s q=%.30s',
                                best_entry.id,
                                channel,
                                state['question'],
                            )
                            session.delete(best_entry)
                            return {
                                'cache_hit': False,
                                'question_hash': question_hash,
                                'question_embedding': embedding,
                                'cache_channel': 'miss',
                                'cache_similarity': best_score,
                                'cache_invalidated': True,
                            }
                        touch_qa_cache_hit(session, best_entry)
                        # 必须在commit之前取值（DetachedInstanceError）
                        answer = best_entry.answer
                        citations = list(best_entry.citations or [])
                        needs_clarification = best_entry.needs_clarification
                        logger.info(
                            '问答[缓存] %s命中 score=%.4f q=%.30s',
                            channel,
                            best_score,
                            state['question'],
                        )
                        return {
                            'cache_hit': True,
                            'question_hash': question_hash,
                            'question_embedding': embedding,
                            'answer': answer,
                            'citations': citations,
                            'needs_clarification': needs_clarification,
                            'cache_channel': channel,
                            'cache_similarity': best_score,
                        }
                    return {
                        'cache_hit': False,
                        'question_hash': question_hash,
                        'question_embedding': embedding,
                        'cache_channel': 'miss',
                        'cache_similarity': best_score,
                    }
                return {'cache_hit': False, 'question_hash': question_hash, 'cache_channel': 'miss'}
            if _is_cache_expired(entry, settings.cache_ttl_days):
                session.delete(entry)
                logger.info('问答[缓存] 精确条目已过期丢弃 q=%.30s', state['question'])
                return {'cache_hit': False, 'question_hash': question_hash, 'cache_channel': 'miss'}
            if _is_stale_refusal(entry, state):
                # 环境性失败期间写入的「答不了」，环境已恢复 → 删除并重新回答
                logger.warning(
                    '问答[缓存] 丢弃陈旧拒答缓存 id=%s channel=%s q=%.30s',
                    entry.id,
                    state['question'],
                )
                session.delete(entry)
                return {
                    'cache_hit': False,
                    'question_hash': question_hash,
                    'cache_channel': 'miss',
                    'cache_invalidated': True,
                }
            touch_qa_cache_hit(session, entry)
            answer = entry.answer
            citations = list(entry.citations or [])
            needs_clarification = entry.needs_clarification

        logger.info('问答[缓存] 精确命中 q=%.30s', state['question'])
        return {
            'cache_hit': True,
            'question_hash': question_hash,
            'answer': answer,
            'citations': citations,
            'needs_clarification': needs_clarification,
            'cache_channel': 'exact',
        }

    def retrieve(state: QAState) -> dict:
        question = state['question']
        if state.get('clarification'):
            question = f'{question}\n补充说明：{state["clarification"]}'
        doc_ids = state.get('document_ids') or []
        # 检索白名单：每请求查一次 PG，只保留 index_status == 'indexed' 的文档命中，
        # 避免「入库中断/失败的文档」用其残缺向量冒充来源。
        # ⚠️ 依赖顺序：启用前必须先跑 scripts/reconcile_index_status.py 清洗历史假阳性
        #    （历史库里 3/6 文档卡在 indexing，实际向量完整），否则会误杀这些正常文档。
        with session_scope(session_factory) as session:
            indexed_ids = list_indexed_document_ids(session)
        if len(doc_ids) > 1:
            # 多文档问答：并行 fan-out，每书独立检索后按相似度合流取全局 top_k
            from concurrent.futures import ThreadPoolExecutor

            settings = get_settings()
            total_k = settings.top_k
            per_doc = max(2, -(-total_k // len(doc_ids)))  # 按文档数 ceil 分配

            def _retrieve_one(doc_id: int):
                if doc_id not in indexed_ids:
                    return []
                return retriever.retrieve(question, top_k=per_doc, document_id=doc_id)

            with ThreadPoolExecutor(max_workers=min(len(doc_ids), 8)) as executor:
                grouped = list(executor.map(_retrieve_one, doc_ids))
            merged = sorted(
                (hit for group in grouped for hit in group if hit.document_id in indexed_ids),
                key=lambda hit: hit.score,
                reverse=True,
            )[:total_k]
            titles: dict[int, str] = {}
            with session_scope(session_factory) as session:
                for document in list_documents(session):
                    if document.id in doc_ids:
                        titles[document.id] = document.filename
            logger.info(
                '问答[检索] q=%.30s docs=%s indexed=%s merged=%d',
                question,
                doc_ids,
                sorted(indexed_ids),
                len(merged),
            )
            return {'chunks': [_chunk_to_dict(hit) for hit in merged], 'doc_titles': titles}
        hits = retriever.retrieve(question, document_id=state.get('document_id'))
        hits = [hit for hit in hits if hit.document_id in indexed_ids]
        titles: dict[int, str] = {}
        doc_id = state.get('document_id')
        if doc_id is not None:
            with session_scope(session_factory) as session:
                document = get_document(session, doc_id)
                if document is not None:
                    titles[document.id] = document.filename
        logger.info(
            '问答[检索] q=%.30s doc=%s hit=%d', question, state.get('document_id'), len(hits)
        )
        return {'chunks': [_chunk_to_dict(hit) for hit in hits], 'doc_titles': titles or None}

    def judge(state: QAState) -> dict:
        # 有澄清说明时视为信息已补充；否则无检索结果即为信息不足
        needs = not state.get('chunks') and not state.get('clarification')
        return {'needs_clarification': needs}

    def create_hitl(state: QAState, config: RunnableConfig) -> dict:
        """信息不足 → 建任务 → **真挂起**（interrupt），等澄清后从断点继续。

        与「重跑」的本质区别：挂起时状态留在 checkpoint 里，恢复时从本节点继续，
        ``retrieve`` 用的是**本次会话真实的** document_ids / 历史 / 已算好的向量，
        不需要客户端重发问题、也不会把 gate/context_load/summarize 重付一遍。

        ⚠️ ``interrupt()`` 恢复时**本节点从头重跑**（LangGraph 用重放重建入口状态），
        所以 ``interrupt()`` 之前的一切副作用都必须幂等 —— 见
        ``get_or_create_hitl_task``（按 thread_id 取或建）。
        """
        thread_id = (config or {}).get('configurable', {}).get('thread_id')
        logger.info(
            '问答[HITL] 信息不足，进入澄清流程 q=%.30s thread=%.24s',
            state['question'],
            thread_id or '(无)',
        )
        with session_scope(session_factory) as session:
            if thread_id:
                task = get_or_create_hitl_task(
                    session, state.get('session_id'), state['question'], thread_id
                )
            else:
                # 没有 thread_id 就没法恢复 → 不挂起，退回旧行为（建任务后结束本轮）
                task = create_hitl_task(
                    session, session_id=state.get('session_id'), question=state['question']
                )
            settings = get_settings()
            # ⚠️ 退化环境下（目标文档没入库完成）的「答不了」不写缓存：
            # 那不是关于书内容的结论，而是关于系统状态的结论，修好索引后必然过期。
            # 2026-09-17 实测：文档被误置为 indexing 期间整本书的检索被打空，
            # 这些假拒答落进 qa_cache 后会永久回放（缓存命中直接转 HITL、跳过检索）。
            doc_id = state.get('document_id')
            degraded = _retrieval_was_degraded(session, doc_id)
            if degraded:
                logger.warning(
                    '问答[HITL] 检索环境退化（doc=%s 未入库完成），本次判定不写缓存', doc_id
                )
            if (
                settings.cache_enabled
                and state.get('question_hash')
                and not state.get('cache_hit')
                and not degraded
            ):
                doc_id = state.get('document_id')
                save_qa_cache_entry(
                    session,
                    question_raw=state['question'],
                    question_normalized=normalize_question(state['question']),
                    question_hash=state['question_hash'],
                    answer=None,
                    needs_clarification=True,
                    document_id=doc_id,
                    content_hash=_document_content_hash(session, doc_id),
                    question_embedding=state.get('question_embedding'),
                    cached_chunk_count=len(state.get('chunks') or []),
                )
                prune_qa_cache(session, settings.cache_max_entries)
            task_id = task.id

        if not thread_id or state.get('clarification'):
            # 无 thread_id（不可恢复）或澄清已用尽（防再次挂起成死循环）→ 正常结束本轮
            return {'hitl_task_id': task_id}
        # 挂起。载荷把 task_id 带出去 —— 节点被中断时它的返回值不会写进 state，
        # 所以 API 层只能从挂起载荷里拿这个指针（或回查 DB）。
        clarification = interrupt({'hitl_task_id': task_id, 'question': state['question']})
        logger.info('问答[HITL] 收到澄清，从断点继续 thread=%.24s', thread_id)
        # 澄清是用户的一次独立发言，记进历史（本段只在恢复时执行一次：
        # 任务要能被 approved 必须先处于 awaiting，重复提交会被拒）
        if state.get('session_id') and clarification:
            with session_scope(session_factory) as session:
                session.add(
                    ChatMessage(
                        session_id=state['session_id'],
                        role='user',
                        content=f'补充说明：{clarification}',
                    )
                )
        return {
            'hitl_task_id': task_id,
            'clarification': clarification,
            'needs_clarification': False,
        }

    def answer(state: QAState) -> dict:
        import time

        prompt = _build_answer_prompt(
            question=state['question'],
            chunks=state.get('chunks') or [],
            clarification=state.get('clarification'),
            doc_titles=state.get('doc_titles'),
            history=state.get('history') or None,
            summary=state.get('summary') or None,
        )
        start = time.perf_counter()
        tool_calls = []
        if hasattr(chat_model, 'bind_tools'):
            try:
                response = chat_model.bind_tools([final_answer, request_clarification]).invoke(
                    prompt
                )
                tool_calls = getattr(response, 'tool_calls', None) or []
            except Exception:
                logger.exception('问答[answer] 工具调用异常，回退纯文本')
                response = chat_model.invoke(prompt)
        else:
            response = chat_model.invoke(prompt)
        cost_ms = (time.perf_counter() - start) * 1000

        if tool_calls:
            # 主判定：LLM 通过工具调用显式表态
            call = tool_calls[0]
            args = call.get('args') or {}
            if call.get('name') == 'request_clarification':
                missing = args.get('missing_info') or ''
                logger.info(
                    '问答[answer] 工具判定信息不足 → HITL missing=%.50s (%.0fms)', missing, cost_ms
                )
                if not state.get('clarification'):
                    # 转入 HITL 澄清，不缓存该回答、不写入聊天记录
                    return {'needs_clarification': True, 'answer': None, 'citations': []}
                # 澄清后模型仍判定不足：不再重复建任务，以说明文本作回答
                content = missing or '澄清后原文信息仍然不足。'
            else:
                content = args.get('answer') or ''
                logger.info('问答[answer] 工具 final_answer len=%d (%.0fms)', len(content), cost_ms)
        else:
            # 兜底 1：模型未走工具，按纯文本处理
            content = response.content if hasattr(response, 'content') else str(response)
            logger.info('问答[answer] 纯文本回答 len=%d (%.0fms)', len(content), cost_ms)
        titles = state.get('doc_titles') or {}
        all_citations = []
        for index, chunk in enumerate(state.get('chunks') or [], start=1):
            all_citations.append(
                {
                    # index 为 prompt 中该片段的编号，用于与答案里的 [n] 标记对应展示
                    'index': index,
                    'chunk_id': chunk['chunk_id'],
                    'chapter': chunk.get('chapter'),
                    'page': None,
                    'excerpt': chunk['text'][:120],
                    'document': titles.get(chunk.get('document_id')),
                }
            )
        # 只展示答案实际引用过的片段；回答未标注引用时保留全部，避免丢失溯源线索
        used_indexes = _used_citation_indexes(content, len(all_citations))
        if used_indexes:
            used_set = set(used_indexes)
            citations = [c for c in all_citations if c['index'] in used_set]
        else:
            citations = all_citations
        # 编号重排为连续的 1..n，回答里的 [n] 标记同步改写：
        # 避免"只引用 1 条却显示 [3]"这类让用户以为丢了引用的显示
        content, citations = _renumber_citations(content, citations)
        settings = get_settings()
        if settings.cache_enabled and state.get('question_hash') and not state.get('cache_hit'):
            with session_scope(session_factory) as session:
                doc_id = state.get('document_id')
                save_qa_cache_entry(
                    session,
                    question_raw=state['question'],
                    question_normalized=normalize_question(state['question']),
                    question_hash=state['question_hash'],
                    answer=content,
                    citations=citations,
                    document_id=doc_id,
                    content_hash=_document_content_hash(session, doc_id),
                    question_embedding=state.get('question_embedding'),
                    cached_chunk_count=len(state.get('chunks') or []),
                )
                prune_qa_cache(session, settings.cache_max_entries)
        return {'answer': content, 'citations': citations}

    def record_question(state: QAState) -> dict:
        """轮次开头即记下用户的问题（**不能**等到 record 才写）。

        为什么必须提前：真挂起后 ``record`` 在本轮不会执行（图停在 create_hitl），
        问题就会**丢失** —— 用户放弃澄清时历史里查无此问，下一轮问「刚才我问了什么」
        会得到错误答案。今天之所以没这个问题，是因为旧流程「挂起」其实跑到了 END。
        """
        if not state.get('session_id'):
            return {}
        with session_scope(session_factory) as session:
            session.add(
                ChatMessage(
                    session_id=state['session_id'],
                    role='user',
                    content=state['question'],
                )
            )
        return {}

    def record(state: QAState) -> dict:
        if not state.get('session_id'):
            return {}
        with session_scope(session_factory) as session:
            if state.get('answer'):
                session.add(
                    ChatMessage(
                        session_id=state['session_id'],
                        role='assistant',
                        content=state['answer'],
                        meta={
                            'citations': state.get('citations') or [],
                            'cache_hit': bool(state.get('cache_hit')),
                            'cache_channel': state.get('cache_channel'),
                        },
                    )
                )
        return {}

    # 构图期确定：模型是否支持工具调用（测试里的 fake LLM 多数只有 invoke）
    agent_tool_support = hasattr(chat_model, 'bind_tools')
    state_scope: dict = {}

    def _agent_search(query: str) -> list:
        """agent 的语义检索通道：复用既有 retriever 与文档范围白名单。

        与 ``retrieve`` 同源（hybrid 检索、同一 min_score、同一 indexed 白名单），
        保证 agent 拿到的片段与主链路质量一致 —— agent 只是多给了「换策略」的能力，
        不引入第二套检索语义。
        """
        with session_scope(session_factory) as session:
            indexed_ids = set(list_indexed_document_ids(session))
        doc_ids = state_scope.get('document_ids') or []
        if len(doc_ids) > 1:
            merged: list = []
            for doc_id in doc_ids:
                if doc_id in indexed_ids:
                    merged.extend(retriever.retrieve(query, document_id=doc_id))
            merged.sort(key=lambda hit: hit.score, reverse=True)
            return merged[: get_settings().top_k]
        single = state_scope.get('document_id')
        hits = retriever.retrieve(query, document_id=single)
        return [hit for hit in hits if hit.document_id in indexed_ids]

    def agent(state: QAState, config: RunnableConfig) -> dict:
        """有界 ReAct 检索：模型自己选工具（search/grep/list_chapters/read_chapter/finish）。

        位置：它是**「一次检索」与「问人」之间的那一级台阶**。
        - 由 ``judge`` 进来 = 一次检索什么都没拿到；
        - 由 ``answer`` 进来 = 拿到了但模型判定不足以回答；
        - 走完仍不够 → 才落到 ``create_hitl``（今天的行为）。

        ⚠️ 每题至多一次（``agent_used`` 闸）：agent 内部已有 ``max_steps`` 预算，
        但若允许它被反复进入，单题成本会乘以进入次数 —— 而收益递减。
        一次打不开就交给人，比无限加码检索更像生产系统该有的样子。

        ⚠️ **重放安全性**：本节点全部工具只读（检索/字面扫描/取章），
        所以 ``interrupt()`` 恢复时即便被整体重跑，也只是重算一遍、无副作用。
        这是把循环放在节点内部（而非图上新增环）的前提 —— 图上的环会让恢复变成整圈重放。
        """
        nonlocal state_scope
        state_scope = {
            'document_id': state.get('document_id'),
            'document_ids': state.get('document_ids') or [],
        }
        settings = get_settings()
        doc_ids = state.get('document_ids') or []
        titles: dict[int, str] = dict(state.get('doc_titles') or {})
        if not titles:
            with session_scope(session_factory) as session:
                wanted = set(doc_ids) or {d.id for d in list_documents(session)}
                for document in list_documents(session):
                    if document.id in wanted:
                        titles[document.id] = document.filename

        toolbox = ToolBox(
            vector_store=vector_store,
            search_fn=_agent_search,
            document_ids=doc_ids or None,
            doc_titles=titles,
        )
        budget = AgentBudget(max_steps=settings.agent_max_steps, max_chunks=settings.top_k)
        start = time.perf_counter()
        run = run_agent_loop(
            model=chat_model,
            toolbox=toolbox,
            question=state['question'],
            history=state.get('history'),
            budget=budget,
            seed_chunks=state.get('chunks') or None,
        )
        cost_ms = (time.perf_counter() - start) * 1000
        logger.info(
            '问答[agent] q=%.30s steps=%d stopped=%s finished=%s chunks=%d (%.0fms)',
            state['question'], run.steps, run.stopped_by, run.finished,
            len(run.chunks), cost_ms,
        )

        # 合并策略：agent 片段优先，但保证原有片段不被清空（见 merge_agent_chunks 的说明）
        merged = merge_agent_chunks(
            run.chunks, list(state.get('chunks') or []), budget.max_chunks
        )
        kept_original = sum(
            1 for c in merged
            if str(c.get('chunk_id') or '') in
            {str(o.get('chunk_id') or '') for o in (state.get('chunks') or [])}
        )
        logger.info(
            '问答[agent] 合并后 %d 条（其中原有 %d 条）—— 保证原有片段不被清空',
            len(merged), kept_original,
        )
        return {
            'chunks': merged,
            'doc_titles': titles or None,
            'agent_used': True,
            'agent_trace': list(run.trace),
            'agent_steps': run.steps,
            'needs_clarification': False,  # 交给 answer 重新判定
        }

    def route_after_cache(state: QAState) -> str:
        if not state.get('cache_hit'):
            return 'retrieve'
        return 'create_hitl' if state.get('needs_clarification') else 'record'

    def _agent_available(state: QAState) -> bool:
        """能否进 agent：开启 + 模型支持工具 + 未用过（每题至多一次）。

        三者缺一不可：
        - ``agent_enabled`` 是配置开关；
        - ``agent_tool_support`` 在**构图期**确定（模型有没有 ``bind_tools``）——
          没有工具能力的模型进 agent 只会白跑一轮且改变原有分流，必须在入口挡住；
        - ``agent_used`` 保证每题至多一次：agent 内部已有 max_steps 预算，
          但允许反复进入会让单题成本乘以进入次数，而收益递减。
        """
        if not get_settings().agent_enabled or not agent_tool_support:
            return False
        return not state.get('agent_used')

    def route_after_judge(state: QAState) -> str:
        if not state.get('needs_clarification'):
            return 'answer'
        # 一次检索什么都没拿到 —— 先让 agent 换个策略自己再试
        return 'agent' if _agent_available(state) else 'create_hitl'

    def route_after_answer(state: QAState) -> str:
        if not state.get('needs_clarification'):
            return 'record'
        # 模型判定「现有片段不足以回答」—— 交给 agent 补检，仍不够才问人
        return 'agent' if _agent_available(state) else 'create_hitl'

    def route_after_create_hitl(state: QAState) -> str:
        """恢复后带澄清说明回到检索重跑；未挂起（不可恢复 / 澄清已用尽）则收尾。

        ⚠️ 恢复路径**刻意跳过 cache_check**：本轮是同一个 thread 的续跑，
        而缓存里那条正是上次「答不了」的判定，再去查一遍只会把它捞回来。
        """
        return 'retrieve' if state.get('clarification') else 'record'

    def context_load(state: QAState) -> dict:
        """读回最近对话 + 长会话滚动摘要任务（时间正序，不含本轮）。"""
        sid = state.get('session_id')
        if not sid:
            # 如果非本轮会话，则无上下文
            return {'history': [], 'summary': None, 'summary_pending': None}
        with session_scope(session_factory) as session:
            # 查询本session最近HISTORY_TURNS轮对话内容，order by 会话内容id 倒排；最新的对话在前面
            rows = list(
                session.scalars(
                    select(ChatMessage)
                    .where(ChatMessage.session_id == sid)
                    .order_by(ChatMessage.id.desc())
                    .limit(HISTORY_TURNS)
                )
            )
            # 倒序一下，按时间顺序排
            rows.reverse()  # 时间正序（最早在前）
            # 计算本轮对话一共多少条对话内容
            total = (
                session.scalar(
                    select(func.count(ChatMessage.id)).where(ChatMessage.session_id == sid)
                )
                or 0
            )
            truncated = total > len(rows)  # 判断是否被截断，即对话内容超出HISTORY_TURNS
            summary_text = None
            pending = None
            if truncated:
                anchor = rows[0].id  # 锚点，窗口内最早一条消息 id
                logger.info('anchor:%d', anchor)
                stored = _load_summary(session, sid)  # 获取本轮对话的总结摘要
                if stored:
                    summary_text = stored.get('text')
                upto = int((stored or {}).get('upto') or 0)
                logger.info('upto:%d', upto)

                if anchor - 1 > upto:
                    # 增量：仅摘要 (upto, anchor-1] 区间内滚出窗口的消息
                    old_rows = list(
                        session.scalars(
                            select(ChatMessage)
                            .where(
                                ChatMessage.session_id == sid,
                                ChatMessage.id > upto,
                                ChatMessage.id <= anchor - 1,
                            )
                            .order_by(ChatMessage.id)
                        )
                    )
                    pending = {
                        'upto': upto,
                        'anchor': anchor,
                        'text': _render_history(
                            [{'role': m.role, 'content': m.content} for m in old_rows], cap=160
                        ),
                    }
        return {
            'history': [{'role': m.role, 'content': m.content} for m in rows],
            'summary': summary_text,
            'summary_pending': pending,
        }

    def summarize(state: QAState) -> dict:
        """长会话增量摘要：把新滚出窗口的消息并入既有摘要并落库。"""
        pending = state.get('summary_pending')
        if not pending:
            return {'summary': state.get('summary')}
        current = state.get('summary') or ''
        prompt = _build_summary_prompt(current, pending.get('text') or '')
        text = current
        start = time.perf_counter()
        try:
            resp = chat_model.invoke(prompt)
            candidate = (getattr(resp, 'content', None) or str(resp)).strip()
            if candidate:
                text = candidate
        except Exception:
            logger.exception('问答[摘要] 生成异常，沿用旧摘要')
        cost_ms = (time.perf_counter() - start) * 1000
        with session_scope(session_factory) as session:
            chat = session.get(ChatSession, state.get('session_id'))
            if chat is not None:
                chat.summary = json.dumps(
                    {'upto': pending['anchor'] - 1, 'text': text}, ensure_ascii=False
                )
        logger.info(
            '问答[摘要] 增量 upto=%d chars=%d (%.0fms)', pending['anchor'] - 1, len(text), cost_ms
        )
        return {'summary': text}

    def gate(state: QAState) -> dict:
        """检索门：只要在会话里就分类本轮意图(含第一问)；无会话(如 MCP 工具调用)直接 book。

        history=询问会话历史本身；chat=寒暄闲聊；book=书内容(默认)。
        **三级级联**(见 graph/intent.py)：规则 → 向量原型 → LLM；前两层零 LLM 调用，
        只有它们都拿不准时才付一次 LLM。意图判错的代价不对称（判成 chat/history 会
        直接吞掉用户的问题，判成 book 只是多检索一次），所以：
        - 拿不准一律回落 book——宁可进检索，也不吞问题；
        - L2 原型层只敢自动判 book，chat/history 只由高精度规则层裁。
        注:第一问历史为空,同样要过门——否则首轮"你好"会掉进书问答链路被判信息不足。
        """
        if not state.get('session_id'):
            return {'intent': 'book', 'intent_channel': 'llm'}
        question = state['question']
        settings = get_settings()

        # ---- L1/L2 快通道：命中即出意图，零 LLM 调用 ----
        decision = route_fast(
            question,
            prototype_router,
            lambda: retriever.embed(question),
            rules_enabled=settings.intent_rules_enabled,
        )
        if decision.intent:
            logger.info(
                '问答[gate] q=%.30s intent=%s channel=%s rule=%s score=%s (0 次 LLM)',
                question,
                decision.intent,
                decision.channel,
                decision.rule_label or '-',
                f'{decision.score:.4f}' if decision.score is not None else '-',
            )
            return {
                'intent': decision.intent,
                'intent_channel': decision.channel,
                'intent_score': decision.score,
                'intent_embedding': decision.vector,
            }

        # ---- L3 LLM 兜底：只处理前两层拿不准的余量 ----
        raw = None
        prompt_text = _build_gate_prompt(question, state.get('history'), state.get('summary'))
        start_ts = time.perf_counter()
        try:
            resp = chat_model.invoke(prompt_text)
            raw = (getattr(resp, 'content', None) or str(resp)).strip().lower()
        except Exception:
            logger.exception('问答[gate] 意图分类异常，回落 book')
        intent = 'book'
        if raw:
            for cand in ('history', 'chat', 'book'):
                if cand in raw:
                    intent = cand
                    break
        logger.info(
            '问答[gate] q=%.30s intent=%s raw=%s channel=llm in_chars=%d (%.0fms)',
            question,
            intent,
            raw,
            len(prompt_text),
            (time.perf_counter() - start_ts) * 1000,
        )
        return {
            'intent': intent,
            'intent_channel': 'llm',
            'intent_score': decision.score,
            'intent_embedding': decision.vector,
        }

    def context_answer(state: QAState) -> dict:
        """history/chat 分支：不检索书籍，仅凭对话记录/寒暄作答；结果写回记录。"""
        question = state['question']
        prompt = _build_context_prompt(question, state.get('history') or [], state.get('summary'))
        content = ''
        response = None
        try:
            if hasattr(chat_model, 'bind_tools'):
                response = chat_model.bind_tools([final_answer]).invoke(prompt)
                tool_calls = getattr(response, 'tool_calls', None) or []
                if tool_calls:
                    call = tool_calls[0]
                    if call.get('name') == 'final_answer':
                        content = (call.get('args') or {}).get('answer') or ''
            else:
                response = chat_model.invoke(prompt)
        except Exception:
            logger.exception('问答[context_answer] 模型调用异常')
        if not content and response is not None:
            content = getattr(response, 'content', None) or ''
            if not isinstance(content, str):
                content = str(content)
        if not content:
            content = '抱歉，我暂时没理解你的意思，换个说法试试？'
        logger.info(
            '问答[context_answer] intent=%s q=%.30s len=%d',
            state.get('intent'),
            question,
            len(content),
        )
        return {'answer': content, 'citations': []}

    def route_after_agent(state: QAState) -> str:
        """agent 交回答案节点前先看是否真有收获。

        若 agent 一圈下来**一个片段都没有**（例如书里确实没有），回到 answer 只是
        用同样的空上下文再判一次「信息不足」，纯属浪费一次 LLM → 直接转 HITL。
        """
        return 'answer' if state.get('chunks') else 'create_hitl'

    def route_after_gate(state: QAState) -> str:
        intent = state.get('intent')
        return intent if intent in ('book', 'history', 'chat') else 'book'

    graph = StateGraph(QAState)
    graph.add_node('cache_check', cache_check)
    graph.add_node('retrieve', retrieve)
    graph.add_node('judge', judge)
    graph.add_node('create_hitl', create_hitl)
    graph.add_node('answer', answer)
    graph.add_node('agent', agent)
    graph.add_node('record', record)
    graph.add_node('record_question', record_question)
    graph.add_node('context_load', context_load)
    graph.add_node('summarize', summarize)
    graph.add_node('gate', gate)
    graph.add_node('context_answer', context_answer)

    graph.add_edge(START, 'context_load')
    graph.add_edge('context_load', 'record_question')
    graph.add_edge('record_question', 'summarize')
    graph.add_edge('summarize', 'gate')
    graph.add_conditional_edges(
        'gate',
        route_after_gate,
        {'book': 'cache_check', 'history': 'context_answer', 'chat': 'context_answer'},
    )
    graph.add_edge('context_answer', 'record')
    graph.add_conditional_edges(
        'cache_check',
        route_after_cache,
        {'retrieve': 'retrieve', 'create_hitl': 'create_hitl', 'record': 'record'},
    )
    graph.add_edge('retrieve', 'judge')
    # agent 补检完回到 answer 重新判定；必要时由 answer 的 route 再决定是否问人
    graph.add_conditional_edges(
        'agent',
        route_after_agent,
        {'answer': 'answer', 'create_hitl': 'create_hitl'},
    )
    graph.add_conditional_edges(
        'judge',
        route_after_judge,
        {'answer': 'answer', 'agent': 'agent', 'create_hitl': 'create_hitl'},
    )
    graph.add_conditional_edges(
        'answer',
        route_after_answer,
        {'agent': 'agent', 'create_hitl': 'create_hitl', 'record': 'record'},
    )
    graph.add_conditional_edges(
        'create_hitl',
        route_after_create_hitl,
        {'retrieve': 'retrieve', 'record': 'record'},
    )
    graph.add_edge('record', END)
    return graph.compile(checkpointer=checkpointer or InMemorySaver())


if __name__ == '__main__':
    print(build_qa_graph().get_graph().draw_mermaid())
