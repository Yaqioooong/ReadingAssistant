"""问答流水线：检索 → 充足性判断 → 回答 / HITL 澄清 → 记录。"""

import math
from datetime import datetime, timedelta, timezone
from typing import TypedDict

from langchain_core.embeddings import Embeddings
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from nbconvert.filters import citation
from sqlalchemy.orm import Session, sessionmaker

from reading_assistant.config import get_settings
from reading_assistant.model.factory import get_chat_model, get_embedding_model
from reading_assistant.rag import Retriever
from reading_assistant.storage import (
    ChatMessage,
    create_hitl_task,
    get_document,
    get_qa_cache_entry,
    list_qa_cache_entries,
    normalize_question,
    prune_qa_cache,
    save_qa_cache_entry,
    session_scope,
    sha256_hex,
    touch_qa_cache_hit,
)
from reading_assistant.storage.vector_store import VectorStore


class QAState(TypedDict, total=False):
    """问答图的共享状态。"""

    question: str
    session_id: int | None
    document_id: int | None
    clarification: str | None
    chunks: list[dict]
    needs_clarification: bool
    hitl_task_id: int | None
    answer: str | None
    citations: list[dict]
    cache_hit: bool
    question_hash: str | None
    question_embedding: list[float] | None


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
    question: str, chunks: list[dict], clarification: str | None = None
) -> str:
    context = (
        '\n\n'.join(
            f'[{index}]（{chunk.get("citation", "")}）\n{chunk["text"]}'
            for index, chunk in enumerate(chunks, start=1)
        )
        or '（未检索到相关原文片段）'
    )
    lines = ['你是阅读助手，请仅根据提供的原文片段回答问题。', f'问题：{question}']
    if clarification:
        lines.append(f'用户的补充说明：{clarification}')
    lines.extend(
        [
            f'原文片段：\n{context}',
            '回答时注明引用片段编号；若原文不足以回答，请直接说明。',
        ]
    )
    return '\n\n'.join(lines)


def _is_cache_expired(entry, ttl_days: int) -> bool:
    """TTL判断：超过cache_ttl_days天视为过期
    兼容 SQLite（naive）与 Postgres（aware）的时区差异。"""
    if ttl_days <= 0:
        return False
    created = entry.created_at
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc) - created > timedelta(days=ttl_days)


def _document_content_hash(session: Session, document_id: int | None) -> str:
    """取文档版本hash, 全库问答(无document_id)时返回空串"""
    if document_id is None:
        return ''
    document = get_document(session, document_id)
    return document.content_hash if document else ''


def _full_question_text(question: str, clarification: str | None) -> str:
    """租种与retrieve节点一致的完整问题文本（含补充说明）"""
    if clarification:
        return f'{question}\n补充说明：{clarification}'
    return question


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """余弦相似度；向量缺失/长度不一致/零向量时返回 0"""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def build_qa_graph(
    session_factory: sessionmaker[Session],
    vector_store: VectorStore,
    llm=None,
    embedding_model: Embeddings | None = None,
    checkpointer=None,
):
    """构建问答图。

    ``llm`` 与 ``embedding_model`` 可注入（测试用 mock）；缺省使用配置的真实模型。
    """
    chat_model = llm or get_chat_model()
    retriever = Retriever(vector_store, embedding_model or get_embedding_model())

    def cache_check(state: QAState) -> dict:
        settings = get_settings()
        if not settings.cache_enabled:
            return {'cache_hit': False, 'question_hash': None}
        question_hash = sha256_hex(
            normalize_question(state['question'])
            + '|'
            + normalize_question(state.get('clarification') or '')
        )
        with session_scope(session_factory) as session:
            doc_id = state.get('document_id')
            content_hash = _document_content_hash(session, doc_id)
            entry = get_qa_cache_entry(session, question_hash, content_hash, doc_id)
            if entry is None:
                # 精确未命中 -> 语义层：同文档内余弦相似度匹配
                if settings.cache_similarity_threshold > 0:
                    embedding = retriever.embed(
                        _full_question_text(state['question'], state.get('clarification')),
                    )
                    best_entry, best_score = None, 0.0
                    for cand in list_qa_cache_entries(session, content_hash, doc_id):
                        score = _cosine_similarity(embedding, cand.question_embedding or [])
                        if score > best_score:
                            best_score, best_entry = score, cand
                    if best_entry is not None and best_score >= settings.cache_similarity_threshold:
                        if _is_cache_expired(best_entry, settings.cache_ttl_days):
                            session.delete(best_entry)
                            return {
                                'cache_hit': False,
                                'question_hash': question_hash,
                                'question_embedding': embedding,
                            }
                        touch_qa_cache_hit(session, best_entry)
                        # 必须在commit之前取值（DetachedInstanceError）
                        answer = best_entry.answer
                        citations = list(best_entry.citations or [])
                        needs_clarification = best_entry.needs_clarification
                        return {
                            'cache_hit': True,
                            'question_hash': question_hash,
                            'question_embedding': embedding,
                            'answer': answer,
                            'citations': citations,
                            'needs_clarification': needs_clarification,
                        }
                    return {
                        'cache_hit': False,
                        'question_hash': question_hash,
                        'question_embedding': embedding,
                    }
                return {'cache_hit': False, 'question_hash': question_hash}
            if _is_cache_expired(entry, settings.cache_ttl_days):
                session.delete(entry)
                return {'cache_hit': False, 'question_hash': question_hash}
            touch_qa_cache_hit(session, entry)
            answer = entry.answer
            citations = list(entry.citations or [])
            needs_clarification = entry.needs_clarification

        return {
            'cache_hit': True,
            'question_hash': question_hash,
            'answer': answer,
            'citations': citations,
            'needs_clarification': needs_clarification,
        }

    def retrieve(state: QAState) -> dict:
        question = state['question']
        if state.get('clarification'):
            question = f'{question}\n补充说明：{state["clarification"]}'
        hits = retriever.retrieve(question, document_id=state.get('document_id'))
        return {'chunks': [_chunk_to_dict(hit) for hit in hits]}

    def judge(state: QAState) -> dict:
        # 有澄清说明时视为信息已补充；否则无检索结果即为信息不足
        needs = not state.get('chunks') and not state.get('clarification')
        return {'needs_clarification': needs}

    def create_hitl(state: QAState) -> dict:
        with session_scope(session_factory) as session:
            task = create_hitl_task(
                session, session_id=state.get('session_id'), question=state['question']
            )
            settings = get_settings()
            if settings.cache_enabled and state.get('question_hash') and not state.get('cache_hit'):
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
                )
                prune_qa_cache(session, settings.cache_max_entries)
            return {'hitl_task_id': task.id}

    def answer(state: QAState) -> dict:
        prompt = _build_answer_prompt(
            question=state['question'],
            chunks=state.get('chunks') or [],
            clarification=state.get('clarification'),
        )
        response = chat_model.invoke(prompt)
        content = response.content if hasattr(response, 'content') else str(response)
        citations = [
            {
                'chunk_id': chunk['chunk_id'],
                'chapter': chunk.get('chapter'),
                'page': None,
                'excerpt': chunk['text'][:120],
            }
            for chunk in (state.get('chunks') or [])
        ]
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
                )
                prune_qa_cache(session, settings.cache_max_entries)
        return {'answer': content, 'citations': citations}

    def record(state: QAState) -> dict:
        if not state.get('session_id'):
            return {}
        with session_scope(session_factory) as session:
            session.add(
                ChatMessage(session_id=state['session_id'], role='user', content=state['question'])
            )
            if state.get('answer'):
                session.add(
                    ChatMessage(
                        session_id=state['session_id'],
                        role='assistant',
                        content=state['answer'],
                        meta={'citations': state.get('citations') or []},
                    )
                )
        return {}

    def route_after_cache(state: QAState) -> str:
        if not state.get('cache_hit'):
            return 'retrieve'
        return 'create_hitl' if state.get('needs_clarification') else 'record'

    def route_after_judge(state: QAState) -> str:
        return 'create_hitl' if state.get('needs_clarification') else 'answer'

    graph = StateGraph(QAState)
    graph.add_node('cache_check', cache_check)
    graph.add_node('retrieve', retrieve)
    graph.add_node('judge', judge)
    graph.add_node('create_hitl', create_hitl)
    graph.add_node('answer', answer)
    graph.add_node('record', record)
    graph.add_edge(START, 'cache_check')
    graph.add_conditional_edges(
        'cache_check',
        route_after_cache,
        {'retrieve': 'retrieve', 'create_hitl': 'create_hitl', 'record': 'record'},
    )
    graph.add_edge('retrieve', 'judge')
    graph.add_conditional_edges(
        'judge',
        route_after_judge,
        {'answer': 'answer', 'create_hitl': 'create_hitl'},
    )
    graph.add_edge('answer', 'record')
    graph.add_edge('create_hitl', 'record')
    graph.add_edge('record', END)
    return graph.compile(checkpointer=checkpointer or InMemorySaver())
