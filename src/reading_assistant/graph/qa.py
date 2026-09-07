"""问答流水线：检索 → 充足性判断 → 回答 / HITL 澄清 → 记录。"""

import math
from datetime import datetime, timedelta, timezone
from typing import TypedDict

from langchain_core.embeddings import Embeddings
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from sqlalchemy.orm import Session, sessionmaker

from reading_assistant.config import get_settings
from reading_assistant.model.factory import get_chat_model, get_embedding_model
from reading_assistant.rag import create_retriever
from reading_assistant.storage import (
    ChatMessage,
    create_hitl_task,
    get_document,
    get_qa_cache_entry,
    list_documents,
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
                '3. 若原文片段足以回答，请调用 final_answer 工具提交回答；'
                '若不足以回答，请调用 request_clarification 工具并说明缺少什么。'
            ),
        ]
    )
    if doc_titles and len(doc_titles) > 1:
        lines.append(
            '注意：本次问题涉及多本书籍。回答时请明确区分不同书籍各自的内容/立场，'
            '原文片段前已标注书名《...》，引用时对应到正确的书籍。'
        )
    return '\n\n'.join(lines)


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


_NO_INFO_MARKERS = (
    '原文中没有相关信息',
    '没有相关信息',
    '原文未提及',
    '原文没有',
    '没有找到',
    '无法根据原文',
    '无法回答',
    '信息不足',
    '不足以回答',
)


def _looks_like_no_info(content: str) -> bool:
    """启发式检测：LLM 是否判定原文信息不足（只看回答开头 80 字符）。"""
    head = (content or '').strip()[:80]
    return any(marker in head for marker in _NO_INFO_MARKERS)


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
    retriever = create_retriever(vector_store, embedding_model or get_embedding_model())

    def cache_check(state: QAState) -> dict:
        if len(state.get('document_ids') or []) > 1:
            # 多文档问答缓存键需绑定全部文档版本，MVP 直接跳过缓存（每次实时检索）
            return {'cache_hit': False, 'question_hash': None}
        settings = get_settings()
        if not settings.cache_enabled:
            return {'cache_hit': False, 'question_hash': None}
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
                        if best_entry.answer and _looks_like_no_info(best_entry.answer):
                            # 无效缓存：信息不足式回答不应被缓存，删除并重新回答
                            logger.warning('问答[缓存] 丢弃无效回答缓存 id=%s q=%.30s',
                                           best_entry.id, state['question'])
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
            if entry.answer and _looks_like_no_info(entry.answer):
                # 无效缓存：信息不足式回答不应被缓存，删除并重新回答
                logger.warning('问答[缓存] 丢弃无效回答缓存 id=%s q=%.30s',
                               entry.id, state['question'])
                session.delete(entry)
                return {'cache_hit': False, 'question_hash': question_hash}
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
        }

    def retrieve(state: QAState) -> dict:
        question = state['question']
        if state.get('clarification'):
            question = f'{question}\n补充说明：{state["clarification"]}'
        doc_ids = state.get('document_ids') or []
        if len(doc_ids) > 1:
            # 多文档问答：并行 fan-out，每书独立检索后按相似度合流取全局 top_k
            from concurrent.futures import ThreadPoolExecutor

            settings = get_settings()
            total_k = settings.top_k
            per_doc = max(2, -(-total_k // len(doc_ids)))  # 按文档数 ceil 分配

            def _retrieve_one(doc_id: int):
                return retriever.retrieve(question, top_k=per_doc, document_id=doc_id)

            with ThreadPoolExecutor(max_workers=min(len(doc_ids), 8)) as executor:
                grouped = list(executor.map(_retrieve_one, doc_ids))
            merged = sorted(
                (hit for group in grouped for hit in group),
                key=lambda hit: hit.score,
                reverse=True,
            )[:total_k]
            titles: dict[int, str] = {}
            with session_scope(session_factory) as session:
                for document in list_documents(session):
                    if document.id in doc_ids:
                        titles[document.id] = document.filename
            logger.info('问答[检索] q=%.30s docs=%s merged=%d', question, doc_ids, len(merged))
            return {'chunks': [_chunk_to_dict(hit) for hit in merged], 'doc_titles': titles}
        hits = retriever.retrieve(question, document_id=state.get('document_id'))
        titles: dict[int, str] = {}
        doc_id = state.get('document_id')
        if doc_id is not None:
            with session_scope(session_factory) as session:
                document = get_document(session, doc_id)
                if document is not None:
                    titles[document.id] = document.filename
        logger.info('问答[检索] q=%.30s doc=%s hit=%d', question,
                    state.get('document_id'), len(hits))
        return {'chunks': [_chunk_to_dict(hit) for hit in hits], 'doc_titles': titles or None}

    def judge(state: QAState) -> dict:
        # 有澄清说明时视为信息已补充；否则无检索结果即为信息不足
        needs = not state.get('chunks') and not state.get('clarification')
        return {'needs_clarification': needs}

    def create_hitl(state: QAState) -> dict:
        logger.info('问答[HITL] 信息不足，进入澄清流程 q=%.30s', state['question'])
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
        import time

        prompt = _build_answer_prompt(
            question=state['question'],
            chunks=state.get('chunks') or [],
            clarification=state.get('clarification'),
            doc_titles=state.get('doc_titles'),
        )
        start = time.perf_counter()
        tool_calls = []
        if hasattr(chat_model, 'bind_tools'):
            try:
                response = chat_model.bind_tools(
                    [final_answer, request_clarification]
                ).invoke(prompt)
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
                logger.info('问答[answer] 工具判定信息不足 → HITL missing=%.50s (%.0fms)',
                            missing, cost_ms)
                if not state.get('clarification'):
                    # 转入 HITL 澄清，不缓存该回答、不写入聊天记录
                    return {'needs_clarification': True, 'answer': None, 'citations': []}
                # 澄清后模型仍判定不足：不再重复建任务，以说明文本作回答
                content = missing or '澄清后原文信息仍然不足。'
            else:
                content = args.get('answer') or ''
                logger.info('问答[answer] 工具 final_answer len=%d (%.0fms)', len(content),
                            cost_ms)
        else:
            # 兜底 1：模型未走工具，按纯文本处理
            content = response.content if hasattr(response, 'content') else str(response)
            logger.info('问答[answer] 纯文本回答 len=%d (%.0fms)', len(content), cost_ms)
            # 兜底 2：纯文本含"信息不足"措辞 → 仍转 HITL
            if not state.get('clarification') and _looks_like_no_info(content):
                logger.info('问答[answer] 文本兜底判定信息不足，转入 HITL q=%.30s',
                            state['question'])
                return {'needs_clarification': True, 'answer': None, 'citations': []}
        titles = state.get('doc_titles') or {}
        citations = []
        for chunk in state.get('chunks') or []:
            citations.append(
                {
                    'chunk_id': chunk['chunk_id'],
                    'chapter': chunk.get('chapter'),
                    'page': None,
                    'excerpt': chunk['text'][:120],
                    'document': titles.get(chunk.get('document_id')),
                }
            )
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

    def route_after_answer(state: QAState) -> str:
        return 'create_hitl' if state.get('needs_clarification') else 'record'

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
    graph.add_conditional_edges(
        'answer',
        route_after_answer,
        {'create_hitl': 'create_hitl', 'record': 'record'},
    )
    graph.add_edge('create_hitl', 'record')
    graph.add_edge('record', END)
    return graph.compile(checkpointer=checkpointer or InMemorySaver())
