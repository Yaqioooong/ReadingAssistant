"""问答流水线：检索 → 充足性判断 → 回答 / HITL 澄清 → 记录。"""

from typing import TypedDict

from langchain_core.embeddings import Embeddings
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from sqlalchemy.orm import Session, sessionmaker

from reading_assistant.model.factory import get_chat_model, get_embedding_model
from reading_assistant.rag import Retriever
from reading_assistant.storage import (
    ChatMessage,
    create_hitl_task,
    session_scope,
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

    def route_after_judge(state: QAState) -> str:
        return 'create_hitl' if state.get('needs_clarification') else 'answer'

    graph = StateGraph(QAState)
    graph.add_node('retrieve', retrieve)
    graph.add_node('judge', judge)
    graph.add_node('create_hitl', create_hitl)
    graph.add_node('answer', answer)
    graph.add_node('record', record)
    graph.add_edge(START, 'retrieve')
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
