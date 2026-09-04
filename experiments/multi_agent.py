"""多 Agent 读书问答实验：Supervisor 规划 → 检索 Worker → 总结 Worker。

展示 LangGraph 多 Agent 协作：
- supervisor：查看馆藏书籍，规划"查哪本书 + 最优检索词"（结构化 Plan 输出）
- retrieval_worker：按计划执行向量检索
- synthesis_worker：基于检索结果生成带引用的最终回答

运行：uv run python experiments/multi_agent.py
"""
import json

from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from typing import TypedDict

from reading_assistant.model.factory import get_chat_model, get_embedding_model
from reading_assistant.rag import Retriever
from reading_assistant.storage import create_db_engine, create_session_factory, list_documents
from reading_assistant.storage.vector_store import create_vector_store
from reading_assistant.utils.logger_handler import get_logger

logger = get_logger('multi_agent')


class MultiAgentState(TypedDict, total=False):
    question: str
    plan: dict
    chunks: list[dict]
    answer: str
    citations: list[dict]


@tool
def make_plan(doc_scope: str, search_query: str, note: str) -> str:
    """制定检索计划。

    Args:
        doc_scope: 'all' 表示全库检索；或书籍文件名精确匹配（如 张三爱情故事.txt）
        search_query: 最适合向量检索的关键问题（含人名/事件）
        note: 规划理由，简要说明判断依据
    """
    return json.dumps({'doc_scope': doc_scope, 'search_query': search_query, 'note': note}, ensure_ascii=False)


def _book_listing() -> str:
    factory = create_session_factory(create_db_engine())
    with factory() as session:
        docs = list_documents(session)
    return '\n'.join(f'  - {d.id}: {d.filename}（{d.chunk_count} 分块）' for d in docs) or '  （暂无书籍）'


def _resolve_doc_id(scope: str) -> int | None:
    if scope == 'all':
        return None
    factory = create_session_factory(create_db_engine())
    with factory() as session:
        docs = list_documents(session)
    return next((d.id for d in docs if scope in d.filename), None)


def build_multi_agent_graph(llm=None, retriever=None):
    chat = llm or get_chat_model()
    retriever = retriever or Retriever(create_vector_store(), get_embedding_model())

    def supervisor(state: MultiAgentState) -> dict:
        """1. 查看馆藏 → 输出检索计划。"""
        question = state['question']
        prompt = (
            '你是阅读问答的主管 Agent，负责为问题制定检索计划。\n'
            f'图书馆当前馆藏：\n{_book_listing()}\n\n'
            f'用户问题：{question}\n\n'
            '请调用 make_plan 制定计划：判断该问题应全库检索还是某本书内检索，'
            '并给出最适合向量检索的查询词。'
        )
        try:
            resp = chat.bind_tools([make_plan]).invoke(prompt)
            calls = getattr(resp, 'tool_calls', None) or []
            plan = calls[0]['args'] if calls and calls[0].get('name') == 'make_plan' else {}
        except Exception:
            plan = {}
        if not plan:
            plan = {'doc_scope': 'all', 'search_query': question, 'note': '规划失败，全库兜底'}
        logger.info('[supervisor] 规划完成 doc_scope=%s query=%s note=%s',
                    plan['doc_scope'], plan['search_query'], plan['note'][:40])
        print(f'\n🧠 [Supervisor] 判断：查「{plan["doc_scope"]}」, 检索词「{plan["search_query"]}」\n'
              f'   理由：{plan["note"]}')
        return {'plan': plan}

    def retrieval_worker(state: MultiAgentState) -> dict:
        """2. 按计划检索。"""
        plan = state['plan']
        doc_id = _resolve_doc_id(plan.get('doc_scope', 'all'))
        hits = retriever.retrieve(plan.get('search_query', state['question']), document_id=doc_id)
        chunks = [
            {
                'chunk_id': h.chunk_id,
                'text': h.text,
                'document_id': h.document_id,
                'chapter': h.chapter,
                'score': h.score,
            }
            for h in hits
        ]
        logger.info('[retrieval_worker] 检索 doc_scope=%s hit=%d', plan.get('doc_scope'), len(chunks))
        print(f'🔍 [检索Worker] 命中 {len(chunks)} 条片段'
              + (f'（最高分 {chunks[0]["score"]:.2f}）' if chunks else ''))
        return {'chunks': chunks}

    def synthesis_worker(state: MultiAgentState) -> dict:
        """3. 基于片段总结回答。"""
        chunks = state.get('chunks') or []
        if not chunks:
            logger.info('[synthesis_worker] 无检索结果，如实说明')
            print('✍️  [总结Worker] 未检索到相关内容')
            return {'answer': '未检索到相关内容，无法回答。', 'citations': []}
        context = '\n\n'.join(
            f'[{i}]（{c.get("chapter", "")}，来源片段 {c["chunk_id"]}）\n{c["text"]}'
            for i, c in enumerate(chunks, start=1)
        )
        prompt = (
            '你是阅读问答的总结 Agent。请仅根据检索到的原文片段回答用户问题。\n'
            f'问题：{state["question"]}\n\n原文片段：\n{context}\n\n'
            '请给出最终回答，需要引用时用 [n] 标注片段编号。'
        )
        resp = chat.invoke(prompt)
        content = resp.content if hasattr(resp, 'content') else str(resp)
        citations = [
            {
                'chunk_id': c['chunk_id'],
                'chapter': c.get('chapter'),
                'excerpt': c['text'][:120],
            }
            for c in chunks
        ]
        logger.info('[synthesis_worker] 回答完成 len=%d cit=%d', len(content), len(citations))
        print(f'✍️  [总结Worker] 回答（{len(content)} 字）：{content[:80]}…')
        return {'answer': content, 'citations': citations}

    graph = StateGraph(MultiAgentState)
    graph.add_node('supervisor', supervisor)
    graph.add_node('retrieval_worker', retrieval_worker)
    graph.add_node('synthesis_worker', synthesis_worker)
    graph.add_edge(START, 'supervisor')
    graph.add_edge('supervisor', 'retrieval_worker')
    graph.add_edge('retrieval_worker', 'synthesis_worker')
    graph.add_edge('synthesis_worker', END)
    return graph.compile()


def run(question: str, llm=None, retriever=None) -> dict:
    """跑一次多 Agent 问答，返回结构化结果（供 API / 页面调用）。"""
    agent = build_multi_agent_graph(llm=llm, retriever=retriever)
    result = agent.invoke({'question': question})
    plan = result.get('plan') or {}
    chunks = result.get('chunks') or []
    return {
        'question': question,
        'plan': plan,
        'chunks': chunks,
        'answer': result.get('answer') or '',
        'citations': result.get('citations') or [],
    }


def demo():
    print('=' * 60)
    print('多 Agent 读书问答实验（Supervisor → 检索Worker → 总结Worker）')
    print('=' * 60)
    for q in ['张三喜欢谁？', '罗辑在第三章放下了什么？']:
        print(f'\n{"-" * 60}\n用户问题：{q}')
        result = run(q)
        print(f'\n最终回答：{result["answer"]}\n')
    print('\n实验完成。查看 logs/multi_agent_*.log 获取完整 trace。')


if __name__ == '__main__':
    demo()
