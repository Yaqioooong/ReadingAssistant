"""历史窗口预算的端到端 A/B：窗口把早期轮次挤掉时，答案**真的**会变吗？

为什么不复用现成夹具：``eval/multi_turn_gold.json`` 只断言 **route**、
最长 6 轮，窗口装得下，所以 budget 调到任何值结果都相同 —— 测不到。

本探针的用例刻意做成**无噪声**的形态：
- 第 1 轮用户说「请记住代号蓝鲸-7。」，末尾问「我第一个问题是什么？」
- 该问命中 ``history`` **规则快通道**（词表里有「第一个问题」）→ 零 LLM 门、
  不走检索 → 结果只取决于「第 1 轮原文还在不在窗口里」，没有判官/检索噪声。
- 填轮用的是**库里真实的长助手回复**，长度分布真实（不是合成文本）。
- 判据是确定性的：答案里有没有「蓝鲸-7」。

跑法（**必须用环境变量切预算**，这才是真实的 A/B 姿势）：
    HISTORY_TOKEN_BUDGET=4000  uv run python scripts/probe_context_budget_e2e.py
    HISTORY_TOKEN_BUDGET=16000 uv run python scripts/probe_context_budget_e2e.py

副作用：会在库里建一个会话并在结束时删除（含其消息）。
"""
from __future__ import annotations

import json

from fastapi.testclient import TestClient
from sqlalchemy import select

from reading_assistant.api import create_app
from reading_assistant.config import get_settings
from reading_assistant.graph.qa import _count_tokens, _HISTORY_ROLE_OVERHEAD, _select_window
from reading_assistant.model.factory import get_chat_model, get_embedding_model
from reading_assistant.storage import create_db_engine, create_session_factory
from reading_assistant.storage.models import ChatMessage, ChatSession
from reading_assistant.storage.vector_store import create_vector_store

CODE = '蓝鲸-7'
INTRO = f'请记住代号{CODE}。'
PROBE = '我第一个问题是什么？'
FILLER_TURNS = 12  # 足够超 4000、远低于 16000


def _filler_messages(messages: list[ChatMessage]) -> list[tuple[str, str]]:
    """从库里挑真实的长助手回复当填轮 —— 保证 token 开销分布真实。"""
    long_ones = sorted(
        ((_count_tokens(m.content or ''), m.content or '') for m in messages if m.role == 'assistant'),
        reverse=True,
    )
    pool = [t for _, t in long_ones[:40]] or ['（无历史回复可复用）' * 30]
    turns = []
    for i in range(FILLER_TURNS):
        turns.append(('user', f'西游记第{i + 10}回讲了什么？'))
        turns.append(('assistant', pool[i % len(pool)]))
    return turns


def main() -> None:
    settings = get_settings()
    engine = create_db_engine(settings.database_url)
    factory = create_session_factory(engine)

    with factory() as db:
        all_msgs = list(db.scalars(select(ChatMessage)))
    filler = _filler_messages(all_msgs)

    # 先算清楚：这个会话在两种预算下各保留哪些 —— 确定性的部分，不依赖模型
    rows = [('user', INTRO), *filler]
    kept = _select_window(rows, settings.history_token_budget)
    total = sum(_count_tokens(c) + _HISTORY_ROLE_OVERHEAD for _, c in rows)
    # ⚠️ _select_window 返回的是 **dict 列表**，不是 (role, content) 元组 ——
    # 按 tuple 解包会拿到键名（'role'/'content'），等于这条判据恒为 False。
    # 这个坑会让报告里出现一个「确定性」的假数，所以留注释钉住。
    intro_in_window = any(d['content'] == INTRO for d in kept)

    app = create_app(
        session_factory=factory,
        vector_store=create_vector_store(),
        llm=get_chat_model(),
        embedding_model=get_embedding_model(),
        upload_dir=None,
    )
    sid = None
    try:
        with TestClient(app) as client:
            sid = client.post('/api/sessions').json()['session_id']
            with factory() as db:
                for role, content in rows:
                    db.add(ChatMessage(session_id=sid, role=role, content=content))
                db.commit()
            resp = client.post(
                f'/api/sessions/{sid}/messages', json={'question': PROBE}
            ).json()
        answer = resp.get('answer') or ''
        result = {
            'budget': settings.history_token_budget,
            'seed_total_tokens': total,
            '窗口保留条数': f'{len(kept)}/{len(rows)}',
            '第1轮在窗口内(确定性)': intro_in_window,
            'intent': resp.get('intent'),
            'intent_channel': resp.get('intent_channel'),
            '答案含代号(端到端)': CODE in answer,
            'answer': answer[:160],
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        if sid is not None:
            # ⚠️ API 返回的 session_id 是**字符串**，而列是 Integer ——
            # 不转 int 会在 PG 上炸 operator does not exist: integer = character varying
            sid_int = int(sid)
            with factory() as db:
                for m in db.scalars(
                    select(ChatMessage).where(ChatMessage.session_id == sid_int)
                ):
                    db.delete(m)
                sess = db.get(ChatSession, sid_int)
                if sess is not None:
                    db.delete(sess)
                db.commit()
            print(f'（已清理会话 {sid}）')


if __name__ == '__main__':
    main()
