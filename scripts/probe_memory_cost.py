"""成本形状探针：多轮对话里每轮的 LLM 调用次数。

独立于 tests/（所以能在改造前后的代码上跑同一份），只走公开装配路径。
跑法：  PYTHONPATH=. uv run python /tmp/probe_memory_cost.py
"""
from __future__ import annotations

import math
import tempfile
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient
from langchain_core.embeddings import Embeddings

from reading_assistant.api import create_app
from reading_assistant.storage import create_db_engine, create_session_factory, init_db
from reading_assistant.storage.models import Document
from reading_assistant.storage.vector_store import InMemoryVectorStore, StoredChunk

TURNS = 12
NAMES = ['张三', '李四', '王五', '赵六', '孙七', '周八', '吴九', '郑十',
         '钱一', '冯二', '陈三', '褚四']


class CountingLLM:
    """统计每次 invoke，并按 prompt 归因到具体环节。"""

    def __init__(self) -> None:
        self.calls = 0
        self.kinds: dict[str, int] = {}
        self.prompts: list[str] = []

    def invoke(self, prompt: str) -> SimpleNamespace:
        self.calls += 1
        self.prompts.append(prompt)
        if '对话摘要任务' in prompt:
            kind = 'summarize'
        elif '意图分类任务' in prompt:
            kind = 'gate'
        elif '不需要检索书籍内容' in prompt:
            kind = 'history/chat 作答'
        else:
            kind = '书问题作答'
        self.kinds[kind] = self.kinds.get(kind, 0) + 1
        if kind == 'gate':
            return SimpleNamespace(content='book')
        if kind == 'summarize':
            return SimpleNamespace(content='（旧方案的滚动摘要）')
        return SimpleNamespace(content='模拟回答。')


class SpreadEmb(Embeddings):
    """按问题内容铺开角度：与文档块相似度够高，但问题之间彼此可分。"""

    def __init__(self) -> None:
        self.seen: list[str] = []

    def embed_query(self, text: str) -> list[float]:
        if text not in self.seen:
            self.seen.append(text)
        angle = (self.seen.index(text) - TURNS / 2) * 0.12
        return [math.cos(angle), math.sin(angle)]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


def main() -> None:
    tmp = Path(tempfile.mkdtemp())
    engine = create_db_engine('sqlite:///:memory:')
    init_db(engine)
    factory = create_session_factory(engine)
    with factory() as session:
        session.add_all([
            Document(filename='d1.docx', title='d1', file_hash='f1',
                     content_hash='c1', index_status='indexed'),
            Document(filename='d2.docx', title='d2', file_hash='f2',
                     content_hash='c2', index_status='indexed'),
        ])
        session.commit()
    store = InMemoryVectorStore()
    store.add([
        StoredChunk(id=f'd{i}-0', text=f'文档{i}内容',
                    metadata={'document_id': i, 'chapter': '正文', 'chapter_index': 0},
                    embedding=[1.0, 0.0])
        for i in (1, 2)
    ])
    llm = CountingLLM()
    app = create_app(session_factory=factory, vector_store=store, llm=llm,
                     embedding_model=SpreadEmb(), upload_dir=tmp / 'uploads')

    print(f'{"轮":>3} {"本轮LLM调用":>11} {"累计":>6}  明细')
    print('-' * 62)
    with TestClient(app) as client:
        sid = client.post('/api/sessions').json()['session_id']
        prev = 0
        for i, name in enumerate(NAMES[:TURNS], start=1):
            before_kinds = dict(llm.kinds)
            resp = client.post(f'/api/sessions/{sid}/messages',
                               json={'question': f'{name}喜欢谁？',
                                     'document_ids': [1 + (i % 2)]}).json()
            delta_calls = llm.calls - prev
            prev = llm.calls
            delta_kinds = {
                k: v - before_kinds.get(k, 0)
                for k, v in llm.kinds.items() if v - before_kinds.get(k, 0) > 0
            }
            detail = ' '.join(f'{k}×{v}' for k, v in delta_kinds.items()) or '（缓存命中，0 调用）'
            if delta_kinds.get('summarize'):
                np_ = llm.prompts[-1]
                detail += f'  [摘要prompt {len(np_)} 字符]'
            flag = '' if resp.get('answer') else '  ⚠ 无回答'
            print(f'{i:>3} {delta_calls:>11} {llm.calls:>6}  {detail}{flag}')

    print('-' * 62)
    print(f'总计 {llm.calls} 次 LLM 调用 / {TURNS} 轮')
    print('分环节：' + '，'.join(f'{k}={v}' for k, v in sorted(llm.kinds.items())))
    if 'summarize' in llm.kinds:
        print(f'⚠ 滚动摘要被调用了 {llm.kinds["summarize"]} 次 —— 这就是要消除的成本')
    else:
        print('✓ 零摘要调用')


if __name__ == '__main__':
    main()
