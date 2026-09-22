"""检索 agent（有界 ReAct）契约测试。

守三件事：
1. **工具集是重点**，不是循环 —— 实测只有 `search` 时 ReAct 退化成查询改写、8/8 失败，
   所以每个工具的返回形状与降级行为都要钉住；
2. **协议正确性** —— OpenAI 兼容协议要求逐个回填 tool_call，只回一个直接 400；
3. **不伤害主链路** —— 简单问题不该被 agent 拖下水。
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage

from reading_assistant.graph.agent import (
    AGENT_TOOLS,
    AgentBudget,
    ToolBox,
    _to_chunk,
    merge_agent_chunks,
    run_agent_loop,
)
from reading_assistant.rag import RetrievedChunk
from reading_assistant.storage.vector_store import (
    InMemoryVectorStore,
    SearchHit,
    StoredChunk,
    _and_where,
)

DOC_ID = 7


# --------------------------------------------------------------- 语料/工具环境


def _store() -> InMemoryVectorStore:
    store = InMemoryVectorStore()
    store.add([
        StoredChunk(
            id='doc7-h-10', text='却说三藏骑马前行，忽见一只老虎精，唤作寅将军。',
            metadata={'document_id': DOC_ID, 'chapter_index': 13,
                      'chapter': '第十三回 陷虎城金星解厄'},
        ),
        StoredChunk(
            id='doc7-h-11', text='寅将军身披锦绣花纹，锯牙凿齿，目光如电。',
            metadata={'document_id': DOC_ID, 'chapter_index': 13,
                      'chapter': '第十三回 陷虎城金星解厄'},
        ),
        StoredChunk(
            id='doc7-h-12', text='太白金星化作老叟，拂断绳索救了三藏。',
            metadata={'document_id': DOC_ID, 'chapter_index': 13,
                      'chapter': '第十三回 陷虎城金星解厄'},
        ),
        StoredChunk(
            id='doc7-h-40', text='第四十回 婴儿戏化禅心乱，红孩儿登场。',
            metadata={'document_id': DOC_ID, 'chapter_index': 40,
                      'chapter': '第四十回 婴儿戏化禅心乱'},
        ),
        # 前言：含「寅将军」但**不是正文回目** —— 序数判断必须把它排除
        StoredChunk(
            id='doc7-h-02', text='前言：本书讲唐僧师徒西行，寅将军是最早出场的妖怪之一。',
            metadata={'document_id': DOC_ID, 'chapter_index': 2, 'chapter': '前言'},
        ),
    ])
    return store


class ScriptedModel:
    """按脚本返回 tool_calls 的假模型；暴露 bind_tools 使 agent 认为它支持工具。"""

    def __init__(self, script: list[list[dict]]) -> None:
        self.script = list(script)
        self.seen_messages: list[list] = []
        self.bound = None

    def bind_tools(self, tools):
        self.bound = tools
        return self

    def invoke(self, messages):
        self.seen_messages.append(list(messages))
        calls = self.script.pop(0) if self.script else []
        return AIMessage(
            content='',
            tool_calls=[
                {'name': c['name'], 'args': c.get('args', {}), 'id': f'call-{i}'}
                for i, c in enumerate(calls)
            ],
        )


def _toolbox(store=None) -> ToolBox:
    store = store or _store()
    return ToolBox(
        vector_store=store,
        search_fn=lambda q: [],
        document_ids=[DOC_ID],
        doc_titles={DOC_ID: '西游记'},
    )


# --------------------------------------------------------------- 结果类型兼容


def test_to_chunk_accepts_search_hit() -> None:
    """SearchHit：id + metadata 字典。"""
    chunk = _to_chunk(SearchHit(
        id='c1', score=0.5, text='正文',
        metadata={'document_id': 1, 'chapter': '甲', 'chapter_index': 2},
    ))
    assert chunk['chunk_id'] == 'c1'
    assert chunk['document_id'] == 1
    assert chunk['chapter_index'] == 2


def test_to_chunk_accepts_flat_retrieved_chunk() -> None:
    """回归：检索器返回的是 RetrievedChunk（chunk_id + 扁平字段，**没有 metadata**）。

    2026-09-21 实测：只按 SearchHit 写（hit.metadata）会让 search 工具每次都抛
    AttributeError，而工具级 try/except 把它变成「工具执行失败」的观察 ——
    **检索静默降级**，日志上只是 new_chunks=0。
    """
    chunk = _to_chunk(RetrievedChunk(
        chunk_id='c9', score=0.42, text='正文',
        document_id=3, chapter='乙', chapter_index=7,
    ))
    assert chunk['chunk_id'] == 'c9'
    assert chunk['document_id'] == 3
    assert chunk['chapter'] == '乙'
    assert chunk['chapter_index'] == 7
    assert chunk['score'] == pytest.approx(0.42)


# --------------------------------------------------------------- 工具行为


def test_grep_groups_by_chapter_in_book_order() -> None:
    """grep 的观察必须给出「命中回 + 书内顺序」——序数推理靠的就是它。"""
    result = _toolbox().call('grep', {'term': '寅将军'})
    assert '第1个命中回' in result.observation
    assert 'chapter_index=13' in result.observation
    assert '最早出现的回目' in result.observation
    assert result.chunks, 'grep 应贡献片段'


def test_grep_excludes_front_matter_from_ordinal_ordering() -> None:
    """回归：前言/附录里出现该词**不计入**「第几个」。

    实测（2026-09-21）：「前言」里含「妖怪」，未排除时它会成为「最早命中章」，
    把「第一个妖怪」的推理带偏。排除规则是确定性的（回目形如「第…回」），交给代码。
    """
    result = _toolbox().call('grep', {'term': '寅将军'})
    narrative_part = result.observation.split('【非正文')[0]
    assert '第1个命中回' in narrative_part
    assert 'chapter_index=13' in narrative_part, '正文部分是第十三回'
    assert 'chapter_index=2' not in narrative_part, '前言不该出现在正文回目里'
    assert '【非正文' in result.observation, '前言应被归入非正文并说明'


def test_grep_compares_multiple_candidates_and_ranks_by_book_order() -> None:
    """新增能力：一次给多个候选名，工具直接排出「谁最早出现」。

    「第几个」本质是**比较先后**，让代码算比让模型心算可靠 ——
    实测模型曾自称「第20-21回」却读的是 25/26 回。
    """
    result = _toolbox().call('grep', {'term': '寅将军,黄风怪,红孩儿'})
    assert '【候选比较】' in result.observation
    block = result.observation.split('【候选比较】')[1]
    assert '1. 「寅将军」' in block, '寅将军(第13回)应排第一'
    assert '最早，就是它' in block
    # 排序必须是书内顺序，而不是输入顺序
    first = block.index('寅将军')
    assert first < block.index('红孩儿'), '红孩儿(第40回)应排在寅将军之后'


def test_grep_handles_multi_candidate_with_no_narrative_hit() -> None:
    """有命中、但都在非正文（前言）里 → 比较段要明说「正文回目里没有命中」。"""
    result = _toolbox().call('grep', {'term': '前言,不存在的候选'})
    assert '【候选比较】' in result.observation
    assert '都没有命中' in result.observation
    assert '【非正文' in result.observation


def test_grep_reports_absence_clearly() -> None:
    result = _toolbox().call('grep', {'term': '不会出现的词'})
    assert '没有任何位置出现' in result.observation
    assert not result.chunks


def test_read_chapter_returns_full_chapter_without_score_filter() -> None:
    """read_chapter 的存在意义就是绕开 min_score：**整章文本完整取回**、不做相似度过滤。

    ⚠️ 断言的是**文本完整性**而不是 chunk 数 —— 整章会被合并成少量「分段」
    （见 coalesce_chapter），所以段数取决于字数而非章内 chunk 数。
    2026-09-21 实测：原实现 `found[:6]` 把第六章十六回的第 8 段（谜底：黄眉之名/
    弥勒座下司磬童儿/人种袋/设瓜田）切掉了，于是 agent 答对、answer 却说「未出现」。
    """
    result = _toolbox().call('read_chapter', {'chapter_index': 13})
    assert '第13章' in result.observation
    all_text = '\n'.join(c['text'] for c in result.chunks)
    for needle in ('寅将军', '身披锦绣花纹', '太白金星'):
        assert needle in all_text, f'整章文本应含「{needle}」——不得按段数截断'
    assert all(c['source'] == 'read_chapter' for c in result.chunks)
    assert all(c['chapter_index'] == 13 for c in result.chunks)


def test_coalesce_chapter_keeps_every_chunk_text() -> None:
    """合并分段不得丢字：分段的文本总长必须等于原章全部 chunk 之和。"""
    from reading_assistant.graph.agent import coalesce_chapter
    from reading_assistant.storage.vector_store import StoredChunk

    chunks = [
        StoredChunk(id=f'd1-{i}', text='甲' * 1500,
                    metadata={'document_id': 1, 'chapter': '第一回', 'chapter_index': 1})
        for i in range(4)
    ]
    segments = coalesce_chapter(chunks, max_chars=4000)
    assert len(segments) == 2, '6000 字应合并成 2 段（目标 4000/段）'
    # 不丢字：每个原始 chunk 的文本都要在某个分段里**逐字出现**
    # （不能拿总长度比 —— 分段用 '\n'.join 拼接，会多出分隔符）
    merged_text = '\n'.join(seg['text'] for seg in segments)
    for chunk in chunks:
        assert chunk.text in merged_text, f'{chunk.id} 的正文被合并弄丢了'
    # 分段 = [[chunk0, chunk1], [chunk2, chunk3]]（每段目标 4000 字、每 chunk 1500 字）
    assert segments[0]['chunk_id'] == 'd1-0', '分段 id 沿用该段首个真实 chunk'
    assert segments[1]['chunk_id'] == 'd1-2'
    assert segments[0]['segment_of'] == 2, '记录该段由几个原 chunk 合并而来'


def test_read_chapter_fits_long_chapter_in_few_slots() -> None:
    """回归：整章必须能塞进 agent 的少量名额。

    2026-09-21 失败现场：一章 11~12 个 chunk，而 agent 最终只有约 8 个名额，
    **一整章塞不下**，丢哪段取决于顺序（靠运气）。合并成分段后一章只占 2~3 个名额。
    """
    result = _toolbox().call('read_chapter', {'chapter_index': 13})
    assert len(result.chunks) <= 3, f'整章应只占少量名额，实际 {len(result.chunks)}' 


def test_read_chapter_rejects_bad_argument() -> None:
    result = _toolbox().call('read_chapter', {'chapter_index': '不是数字'})
    assert '必须是整数' in result.observation


def test_read_chapter_missing_chapter_is_reported() -> None:
    result = _toolbox().call('read_chapter', {'chapter_index': 9999})
    assert '没有找到' in result.observation


def test_list_chapters_lists_all_in_order() -> None:
    result = _toolbox().call('list_chapters', {})
    indexes = [
        int(line.split('chapter_index=')[1].split()[0])
        for line in result.observation.splitlines() if 'chapter_index=' in line
    ]
    assert indexes == sorted(indexes)
    assert indexes == [2, 13, 40]  # 含前言（chapter_index=2）


def test_unknown_tool_returns_structured_error() -> None:
    """工具名幻觉不能炸掉问答，要把可用工具名告诉模型。"""
    result = _toolbox().call('不存在的工具', {})
    assert '没有名为' in result.observation
    assert 'grep' in result.observation


def test_tool_exception_is_contained() -> None:
    """工具内部异常 → 结构化错误（项目原则：模型读得到才能自我修正）。"""
    def boom(_query):
        raise RuntimeError('底层检索炸了')

    box = _toolbox()
    box.search_fn = boom
    result = box.call('search', {'query': '任意'})
    assert '执行失败' in result.observation
    assert 'RuntimeError' in result.observation


# --------------------------------------------------------------- 循环与预算


def test_loop_stops_on_finish() -> None:
    model = ScriptedModel([
        [{'name': 'grep', 'args': {'term': '寅将军'}}],
        [{'name': 'finish', 'args': {'reason': '够了'}}],
    ])
    run = run_agent_loop(model=model, toolbox=_toolbox(), question='第一个妖怪是谁？')
    assert run.finished is True
    assert run.stopped_by == 'finish'
    assert run.steps == 2
    assert run.chunks, 'grep 的片段应被收集'


def test_loop_answers_every_tool_call_in_a_round() -> None:
    """回归：一轮里多个 tool_call **必须逐条**回填 tool 消息。

    OpenAI 兼容协议（DeepSeek 同）要求 assistant 的 tool_calls 后面
    「每条都有对应 tool 消息」，否则下一次请求 400：
    "An assistant message with 'tool_calls' must be followed by tool messages
     responding to each 'tool_call_id'"
    实测模型很自然地一次返回两个（search + list_chapters）。
    """
    model = ScriptedModel([
        [{'name': 'grep', 'args': {'term': '寅将军'}},
         {'name': 'list_chapters', 'args': {}}],
        [{'name': 'finish', 'args': {'reason': 'ok'}}],
    ])
    run_agent_loop(model=model, toolbox=_toolbox(), question='q')
    second_round = model.seen_messages[1]
    tool_messages = [m for m in second_round if type(m).__name__ == 'ToolMessage']
    assert len(tool_messages) == 2, '两个 tool_call 必须各有一条 tool 消息'
    assert len({m.tool_call_id for m in tool_messages}) == 2


def test_loop_respects_max_steps() -> None:
    """预算硬上限：模型一直不停也必须被截断。"""
    model = ScriptedModel([[{'name': 'grep', 'args': {'term': '寅将军'}}] for _ in range(20)])
    run = run_agent_loop(
        model=model, toolbox=_toolbox(), question='q',
        budget=AgentBudget(max_steps=3),
    )
    assert run.steps == 3
    assert run.stopped_by == 'budget'


def test_loop_stops_when_model_returns_no_tool_call() -> None:
    model = ScriptedModel([[]])
    run = run_agent_loop(model=model, toolbox=_toolbox(), question='q')
    assert run.stopped_by == 'no_tool_call'
    assert run.steps == 0


def test_loop_survives_model_exception() -> None:
    """agent 是增强项：模型异常要能回退，不能把整次问答带崩。"""
    class BoomModel:
        def bind_tools(self, tools):
            return self

        def invoke(self, messages):
            raise RuntimeError('模型挂了')

    run = run_agent_loop(model=BoomModel(), toolbox=_toolbox(), question='q')
    assert run.stopped_by == 'error'
    assert run.chunks == []


def test_loop_dedups_chunks_and_leaves_capping_to_merge() -> None:
    """循环只去重、**不截断** —— 最终裁剪由 merge_agent_chunks 负责。

    ⚠️ 这条契约在 2026-09-21 改过：原实现 `run.chunks = deduped[:max_chunks]`
    在 merge **之前**就截到 16 条，且取最先累积的（grep 的候选 + 第一次读章的前半），
    于是 agent 随后读到的第二章**根本到不了 merge**。
    实测：工具共贡献 31 条，到 merge 只剩 16 条，而答案在「第二章第 8 段」。
    累计不设限（只是些轻量 dict），裁剪交给知道来源优先级的 merge。
    """
    model = ScriptedModel([
        [{'name': 'grep', 'args': {'term': '寅将军'}}],
        [{'name': 'grep', 'args': {'term': '寅将军'}}],
        [{'name': 'finish', 'args': {'reason': 'ok'}}],
    ])
    run = run_agent_loop(
        model=model, toolbox=_toolbox(), question='q',
        budget=AgentBudget(max_steps=6, max_chunks=2),
    )
    ids = [c['chunk_id'] for c in run.chunks]
    assert len(ids) == len(set(ids)), '同一片段不该重复'
    assert len(ids) > 2, '循环不应按 max_chunks 截断（裁剪是 merge 的职责）' 


def test_loop_records_trace_for_observability() -> None:
    model = ScriptedModel([
        [{'name': 'grep', 'args': {'term': '寅将军'}}],
        [{'name': 'finish', 'args': {'reason': 'ok'}}],
    ])
    run = run_agent_loop(model=model, toolbox=_toolbox(), question='q')
    assert run.trace and run.trace[0]['action'] == 'grep'
    assert 'observation_chars' in run.trace[0]


def test_tool_schemas_are_exposed() -> None:
    names = {t.name for t in AGENT_TOOLS}
    assert names == {'search', 'grep', 'list_chapters', 'read_chapter', 'finish'}


# --------------------------------------------------------------- where 组装


def test_and_where_wraps_multiple_conditions() -> None:
    """回归：Chroma 的 where 只接受一个操作符，双条件必须 $and。

    实测：`{'document_id': 7, 'chapter_index': 18}` 会被 Chroma 拒绝
    （Expected where to have exactly one operator），而 InMemory 实现自己线性过滤、
    不校验形状 —— 所以单测全绿、线上 read_chapter 每次都失败。
    """
    assert _and_where(document_id=7) == {'document_id': 7}
    assert _and_where(document_id=7, chapter_index=18) == {
        '$and': [{'document_id': 7}, {'chapter_index': 18}]
    }
    assert _and_where() is None
    assert _and_where(document_id=None) is None


# --------------------------------------------------------------- 合并策略
#
# 2026-09-21 实测：agent 走错方向时一轮贡献 28 条片段（两次 grep + 两章全文），
# 把上限 16 全占满，**原有的检索结果被整体挤出** —— 于是「agent 失败」升级成
# 「agent 失败且销毁了原有证据」，answer 只能转澄清。


def _chunks(prefix: str, n: int) -> list[dict]:
    return [{'chunk_id': f'{prefix}-{i}', 'text': f'{prefix}{i}'} for i in range(n)]


def test_merge_never_lets_agent_wipe_originals() -> None:
    """核心契约：agent 片段再多，原有片段也必须留有一席之地。"""
    originals = _chunks('orig', 16)
    agent = _chunks('agent', 28)
    merged = merge_agent_chunks(agent, originals, cap=16)

    assert len(merged) == 16
    kept = [c for c in merged if c['chunk_id'].startswith('orig')]
    assert kept, '原有片段被完全挤出了 —— 这正是要防的回归'
    assert len(kept) >= 8, f'应保留约一半名额给原有片段，实际 {len(kept)}'


def test_merge_prefers_agent_chunks_when_originals_are_few() -> None:
    """原有片段很少时不必强留一半 —— agent 的定向片段优先。"""
    merged = merge_agent_chunks(_chunks('agent', 10), _chunks('orig', 2), cap=6)
    assert len(merged) == 6
    assert len([c for c in merged if c['chunk_id'].startswith('orig')]) == 2


def test_merge_dedups_by_chunk_id() -> None:
    shared = {'chunk_id': 'same', 'text': '重叠片段'}
    merged = merge_agent_chunks([shared], [dict(shared)], cap=5)
    assert len(merged) == 1


def test_merge_without_originals_keeps_the_most_recent() -> None:
    """没有原有片段时，保留 agent **最新**的一批（轨迹收敛 → 越晚越接近结论）。"""
    merged = merge_agent_chunks(_chunks('agent', 20), [], cap=4)
    assert [c['chunk_id'] for c in merged] == ['agent-19', 'agent-18', 'agent-17', 'agent-16']


def test_merge_handles_zero_cap() -> None:
    assert merge_agent_chunks(_chunks('agent', 3), _chunks('orig', 3), cap=0) == []


def test_merge_skips_chunks_without_id() -> None:
    merged = merge_agent_chunks([{'text': '没有 id'}], _chunks('orig', 1), cap=5)
    assert [c['chunk_id'] for c in merged] == ['orig-0']


def test_merge_prefers_later_agent_findings() -> None:
    """回归（2026-09-21 17:11 实测）：agent 的**后出**片段必须优先。

    失败现场：问「收悟空后孙悟空消灭的第一个妖精是谁」，agent 贡献 28 条，
    结论正确（白衣秀士，第十七回）。旧的「取前 8 条」只放进了 step1 的宽网候选，
    把 step2 read_chapter(22) 与 step4 grep 的**含答案片段**丢掉 →
    answer 看不到依据，再次判「信息不足」转澄清。

    轨迹是收敛的：越晚的观察越接近结论。
    """
    early = [{'chunk_id': f'early-{i}', 'text': '宽网候选'} for i in range(8)]
    late = [{'chunk_id': f'late-{i}', 'text': '含答案片段'} for i in range(8)]
    originals = _chunks('orig', 8)

    merged = merge_agent_chunks(early + late, originals, cap=16)
    ids = [c['chunk_id'] for c in merged]
    assert any(i.startswith('late-') for i in ids), (
        '后出的（更接近结论的）片段被丢掉了 —— 这正是要防的回归'
    )
    assert all(i.startswith('late-') for i in ids if i.startswith(('late-', 'early-'))), (
        'agent 名额应全部给最新一批'
    )


def test_merge_prioritizes_read_chapter_over_candidates() -> None:
    """回归（2026-09-21 第二轮）：`read_chapter` 的章正文必须优先于 grep/search 候选。

    失败现场：小雷音寺那题跑 5 次只对 3 次。失败的两次里，agent 明明
    `read_chapter(71)/(72)` 读到了答案所在的两回，**正文却一条都没进最终上下文**
    （或只进半章）—— 因为排名靠后的收尾 `search` 返回大批候选，把章正文挤掉了。
    agent 的结论正是从读完的章里得出的，所以章正文的证据等级高于候选。
    """
    candidates = [dict(c, source='search') for c in _chunks('cand', 20)]
    chapters = [dict(c, source='read_chapter') for c in _chunks('chap', 6)]
    originals = _chunks('orig', 8)

    # 收尾的 search 排在最后（= 原实现「取最新」会优先它）
    merged = merge_agent_chunks(candidates + chapters, originals, cap=16)
    ids = [c['chunk_id'] for c in merged]

    kept_chapters = [i for i in ids if i.startswith('chap-')]
    assert len(kept_chapters) == 6, (
        f'read_chapter 的章正文应全部保留（agent 的答案依据），实际 {kept_chapters}'
    )
    assert all(i.startswith('chap-') for i in ids[:6]), '章正文应排在最前'


def test_merge_read_chapter_priority_does_not_break_original_reserve() -> None:
    """分层优先级不能破坏「原有片段留名额」这条早先的契约。"""
    chapters = [dict(c, source='read_chapter') for c in _chunks('chap', 20)]
    merged = merge_agent_chunks(chapters, _chunks('orig', 8), cap=16)
    kept_originals = [c for c in merged if c['chunk_id'].startswith('orig')]
    assert kept_originals, '原有片段被章正文完全挤出了 —— 两条规则必须同时成立'


def test_merge_keeps_later_chunk_when_reading_same_chapter_twice() -> None:
    """同一章读两次（agent 常见重试）时，后一次的版本优先。"""
    first = [{'chunk_id': 'c1', 'text': '旧', 'source': 'read_chapter'}]
    second = [{'chunk_id': 'c1', 'text': '新', 'source': 'read_chapter'}]
    merged = merge_agent_chunks(first + second, [], cap=4)
    assert len(merged) == 1
    assert merged[0]['text'] == '新'


def test_merge_honours_exact_reserve_quota() -> None:
    """回归（2026-09-22）：原有片段的名额是**硬配额**，不是「尽量」。

    真实日志证据：`合并后 16 条（其中原有 5 条）`（agent 11 条、原有 12 条、cap 16）。
    按 `reserve = min(12, 16//2) = 8` 应该保 8 条，实际只保了 5 条。

    根因是旧写法 `head[:cap - len(tail)] + tail` —— `head` 里仍然包含**全部** agent 片段
    （它们占着最前面几个位置），截断后 agent 原封不动、只给原有片段留了 2 个空位。
    必须显式分三段取：agent（≤cap-reserve）→ 原有（拿满 reserve）→ 空位回填 agent。
    """
    agent = ([{'chunk_id': f'ag-{i}', 'text': 'x', 'source': 'grep'} for i in range(8)]
             + [{'chunk_id': f'ch-{i}', 'text': 'x', 'source': 'read_chapter'} for i in range(3)])
    originals = [{'chunk_id': f'or-{i}', 'text': 'x'} for i in range(12)]

    out = merge_agent_chunks(agent, originals, cap=16)
    kept = [c['chunk_id'] for c in out if c['chunk_id'].startswith('or-')]
    assert len(kept) == 8, f'原有片段应拿满 reserve=8，实际 {len(kept)}'
    assert len(out) == 16
    # read_chapter 仍在最前（它比 grep 候选更接近结论）
    assert [c['chunk_id'] for c in out[:3]] == ['ch-2', 'ch-1', 'ch-0']


def test_merge_backfills_with_agent_when_originals_are_short() -> None:
    """原有片段不够 reserve 时，空位要回填 agent 片段（不能浪费名额）。"""
    agent = [{'chunk_id': f'ag-{i}', 'text': 'x', 'source': 'search'} for i in range(20)]
    originals = [{'chunk_id': f'or-{i}', 'text': 'x'} for i in range(2)]

    out = merge_agent_chunks(agent, originals, cap=10)
    assert len(out) == 10, '应把名额填满'
    assert sum(1 for c in out if c['chunk_id'].startswith('or-')) == 2
    assert sum(1 for c in out if c['chunk_id'].startswith('ag-')) == 8
