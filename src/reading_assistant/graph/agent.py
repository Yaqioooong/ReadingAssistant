"""检索 agent：带工具的有界 ReAct 循环（Thought → Action → Observation）。

## 为什么需要它

一次向量检索对某些问法是**结构性无能**，与检索算法好坏无关。2026-09-21 实测把失败分成两类：

- **查询里有实体**（「寅将军是什么妖怪？」）→ 检索没问题，只是被 ``retrieval_min_score``
  之类的阈值挡住 → 换个通道/阈值就能救；
- **查询里没有实体**（「第一个妖怪是什么？」「它的特点是什么？」「三问合一」）
  → 在 ``min_score=0``、``top_k=200`` 下**任何阈值都捞不到**，
  因为答案段落里**从来不说「第一个」**，也没有「它」的指代对象。

第二类是**策略问题**：要拿到答案，必须先决定「换个词搜」「先查目录定位到哪一章」
「把整章读掉」—— 也就是**多步、且下一步依赖上一步的观察**。这正是 ReAct 的形态。

## 为什么工具比循环更关键

实测（同日）：让模型只拿 `search` 一个工具去改写「第一个妖怪」，生成 8 条查询，
**8/8 全部未命中**。只有 `search` 时，ReAct 退化成「昂贵的查询改写」，救不了序数问题。

而加上字面通道后：`grep('寅将军')` **恰好命中 1 章**，确定性且精确。
所以本模块的设计重心在**工具集**上：

| 工具 | 解决什么 | 为什么它行 |
|---|---|---|
| ``grep`` | 存在性 / 定位（序数类） | 字面匹配，不依赖相似度；答案段落不必「像」问题 |
| ``list_chapters`` | 书内顺序 | ``chapter_index`` 是序数推理的**唯一**可靠依据 |
| ``read_chapter`` | 低余弦的正确段落 | 绕开 ``min_score``；整章取回，不做相似度过滤 |
| ``search`` | 语义召回（原有能力） | hybrid 检索，池 50 |
| ``finish`` | 终止 | 由模型判断「够了」 |

## 边界与安全性

- **循环在节点内部**，不在图拓扑上新增环。原因：``interrupt()`` 恢复时**节点从头重跑**，
  把循环做成图上的环会让「恢复」变成整圈重放（重复副作用 + 重复付费）。
  放在节点内 + 全部工具**只读**，重放天然安全。
- **硬预算**：``max_steps`` 封顶 LLM 轮数；观察文本总量封顶，防止上下文被撑爆。
- **工具失败不抛异常**：返回结构化错误让模型自己改（项目既有原则：
  「工具失败返回结构化错误而非抛异常——模型读得到才能自我修正」）。
- **可观测**：每步写进 ``state['agent_trace']``，事后能回答「它为什么绕了三圈」。
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, NamedTuple

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from reading_assistant.utils.logger_handler import get_logger

logger = get_logger('agent')

# ---- 预算（全部可配，见 config.Settings）----
DEFAULT_MAX_STEPS = 6
MAX_OBSERVATION_CHARS = 3000  # 单次观察上限，防上下文被单章全文撑爆
MAX_CHAPTER_CHARS = 2000  # read_chapter 的 observation 预览上限
CHAPTER_SEGMENT_CHARS = 4000  # read_chapter 把整章合并成「分段」时每段的目标字数
MAX_TOOL_CALLS_PER_ROUND = 3  # 单轮最多真正执行几个 tool_call（其余回占位，协议要求）


@dataclass
class AgentBudget:
    """一次问答的 agent 预算。硬上限，不依赖模型自觉。"""

    max_steps: int = DEFAULT_MAX_STEPS
    max_observation_chars: int = MAX_OBSERVATION_CHARS * DEFAULT_MAX_STEPS
    max_chunks: int = 16  # 最终并入答案的片段数上限

    spent_observation_chars: int = 0
    spent_steps: int = 0

    def can_continue(self) -> bool:
        return (
            self.spent_steps < self.max_steps
            and self.spent_observation_chars < self.max_observation_chars
        )


class ToolResult(NamedTuple):
    """工具执行结果：observation 给模型看，chunks 进入检索结果集。

    ⚠️ 是 ``NamedTuple`` 不是 dataclass —— 里面**不能**用 ``field(default_factory=...)``
    （那是 dataclass 的语法，NamedTuple 里会得到一个未展开的 Field 对象，
    调用方迭代时报 ``TypeError: 'Field' object is not iterable``）。
    默认值用具名元组自带的写法：不可变默认 ``()``。
    """

    observation: str
    chunks: tuple = ()


# 正文回目标识：`第一回 …` / `第 12 回 …`。前言、修订说明、附录、版权页都不匹配。
# 序数类问题（「第一个妖怪」）必须把**前后附文**排除掉，否则「前言」里提到
# 妖怪就会成为「最早命中章」—— 实测（2026-09-21）前言确实含「妖怪」，
# 而它显然不是答案所在。这个判断是确定性的，交给代码，不该让模型去猜。
_NARRATIVE_CHAPTER_RE = re.compile(r'^\s*第\s*[一二三四五六七八九十百千零〇\d]+\s*回')


def is_narrative_chapter(chapter_name: str) -> bool:
    """章名是否像「正文回目」（第一回 / 第 12 回）。用于排除前言、附录等。"""
    return bool(_NARRATIVE_CHAPTER_RE.match(str(chapter_name or '')))


def _clip(text: str, limit: int) -> str:
    text = text or ''
    return text if len(text) <= limit else text[:limit] + '…（已截断）'


def coalesce_chapter(chunks: list[Any], max_chars: int = CHAPTER_SEGMENT_CHARS) -> list[dict]:
    """把一章的若干 chunk 合并成少量「分段」，**完整保留章内文本**。

    为什么需要（2026-09-21 实测）：一章有 11~12 个 chunk，而 agent 在最终上下文里
    只有约 8 个名额 —— **一整章塞不下**，必然丢掉一部分，丢哪一段取决于顺序，靠运气。
    实测：第六十六回的谜底（黄眉之名 / 弥勒座下司磬童儿 / 人种袋 / 设瓜田）
    **全部集中在第 8 段**，一旦被丢，answer 就只能答「片段中未出现」。

    合并成分段后，一章只占 2~3 个名额（7200 字 ÷ 4000 ≈ 2 段），
    于是「agent 读过的整章」能确定性地全部进入上下文。
    分段的 ``chunk_id`` 沿用该段首个真实 chunk 的 id，保证引用仍可回溯到原文。
    """
    segments: list[dict] = []
    buffer: list[Any] = []
    buffer_len = 0

    def flush() -> None:
        if not buffer:
            return
        head = buffer[0]
        segments.append({
            'chunk_id': head.id,
            'text': '\n'.join(c.text or '' for c in buffer),
            'score': 0.0,
            'document_id': head.metadata.get('document_id'),
            'chapter': head.metadata.get('chapter'),
            'chapter_index': head.metadata.get('chapter_index'),
            'citation': None,
            'source': 'read_chapter',
            'segment_of': len(buffer),
        })
        buffer.clear()

    for chunk in chunks:
        text_len = len(chunk.text or '')
        if buffer and buffer_len + text_len > max_chars:
            flush()
            buffer_len = 0
        buffer.append(chunk)
        buffer_len += text_len
    flush()
    return segments


def _to_chunk(hit, source: str = '') -> dict:
    """把上游两种检索结果统一成 QA state 里 chunks 的形状。

    ⚠️ 上游有**两个不同**的结果类型，字段名还不一样，这里必须都接住：

    - ``storage.vector_store.SearchHit``（``search_text`` / ``get_chapter`` 返回）：
      ``id`` + ``metadata`` 字典；
    - ``rag.retriever.RetrievedChunk``（``search`` 通道返回）：``chunk_id`` +
      **扁平**的 ``document_id``/``chapter``/``chapter_index`` 字段，没有 ``metadata``。

    实测踩坑（2026-09-21）：只按 ``SearchHit`` 写（``hit.id``）会让 ``search`` 工具
    每次都抛 ``AttributeError`` —— 而工具级 try/except 会把它变成「工具执行失败」的观察，
    于是**检索静默降级**：agent 仍然在跑、日志上只是 ``new_chunks=0``，看不出坏了。
    这类「被兜底吞掉的错」最难查，所以这里显式区分类型而不是靠属性猜测。
    """
    metadata = getattr(hit, 'metadata', None)
    chunk_id = getattr(hit, 'id', None) or getattr(hit, 'chunk_id', None)
    if metadata is None:
        metadata = {
            'document_id': getattr(hit, 'document_id', None),
            'chapter': getattr(hit, 'chapter', None),
            'chapter_index': getattr(hit, 'chapter_index', None),
        }
    return {
        'chunk_id': chunk_id,
        'text': getattr(hit, 'text', '') or '',
        'score': float(getattr(hit, 'score', 0.0) or 0.0),
        'document_id': metadata.get('document_id'),
        'chapter': metadata.get('chapter'),
        'chapter_index': metadata.get('chapter_index'),
        'citation': None,
        # 由哪个工具取到的。合并时要按来源分层（见 merge_agent_chunks），
        # 同时也是有用的观测信息。
        'source': source,
    }


# ============================================================ 工具 schema
# 用 @tool 只为拿到 bind_tools 需要的 schema；真正的执行在 ToolBox 里按 name 分发。
# 这样副作用（chunks 累积）集中管理，也和项目里 final_answer / request_clarification 的写法一致。


@tool
def search(query: str) -> str:
    """按语义检索原文片段。适合你知道大概在讲什么、但不确定原文用词时。

    Args:
        query: 检索语句，用原文可能出现的说法
    """
    return ''


@tool
def grep(term: str) -> str:
    """按字面查找原文里**确实出现**的词，返回命中的章与**书内顺序**。

    凡是涉及「有没有提到过」「第一个/最早/最后一个」「出现几次」这类问题，
    必须用本工具判定——语义检索对这类问题会全部落空。

    **一次可以给多个候选词**（用空格、逗号或 `|` 分隔）。当给多个时，
    返回里会多一段「候选比较」，直接告诉你哪一个**最早出现在正文**。
    「第一个/最早/第 N 个」这类问题强烈建议这样用：先凭你对这本书的了解列出
    2~4 个候选名，再一次查完让工具替你比较先后。

    ⚠️ 不要只用一个**类别词**（如「妖怪」）来回答「第几个」——
    作者在不同章用词不同（实测《西游记》第十三回写寅将军时
    「妖怪/妖魔/妖精」都是 0 次，只有「妖邪」「魔」），
    类别词会把真正最早的那一章整章漏掉。

    Args:
        term: 要查找的词；多个候选词用空格、逗号或 | 分隔（如「寅将军,黄风怪,黑熊精」）
    """
    return ''


@tool
def list_chapters() -> str:
    """列出本书全部章节目录（含章序号）。用于先看清全书结构再决定查哪一章。"""
    return ''


@tool
def read_chapter(chapter_index: int) -> str:
    """把某一章的正文完整读回来（不做相似度过滤）。

    当你已由 grep 或目录定位到具体章、但语义检索拿不到其中细节时使用。

    Args:
        chapter_index: 章序号（来自 list_chapters 或 grep 的返回）
    """
    return ''


@tool
def locate_event(term: str) -> str:
    """定位一个**事件或人物首次出现的回目**，并给出「从这一回起，接下来若干回」的目录。

    **带锚点的序数问题必须先用本工具**（例如「唐僧收孙悟空为徒后，消灭的第一个妖精是谁？」——
    锚点事件是「收孙悟空为徒」）。三步：
    ① 本工具定位锚点 → ② 从锚点回目往后看目录，**紧接的那几回** →
    ③ read_chapter 读最早的候选回，确认第一个被消灭的妖精。

    ⚠️ 为什么不能靠「凭记忆列候选名再 grep」：那依赖你对这本书的先验知识，
    会**漏掉你没想起来的角色**。实测（2026-09-21）：问「收悟空后第一个被消灭的妖精」，
    agent 列的候选是 `黑熊精/白花蛇怪/白骨夫人/黄风怪/虎先锋/凌虚子/玉龙`，
    **漏了第十四回紧接收徒的那只猛虎** —— 而正确答案恰恰是它。

    顺序原则：**按回目顺序往后找，而不是按语义相似度找。**

    Args:
        term: 锚点事件的字面关键词（如「收了孙悟空」「拜师」「两界山」）
    """
    return ''


@tool
def finish(reason: str) -> str:
    """已收集到足够回答问题的原文，或确认书里确实没有，结束检索。

    Args:
        reason: 一句话说明为什么可以结束
    """
    return ''


AGENT_TOOLS = [search, grep, list_chapters, read_chapter, finish]
# ⚠️ `locate_event` 已实现但**刻意不暴露**：
#   实测 2026-09-21，加入它并强制「先定位锚点再往后扫」的流程后，同一道带锚点的
#   序数题从「0/3 转澄清、1/3 答对」**退化为 4/4 转澄清** —— 因为定位锚点本身又是
#   一次「词汇鸿沟」：问「收孙悟空为徒」时正文写的是「心猿归正」，字面找不到，
#   模型于是反复重试 locate_event，把 6 步预算耗尽，根本没能去读正文。
#   教训：**给 agent 加能力之前，先确认那个能力的前提在这个语料上成立。**
#   要复活它，前提是先解决「锚点事件的语义定位」（当前只做了字面定位）。


# ============================================================ 工具执行


@dataclass
class ToolBox:
    """把工具绑定到一次具体问答的语料范围上。

    ⚠️ 全部工具**只读**（不写库、不改状态），这是「节点重放安全」的前提：
    ``interrupt()`` 恢复时 agent 节点可能被整体重跑，只读则重放无副作用。
    """

    vector_store: Any
    search_fn: Callable[[str], list[Any]]  # 注入的语义检索（带 hybrid/阈值等既有行为）
    document_ids: list[int] | None = None
    doc_titles: dict[int, str] = field(default_factory=dict)

    def call(self, name: str, args: dict) -> ToolResult:
        """按 name 分发。任何异常都收敛成结构化错误，绝不冒泡炸掉整次问答。"""
        try:
            handler = getattr(self, f'_tool_{name}', None)
            if handler is None:
                return ToolResult(f'错误：没有名为 {name} 的工具。可用工具：'
                                  'search / grep / list_chapters / read_chapter / finish')
            return handler(args or {})
        except Exception as exc:  # noqa: BLE001 工具失败要让模型看到并可自我修正
            logger.exception('检索agent[工具异常] tool=%s', name)
            return ToolResult(f'错误：{name} 执行失败（{type(exc).__name__}: {exc}）。'
                              '可以换个参数或换个工具重试。')

    # ---- 各工具 ----

    def _tool_search(self, args: dict) -> ToolResult:
        query = str(args.get('query') or '').strip()
        if not query:
            return ToolResult('错误：query 不能为空。')
        hits = self.search_fn(query)
        chunks = [_to_chunk(h, 'search') for h in hits]
        if not hits:
            return ToolResult(
                f'语义检索「{query}」没有任何命中（可能被相似度阈值挡住）。\n'
                '建议改用 grep 按字面查关键词。'
            )
        # 统一走 _to_chunk 的取字段逻辑，别在这里再摸一次上游字段名 ——
        # 2026-09-21 这里曾按 SearchHit 写（hit.metadata），而 search_fn 返回的是
        # RetrievedChunk（扁平字段、无 metadata），导致 search 工具每次都抛异常。
        lines = [f'语义检索「{query}」命中 {len(hits)} 段：']
        for index, chunk in enumerate(chunks[:8], start=1):
            lines.append(
                f'[{index}] 第{chunk["chapter_index"]}章'
                f'《{_clip(str(chunk["chapter"] or ""), 24)}》'
                f' 相似度{chunk["score"]:.3f}\n{_clip(chunk["text"], 220)}'
            )
        return ToolResult('\n'.join(lines), chunks)

    def _tool_grep(self, args: dict) -> ToolResult:
        raw = str(args.get('term') or '').strip()
        if not raw:
            return ToolResult('错误：term 不能为空。')
        # 支持一次给多个候选词（空格 / 逗号 / 、 / | 分隔）——
        # 「第几个」本质是**比较先后**，让工具做比较比让模型心算可靠得多
        # （实测 2026-09-21：模型曾自称「第20-21回」却读的是 25/26 回）。
        terms = [t for t in re.split(r'[\s,、|]+', raw) if t][:6]
        if not terms:
            return ToolResult('错误：term 不能为空。')
        term = terms[0] if len(terms) == 1 else raw

        merged_hits: dict[str, Any] = {}
        per_term_earliest: list[tuple[str, int, str]] = []
        for one in terms:
            one_hits = self.vector_store.search_text(one, self.document_ids, limit=400)
            for hit in one_hits:
                merged_hits.setdefault(hit.id, hit)
            narrative = sorted(
                (int(h.metadata.get('chapter_index')), str(h.metadata.get('chapter') or ''))
                for h in one_hits
                if h.metadata.get('chapter_index') is not None
                and is_narrative_chapter(h.metadata.get('chapter'))
            )
            if narrative:
                per_term_earliest.append((one, narrative[0][0], narrative[0][1]))
        hits = list(merged_hits.values())
        if not hits:
            return ToolResult(
                f'原文里没有任何位置出现「{term}」。\n'
                '可以换一个更常见的说法再试（例如把具体法宝名换成类别词）。'
            )
        # 关键设计：**按章聚合并按书内顺序排列** —— 序数推理（第 N 个）
        # 需要的正是这个「章集合 + 顺序」，而不是一堆片段文本。
        by_doc: dict[int, dict[int, list[Any]]] = {}
        for hit in hits:
            doc_id = hit.metadata.get('document_id')
            chapter_index = hit.metadata.get('chapter_index')
            if doc_id is None or chapter_index is None:
                continue
            by_doc.setdefault(doc_id, {}).setdefault(int(chapter_index), []).append(hit)

        lines = [f'字面查找「{term}」：全书共 {len(hits)} 处命中。']
        for doc_id, chapters in by_doc.items():
            title = self.doc_titles.get(doc_id) or f'文档{doc_id}'
            ordered = sorted(chapters)
            narrative = [ci for ci in ordered
                         if is_narrative_chapter(chapters[ci][0].metadata.get('chapter'))]
            front_matter = [ci for ci in ordered if ci not in narrative]

            lines.append(f'\n《{_clip(title, 30)}》：{len(ordered)} 章命中。')
            if narrative:
                lines.append(
                    f'【正文回目（按书内顺序，共 {len(narrative)} 回）】'
                    '—— 回答「第一个/最早」这类问题，请看这里的第一项：'
                )
                for position, chapter_index in enumerate(narrative, start=1):
                    chapter_hits = chapters[chapter_index]
                    chapter_name = str(chapter_hits[0].metadata.get('chapter') or '')
                    mark = '  ★★★ 这就是全书最早出现的回目' if position == 1 else ''
                    lines.append(
                        f'  [第{position}个命中回] chapter_index={chapter_index}'
                        f' 《{_clip(chapter_name, 30)}》({len(chapter_hits)}处){mark}'
                    )
                    if position <= 3:
                        lines.append(f'      {_clip(chapter_hits[0].text or "", 150)}')
                if len(narrative) > 6:
                    lines.append(f'  ……（另有 {len(narrative) - 6} 回，未列出）')
            if front_matter:
                # 前言/附录里出现该词**不代表**正文讲过 —— 必须显式排除，
                # 否则模型会把「前言」当成最早命中章（实测踩到）。
                names = '、'.join(
                    _clip(str(chapters[ci][0].metadata.get('chapter') or ''), 14)
                    for ci in front_matter[:4]
                )
                lines.append(
                    f'【非正文（前言/附录/版权等，共 {len(front_matter)} 处）】'
                    f'{names} —— 这些**不计入**「第几个」的判断。'
                )
        if len(terms) > 1:
            if per_term_earliest:
                ranked = sorted(per_term_earliest, key=lambda item: item[1])
                lines.append('\n【候选比较】按「最早出现在正文的位置」排序：')
                for position, (one, index, name) in enumerate(ranked, start=1):
                    mark = '  ★★★ 最早，就是它' if position == 1 else ''
                    lines.append(
                        f'  {position}. 「{one}」最早出现于 chapter_index={index}'
                        f' 《{_clip(name, 26)}》{mark}'
                    )
                lines.append(
                    '→ 「第一个/最早」的答案取上面第 1 项。若它与你预期的角色不符，'
                    '请再补充几个候选名重查，不要直接沿用最初检索到的片段。'
                )
            else:
                lines.append('\n【候选比较】以上候选在正文回目里都没有命中。')
        lines.append(
            '\n提示：判断「第 N 个」请用【正文回目】的顺序（它等于书的顺序），'
            '取第一项即最早；确定后用 read_chapter 取该章细节。'
        )
        chunks = [_to_chunk(hit, 'grep') for hit in hits[:8]]
        return ToolResult('\n'.join(lines), chunks)

    def _tool_list_chapters(self, args: dict) -> ToolResult:
        docs = self.document_ids or sorted(self.doc_titles)
        if not docs:
            return ToolResult('错误：当前没有可查的书籍范围。')
        lines = []
        for doc_id in docs:
            chapters = self.vector_store.list_chapters(doc_id)
            if not chapters:
                continue
            title = self.doc_titles.get(doc_id) or f'文档{doc_id}'
            lines.append(f'《{_clip(title, 34)}》共 {len(chapters)} 章：')
            for chapter_index, chapter in chapters:
                lines.append(f'  chapter_index={chapter_index}  {_clip(chapter, 40)}')
        if not lines:
            return ToolResult('该书没有可用的章节信息。')
        return ToolResult('\n'.join(lines))

    def _tool_locate_event(self, args: dict) -> ToolResult:
        """定位锚点事件所在回目，并给出「从该回起接下来若干回」的目录。

        实现上只需要「字面找到最早正文回目 + 附上该回之后的目录」——
        不需要模型先猜候选名，所以不会因为「想不起某个角色」而漏掉。
        它是序数题从「猜候选」转向「按顺序扫」的关键一步。
        """
        term = str(args.get('term') or '').strip()
        if not term:
            return ToolResult('错误：term 不能为空。')
        docs = self.document_ids or sorted(self.doc_titles)
        lines: list[str] = []
        chunks: list[dict] = []
        for doc_id in docs:
            hits = self.vector_store.search_text(term, [doc_id], limit=200)
            narrative = sorted(
                (int(h.metadata.get('chapter_index')), str(h.metadata.get('chapter') or ''))
                for h in hits
                if h.metadata.get('chapter_index') is not None
                and is_narrative_chapter(h.metadata.get('chapter'))
            )
            if not narrative:
                continue
            title = self.doc_titles.get(doc_id) or f'文档{doc_id}'
            anchor_index, anchor_name = narrative[0]
            lines.append(
                f'《{_clip(title, 30)}》：锚点「{term}」最早出现于 **chapter_index={anchor_index}**'
                f' 《{_clip(anchor_name, 30)}》（共 {len(narrative)} 回命中）。'
            )
            # ⚠️ 「首次**提及**」常常早于「事件**发生**」的那一回。
            # 实测（2026-09-21）：查「两界山」时首次提及是第十三回，
            # 而「收孙悟空」这件事发生在第十四回。所以把命中回都列出来，
            # 让模型能看出这个词横跨了哪几回，而不是只信赖第一个。
            if len(narrative) > 1:
                spread = '、'.join(
                    f'{i}({_clip(n, 12)})' for i, n in narrative[:6]
                )
                lines.append(
                    f'  ⚠️ 注意：「{term}」在 {len(narrative)} 回里出现 —— {spread}'
                    f'{"…" if len(narrative) > 6 else ""}。'
                    '「首次提及」**可能早于**「事件发生」的那一回，请结合回目名判断。'
                )
            chapters = self.vector_store.list_chapters(doc_id)
            after = [(i, n) for i, n in chapters
                     if i >= anchor_index and is_narrative_chapter(n)]
            if after:
                lines.append(
                    f'\n【从锚点起、按书内顺序的接下来 {min(len(after), 12)} 回】'
                    '——「锚点之后第一个…」的答案，从**紧接锚点的下一回开始**找：'
                )
                for i, name in after[:12]:
                    tag = '  ← 锚点所在回' if i == anchor_index else ''
                    lines.append(f'  chapter_index={i}  {_clip(name, 40)}{tag}')
                lines.append(
                    '\n→ 请按上面顺序，从锚点回**往下**逐回检查；'
                    '不要凭记忆列候选名（会漏掉没想起来的角色）。'
                    '先用 read_chapter 读最靠前的候选回。'
                )
            anchor_chunks = [
                _to_chunk(h) for h in hits if h.metadata.get('chapter_index') == anchor_index
            ]
            chunks.extend(anchor_chunks[:4])
        if not lines:
            return ToolResult(
                f'没有在正文回目里找到「{term}」。可以换个说法（如改用书名里出现的人名/地名）再试。'
            )
        return ToolResult('\n'.join(lines), chunks)

    def _tool_read_chapter(self, args: dict) -> ToolResult:
        raw = args.get('chapter_index')
        try:
            chapter_index = int(raw)
        except (TypeError, ValueError):
            return ToolResult(f'错误：chapter_index 必须是整数，收到 {raw!r}。')
        docs = self.document_ids or sorted(self.doc_titles)
        found = []
        for doc_id in docs:
            found.extend(self.vector_store.get_chapter(doc_id, chapter_index))
        if not found:
            return ToolResult(
                f'没有找到 chapter_index={chapter_index} 的章节。'
                '请先用 list_chapters 确认章序号。'
            )
        text = '\n'.join(c.text or '' for c in found)
        chapter_name = str(found[0].metadata.get('chapter') or '')

        # ⚠️ **不要截断成「前 N 段」** —— 这是 2026-09-21 实测踩到的致命点：
        # 原实现 `found[:6]` 只回前 6 段，而第六章十六回（小雷音寺）的谜底
        # ——妖怪名（黄眉）、来历（弥勒座下司磬童儿）、法宝（人种袋）、结局（设瓜田）
        # ——**全部集中在第 8 段**，正好被切掉。于是 agent 的结论是对的（它读到了），
        # 但进入最终上下文的正文里没有答案，answer 只能答「片段中未出现」。
        # 章内 chunk 是切分产物、顺序＝书内顺序，**答案没有理由偏在前半章**。
        chunks = coalesce_chapter(found)
        # observation 给「分段开头」的概览（模型据此判断够不够），
        # **完整正文进 chunks**（供 answer 使用）。两者职责不同，不要混。
        preview = '\n'.join(
            f'  [{i + 1}/{len(found)}] {_clip(c.text or "", 200)}'
            for i, c in enumerate(found)
        )
        return ToolResult(
            f'第{chapter_index}章《{_clip(chapter_name, 34)}》全文已取回'
            f'（共 {len(found)} 段、{len(text)} 字）——**完整正文已进入你的可用原文片段**，'
            f'下面只是每段开头，便于你判断还需不需要读别的章：\n'
            f'{_clip(preview, MAX_CHAPTER_CHARS * 2)}',
            chunks,
        )

    def _tool_finish(self, args: dict) -> ToolResult:
        return ToolResult(f'（结束检索：{args.get("reason") or "已收集足够原文"}）')


# ============================================================ ReAct 循环


SYSTEM_PROMPT = """你是阅读助手的检索策略模块。你的唯一任务是**找到能回答用户问题的原文片段**，
不负责撰写最终回答。

你可以调用工具去查书。

## 遇到「第一个 / 最早 / 最后 / 第 N 个」这类问题时，必须按下面三步做
## 【多候选枚举流程】（B 类，全书第一个）

1. **列候选**：先凭你对这本书的了解，写出 2~4 个**具体候选名**
   （例如问「第一个妖怪」→ 寅将军、黑熊精、黄风怪）。
2. **一次查完让工具比较**：`grep("寅将军,黄风怪,黑熊精")` —— 用空格/逗号/`|` 分隔多个词。
   返回里会有【候选比较】，**直接按「最早出现在正文的位置」排好序**，第 1 项就是答案。
3. **读那一章**：read_chapter 取细节（特点、结局等）。

⚠️ **两个最容易犯的错，都会导致答错：**

- **不要只用类别词**。作者在不同章用词不同：实测《西游记》第十三回（写寅将军）
  里「妖怪 / 妖魔 / 妖精」出现 **0 次**，只有「妖邪」「魔」。
  只用「妖怪」这个词搜，会把真正最早的那一章**整章漏掉**，然后把你引向第三回或第十六回。
  → 必须用**具体名字**，或多个同义词一起查。
- **不要把最初检索到的片段当依据**。那些片段只是「语义上最像问题」，**不等于「书里最早」**。
  实测（2026-09-21）：问「第一个妖怪」时初始片段全是第二十回黄风岭的内容（因为问法像），
  顺着它答「黄风怪」就错了 —— 真正的第一个是第十三回的寅将军。
  **判断「第几个」只能依据 grep 给出的回目顺序。**

## 其它规则

- 「有没有提到过」「出现几次」同样以 grep 为准 —— 语义检索对这类问题会全部落空，
  因为答案段落里不会出现「第一个」这种词。
- search 只在你知道大概内容、但不确定原文用词时使用。
- 拿到足以回答的原文后就调用 finish，不要过度检索。每多一步都会增加用户等待。
- 若多次尝试后确认书里确实没有相关内容，也调用 finish 并说明「书中未找到」。

请逐步思考：先想清楚缺什么信息，再选最合适的工具。"""


def merge_agent_chunks(
    agent_chunks: list[dict], originals: list[dict], cap: int
) -> list[dict]:
    """合并 agent 片段与原有片段，**保证原有片段不被清空、且按来源分层取用**。

    ## 两条各自独立的规则（都来自实测失败）

    **规则一：原有片段留名额。**
    原来「agent 在前、截断到 cap」在 agent 走错方向时会把 28 条塞满 16 个名额、
    **整体挤出原有检索结果** —— 「agent 失败」于是升级成「agent 失败且销毁了原有证据」。
    所以预留 `cap//2`（原有片段是 pre-agent 流水线的产出，可能与答案相关）。

    **规则二：`read_chapter` 的片段优先于 `grep`/`search` 的片段。**
    这条是 2026-09-21 第二轮踩出来的。原先改成「按 agent 轨迹取最新 N 条」，
    但**最新那一步常常是收尾用的 `search`**，它会把前面 `read_chapter` 的章正文挤掉。
    实测同一题（小雷音寺）跑 5 次只对 3 次，失败的两次最终上下文里
    **agent 读过的那两章正文一条都没进**（或只进了半章）。

    为什么可以按来源分层：两者的**证据等级不同** ——
    - `read_chapter` 是 agent **主动选定、且该章完整取回**（自足、无相似度过滤）→ 高置信；
    - `grep` / `search` 返回的是**候选**（宽网、含噪）→ 低置信。

    而 agent 的结论正是在读完章之后才得出的（finish 的 reason 里引用的也是回目）。
    所以：**先放读完的章，再用候选填剩余名额。**

    分层内部一律「后出优先」—— agent 轨迹是收敛的，越晚越接近结论。
    原有片段保持原顺序（它们是召回导向的）。
    """
    if cap <= 0:
        return []
    reserve = min(len(originals), max(1, cap // 2))
    # agent 最多占 cap - reserve，**把 reserve 个名额硬留给原有片段**。
    agent_budget = max(0, cap - reserve)

    read_chunks = [c for c in agent_chunks if c.get('source') == 'read_chapter']
    other_chunks = [c for c in agent_chunks if c.get('source') != 'read_chapter']
    ordered_agent = list(reversed(read_chunks)) + list(reversed(other_chunks))

    picked: list[dict] = []
    seen: set[str] = set()

    def _take(chunks) -> None:
        for chunk in chunks:
            if len(picked) >= cap:
                return
            chunk_id = str(chunk.get('chunk_id') or '')
            if not chunk_id or chunk_id in seen:
                continue
            seen.add(chunk_id)
            picked.append(chunk)

    # ① agent 片段（上限 agent_budget）② 原有片段（拿到 reserve）③ 空位回填 agent
    #
    # ⚠️ 这里**必须显式分三段取**，不能写成「head 截断 + tail 追加」。
    # 2026-09-21~22 实测：旧写法 `head[:cap - len(tail)] + tail` 里，
    # `head` 仍然包含**全部** agent 片段（它们占着最前面几个位置），
    # 于是截到 13 条时 agent 的 11 条原封不动、只给原有片段留下 2 个空位 →
    # reserve=8 的意图落空，实际只保 5 条。
    # 真实日志证据：`合并后 16 条（其中原有 5 条）`（agent 11 条、原有 12 条、cap 16）。
    _take(ordered_agent[:agent_budget])
    _take(originals)
    _take(ordered_agent)
    return picked[:cap]


@dataclass
class AgentRun:
    """一次 agent 循环的结果。"""

    chunks: list[dict] = field(default_factory=list)
    trace: list[dict] = field(default_factory=list)
    steps: int = 0
    finished: bool = False
    finish_reason: str = ''
    stopped_by: str = ''  # finish | no_tool_call | budget | error


def run_agent_loop(
    *,
    model: Any,
    toolbox: ToolBox,
    question: str,
    history: list[dict] | None = None,
    budget: AgentBudget | None = None,
    seed_chunks: list[dict] | None = None,
) -> AgentRun:
    """执行有界 ReAct 循环：模型选动作 → 执行 → 观察回灌 → 再决策。

    终止条件（全部硬编码，不依赖模型自觉）：
    - 模型调用 ``finish``；
    - 模型没有返回工具调用（视为它认为够了）；
    - 预算耗尽（``max_steps`` / 观察字符总量）；
    - 模型调用异常。
    """
    budget = budget or AgentBudget()
    run = AgentRun()

    context_lines = [f'用户问题：{question}']
    if history:
        recent = '\n'.join(
            f'{"用户" if m.get("role") == "user" else "助手"}：'
            f'{_clip(str(m.get("content") or ""), 120)}'
            for m in history[-4:]
        )
        context_lines.append(f'【最近对话（用于理解指代）】\n{recent}')
    if toolbox.doc_titles:
        titles = '、'.join(
            f'文档{doc_id}《{_clip(str(name), 34)}》'
            for doc_id, name in toolbox.doc_titles.items()
        )
        context_lines.append(f'【可查范围】{titles}')
    if seed_chunks:
        preview = '\n'.join(
            f'- 第{c.get("chapter_index")}章《{_clip(str(c.get("chapter") or ""), 24)}》：'
            f'{_clip(str(c.get("text") or ""), 100)}'
            for c in seed_chunks[:4]
        )
        context_lines.append(
            f'【初次语义检索已拿到的片段（判断是否已经够了）】\n{preview}'
        )
    context_lines.append('请决定下一步动作。')

    messages: list[Any] = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content='\n\n'.join(context_lines)),
    ]

    bound = model.bind_tools(AGENT_TOOLS) if hasattr(model, 'bind_tools') else model

    while budget.can_continue():
        start = time.perf_counter()
        try:
            response = bound.invoke(messages)
        except Exception:  # noqa: BLE001 —— agent 是增强项，失败要能回退
            logger.exception('检索agent[模型异常] 提前结束，沿用已有片段')
            run.stopped_by = 'error'
            break

        calls = getattr(response, 'tool_calls', None) or []
        if not calls:
            run.stopped_by = 'no_tool_call'
            break

        budget.spent_steps += 1
        run.steps += 1

        # ⚠️ 必须原样回填模型给出的**每一个** tool_call。
        # OpenAI 兼容协议（DeepSeek 也一样）要求「带 tool_calls 的 assistant 消息」后面
        # 必须紧跟**逐条对应**的 tool 消息，否则下一次请求直接 400：
        #   "An assistant message with 'tool_calls' must be followed by tool messages
        #    responding to each 'tool_call_id'"
        # 实测（2026-09-21）模型很自然地一次返回两个（search + list_chapters），
        # 只回一个就炸 —— 即使 prompt 里写了「一次只调用一个工具」也不能指望它遵守。
        messages.append(response if isinstance(response, AIMessage) else AIMessage(
            content=getattr(response, 'content', '') or '', tool_calls=list(calls)))

        finishing = False
        for call in calls[:MAX_TOOL_CALLS_PER_ROUND]:
            name = call.get('name') or ''
            args = call.get('args') or {}
            result = toolbox.call(name, args)

            if name == 'finish':
                run.finished = True
                run.finish_reason = str(args.get('reason') or '')
                run.stopped_by = 'finish'
                finishing = True
                observation = _clip(result.observation, MAX_OBSERVATION_CHARS)
                logger.info(
                    '检索agent[finish] steps=%d reason=%s',
                    run.steps, _clip(run.finish_reason, 80),
                )
            else:
                observation = _clip(result.observation, MAX_OBSERVATION_CHARS)
                budget.spent_observation_chars += len(observation)
                run.chunks.extend(result.chunks)
                run.trace.append({
                    'step': run.steps,
                    'action': name,
                    'args': {k: _clip(str(v), 80) for k, v in args.items()},
                    'observation_chars': len(observation),
                    'new_chunks': len(result.chunks),
                    'cost_ms': round((time.perf_counter() - start) * 1000),
                })
                logger.info(
                    '检索agent[step %d] action=%s args=%s obs=%d字 new_chunks=%d (%.0fms)',
                    run.steps, name, _clip(str(args), 70), len(observation),
                    len(result.chunks), (time.perf_counter() - start) * 1000,
                )
            messages.append(ToolMessage(
                content=observation, tool_call_id=call.get('id') or '',
            ))

        # 超出本轮上限的 tool_call 也必须各回一条，否则下一轮请求同样 400
        for call in calls[MAX_TOOL_CALLS_PER_ROUND:]:
            messages.append(ToolMessage(
                content='（本轮动作数已达上限，此动作被跳过）',
                tool_call_id=call.get('id') or '',
            ))

        if finishing:
            break

    if not run.stopped_by:
        run.stopped_by = 'budget'
        logger.info('检索agent[预算耗尽] steps=%d obs=%d字',
                    run.steps, budget.spent_observation_chars)

    # 去重（同一 chunk 可能被多个工具取到），保持「先取到的优先」。
    deduped: list[dict] = []
    seen: set[str] = set()
    for chunk in run.chunks:
        chunk_id = str(chunk.get('chunk_id') or '')
        if not chunk_id or chunk_id in seen:
            continue
        seen.add(chunk_id)
        deduped.append(chunk)
    # ⚠️ **这里不能按 max_chunks 截断** —— 这是 2026-09-21 实测的致命点：
    # 原实现 `deduped[:budget.max_chunks]` 在 merge **之前**就把 agent 的片段截到 16 条，
    # 而且取的是**最先累积**的（grep 的候选 + 第一次 read_chapter 的前半）。
    # 于是 agent 随后读到的第二章（含答案的那一章）**根本到不了 merge**。
    # 实测：工具共贡献 31 条（grep 8 + 章A 11 + 章B 12），到 merge 只剩 16 条，
    # 而答案在「章B 第 8 段」—— 被这层截断吃掉了。
    #
    # 职责划分：**累积不设限（只是些轻量 dict），最终裁剪只由 merge_agent_chunks 负责**
    # —— 它知道来源优先级与「给原有片段留名额」，这里截断只会让它拿不到完整信息。
    run.chunks = deduped
    return run
