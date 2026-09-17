"""文本分块：按 separators 切分并保留章节元数据。"""

import re
from dataclasses import dataclass, field

from langchain_text_splitters import RecursiveCharacterTextSplitter

from reading_assistant.config import get_settings
from reading_assistant.parsers import ParsedBook
from reading_assistant.parsers.base import TABLE_BEGIN, TABLE_END


@dataclass
class TextChunk:
    """一个分块：文本 + 序号 + 元数据（章节等）。"""

    text: str
    index: int
    metadata: dict = field(default_factory=dict)


def chunk_text(
    text: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    separators: list[str] | None = None,
) -> list[str]:
    """将单段文本切分为若干块；空文本返回空列表。"""
    if not text or not text.strip():
        return []
    settings = get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or settings.chunk_size,
        chunk_overlap=chunk_overlap if chunk_overlap is not None else settings.chunk_overlap,
        separators=separators if separators is not None else settings.separators,
        keep_separator=False,
    )
    return [piece for piece in splitter.split_text(text) if piece.strip()]


# 表格是"不可切分单元"——整表作为单个 chunk 入库，避免被分块器从行中间腰斩。
_TABLE_BLOCK_RE = re.compile(
    r'\s*' + re.escape(TABLE_BEGIN) + r'\n(.*?)\n\s*' + re.escape(TABLE_END), re.S
)

# 表格 chunk 字符上限。DashScope text-embedding-v4 单条输入上限 8192 token，
# 中文按 1 字≈1~2 token 折算，取 3000 字符留足安全余量：
# 超过该上限的整表直接送 embed_documents 会被批量调用整体 400，导致整本书 failed。
_MAX_TABLE_CHARS = 3000

# 未转义的单元格分隔符（docx 解析器会把正文里的 '|' 转义成 '\|'）。
_CELL_SPLIT_RE = re.compile(r'(?<!\\)\|')


def _render_row(cells: list[str]) -> str:
    """把单元格列表渲染成 Markdown 数据行。"""
    return '| ' + ' | '.join(cells) + ' |'


def _split_cells(line: str) -> list[str]:
    """拆出一行的单元格（去首尾竖线）。"""
    stripped = line.strip()
    if stripped.startswith('|'):
        stripped = stripped[1:]
    if stripped.endswith('|'):
        stripped = stripped[:-1]
    return [cell.strip() for cell in _CELL_SPLIT_RE.split(stripped)]


def _parse_table(table: str) -> tuple[str, str, list[str]] | None:
    """解析 Markdown 表格 → (表头行, 分隔行, 数据行)。非标准表格返回 None。"""
    lines = [line.strip() for line in table.splitlines() if line.strip()]
    if len(lines) < 2 or not lines[0].startswith('|') or not lines[1].startswith('|'):
        return None
    return lines[0], lines[1], lines[2:]


def _split_wide_cells(header: str, cells: list[str]) -> list[str]:
    """超长行兜底：按单元格边界分组，每组仍带表头 + 分隔行。"""
    if not cells:
        return [header]
    header_cells = _split_cells(header)
    separator = _render_row(['---'] * max(1, len(header_cells)))
    budget = len(header) + len(separator) + 2
    parts: list[str] = []
    group: list[str] = []
    group_len = budget
    for cell in cells:
        piece_len = len(cell) + 3
        if group and group_len + piece_len > _MAX_TABLE_CHARS:
            parts.append('\n'.join([header, separator, _render_row(group)]))
            group = []
            group_len = budget
        group.append(cell)
        group_len += piece_len
    if group:
        parts.append('\n'.join([header, separator, _render_row(group)]))
    return parts


def _split_table(table: str) -> list[str]:
    """表头感知切分：`> _MAX_TABLE_CHARS` 时按行分批，每批重复表头 + 分隔行。

    - 不超过上限：整表保持原子（调用方原行为）。
    - 超过上限：按行分批，**不从行中间切**；每批都重复 Markdown 表头 + 分隔行。
    - 单个超长行：最后兜底才在单元格边界切，且仍带表头。
    """
    header, separator, rows = _parse_table(table)
    prefix = f'{header}\n{separator}'

    if not rows:
        # 只有表头 + 分隔行：整体超限说明列极多，按单元格切
        return _split_wide_cells(header, _split_cells(header))

    parts: list[str] = []
    group: list[str] = []
    group_len = len(prefix)
    for row in rows:
        row_len = len(row) + 1
        if group and group_len + row_len > _MAX_TABLE_CHARS:
            parts.append('\n'.join([header, separator, *group]))
            group = []
            group_len = len(prefix)
        if not group and group_len + row_len > _MAX_TABLE_CHARS:
            # 单行本身（含表头）就超限 → 单元格边界兜底
            parts.extend(_split_wide_cells(header, _split_cells(row)))
            continue
        group.append(row)
        group_len += row_len
    if group:
        parts.append('\n'.join([header, separator, *group]))
    return parts


def _split_table_blocks(content: str) -> list[tuple[str, str]]:
    """把章节内容切成 (kind, text) 段：'table' 原子块 / 'text' 普通段落。"""
    segments: list[tuple[str, str]] = []
    pos = 0
    for match in _TABLE_BLOCK_RE.finditer(content):
        before = content[pos : match.start()]
        if before.strip():
            segments.append(('text', before))
        table = match.group(1).strip()
        if table:
            segments.append(('table', table))
        pos = match.end()
    tail = content[pos:]
    if tail.strip():
        segments.append(('text', tail))
    return segments


def make_chunk_id(document_id: int, content_hash: str, index: int) -> str:
    """内容寻址 + 版本化 chunk id：``doc{document_id}-{content_hash[:8]}-{index}``。

    ``content_hash`` 为文档版本指纹（``documents.content_hash``）。相对旧的
    位置型 id（``doc{document_id}-{index}``），版本段让「同书不同版本」落在
    不同 id 命名空间，从而可安全地「先写新版本、再删旧版本」。
    """
    return f'doc{document_id}-{(content_hash or "")[:8]}-{index}'


def chunk_book(
    book: ParsedBook,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    separators: list[str] | None = None,
) -> list[TextChunk]:
    """按章节分块，在元数据中记录章节序号与标题。

    表格块（由哨兵包裹）默认作为不可切分单元整块入库（单行不被腰斩、
    单元格数据不散落到不同 chunk）；仅当整表超过 ``_MAX_TABLE_CHARS`` 时，
    才做「表头感知」分批：按行分批且每批重复表头 + 分隔行，保证任何
    产出 chunk 都不超出 embedding 模型单条输入上限。
    """
    chunks: list[TextChunk] = []
    index = 0
    for chapter_index, chapter in enumerate(book.chapters):
        for kind, segment in _split_table_blocks(chapter.content):
            if kind == 'table':
                pieces = [segment] if len(segment) <= _MAX_TABLE_CHARS else _split_table(segment)
                for piece in pieces:
                    chunks.append(
                        TextChunk(
                            text=piece,
                            index=index,
                            metadata={
                                'chapter_index': chapter_index,
                                'chapter': chapter.title,
                                'block_type': 'table',
                            },
                        )
                    )
                    index += 1
                continue
            for piece in chunk_text(segment, chunk_size, chunk_overlap, separators):
                chunks.append(
                    TextChunk(
                        text=piece,
                        index=index,
                        metadata={'chapter_index': chapter_index, 'chapter': chapter.title},
                    )
                )
                index += 1
    return chunks
