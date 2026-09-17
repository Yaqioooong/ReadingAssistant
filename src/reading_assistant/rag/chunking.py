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

    表格块（由哨兵包裹）作为不可切分单元整块入库：单行不被腰斩、
    单元格数据不散落到不同 chunk；代价是超长表格块不参与 800 字切分
    （对常见书籍表格可接受）。
    """
    chunks: list[TextChunk] = []
    index = 0
    for chapter_index, chapter in enumerate(book.chapters):
        for kind, segment in _split_table_blocks(chapter.content):
            if kind == 'table':
                chunks.append(
                    TextChunk(
                        text=segment,
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
