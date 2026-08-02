"""文本分块：按 separators 切分并保留章节元数据。"""

from dataclasses import dataclass, field

from langchain_text_splitters import RecursiveCharacterTextSplitter

from reading_assistant.config import get_settings
from reading_assistant.parsers import ParsedBook


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


def chunk_book(
    book: ParsedBook,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    separators: list[str] | None = None,
) -> list[TextChunk]:
    """按章节分块，在元数据中记录章节序号与标题。"""
    chunks: list[TextChunk] = []
    index = 0
    for chapter_index, chapter in enumerate(book.chapters):
        for piece in chunk_text(chapter.content, chunk_size, chunk_overlap, separators):
            chunks.append(
                TextChunk(
                    text=piece,
                    index=index,
                    metadata={'chapter_index': chapter_index, 'chapter': chapter.title},
                )
            )
            index += 1
    return chunks
