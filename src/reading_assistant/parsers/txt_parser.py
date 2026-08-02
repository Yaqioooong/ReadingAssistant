"""TXT 解析器：自动探测编码，按章节标题切分。"""

import re
from pathlib import Path

from reading_assistant.parsers.base import BookParser, Chapter, ParsedBook, ParseError

# 匹配形如 "第一章 开端"、"第1章"、"Chapter 2"、"序章" 等章节标题行
_CHAPTER_RE = re.compile(
    r'^\s*(?:第[0-9一二三四五六七八九十百千万零]+[章回节卷部]|'
    r'Chapter\s+\d+|CHAPTER\s+\d+|序章|楔子|前言|后记|尾声)(?:\s+.*)?\s*$'
)

# 常见中文电子书编码，按优先级尝试
_ENCODINGS = ('utf-8', 'gb18030', 'big5', 'latin-1')


class TxtParser(BookParser):
    extensions = frozenset({'.txt'})

    def parse(self, path: str | Path) -> ParsedBook:
        path = Path(path)
        if not path.is_file():
            raise ParseError(f'文件不存在: {path}')
        text = self._read_text(path)
        if not text.strip():
            raise ParseError(f'文件为空: {path}')
        chapters = self._split_chapters(text)
        return ParsedBook(title=path.stem, chapters=chapters or [Chapter('正文', text.strip())])

    @staticmethod
    def _read_text(path: Path) -> str:
        for encoding in _ENCODINGS:
            try:
                return path.read_text(encoding=encoding)
            except (UnicodeDecodeError, UnicodeError):
                continue
        raise ParseError(f'无法识别文件编码: {path}')

    @staticmethod
    def _split_chapters(text: str) -> list[Chapter]:
        chapters: list[Chapter] = []
        current_title = ''
        current_lines: list[str] = []

        def flush() -> None:
            nonlocal current_title, current_lines
            content = '\n'.join(current_lines).strip()
            if current_title or content:
                chapters.append(Chapter(current_title or '正文', content))
            current_title = ''
            current_lines = []

        for line in text.splitlines():
            stripped = line.strip()
            if stripped and _CHAPTER_RE.match(stripped):
                flush()
                current_title = stripped
            else:
                current_lines.append(line)
        flush()
        return [chapter for chapter in chapters if chapter.content]
