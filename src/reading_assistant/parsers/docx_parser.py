"""DOCX 解析器：基于 python-docx，按 Heading 样式切分章节。"""

from pathlib import Path

from docx import Document
from docx.opc.exceptions import PackageNotFoundError

from reading_assistant.parsers.base import BookParser, Chapter, ParsedBook, ParseError


class DocxParser(BookParser):
    extensions = frozenset({'.docx'})

    def parse(self, path: str | Path) -> ParsedBook:
        path = Path(path)
        try:
            document = Document(str(path))
        except PackageNotFoundError as exc:
            raise ParseError(f'DOCX 解析失败（文件损坏）: {path}') from exc
        except Exception as exc:
            raise ParseError(f'DOCX 解析失败: {path}') from exc

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

        for para in document.paragraphs:
            style_name = (para.style.name or '').lower() if para.style else ''
            is_heading = style_name.startswith('heading')
            text = para.text.strip()
            if is_heading and text:
                flush()
                current_title = text
            elif para.text:
                current_lines.append(para.text)
        flush()

        chapters = [chapter for chapter in chapters if chapter.content]
        if not chapters:
            raise ParseError(f'DOCX 无可提取文本: {path}')

        title = document.core_properties.title or path.stem
        author = document.core_properties.author or None
        return ParsedBook(title=title, author=author, chapters=chapters)
