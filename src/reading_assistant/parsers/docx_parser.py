"""DOCX 解析器：基于 python-docx，按 Heading 样式切分章节。"""

from pathlib import Path

from docx import Document
from docx.opc.exceptions import PackageNotFoundError
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph

from reading_assistant.parsers.base import (
    TABLE_BEGIN,
    TABLE_END,
    BookParser,
    Chapter,
    ParsedBook,
    ParseError,
    resolve_book_path,
)


class DocxParser(BookParser):
    extensions = frozenset({'.docx'})

    def parse(self, path: str | Path) -> ParsedBook:
        path = resolve_book_path(path)
        if not path.is_file():
            raise ParseError(f'文件不存在: {path}')
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

        for block in _iter_block_items(document):
            if isinstance(block, Paragraph):
                style_name = (block.style.name or '').lower() if block.style else ''
                is_heading = style_name.startswith('heading')
                text = block.text.strip()
                if is_heading and text:
                    flush()
                    current_title = text
                elif block.text:
                    current_lines.append(block.text)
            elif isinstance(block, Table):
                markdown = _table_to_markdown(block)
                if markdown:
                    # 哨兵包裹：chunk 层识别为不可切分原子块
                    current_lines.append(f'{TABLE_BEGIN}\n{markdown}\n{TABLE_END}')
        flush()

        chapters = [chapter for chapter in chapters if chapter.content]
        if not chapters:
            raise ParseError(f'DOCX 无可提取文本: {path}')

        title = document.core_properties.title or path.stem
        author = document.core_properties.author or None
        return ParsedBook(title=title, author=author, chapters=chapters)


def _iter_block_items(parent):
    """按文档 body 顺序产出段落与表格（保持图文/表格交错的原序）。"""
    body = parent.element.body
    for child in body.iterchildren():
        if child.tag == qn('w:p'):
            yield Paragraph(child, parent)
        elif child.tag == qn('w:tbl'):
            yield Table(child, parent)


def _cell_text(cell) -> str:
    text = ' '.join((para.text or '').strip() for para in cell.paragraphs if para.text)
    return text.replace('|', '\\|').replace('\n', ' ').strip()


def _table_to_markdown(table: Table) -> str:
    """python-docx 表格 → Markdown 表格（含表头分隔行）。"""
    rows = [[_cell_text(cell) for cell in row.cells] for row in table.rows]
    if not rows or not rows[0]:
        return ''
    header = rows[0]
    body = rows[1:]
    columns = len(header)
    lines = ['| ' + ' | '.join(header) + ' |']
    lines.append('| ' + ' | '.join(['---'] * columns) + ' |')
    for row in body:
        padded = (row + [''] * columns)[:columns]
        lines.append('| ' + ' | '.join(padded) + ' |')
    return '\n'.join(lines)
