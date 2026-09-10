"""PDF 解析器：基于 pdfplumber（布局感知 + 表格检测），pypdf 兜底。

- 正文优先走 pdfplumber：无表格页取布局文本；有表格页把表格区域从
  文字流中剔除，表格转 Markdown 后以哨兵包裹（chunk 层整表原子入库）。
- pdfplumber 对极小/非常规 PDF 可能失败（如测试用手工构造的最小文件），
  此时回退 pypdf 纯文本提取，行为与旧版一致。
- 加密/损坏检测仍由 pypdf 完成（错误语义不变）。
"""

from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from reading_assistant.parsers.base import (
    TABLE_BEGIN,
    TABLE_END,
    BookParser,
    Chapter,
    EncryptedFileError,
    ParsedBook,
    ParseError,
    resolve_book_path,
)


class PdfParser(BookParser):
    extensions = frozenset({'.pdf'})

    def parse(self, path: str | Path) -> ParsedBook:
        path = resolve_book_path(path)
        if not path.is_file():
            raise ParseError(f'文件不存在: {path}')
        try:
            reader = PdfReader(str(path))
        except PdfReadError as exc:
            raise ParseError(f'PDF 解析失败（文件损坏）: {path}') from exc
        except Exception as exc:
            raise ParseError(f'PDF 解析失败: {path}') from exc

        if reader.is_encrypted:
            raise EncryptedFileError(f'PDF 已加密，暂不支持密码解锁: {path}')

        text = self._extract_text(path)
        if not text:
            raise ParseError(f'PDF 无可提取文本: {path}')

        title = path.stem
        author = None
        try:
            if reader.metadata is not None:
                title = reader.metadata.title or title
                author = reader.metadata.author or None
        except Exception:
            pass
        return ParsedBook(title=title, author=author, chapters=[Chapter('正文', text)])

    @staticmethod
    def _extract_text(path: str | Path) -> str:
        """pdfplumber 主路径；失败/无可提取时回退 pypdf 纯文本。"""
        segments: list[str] = []
        try:
            import pdfplumber

            with pdfplumber.open(str(path)) as pdf:
                for page in pdf.pages:
                    try:
                        seg = _page_content(page)
                    except Exception:  # noqa: BLE001 —— 单页失败不拖垮全书
                        seg = ''
                    if seg:
                        segments.append(seg)
        except Exception:  # noqa: BLE001 —— pdfplumber 打不开就走 pypdf
            segments = []
        if segments:
            return '\n\n'.join(segments).strip()

        # 回退：pypdf 纯文本（旧行为）
        try:
            reader = PdfReader(str(path))
            pages = []
            for page in reader.pages:
                try:
                    pages.append(page.extract_text() or '')
                except Exception:  # noqa: BLE001
                    pages.append('')
            return '\n'.join(pages).strip()
        except Exception:  # noqa: BLE001
            return ''


def _page_content(page) -> str:
    """单页内容：正文（剔除表格区域）+ 表格 Markdown 块（哨兵包裹）。"""
    plain = page.extract_text() or ''
    try:
        tables = page.find_tables()
    except Exception:  # noqa: BLE001
        tables = []
    if not tables:
        return plain

    kept_boxes: list[tuple] = []
    table_blocks: list[str] = []
    for table in tables:
        try:
            rows = table.extract()
        except Exception:  # noqa: BLE001
            continue
        markdown = _rows_to_markdown(rows)
        if not markdown:
            continue
        kept_boxes.append(table.bbox)
        table_blocks.append(f'{TABLE_BEGIN}\n{markdown}\n{TABLE_END}')

    if not kept_boxes:
        return plain

    # 剔除表格区域内的单词后重构正文行，避免表格文本重复入库
    words = []
    try:
        words = page.extract_words() or []
    except Exception:  # noqa: BLE001
        words = []
    outside = [
        word for word in words if not any(_word_in_bbox(word, box) for box in kept_boxes)
    ]
    prose = '\n'.join(_rebuild_lines(outside)).strip()
    parts = [prose] if prose else []
    parts.extend(table_blocks)
    return '\n\n'.join(parts)


def _word_in_bbox(word: dict, bbox: tuple) -> bool:
    x0, top, x1, bottom = bbox
    cx = (word['x0'] + word['x1']) / 2.0
    cy = (word['top'] + word['bottom']) / 2.0
    return x0 <= cx <= x1 and top <= cy <= bottom


def _rebuild_lines(words: list[dict]) -> list[str]:
    """按纵坐标分组、横向排序重建文本行（近似阅读顺序）。"""
    if not words:
        return []
    ordered = sorted(words, key=lambda w: (w['top'], w['x0']))
    lines: list[str] = []
    current_words: list[dict] = []
    line_top: float | None = None
    for word in ordered:
        if line_top is None or abs(word['top'] - line_top) <= 3.0:
            current_words.append(word)
            if line_top is None:
                line_top = word['top']
        else:
            lines.append(' '.join(w['text'] for w in current_words))
            current_words = [word]
            line_top = word['top']
    if current_words:
        lines.append(' '.join(w['text'] for w in current_words))
    return lines


def _rows_to_markdown(rows: list[list]) -> str:
    """pdfplumber 表格行 → Markdown 表格；退化/空表返回空串。"""
    cleaned: list[list[str]] = []
    for row in rows:
        cells = [
            (cell or '').replace('\n', ' ').replace('|', '\\|').strip() for cell in row
        ]
        if any(cells):
            cleaned.append(cells)
    if not cleaned:
        return ''
    columns = max(len(row) for row in cleaned)
    if len(cleaned) < 2 or columns < 2:
        return ''  # 单行/单列多为 find_tables 误报，弃用
    lines = ['| ' + ' | '.join(cleaned[0]) + ' |']
    lines.append('| ' + ' | '.join(['---'] * columns) + ' |')
    for row in cleaned[1:]:
        padded = (row + [''] * columns)[:columns]
        lines.append('| ' + ' | '.join(padded) + ' |')
    return '\n'.join(lines)
