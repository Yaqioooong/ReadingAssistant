"""PDF 表格解析测试：行转 Markdown 助手 + sample_book.pdf 回归。"""

from pathlib import Path

from reading_assistant.parsers.base import TABLE_BEGIN
from reading_assistant.parsers.pdf_parser import PdfParser, _rows_to_markdown

REPO = Path(__file__).resolve().parents[1]


class TestRowsToMarkdown:
    def test_normal_table(self) -> None:
        md = _rows_to_markdown([['人物', '年份'], ['少白公', '1966'], ['舵手', '1969']])
        assert md.startswith('| 人物 | 年份 |')
        assert '| --- | --- |' in md
        assert '| 少白公 | 1966 |' in md

    def test_empty_and_single_row_rejected(self) -> None:
        assert _rows_to_markdown([]) == ''
        assert _rows_to_markdown([['a', 'b']]) == ''  # 无表体，误报

    def test_ragged_rows_padded(self) -> None:
        md = _rows_to_markdown([['a', 'b', 'c'], ['x']])
        assert '| x |  |  |' in md

    def test_pipe_and_newline_escaped(self) -> None:
        md = _rows_to_markdown([['标题', '备注'], ['a|b', '多\n行']])
        assert 'a\\|b' in md
        assert '多 行' in md


class TestPdfParser:
    def test_sample_book_pdf_parses(self) -> None:
        book = REPO / 'data' / 'books' / 'sample_book.pdf'
        parsed = PdfParser().parse(str(book))
        content = parsed.chapters[0].content
        assert len(parsed.chapters) == 1
        assert parsed.chapters[0].title == '正文'
        assert len(content) > 50
        assert TABLE_BEGIN not in content  # 纯文本页不应出现表格哨兵
