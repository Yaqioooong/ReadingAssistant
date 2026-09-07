"""表格结构化测试：docx 表格 → Markdown 原子块，分块不腰斩。"""

from pathlib import Path

from docx import Document

from reading_assistant.parsers.base import TABLE_BEGIN, TABLE_END, Chapter, ParsedBook
from reading_assistant.parsers.docx_parser import DocxParser
from reading_assistant.rag.chunking import chunk_book


class TestChunkTableBlocks:
    def test_table_is_atomic_chunk(self) -> None:
        content = (
            '第一章开头段落。\n\n'
            f'{TABLE_BEGIN}\n'
            '| 姓名 | 职务 | 年份 |\n'
            '| --- | --- | --- |\n'
            '| 少白公 | 国家元首 | 1966 |\n'
            f'{TABLE_END}\n\n'
            '表格之后的正文。'
        )
        book = ParsedBook(title='T', chapters=[Chapter('第一章', content)])
        chunks = chunk_book(book, chunk_size=50, chunk_overlap=0)
        assert any(c.metadata.get('block_type') == 'table' for c in chunks)

    def test_table_rows_not_split(self) -> None:
        # 即使 chunk_size 很小，表格也必须整块保留（不被从行中间切断）
        content = (
            f'{TABLE_BEGIN}\n'
            '| 列一 | 列二 | 列三 |\n'
            '| --- | --- | --- |\n'
            '| A1 | B1 | C1 |\n'
            '| A2 | B2 | C2 |\n'
            f'{TABLE_END}\n'
        )
        book = ParsedBook(title='T', chapters=[Chapter('正文', content)])
        chunks = chunk_book(book, chunk_size=20, chunk_overlap=0)
        tables = [c for c in chunks if c.metadata.get('block_type') == 'table']
        assert len(tables) == 1
        text = tables[0].text
        assert '| A1 | B1 | C1 |' in text
        assert '| A2 | B2 | C2 |' in text
        assert '| --- | --- | --- |' in text
        assert TABLE_BEGIN not in text and TABLE_END not in text  # 哨兵不进正文

    def test_text_segments_around_table_unchanged(self) -> None:
        content = (
            '前文' + '字' * 100 + '\n\n'
            f'{TABLE_BEGIN}\n| A |\n| --- |\n| 1 |\n{TABLE_END}\n\n'
            '后文' + '字' * 100
        )
        book = ParsedBook(title='T', chapters=[Chapter('第一章', content)])
        chunks = chunk_book(book, chunk_size=60, chunk_overlap=0)
        pre_idx = next(i for i, c in enumerate(chunks) if c.text.startswith('前文'))
        table_idx = next(
            i for i, c in enumerate(chunks) if c.metadata.get('block_type') == 'table'
        )
        post_idx = next(i for i, c in enumerate(chunks) if '后文' in c.text)
        assert pre_idx < table_idx < post_idx  # 顺序：前文 → 表格 → 后文
        assert all(not c.text.startswith(TABLE_BEGIN) for c in chunks)


class TestDocxTable:
    def test_docx_table_to_atomic_chunk(self, tmp_path: Path) -> None:
        doc_path = tmp_path / 'table.docx'
        doc = Document()
        doc.add_heading('第一章 数据', level=1)
        doc.add_paragraph('表格前的说明文字。')
        table = doc.add_table(rows=3, cols=3)
        data = [['姓名', '职务', '年份'], ['少白公', '国家元首', '1966'], ['舵手', '领袖', '1969']]
        for r, row in enumerate(data):
            for c, value in enumerate(row):
                table.cell(r, c).text = value
        doc.add_paragraph('表格后的总结文字。')
        doc.save(str(doc_path))

        parsed = DocxParser().parse(str(doc_path))
        content = parsed.chapters[0].content
        assert '| 姓名 | 职务 | 年份 |' in content
        assert TABLE_BEGIN in content and TABLE_END in content

        chunks = chunk_book(parsed)
        tables = [c for c in chunks if c.metadata.get('block_type') == 'table']
        assert len(tables) == 1
        assert '少白公' in tables[0].text and '1966' in tables[0].text
        assert '国家元首' in tables[0].text
