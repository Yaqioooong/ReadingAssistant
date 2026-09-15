"""EPUB 解析器：基于 ebooklib，按 spine 顺序抽取章节纯文本。"""

from pathlib import Path

import ebooklib
from bs4 import BeautifulSoup
from ebooklib import epub

from reading_assistant.parsers.base import (
    BookParser,
    Chapter,
    ParsedBook,
    ParseError,
    resolve_book_path,
)


class EpubParser(BookParser):
    extensions = frozenset({'.epub'})

    def parse(self, path: str | Path) -> ParsedBook:
        path = resolve_book_path(path)
        if not path.is_file():
            raise ParseError(f'文件不存在: {path}')
        try:
            book = epub.read_epub(str(path))
        except Exception as exc:  # ebooklib 对损坏文件抛出多种异常
            raise ParseError(f'EPUB 解析失败（文件损坏）: {path}') from exc

        title = self._first_metadata(book, 'title') or path.stem
        author = self._first_metadata(book, 'creator')

        chapters = []
        for idref, _linear in book.spine:
            item = book.get_item_with_id(idref)
            if item is None:
                continue
            # ebooklib 将 nav/toc 也标记为 ITEM_DOCUMENT，按文件名排除导航页
            name = (item.get_name() or '').lower()
            if item.get_type() != ebooklib.ITEM_DOCUMENT or name.startswith(('nav', 'toc', 'ncx')):
                continue
            chapter = self._to_chapter(item)
            if chapter is not None:
                chapters.append(chapter)

        if not chapters:
            raise ParseError(f'EPUB 中没有可解析的正文: {path}')
        return ParsedBook(title=title, author=author, chapters=chapters)

    @staticmethod
    def _first_metadata(book: epub.EpubBook, name: str) -> str | None:
        values = book.get_metadata('DC', name) # DC：Dublin Core, EPUB规定的命名空间
        return values[0][0] if values else None

    @staticmethod
    def _to_chapter(item) -> Chapter | None:
        soup = BeautifulSoup(item.get_content(), 'html.parser')
        for tag in soup(['script', 'style']):
            tag.decompose()
        content = soup.get_text('\n').strip()
        if not content:
            return None
        heading = soup.find(['h1', 'h2', 'h3', 'title'])
        title = heading.get_text(strip=True) if heading else ''
        if not title:
            title = next((line.strip() for line in content.splitlines() if line.strip()), '')
        return Chapter(title[:60] or item.get_name(), content)
