"""PDF 解析器：基于 pypdf，提取元数据与逐页文本。"""

from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from reading_assistant.parsers.base import (
    BookParser,
    Chapter,
    EncryptedFileError,
    ParsedBook,
    ParseError,
)


class PdfParser(BookParser):
    extensions = frozenset({'.pdf'})

    def parse(self, path: str | Path) -> ParsedBook:
        path = Path(path)
        try:
            reader = PdfReader(str(path))
        except PdfReadError as exc:
            raise ParseError(f'PDF 解析失败（文件损坏）: {path}') from exc
        except Exception as exc:
            raise ParseError(f'PDF 解析失败: {path}') from exc

        if reader.is_encrypted:
            raise EncryptedFileError(f'PDF 已加密，暂不支持密码解锁: {path}')

        pages = []
        for page in reader.pages:
            try:
                pages.append(page.extract_text() or '')
            except Exception:
                pages.append('')
        text = '\n'.join(pages).strip()
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
