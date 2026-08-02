"""电子书解析器：txt / epub / pdf / docx。"""

from reading_assistant.parsers.base import (
    BookParser,
    Chapter,
    EncryptedFileError,
    ParsedBook,
    ParseError,
    UnsupportedFormatError,
    resolve_book_path,
)
from reading_assistant.parsers.docx_parser import DocxParser
from reading_assistant.parsers.epub_parser import EpubParser
from reading_assistant.parsers.factory import get_parser, parse_book
from reading_assistant.parsers.pdf_parser import PdfParser
from reading_assistant.parsers.txt_parser import TxtParser

__all__ = [
    'BookParser',
    'Chapter',
    'DocxParser',
    'EncryptedFileError',
    'EpubParser',
    'ParsedBook',
    'ParseError',
    'PdfParser',
    'TxtParser',
    'UnsupportedFormatError',
    'get_parser',
    'parse_book',
    'resolve_book_path',
]
