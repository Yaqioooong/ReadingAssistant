"""解析器工厂：按扩展名分发到具体解析器。"""

from pathlib import Path

from reading_assistant.parsers.base import (
    BookParser,
    ParsedBook,
    UnsupportedFormatError,
)
from reading_assistant.parsers.docx_parser import DocxParser
from reading_assistant.parsers.epub_parser import EpubParser
from reading_assistant.parsers.pdf_parser import PdfParser
from reading_assistant.parsers.txt_parser import TxtParser

# 注册顺序即匹配顺序；各解析器的 extensions 互斥
_REGISTRY: tuple[type[BookParser], ...] = (TxtParser, EpubParser, PdfParser, DocxParser)


def get_parser(path: str | Path) -> BookParser:
    """按文件扩展名返回对应的解析器实例。"""
    extension = Path(path).suffix.lower()
    for parser_cls in _REGISTRY:
        if extension in parser_cls.extensions:
            return parser_cls()
    raise UnsupportedFormatError(f'不支持的电子书格式: {extension or "(无扩展名)"}')


def parse_book(path: str | Path) -> ParsedBook:
    """解析电子书并返回纯文本 + 元数据。"""
    return get_parser(path).parse(path)
