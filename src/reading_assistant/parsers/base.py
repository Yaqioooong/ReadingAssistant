"""解析器公共定义：数据结构、异常与抽象基类。"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


class ParseError(Exception):
    """解析失败（文件缺失、损坏或不支持）。"""


class UnsupportedFormatError(ParseError):
    """文件格式不受支持。"""


class EncryptedFileError(ParseError):
    """文件已加密，需要密码。"""


@dataclass
class Chapter:
    """书中的一个章节：标题 + 纯文本内容。"""

    title: str
    content: str


@dataclass
class ParsedBook:
    """解析结果：元数据 + 按章节组织的纯文本。"""

    title: str
    author: str | None = None
    chapters: list[Chapter] = field(default_factory=list)

    @property
    def chapter_count(self) -> int:
        """章节数量。"""
        return len(self.chapters)

    @property
    def full_text(self) -> str:
        """拼接全部章节文本（章节之间以两个换行分隔）。"""
        return '\n\n'.join(chapter.content for chapter in self.chapters)


class BookParser(ABC):
    """电子书解析器基类。"""

    #: 支持的扩展名（小写、含点），由工厂用于分发
    extensions: frozenset[str] = frozenset()

    @abstractmethod
    def parse(self, path: str | Path) -> ParsedBook:
        """将电子书解析为纯文本 + 元数据。"""
