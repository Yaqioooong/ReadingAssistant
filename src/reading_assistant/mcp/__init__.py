"""MCP Server：把 ReadingAssistant 能力暴露给任意 MCP 客户端（Claude/Cursor/自建 agent）。

对外工具：list_books / upload_book / ask_book / reindex_book。
入口：`uv run readingassistant-mcp`（stdio 传输）。
"""

from reading_assistant.mcp.server import main, mcp

__all__ = ['main', 'mcp']
