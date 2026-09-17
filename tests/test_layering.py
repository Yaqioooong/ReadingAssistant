"""分层约束：适配器之间不得横向依赖。

HTTP（api/）与 MCP（mcp/）是同层的两个适配器，都只应向下依赖 core。
MCP 是 stdio 进程，若被拖入 fastapi，说明依赖方向反了。

注：starlette 不作为判据 —— fastmcp 自身依赖它（用于 HTTP transport），
即使走 stdio 也会加载，属正常现象。真正的信号是 fastapi（HTTP 适配器专属）。
"""

import json
import subprocess
import sys


def _modules_after_import(target: str) -> set[str]:
    """在干净子进程中 import target，返回其引入的顶层模块名集合。"""
    code = (
        'import sys, json\n'
        f'import {target}\n'
        'print(json.dumps(sorted({m.split(".")[0] for m in sys.modules})))\n'
    )
    result = subprocess.run(
        [sys.executable, '-c', code],
        capture_output=True,
        text=True,
        check=True,
    )
    return set(json.loads(result.stdout))


def test_mcp_server_does_not_import_fastapi():
    """MCP 适配器不得依赖 HTTP 层（api.deps 曾把 fastapi 拖进 stdio 进程）。"""
    modules = _modules_after_import('reading_assistant.mcp.server')
    assert 'fastapi' not in modules, 'mcp/server.py 依赖了 HTTP 层，适配器出现横向耦合'


def test_runtime_is_framework_agnostic():
    """运行时单例工厂层不得依赖 HTTP 框架。"""
    modules = _modules_after_import('reading_assistant.runtime')
    assert 'fastapi' not in modules, 'runtime.py 不应依赖 HTTP 框架'
