""".env 读写工具（供配置页持久化使用）。

- 仓库根 .env 是本项目配置的持久层（优先级：环境变量 > .env > YAML 默认值）。
- 写入采取「读-改-写」，保留原文件未涉及的键与注释行；
  密钥置 None 表示删除该键（清除）。
- 测试可 monkeypatch ENV_PATH 指向临时文件，避免触碰真实 .env。
"""

from __future__ import annotations

import os
from pathlib import Path

from reading_assistant.utils.path_tools import get_abs_path

ENV_PATH = Path(get_abs_path('.env'))


def _lines(path: Path) -> list[str]:
    if not path.is_file():
        return []
    try:
        return path.read_text(encoding='utf-8').splitlines()
    except OSError:
        return []


def load_env(path: Path | None = None) -> dict[str, str]:
    """读取 .env 为 dict（跳过空行/注释，KEY=VALUE，首个 = 分割）。"""
    target = path or ENV_PATH
    result: dict[str, str] = {}
    for raw in _lines(target):
        line = raw.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, _, value = line.partition('=')
        key = key.strip()
        if key:
            result[key] = value.strip()
    return result


def write_env(
    updates: dict[str, str | None],
    path: Path | None = None,
) -> None:
    """应用一批更新：value=None 删除键，否则写入/覆盖。保留无关内容。

    写入的值会被剥离首尾空白与换行，避免破坏 .env 语法。
    """
    target = path or ENV_PATH
    lines = _lines(target)
    keys = set(updates)

    def sanitize(value: str) -> str:
        return value.strip().replace('\n', ' ').replace('\r', ' ')

    out: list[str] = []
    for raw in lines:
        line = raw.strip()
        if line and not line.startswith('#') and '=' in line:
            key, _, _ = line.partition('=')
            key = key.strip()
            if key in keys:
                new_value = updates.pop(key)
                if new_value is not None:
                    out.append(f'{key}={sanitize(new_value)}')
                continue  # None = 删除该行
        out.append(raw)
    # 剩余未出现在原文件中的键追加到末尾
    for key, value in updates.items():
        if value is not None:
            out.append(f'{key}={sanitize(value)}')
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text('\n'.join(out) + '\n', encoding='utf-8')


def apply_to_os_env(updates: dict[str, str | None]) -> None:
    """同步到进程环境变量（供清除 lru 缓存后工厂重建时读取）。"""
    for key, value in updates.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
