"""pytest 公共配置与 fixtures。"""

import sys
from pathlib import Path

import pytest

# 未安装 editable 包时，也能直接运行 pytest
SRC_DIR = Path(__file__).resolve().parents[1] / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from reading_assistant.config import get_settings  # noqa: E402
from reading_assistant.utils.path_tools import get_project_root  # noqa: E402


@pytest.fixture
def repo_root() -> Path:
    """仓库根目录（包含 pyproject.toml 的目录）。"""
    return Path(get_project_root())


@pytest.fixture
def settings():
    """全局配置实例。"""
    return get_settings()
