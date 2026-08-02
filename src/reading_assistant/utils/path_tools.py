"""为全项目提供统一的绝对路径。"""

import os
from pathlib import Path


def get_project_root() -> str:
    """
    返回仓库根目录（包含 pyproject.toml 的目录）。

    从源码仓库（editable 安装或直接运行）调用时返回仓库根目录；
    包被安装到 site-packages 等非源码环境时，回退到包目录的父级。
    """
    here = Path(os.path.abspath(__file__))
    for parent in here.parents:
        if (parent / 'pyproject.toml').is_file():
            return str(parent)
    return str(here.parents[2])


def get_abs_path(relative_path: str) -> str:
    """将相对路径解析为仓库根目录下的绝对路径。"""
    return os.path.join(get_project_root(), relative_path)


if __name__ == '__main__':
    print(get_abs_path('README.md'))
