"""MCP 上传路径安全策略：白名单根目录 + 敏感路径黑名单（纵深防御）。

背景（P3-1：upload_book 任意路径读取）
-------------------------------------
`upload_book` 最初只校验扩展名，因此接受**任意绝对路径**。两个"合法"工具即可
组合出一个外泄原语::

    ① upload_book('/Users/<user>/.ssh/id_rsa.txt')
       → 通过（.txt 在扩展名白名单）→ 复制进 uploads/ → 解析 → 分块 → 向量化入库
    ② ask_book('我上传的那个 txt 文件内容是什么？')
       → 检索命中，原文回显

本模块按业界标准三层防御收敛该原语：

1. **规范化**：``Path(path).expanduser().resolve()`` —— 必须 resolve，
   以解析符号链接与 ``..`` 穿越，避免"字面路径在允许目录内、真实目标在外部"。
2. **白名单根目录**（allowlist，默认拒绝）：只有落在若干受信根目录内的文件才放行。
3. **敏感路径黑名单**（无论白名单如何一律拒绝）：即使文件位于允许目录内
   （例如 ``~/Downloads/id_rsa.pem``），命中 ssh / 密钥 / 凭证特征也一律拒绝。

设计边界
--------
- 本模块**只负责路径策略**（规范化 + 准入判定），**不负责存在性/类型**。
  文件是否存在、是否为普通文件由调用方（``mcp.server.upload_book``）判定，
  这样策略层保持纯函数、易于单测。
- 错误消息"写给模型看"：说明拒绝原因 + 列出允许的根目录，便于模型自我修正重试。
  MCP 世界没有 HTTP 状态码，故一律抛 ``ValueError``，不照搬 FastAPI 的 409/422。

待接线（config.py 合入后自动生效）
---------------------------------
本模块**优先**读取 ``get_settings().mcp_upload_allowed_roots``。
该字段目前**尚未**加入 ``src/reading_assistant/config.py``（由另一路并行工作负责），
因此当前会自动回退到环境变量 / 默认值。待 config.py 合入以下字段后即可生效，
**无需改动本文件**：:

    # src/reading_assistant/config.py -> class Settings(BaseSettings)
    mcp_upload_allowed_roots: list[str] = []   # 空列表 => 回退环境变量/默认值

字段名：``mcp_upload_allowed_roots``，类型 ``list[str]``（绝对或含 ``~`` 的路径字符串）。

配置优先级
----------
1. ``get_settings().mcp_upload_allowed_roots``（非空时）
2. 环境变量 ``READINGASSISTANT_MCP_UPLOAD_ROOTS``（``os.pathsep`` 分隔）
3. 默认值：``<project_root>``（应用工作区，兼容仓库内 fixture）、
   ``<project_root>/data``、``<project_root>/uploads``、
   ``~/Downloads``、``~/Documents``

演进方向（本次不实现）
----------------------
RAG 语料本身即 Indirect Prompt Injection 的载体：恶意 EPUB 正文 → 分块 →
向量化 → 检索命中 → 作为"参考资料"注入 prompt，全程缺少"这是数据、不是指令"
的信任边界。建议后续在**入库前**增加 injection 特征扫描（如"忽略以上指令"
"你现在是…"等越权指令模板），在 chunk 级别打标或隔离。本条仅作记录。
"""

from __future__ import annotations

import fnmatch
import os
from pathlib import Path

from reading_assistant.utils.path_tools import get_project_root

__all__ = ['validate_book_path']

# 环境变量：覆盖白名单根目录（os.pathsep 分隔）。
ENV_ALLOWED_ROOTS = 'READINGASSISTANT_MCP_UPLOAD_ROOTS'

# 敏感**目录组件**黑名单：路径中任一组件命中即拒绝（大小写不敏感）。
_SENSITIVE_DIR_COMPONENTS = frozenset({
    '.ssh',
    '.aws',
    '.gnupg',
    '.kube',
    '.docker',
    '.config',
    '.netrc',
    '.git-credentials',
})

# 敏感**文件名**黑名单（fnmatch glob，大小写不敏感）。
_SENSITIVE_FILE_PATTERNS = (
    'id_rsa*',
    'id_ed25519*',
    'id_ecdsa*',
    '*.pem',
    '*.key',
    '*.p12',
    '.env*',
    'credentials*',
    '*_history',
)


def _settings_allowed_roots() -> list[str]:
    """尝试从 settings 读取白名单根目录；字段缺失/异常时返回空列表。"""
    try:
        # 惰性 import：避免与 config.py 的并行编辑相互影响，也避免 import 期副作用。
        from reading_assistant.config import get_settings
    except Exception:  # noqa: BLE001 —— config 不可用时退回环境变量/默认值
        return []
    try:
        settings = get_settings()
    except Exception:  # noqa: BLE001
        return []
    roots = getattr(settings, 'mcp_upload_allowed_roots', None)
    if not roots:
        return []
    return [str(p) for p in roots if str(p).strip()]


def _default_allowed_roots() -> list[Path]:
    """内置默认白名单根目录。

    - ``<project_root>``：本应用自身的可信工作区。显式列出，除覆盖 data/uploads
      之外也兼容仓库内 fixture（如 ``tests/``）——避免安全策略把本项目自身的
      测试夹具拒之门外。**宿主敏感目录（``~/.ssh`` 等）位于其外，仍被拒绝。**
    - ``<project_root>/data``、``<project_root>/uploads``：应用数据/上传目录。
    - ``~/Downloads``、``~/Documents``：用户常见书籍投放目录。
    """
    project_root = Path(get_project_root())
    return [
        project_root,
        project_root / 'data',
        project_root / 'uploads',
        Path.home() / 'Downloads',
        Path.home() / 'Documents',
    ]


def _allowed_roots() -> list[Path]:
    """解析当前生效的白名单根目录（已 expanduser + resolve）。

    优先级：settings 字段 > 环境变量 > 默认值。
    每次调用重新解析（不缓存），以便测试 monkeypatch 环境变量后立即生效。
    """
    raw: list[str] = _settings_allowed_roots()
    if not raw:
        env_value = os.environ.get(ENV_ALLOWED_ROOTS, '')
        raw = [item for item in env_value.split(os.pathsep) if item.strip()]
    if not raw:
        roots = _default_allowed_roots()
    else:
        roots = [Path(item).expanduser() for item in raw]
    # 统一 resolve，保证 is_relative_to 比较的是真实路径。
    return [root.resolve() for root in roots]


def _reject(reason: str, roots: list[Path]) -> ValueError:
    """构造"写给模型看"的拒绝错误：原因 + 允许的目录列表。"""
    allowed = '\n'.join(f'  - {root}' for root in roots) or '  (无)'
    return ValueError(
        f'拒绝上传：{reason}\n'
        f'upload_book 仅允许读取以下受信目录（含其子目录）内的书籍文件：\n'
        f'{allowed}\n'
        f'请将书籍文件移动到上述任一目录后重试，或通过环境变量 '
        f'{ENV_ALLOWED_ROOTS}（{os.pathsep} 分隔）追加受信目录。'
    )


def _is_sensitive(resolved: Path) -> str | None:
    """返回敏感命中的说明文本；不敏感则返回 None。"""
    lowered_parts = [part.lower() for part in resolved.parts]
    for part in lowered_parts:
        if part in _SENSITIVE_DIR_COMPONENTS:
            return f'路径包含敏感目录「{part}」'
    name = resolved.name.lower()
    for pattern in _SENSITIVE_FILE_PATTERNS:
        if fnmatch.fnmatch(name, pattern):
            return f'文件名「{resolved.name}」匹配敏感文件特征「{pattern}」'
    return None


def validate_book_path(path: str) -> Path:
    """校验书籍路径是否可安全上传，返回规范化后的绝对路径。

    Args:
        path: 调用方传入的本机路径（可为相对路径 / 含 ``~`` / 含符号链接）。

    Returns:
        规范化（expanduser + resolve）后的绝对路径。

    Raises:
        ValueError: 路径为空、命中敏感黑名单、或不在白名单根目录内。
            消息面向 LLM，包含拒绝原因与允许目录列表，便于自我修正重试。

    Note:
        本函数**不检查文件是否存在或是否为普通文件**——存在性由
        ``mcp.server.upload_book`` 负责，保持策略层为纯路径判定。
    """
    if path is None or not str(path).strip():
        raise _reject('路径为空', _allowed_roots())

    # (a) 规范化：resolve 解析符号链接与 .. 穿越。
    resolved = Path(path).expanduser().resolve()
    roots = _allowed_roots()

    # (c) 敏感黑名单优先：无论白名单如何，一律拒绝（纵深防御）。
    hit = _is_sensitive(resolved)
    if hit is not None:
        raise _reject(hit, roots)

    # (b) 白名单根目录：必须落在某个受信根目录内，默认拒绝。
    for root in roots:
        if resolved.is_relative_to(root):
            return resolved

    raise _reject(f'路径不在允许的受信目录内：{resolved}', roots)
