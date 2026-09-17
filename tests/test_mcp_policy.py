"""MCP 上传路径安全策略（policy.validate_book_path）测试。

覆盖 P3-1 三层防御：规范化（resolve / `..` 穿越 / 符号链接）、白名单根目录、
敏感路径黑名单。所有测试均通过 monkeypatch 注入白名单根目录，**不读取真实用户
家目录**。

注意：本机 pytest 的临时目录位于 ``~/.config/...`` 下，会命中 ``.config`` 敏感
组件；因此测试统一在“无敏感组件”的安全基目录（优先 ``/tmp``）下自建工作目录，
避免把环境因素误当作策略缺陷。
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import pytest

from reading_assistant.mcp import policy
from reading_assistant.mcp.policy import validate_book_path


def _safe_base() -> str:
    """选一个不含敏感组件的基目录（优先 /tmp），隔离真实文件系统与家目录。"""
    for base in ('/tmp', tempfile.gettempdir(), str(Path.home())):
        resolved = Path(base).resolve()
        if not any(
            part.lower() in policy._SENSITIVE_DIR_COMPONENTS for part in resolved.parts
        ):
            return str(resolved)
    return str(Path(tempfile.gettempdir()).resolve())


@pytest.fixture
def workdir():
    """不含敏感组件、且不在任何真实受信目录内的临时工作目录。"""
    path = Path(tempfile.mkdtemp(prefix='mcp_policy_', dir=_safe_base()))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def allowed_root(workdir: Path, monkeypatch) -> Path:
    """把白名单根目录指向 <workdir>/allowed，隔离真实文件系统。"""
    root = workdir / 'allowed'
    root.mkdir()
    monkeypatch.setenv(policy.ENV_ALLOWED_ROOTS, str(root))
    return root


class TestAllowlist:
    def test_path_inside_allowed_root_passes(self, allowed_root: Path) -> None:
        book = allowed_root / 'book.txt'
        book.write_text('hello', encoding='utf-8')
        assert validate_book_path(str(book)) == book.resolve()

    def test_path_in_subdirectory_passes(self, allowed_root: Path) -> None:
        nested = allowed_root / 'sub' / 'deep' / 'book.epub'
        nested.parent.mkdir(parents=True)
        nested.write_text('hello', encoding='utf-8')
        assert validate_book_path(str(nested)) == nested.resolve()

    def test_path_outside_allowed_root_rejected(self, allowed_root: Path, workdir: Path) -> None:
        outside = workdir / 'elsewhere' / 'book.txt'
        outside.parent.mkdir()
        outside.write_text('hello', encoding='utf-8')
        with pytest.raises(ValueError, match='不在允许的受信目录内'):
            validate_book_path(str(outside))

    def test_relative_path_in_allowed_root_passes(self, allowed_root: Path, monkeypatch) -> None:
        book = allowed_root / 'rel.txt'
        book.write_text('hello', encoding='utf-8')
        monkeypatch.chdir(allowed_root)
        assert validate_book_path('rel.txt') == book.resolve()

    def test_dotdot_traversal_rejected(self, allowed_root: Path) -> None:
        # <allowed>/../../etc/passwd —— 规范化后应落在白名单外
        evil = allowed_root / '..' / '..' / 'etc' / 'passwd'
        with pytest.raises(ValueError, match='不在允许的受信目录内'):
            validate_book_path(str(evil))

    def test_symlink_escaping_allowed_root_rejected(
        self, allowed_root: Path, workdir: Path
    ) -> None:
        secret = workdir / 'outer' / 'note.txt'
        secret.parent.mkdir()
        secret.write_text('outside', encoding='utf-8')
        link = allowed_root / 'innocent.txt'
        os.symlink(secret, link)
        with pytest.raises(ValueError, match='不在允许的受信目录内'):
            validate_book_path(str(link))

    def test_symlinked_root_is_resolved(self, workdir: Path, monkeypatch) -> None:
        real = workdir / 'real_root'
        real.mkdir()
        (real / 'book.txt').write_text('hello', encoding='utf-8')
        link_root = workdir / 'link_root'
        os.symlink(real, link_root)
        monkeypatch.setenv(policy.ENV_ALLOWED_ROOTS, str(link_root))
        # 根被解析成真实目录，经符号链接进入的路径仍应放行
        assert validate_book_path(str(link_root / 'book.txt')) == (real / 'book.txt').resolve()


class TestSensitiveBlacklist:
    def test_ssh_component_rejected_even_inside_allowed_root(self, allowed_root: Path) -> None:
        ssh_dir = allowed_root / '.ssh'
        ssh_dir.mkdir()
        key = ssh_dir / 'note.txt'
        key.write_text('secret', encoding='utf-8')
        with pytest.raises(ValueError, match='敏感目录'):
            validate_book_path(str(key))

    def test_private_key_name_rejected_inside_allowed_root(self, allowed_root: Path) -> None:
        key = allowed_root / 'id_rsa.pem'
        key.write_text('secret', encoding='utf-8')
        with pytest.raises(ValueError, match='敏感文件特征'):
            validate_book_path(str(key))

    def test_env_file_rejected_inside_allowed_root(self, allowed_root: Path) -> None:
        env_file = allowed_root / '.env.production'
        env_file.write_text('SECRET=1', encoding='utf-8')
        with pytest.raises(ValueError, match='敏感文件特征'):
            validate_book_path(str(env_file))

    def test_credentials_name_rejected_inside_allowed_root(self, allowed_root: Path) -> None:
        creds = allowed_root / 'credentials.json'
        creds.write_text('{}', encoding='utf-8')
        with pytest.raises(ValueError, match='敏感文件特征'):
            validate_book_path(str(creds))

    def test_sensitive_rejected_even_outside_roots(self, workdir: Path) -> None:
        # 会话历史文件既有敏感特征、又在白名单外——无论如何都应拒绝
        history = workdir / 'bash_history'
        history.write_text('history', encoding='utf-8')
        with pytest.raises(ValueError, match='敏感文件特征'):
            validate_book_path(str(history))

    def test_error_message_lists_allowed_roots(self, allowed_root: Path) -> None:
        outside = allowed_root.parent / 'nope.txt'
        outside.write_text('x', encoding='utf-8')
        with pytest.raises(ValueError) as excinfo:
            validate_book_path(str(outside))
        # 消息要“写给模型看”：包含受信目录与自我修正提示
        assert str(allowed_root) in str(excinfo.value)
        assert policy.ENV_ALLOWED_ROOTS in str(excinfo.value)


class TestRootConfiguration:
    def test_env_var_overrides_default_roots(self, workdir: Path, monkeypatch) -> None:
        custom = workdir / 'custom_books'
        custom.mkdir()
        book = custom / 'book.txt'
        book.write_text('hello', encoding='utf-8')
        monkeypatch.setenv(policy.ENV_ALLOWED_ROOTS, str(custom))
        assert validate_book_path(str(book)) == book.resolve()

    def test_multiple_env_roots_split_by_pathsep(self, workdir: Path, monkeypatch) -> None:
        a = workdir / 'a'
        b = workdir / 'b'
        a.mkdir()
        b.mkdir()
        (b / 'book.txt').write_text('hello', encoding='utf-8')
        monkeypatch.setenv(policy.ENV_ALLOWED_ROOTS, os.pathsep.join([str(a), str(b)]))
        assert validate_book_path(str(b / 'book.txt')) == (b / 'book.txt').resolve()

    def test_env_only_root_rejects_other_dirs(self, workdir: Path, monkeypatch) -> None:
        # 环境变量一旦设置，就**替换**默认根目录：项目内外一律按 env 判定
        allowed = workdir / 'only_allowed'
        allowed.mkdir()
        monkeypatch.chdir(allowed)
        monkeypatch.setenv(policy.ENV_ALLOWED_ROOTS, str(allowed))
        other = workdir / 'other' / 'book.txt'
        other.parent.mkdir()
        other.write_text('hello', encoding='utf-8')
        with pytest.raises(ValueError, match='不在允许的受信目录内'):
            validate_book_path(str(other))

    def test_settings_field_takes_priority(self, workdir: Path, monkeypatch) -> None:
        root = workdir / 'from_settings'
        root.mkdir()
        book = root / 'book.txt'
        book.write_text('hello', encoding='utf-8')

        class FakeSettings:
            mcp_upload_allowed_roots = [str(root)]

        import reading_assistant.config as config

        monkeypatch.setattr(config, 'get_settings', lambda: FakeSettings())
        # 环境变量即便设成别的目录，也应被 settings 字段覆盖
        monkeypatch.setenv(policy.ENV_ALLOWED_ROOTS, str(workdir / 'ignored'))
        assert validate_book_path(str(book)) == book.resolve()

    def test_default_roots_are_nonempty_and_absolute(self, monkeypatch) -> None:
        monkeypatch.delenv(policy.ENV_ALLOWED_ROOTS, raising=False)
        roots = policy._allowed_roots()
        assert roots, '默认白名单根目录不应为空'
        assert all(root.is_absolute() for root in roots)


class TestUploadBookIntegration:
    """upload_book 的职责划分与顺序：存在性/类型 → 扩展名 → 路径策略。"""

    def test_unsupported_extension_still_rejected(self, workdir: Path) -> None:
        # 非书籍扩展名保持原有拒绝行为（扩展名校验先于路径策略，且不触碰运行时依赖）
        from reading_assistant.mcp import server

        exe = workdir / 'malware.exe'
        exe.write_text('MZ', encoding='utf-8')
        with pytest.raises(ValueError, match='不支持的书籍格式'):
            server.upload_book(str(exe))

    def test_missing_file_reported_as_not_found(self, workdir: Path) -> None:
        from reading_assistant.mcp import server

        with pytest.raises(ValueError, match='文件不存在'):
            server.upload_book(str(workdir / 'nope.txt'))

    def test_directory_input_reported_as_not_found(self, workdir: Path) -> None:
        # 目录不是普通文件：由 upload_book 给出明确的“文件不存在”拒绝
        from reading_assistant.mcp import server

        a_dir = workdir / 'a_dir'
        a_dir.mkdir()
        with pytest.raises(ValueError, match='文件不存在'):
            server.upload_book(str(a_dir))

    def test_existing_file_outside_roots_rejected_by_policy(
        self, workdir: Path, monkeypatch
    ) -> None:
        # 合法扩展名 + 真实存在、但在白名单外 → 由路径策略拒绝（早于任何读取/复制）
        from reading_assistant.mcp import server

        allowed = workdir / 'only_allowed'
        allowed.mkdir()
        monkeypatch.setenv(policy.ENV_ALLOWED_ROOTS, str(allowed))
        outside = workdir / 'elsewhere' / 'book.txt'
        outside.parent.mkdir()
        outside.write_text('hello', encoding='utf-8')
        with pytest.raises(ValueError, match='不在允许的受信目录内'):
            server.upload_book(str(outside))


class TestPolicyScope:
    def test_empty_path_rejected(self) -> None:
        with pytest.raises(ValueError, match='路径为空'):
            validate_book_path('')

    def test_policy_does_not_require_existence(self, allowed_root: Path) -> None:
        # policy 只做路径策略，不负责存在性：不存在的文件只要在受信目录内即放行，
        # 由 upload_book 负责把它判为“文件不存在”。
        ghost = allowed_root / 'ghost.txt'
        assert validate_book_path(str(ghost)) == ghost.resolve()

    def test_directory_input_passes_policy(self, allowed_root: Path) -> None:
        # 目录不是 policy 的职责：目录同样通过策略，由 upload_book 判定“非普通文件”。
        a_dir = allowed_root / 'a_directory'
        a_dir.mkdir()
        assert validate_book_path(str(a_dir)) == a_dir.resolve()
