"""配置优先级回归：**进程环境变量 > .env > 字段默认值**。

## 为什么要钉这个

2026-09-20 实测踩到：`model/factory.py` 里写了 `load_dotenv(override=True)` ——
它把 `.env` 的值**强写进 ``os.environ``**。后果是「用命令行环境变量覆盖配置」
在某些调用路径下**静默失效**：

- ``uv run python -c "get_settings().top_k"`` → 读到命令行的 16 ✓
- ``run_all --only e2e``（走 create_app → 建图 → import model.factory）→
  被 ``load_dotenv(override=True)`` 改回 .env 的 6 ✗

而 ``get_settings()`` 是 ``@lru_cache``，**谁先调用决定读到哪一份**，所以同一环境变量
在不同入口下结果不同。代价很实：一次 AB 实验两组人跑的是同一配置，
指标逐字相同（还叠加了缓存污染），差点把「无收益」当成结论报出去。

这类「配置看着改了其实没改」不报错、只是悄悄用了别的值 —— 因此必须有测试钉住，
且必须在**完整 import 链**里钉（单测直接调 get_settings 是测不出来的，
因为那时 load_dotenv 还没跑）。
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# .env 里存在的键才有效 —— 否则子进程读到的是字段默认值，测不出覆盖行为
ENV_KEY = 'TOP_K'


def _dotenv_value(key: str) -> str | None:
    env_file = REPO_ROOT / '.env'
    if not env_file.exists():
        return None
    for raw in env_file.read_text(encoding='utf-8').splitlines():
        line = raw.strip()
        if line.startswith(f'{key}='):
            return line.split('=', 1)[1].strip()
    return None


def _override_value() -> str:
    """挑一个**与 .env 不同**的覆盖值。

    ⚠️ 这条必须动态取：若覆盖值恰好等于 .env 里的值，测试就分不清
    「环境变量生效」与「被 .env 覆盖」—— 形同虚设。
    （2026-09-20 实际踩到：把 .env 的 TOP_K 改成 16 后，覆盖值也是 16。）
    """
    base = _dotenv_value(ENV_KEY)
    if base is None:
        return '16'
    try:
        return str(int(base) + 7)
    except ValueError:
        return f'{base}-override'

# 覆盖整条会 import model.factory 的路径（这正是 override=True 生效的地方）
FULL_CHAIN = (
    'from reading_assistant.config import get_settings\n'
    'from reading_assistant.api import create_app\n'
    'from reading_assistant.model.factory import get_chat_model\n'
    'print(get_settings().top_k)\n'
)


def _env_file_has(key: str) -> bool:
    env_file = REPO_ROOT / '.env'
    if not env_file.exists():
        return False
    return any(
        line.strip().startswith(f'{key}=')
        for line in env_file.read_text(encoding='utf-8').splitlines()
    )


@pytest.mark.skipif(not _env_file_has(ENV_KEY), reason=f'.env 无 {ENV_KEY}，测不出覆盖')
def test_process_env_beats_dotenv_in_full_import_chain() -> None:
    """命令行环境变量必须在完整 import 链之后依然胜出（钉住 load_dotenv(override=True)）。"""
    want = _override_value()
    env = {**os.environ, ENV_KEY: want, 'PYTHONPATH': str(REPO_ROOT / 'src')}
    proc = subprocess.run(
        [sys.executable, '-c', FULL_CHAIN],
        capture_output=True, text=True, env=env, cwd=REPO_ROOT, timeout=180,
    )
    assert proc.returncode == 0, proc.stderr[-800:]
    got = proc.stdout.strip().splitlines()[-1]
    assert got == want, (
        f'环境变量被 .env 覆盖了 —— load_dotenv 又变成了 override=True？\n'
        f'期望 {want}，实际 {got}\nstdout={proc.stdout[-300:]!r}'
    )


@pytest.mark.skipif(not _env_file_has(ENV_KEY), reason=f'.env 无 {ENV_KEY}')
def test_dotenv_still_fills_missing_keys() -> None:
    """反向保障：.env 仍要能补上「环境里没有」的键 —— 别把 dotenv 整个关掉。"""
    env = {k: v for k, v in os.environ.items() if k != ENV_KEY}
    env['PYTHONPATH'] = str(REPO_ROOT / 'src')
    code = (
        'import os\n'
        f'os.environ.pop({ENV_KEY!r}, None)\n'
        'from reading_assistant.config import get_settings\n'
        'print(get_settings().top_k)\n'
    )
    proc = subprocess.run(
        [sys.executable, '-c', code],
        capture_output=True, text=True, env=env, cwd=REPO_ROOT, timeout=180,
    )
    assert proc.returncode == 0, proc.stderr[-800:]
    # 环境里没有该键时应从 .env 读到它写的值（而非字段默认值）
    want = _dotenv_value(ENV_KEY) or ''
    assert want, f'.env 没有 {ENV_KEY}，本测试无意义'
    assert proc.stdout.strip().splitlines()[-1] == want, proc.stdout[-300:]


# --------------------------------------------------------------- 环境覆盖告警
#
# 2026-09-21 事故：shell 里 export 了**轮换前的旧 DashScope key**（恰好是 .env 里
# 被注释掉的那一个），而优先级是「进程环境 > .env」→ 坏 key 盖掉好 key，
# 症状是远端 401 InvalidApiKey / API-key is blocked，排查要一路挖到
# 「读运行中进程的 environ」才定位。护栏让这件事**变响**。


def _write_env(tmp_path: Path, body: str) -> Path:
    env_file = tmp_path / '.env'
    env_file.write_text(body, encoding='utf-8')
    return env_file


def _check(monkeypatch, env_file: Path) -> dict:
    import reading_assistant.config as config

    monkeypatch.setattr(config, '_ENV_FILE', str(env_file))
    monkeypatch.setattr(config, '_shadow_check_done', False)
    return config.warn_if_process_env_shadows_dotenv()


def test_no_warning_when_process_env_matches(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv('TOP_K', raising=False)
    monkeypatch.delenv('KEEP_ME_KEY', raising=False)
    result = _check(monkeypatch, _write_env(tmp_path, 'TOP_K=16\n'))
    assert result['mismatched'] == []
    assert result['stale_shadowed'] == []


def test_warns_when_process_env_differs(tmp_path, monkeypatch) -> None:
    """进程环境与 .env 不同值 → 必须告警（覆盖能力保留，但不能静默）。"""
    monkeypatch.setenv('TOP_K', '6')
    result = _check(monkeypatch, _write_env(tmp_path, 'TOP_K=16\n'))
    assert result['mismatched'] == ['TOP_K']


def test_identifies_process_env_holding_commented_old_value(tmp_path, monkeypatch) -> None:
    """最有价值的一条：进程环境的值 == .env 中被注释掉的旧值。"""
    monkeypatch.setenv('DASHSCOPE_API_KEY', 'sk-old-rotated-away')
    result = _check(monkeypatch, _write_env(
        tmp_path,
        '#DASHSCOPE_API_KEY=sk-old-rotated-away\nDASHSCOPE_API_KEY=sk-new-and-good\n',
    ))
    assert result['mismatched'] == ['DASHSCOPE_API_KEY']
    assert result['stale_shadowed'] == ['DASHSCOPE_API_KEY']


def test_warning_fires_only_once(tmp_path, monkeypatch) -> None:
    """配置加载路径被调多次，告警只该出一次（否则日志被刷屏）。"""
    monkeypatch.setenv('TOP_K', '6')
    env_file = _write_env(tmp_path, 'TOP_K=16\n')
    first = _check(monkeypatch, env_file)
    assert first['mismatched'] == ['TOP_K']
    # 不重置 _shadow_check_done：第二次应短路
    import reading_assistant.config as config
    assert config.warn_if_process_env_shadows_dotenv() == {
        'mismatched': [], 'stale_shadowed': [],
    }


def test_missing_env_file_is_silent(tmp_path, monkeypatch) -> None:
    result = _check(monkeypatch, tmp_path / '不存在.env')
    assert result == {'mismatched': [], 'stale_shadowed': []}


# --------------------------------------------------------------- 密钥形状校验
#
# 2026-09-21 实测：有人把一句**问题文本**粘进了 DASHSCOPE_API_KEY
# （`DASHSCOPE_API_KEY=西游记中师徒四人遇到的第一个妖精是谁？…`）。
# 配置「读得到值、不为空」，所有既有检查都过得去，症状要等到每个请求打远端
# API 时才以 401/400 爆出来，而报错完全指不到「配置里躺着一句中文」。


def test_flags_question_text_pasted_into_secret_field(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv('DASHSCOPE_API_KEY', raising=False)
    monkeypatch.delenv('DEEPSEEK_API_KEY', raising=False)
    env_file = _write_env(
        tmp_path,
        'DASHSCOPE_API_KEY=西游记中师徒四人遇到的第一个妖精是谁？它有什么技能？\n'
        'DEEPSEEK_API_KEY=sk-b82abcdefghijklmnopqrstuvwxyz0123\n',
    )
    import reading_assistant.config as config

    monkeypatch.setattr(config, '_ENV_FILE', str(env_file))
    result = config.warn_if_secret_values_look_malformed()
    assert result['malformed'] == ['.env:DASHSCOPE_API_KEY']
    assert '.env:DEEPSEEK_API_KEY' not in result['malformed'], '正常形状的 key 不该被误报'
    assert '.env:DEEPSEEK_API_KEY' not in result['empty']


def test_flags_empty_secret(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv('DASHSCOPE_API_KEY', raising=False)
    env_file = _write_env(tmp_path, 'DASHSCOPE_API_KEY=\n')  # 空值
    import reading_assistant.config as config

    monkeypatch.setattr(config, '_ENV_FILE', str(env_file))
    result = config.warn_if_secret_values_look_malformed()
    assert result['empty'] == ['.env:DASHSCOPE_API_KEY']


def test_accepts_malformed_secret_from_process_env(tmp_path, monkeypatch) -> None:
    """进程环境里的密钥也要检查（它可能盖掉 .env）。"""
    for key in ('DASHSCOPE_API_KEY', 'DEEPSEEK_API_KEY'):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv('SOME_SERVICE_TOKEN', '这 是一句 话')
    env_file = _write_env(tmp_path, 'TOP_K=16\n')
    import reading_assistant.config as config

    monkeypatch.setattr(config, '_ENV_FILE', str(env_file))
    result = config.warn_if_secret_values_look_malformed()
    assert '进程环境:SOME_SERVICE_TOKEN' in result['malformed']


def test_non_secret_keys_are_not_shape_checked(tmp_path, monkeypatch) -> None:
    """非密钥类配置（如 DATABASE_URL）含非 ASCII 也不该被报 —— 判据只针对密钥。"""
    # 必须清掉环境里可能存在的密钥（load_dotenv 会把 .env 的键灌进 os.environ），
    # 否则本测试会读到宿主机状态而假失败
    for key in ('DASHSCOPE_API_KEY', 'DEEPSEEK_API_KEY'):
        monkeypatch.delenv(key, raising=False)
    env_file = _write_env(tmp_path, 'BM25_TOKENIZER=jieba\n')
    import reading_assistant.config as config

    monkeypatch.setattr(config, '_ENV_FILE', str(env_file))
    result = config.warn_if_secret_values_look_malformed()
    assert result == {'empty': [], 'malformed': []}
