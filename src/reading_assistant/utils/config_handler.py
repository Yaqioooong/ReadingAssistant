"""读取 config/ 目录下的 YAML 配置。"""

from pathlib import Path

import yaml

# YAML 配置随包分发，位于 reading_assistant/config/ 目录
_CONFIG_DIR = Path(__file__).resolve().parent.parent / 'config'


def _load_yaml(path: Path) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def get_model_config(config_path: str | None = None) -> dict:
    """读取 model.yml（LLM 与 Embedding 模型名）。"""
    path = Path(config_path) if config_path else _CONFIG_DIR / 'model.yml'
    return _load_yaml(path)


def get_chroma_config(config_path: str | None = None) -> dict:
    """读取 chroma.yml（向量库与分块参数）。"""
    path = Path(config_path) if config_path else _CONFIG_DIR / 'chroma.yml'
    return _load_yaml(path)


def get_prompt_config(config_path: str | None = None) -> dict:
    """读取 prompt.yml（prompt 文件路径）。"""
    path = Path(config_path) if config_path else _CONFIG_DIR / 'prompt.yml'
    return _load_yaml(path)


def get_agent_config(config_path: str | None = None) -> dict:
    """读取 agent.yml（应用级配置）。"""
    path = Path(config_path) if config_path else _CONFIG_DIR / 'agent.yml'
    return _load_yaml(path)
