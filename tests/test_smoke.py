"""P0 工程基础的冒烟测试：包导入、路径解析、配置加载。"""

from pathlib import Path


def test_project_root_points_to_repo(repo_root: Path) -> None:
    assert (repo_root / 'pyproject.toml').is_file()


def test_settings_loaded_from_env_and_yaml(settings) -> None:
    # 来自 agent.yml / model.yml 的默认值
    assert settings.app_name == 'ReadingAssistant'
    assert settings.llm_model == 'deepseek-v4-flash'
    assert settings.embedding_model == 'text-embedding-v4'
    # 来自 chroma.yml 的默认值
    assert settings.chunk_size == 800
    assert settings.chunk_overlap == 100
    assert settings.top_k == 6
    assert settings.chroma_collection_name == 'reading_agent_chunks'
    assert '。' in settings.separators
    assert 'txt' in settings.allow_knowledge_file_type
    # 敏感字段不因 repr 泄露
    assert 'api_key' not in repr(settings.deepseek_api_key or '')


def test_package_modules_importable() -> None:
    import reading_assistant.model.factory  # noqa: F401
    from reading_assistant.utils import config_handler  # noqa: F401
