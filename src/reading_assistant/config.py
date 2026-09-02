"""统一配置入口：以环境变量/.env 为准，YAML 配置作为默认值。"""

from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from reading_assistant.utils.config_handler import (
    get_agent_config,
    get_chroma_config,
    get_model_config,
    get_prompt_config,
)
from reading_assistant.utils.path_tools import get_abs_path

_agent_cfg = get_agent_config()
_chroma_cfg = get_chroma_config()
_model_cfg = get_model_config()
_prompt_cfg = get_prompt_config()

_ENV_FILE = str(Path(get_abs_path('.env')))


class Settings(BaseSettings):
    """应用配置。

    优先级：环境变量 / .env > YAML 默认值 > 字段默认值。
    """

    model_config = SettingsConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding='utf-8',
        extra='ignore',
    )

    # --- 应用 ---
    app_name: str = _agent_cfg.get('app_name', 'ReadingAssistant')
    external_data_path: str = _agent_cfg.get('external_data_path', 'data/external/records.csv')
    database_url: str = (
        'postgresql+psycopg://psql_reading:123456@localhost:5432/reading_agent?sslmode=disable'
    )

    # --- 模型 ---
    llm_model: str = _model_cfg.get('chat_model_name', 'deepseek-v4-flash')
    embedding_model: str = _model_cfg.get('embedding_model_name', 'text-embedding-v4')
    deepseek_api_key: SecretStr | None = None
    dashscope_api_key: SecretStr | None = None

    # --- 向量库 / 分块 ---
    chroma_collection_name: str = _chroma_cfg.get('collection_name', 'reading_agent_chunks')
    chroma_persist_dir: str = _chroma_cfg.get('persist_directory', 'rag/chroma_db')
    vector_store_backend: str = 'chroma'
    top_k: int = int(_chroma_cfg.get('k', 6))
    retrieval_min_score: float = float(_chroma_cfg.get('min_score', 0.45))
    data_path: str = _chroma_cfg.get('data_path', 'data')
    allow_knowledge_file_type: list[str] = _chroma_cfg.get(
        'allow_knowledge_file_type', ['txt', 'pdf', 'epub']
    )
    chunk_size: int = int(_chroma_cfg.get('chunk_size', 800))
    chunk_overlap: int = int(_chroma_cfg.get('chunk_overlap', 100))
    separators: list[str] = _chroma_cfg.get(
        'separators', ['\n\n', '\n', '.', '!', '?', '。', '！', '？', ' ', '']
    )

    # --- Prompt 路径 ---
    main_prompt_path: str = _prompt_cfg.get('main_prompt_path', 'prompts/main_prompt.txt')
    rag_summarize_prompt_path: str = _prompt_cfg.get(
        'rag_summarize_prompt_path', 'prompts/rag_summarize.txt'
    )
    report_prompt_path: str = _prompt_cfg.get('report_prompt_path', 'prompts/report_prompt.txt')

    # --- 服务 ---
    api_host: str = '0.0.0.0'
    api_port: int = 8000
    log_level: str = 'INFO'

    # --- QA 缓存 ---
    cache_enabled: bool = True  # 是否启用缓存
    cache_ttl_days: int = 30  # 缓存天数
    cache_max_entries: int = 10000  # 最大缓存键数
    cache_similarity_threshold: float = 0.95

    @property
    def chroma_persist_path(self) -> Path:
        """向量库持久化目录（仓库根目录下）。"""
        return Path(get_abs_path(self.chroma_persist_dir))


@lru_cache
def get_settings() -> Settings:
    """获取全局唯一的配置实例（惰性加载并缓存）。"""
    return Settings()
