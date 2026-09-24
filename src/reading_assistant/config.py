"""统一配置入口：以环境变量/.env 为准，YAML 配置作为默认值。"""

import os
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
from reading_assistant.utils.logger_handler import get_logger
from reading_assistant.utils.path_tools import get_abs_path

logger = get_logger('config')

_agent_cfg = get_agent_config()
_chroma_cfg = get_chroma_config()
_model_cfg = get_model_config()
_prompt_cfg = get_prompt_config()

_ENV_FILE = str(Path(get_abs_path('.env')))

_SECRET_HINTS = ('KEY', 'TOKEN', 'SECRET', 'PASSWORD')
_shadow_check_done = False


def _looks_secret(name: str) -> bool:
    upper = name.upper()
    return any(hint in upper for hint in _SECRET_HINTS)


def _dotenv_raw_values() -> tuple[dict[str, str], dict[str, str]]:
    """返回 ``(生效值, 被注释掉的值)`` 两个字典。

    **必须把两者分开存**：注释行的值常是「轮换前的旧值」，
    而 shell 里 export 的往往正是它 —— 这是本函数存在的全部意义。
    早期版本用一个 dict 混存并按「生效行优先」覆盖，
    结果旧值被丢掉、那条最有用的告警永远不触发（2026-09-21 实测）。
    """
    active: dict[str, str] = {}
    commented: dict[str, str] = {}
    try:
        env_file = Path(_ENV_FILE)
        if not env_file.exists():
            return active, commented
        for raw in env_file.read_text(encoding='utf-8').splitlines():
            line = raw.strip()
            if not line or '=' not in line:
                continue
            is_comment = line.startswith('#')
            if is_comment:
                line = line.lstrip('#').strip()
                if '=' not in line:
                    continue
            key, value = (part.strip() for part in line.split('=', 1))
            if not key:
                continue
            (commented if is_comment else active)[key] = value
    except Exception:  # noqa: BLE001 诊断代码不能拖垮配置加载
        pass
    return active, commented


def warn_if_secret_values_look_malformed() -> dict[str, list[str]]:
    """密钥/令牌类配置的值必须是「token 形状」：非空、全 ASCII、无空白。

    为什么需要（2026-09-21 实测）：有人把一句**问题文本**粘进了
    ``DASHSCOPE_API_KEY``：

        DASHSCOPE_API_KEY=西游记中师徒四人遇到的第一个妖精是谁？…

    配置本身「读得到值、不为空」，所有既有检查都过得去；
    症状要等到每个请求打远端 API 时才以 ``401/400`` 爆出来，
    而报错完全指不到「配置里躺着一句中文」。

    判据刻意选得很粗（非空 + 全 ASCII + 无空白）而不是「sk- 开头」之类：
    密钥格式会随供应商轮换而变（本项目就从 35 字符 ``sk-`` 换到 115 字符 ``sk-ws-``），
    但**任何密钥都不会含中文或空格** —— 这条永远不会误伤，却挡得住粘贴事故。

    同时检查 ``.env`` 与进程环境两个来源。
    返回值：``{"empty": [...], "malformed": [...]}``。
    """
    active, _ = _dotenv_raw_values()
    checked: dict[str, str] = {}
    for key, value in active.items():
        if _looks_secret(key):
            checked[f'.env:{key}'] = value
    for key, value in os.environ.items():
        if _looks_secret(key) and key not in active:
            checked[f'进程环境:{key}'] = value

    empty: list[str] = []
    malformed: list[str] = []
    for label, value in checked.items():
        if not value:
            empty.append(label)
            logger.warning(
                '配置[无效值] %s 为空 —— 若该项必填，启动后调用远端 API 会失败。', label,
            )
            continue
        if not value.isascii() or any(ch.isspace() for ch in value):
            malformed.append(label)
            logger.error(
                '配置[无效值] %s 的值不像密钥（含非 ASCII 字符或空白，长度 %d）—— '
                '**很可能是粘贴错误**（例如把一句话粘进了密钥字段）。请检查后重启。',
                label, len(value),
            )
    return {'empty': empty, 'malformed': malformed}


def warn_if_process_env_shadows_dotenv() -> dict[str, list[str]]:
    """启动时检查「进程环境变量」与 ``.env`` 是否**同名不同值**，不一致则告警。

    为什么需要（2026-09-21 实测踩到，且是静默型故障）：

    进程环境变量与 ``.env`` 的优先级是「进程 > .env」（dotenv 的标准约定，
    也是本项目 ``load_dotenv()`` 采用的行为）。但**如果 shell 里 export 了一个
    过期的密钥**，进程环境就会盖掉 ``.env`` 里刚轮换的新值 —— 而症状是
    调用远端 API 时一个语焉不详的 ``401 InvalidApiKey / API-key is blocked``，
    排查要一路挖到「读运行中进程的 environ」才能定位。

    更隐蔽的是：``.env`` 里往往**留着被注释掉的旧值**，shell 里 export 的
    恰恰就是那一个 —— 所以这里额外识别这种情形并高亮。

    仅告警、不阻断启动，也**不改优先级**（覆盖能力必须保留，评测/实验要用）。
    返回值：``{"mismatched": [...], "stale_shadowed": [...]}``（供测试断言与观测）。
    ``stale_shadowed`` 指「进程环境的值 == .env 中被注释掉的旧值」这种最可疑的情形。
    """
    global _shadow_check_done
    if _shadow_check_done:
        return {'mismatched': [], 'stale_shadowed': []}
    _shadow_check_done = True

    file_values, commented_values = _dotenv_raw_values()
    mismatched: list[str] = []
    for key, file_value in file_values.items():
        if not file_value:
            continue  # 只看 .env 里**生效**的行
        process_value = os.environ.get(key)
        if process_value is None or process_value == file_value:
            continue

        mismatched.append(key)
        shown = '<已隐藏>' if _looks_secret(key) else f'进程={process_value!r} .env={file_value!r}'
        logger.warning(
            '配置[环境覆盖] %s 在进程环境与 .env 中不一致（%s）——当前生效的是**进程环境**的值。'
            '若是无意为之（例如 shell 里 export 了旧值），请 unset 后重启。',
            key, shown,
        )

    # 额外识别：进程环境里的值 == .env 里**被注释掉**的旧值（本次事故的原型）
    stale_shadowed: list[str] = []
    for key, old_value in commented_values.items():
        if not old_value:
            continue
        if os.environ.get(key) != old_value:
            continue
        stale_shadowed.append(key)
        logger.warning(
            '配置[环境覆盖] %s 在进程环境里的值**等于 .env 中被注释掉的旧值** —— '
            '这通常是 shell 里 export 了轮换前的旧值，且它会盖掉 .env 里的新值。'
            '请 unset %s（并检查 shell 配置）后重启。',
            key, key,
        )
    return {'mismatched': mismatched, 'stale_shadowed': stale_shadowed}


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
    # LangGraph checkpoint 后端：'memory'（进程内，仅单进程/测试可恢复）
    # 或 'postgres'（跨进程/跨重启可恢复，生产必须用这个）。
    # 设 memory 时「中断-恢复」只在同一进程内有效。
    checkpoint_backend: str = 'memory'
    top_k: int = int(_chroma_cfg.get('k', 6))
    retrieval_min_score: float = float(_chroma_cfg.get('min_score', 0.45))
    # 检索模式：'vector'（纯稠密，默认）| 'hybrid'（BM25 + 稠密 + RRF 融合）
    retrieval_mode: str = _chroma_cfg.get('retrieval_mode', 'vector')
    # hybrid 每路候选池大小与 RRF 融合常数
    hybrid_pool_size: int = int(_chroma_cfg.get('hybrid_pool_size', 50))
    rrf_k: int = int(_chroma_cfg.get('rrf_k', 60))
    bm25_tokenizer: str = _chroma_cfg.get('bm25_tokenizer', 'jieba')
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

    # --- 意图识别级联（规则 → 向量原型 → LLM，见 graph/intent.py）---
    intent_rules_enabled: bool = True  # L1 规则快通道（高精度，命中即省一次 LLM）
    intent_prototype_threshold: float = 0.75  # L2 原型层置信阈值；<=0 停用该层
    intent_prototype_margin: float = 0.08  # L2 判 book 需领先 chat/history 的最小幅度

    # --- 检索 agent（有界 ReAct，见 graph/agent.py）---
    # 位置：「一次检索」与「问人」之间的台阶。一次检索拿不到 → agent 换策略再试 → 仍不够才问人。
    agent_enabled: bool = True
    agent_max_steps: int = 6  # 单题 agent 的 LLM 轮数硬上限（防失控成本）

    # --- 多轮上下文：原文窗口（token 预算）+ 会话结构化状态 ---
    # 取代原「超出 6 条消息就滚动摘要」方案。理由：原文无损且能吃 prefix cache，
    # 而摘要每轮一次 LLM、且位于 prompt 前缀会主动让缓存失效 —— 更贵且有损。
    history_token_budget: int = 4000  # 历史原文窗口总预算（token）
    history_msg_token_cap: int = 800  # 单条历史消息上限（token），防一条长回答吃掉整个窗口
    history_max_messages: int = 200  # 单次最多从库里取回的消息条数（规模护栏，不参与截断语义）

    # --- 对话式查询改写（CQR：指代消解，见 graph/rewrite.py）---
    # 目标：让「它的特点是什么？」在检索前变成自带实体的查询。
    # 代词检测 + 实体抽取都是确定性规则，**零新增 LLM 调用**。
    cqr_enabled: bool = True  # 含代词的追问是否消解成自带实体的查询
    # 最多并入几个上文实体。**实测（禁缓存，8 条代词用例离线量）**指称召回：
    #   4 → 6/8 ；6 → 7/8 ；8 → 7/8（无进一步增益，白白扩大查询）
    # 故取 6：召回已到上限，而注入的候选数比 4 只多两个。
    cqr_max_entities: int = 6

    @property
    def chroma_persist_path(self) -> Path:
        """向量库持久化目录（仓库根目录下）。"""
        return Path(get_abs_path(self.chroma_persist_dir))


@lru_cache
def get_settings() -> Settings:
    # 首次加载配置时做两项检查（都只告警，不改行为）：
    # ① 值形状：密钥是不是被粘错了；② 来源一致性：进程环境有没有悄悄盖掉 .env
    warn_if_secret_values_look_malformed()
    warn_if_process_env_shadows_dotenv()
    """获取全局唯一的配置实例（惰性加载并缓存）。"""
    return Settings()
