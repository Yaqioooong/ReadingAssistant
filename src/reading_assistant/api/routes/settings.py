"""设置路由：读取 / 更新应用运行时配置（脱敏、即时生效）。

- GET  /api/settings  返回分组配置；密钥只回「已配置 + 脱敏尾号」
- PUT  /api/settings  部分更新；写入 .env → 同步进程环境 → 清空各级
  lru 缓存（settings / 模型工厂 / 依赖注入），下一次请求即用新配置
- 敏感项语义：传新值 = 覆盖；传 "__CLEAR__" = 清除该密钥
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from reading_assistant.config import get_settings
from reading_assistant.utils.env_store import apply_to_os_env, write_env

router = APIRouter(prefix='/api/settings', tags=['settings'])

CLEAR_SENTINEL = '__CLEAR__'

# 可编辑配置注册表：env 名 -> 元信息（type 用于校验/序列化）
FIELD_TYPES: dict[str, str] = {
    'LLM_MODEL': 'str',
    'EMBEDDING_MODEL': 'str',
    'DEEPSEEK_API_KEY': 'secret',
    'DASHSCOPE_API_KEY': 'secret',
    'RETRIEVAL_MODE': 'choice',
    'RETRIEVAL_MIN_SCORE': 'float',
    'TOP_K': 'int',
    'HYBRID_POOL_SIZE': 'int',
    'RRF_K': 'int',
    'BM25_TOKENIZER': 'str',
    'CACHE_ENABLED': 'bool',
    'CACHE_TTL_DAYS': 'int',
    'CACHE_MAX_ENTRIES': 'int',
    'CACHE_SIMILARITY_THRESHOLD': 'float',
    'CHUNK_SIZE': 'int',
    'CHUNK_OVERLAP': 'int',
}

GROUPS: list[dict[str, Any]] = [
    {
        'id': 'model',
        'title': '模型与密钥',
        'desc': '对话 / 向量模型选择与厂商 API 密钥。保存后即时生效。',
        'items': [
            {
                'key': 'LLM_MODEL',
                'label': '对话模型',
                'type': 'str',
                'hint': 'DeepSeek 模型名，如 deepseek-chat',
            },
            {
                'key': 'EMBEDDING_MODEL',
                'label': '向量模型',
                'type': 'str',
                'hint': 'DashScope 向量模型，如 text-embedding-v4',
            },
            {'key': 'DEEPSEEK_API_KEY', 'label': 'DeepSeek API Key', 'type': 'secret'},
            {'key': 'DASHSCOPE_API_KEY', 'label': 'DashScope API Key', 'type': 'secret'},
        ],
    },
    {
        'id': 'retrieval',
        'title': '检索',
        'desc': '召回方式与质量闸门：纯稠密 or BM25 混合，阈值越低召回越多但噪音越高。',
        'items': [
            {
                'key': 'RETRIEVAL_MODE',
                'label': '检索模式',
                'type': 'choice',
                'choices': [
                    ('vector', '纯稠密向量（默认，稳）'),
                    ('hybrid', 'BM25 + 稠密 RRF 融合'),
                ],
            },
            {
                'key': 'RETRIEVAL_MIN_SCORE',
                'label': '最低相似度 min_score',
                'type': 'float',
                'min': 0.0,
                'max': 1.0,
                'step': 0.05,
                'hint': '低于该余弦相似度的片段不参与回答',
            },
            {'key': 'TOP_K', 'label': '召回条数 top_k', 'type': 'int', 'min': 1, 'max': 30},
            {
                'key': 'HYBRID_POOL_SIZE',
                'label': '混合检索候选池',
                'type': 'int',
                'min': 10,
                'max': 300,
                'hint': 'hybrid 模式每路候选数量',
            },
            {
                'key': 'RRF_K',
                'label': 'RRF 融合常数',
                'type': 'int',
                'min': 10,
                'max': 200,
                'hint': 'hybrid 模式排名融合参数',
            },
            {
                'key': 'BM25_TOKENIZER',
                'label': 'BM25 分词器',
                'type': 'str',
                'hint': '目前支持 jieba',
            },
        ],
    },
    {
        'id': 'cache',
        'title': '问答缓存（记忆）',
        'desc': '命中后跳过整条 LLM 链路；语义阈值越高越保守。',
        'items': [
            {'key': 'CACHE_ENABLED', 'label': '启用缓存', 'type': 'bool'},
            {
                'key': 'CACHE_TTL_DAYS',
                'label': '缓存有效期（天）',
                'type': 'int',
                'min': 0,
                'max': 365,
            },
            {
                'key': 'CACHE_MAX_ENTRIES',
                'label': '缓存上限（条）',
                'type': 'int',
                'min': 10,
                'max': 100000,
            },
            {
                'key': 'CACHE_SIMILARITY_THRESHOLD',
                'label': '语义命中阈值',
                'type': 'float',
                'min': 0.5,
                'max': 1.0,
                'step': 0.01,
            },
        ],
    },
    {
        'id': 'chunking',
        'title': '分块',
        'desc': '新入库书籍生效；已在库文档需重建索引（reindex）后生效。',
        'items': [
            {
                'key': 'CHUNK_SIZE',
                'label': '分块大小（字符）',
                'type': 'int',
                'min': 100,
                'max': 4000,
            },
            {
                'key': 'CHUNK_OVERLAP',
                'label': '分块重叠（字符）',
                'type': 'int',
                'min': 0,
                'max': 800,
            },
        ],
    },
]

# 只读高级信息（不回明文，DB URL 含凭据时脱敏）
_ADVANCED_KEYS = (
    'vector_store_backend',
    'chroma_collection_name',
    'database_summary',
    'server_api_port',
)


class UpdateRequest(BaseModel):
    """部分更新：{KEY: value | __CLEAR__}。"""

    updates: dict[str, Any] = {}


def _mask_secret(value: str) -> str:
    if not value:
        return ''
    if len(value) <= 6:
        return '••••••'
    return f'{value[:3]}••••••{value[-4:]}'


def _settings_value(env_key: str) -> str:
    """从当前 Settings 取字段值（env 键名与 pydantic 字段大小写不敏感对应）。"""
    settings = get_settings()
    return str(getattr(settings, env_key.lower(), ''))


def _secret_state(settings, env_key: str) -> dict[str, Any]:
    secret = getattr(settings, env_key.lower(), None)
    raw = secret.get_secret_value() if secret is not None else ''
    return {'configured': bool(raw), 'masked': _mask_secret(raw)}


def _mask_database_url(url: str) -> str:
    """数据库 URL 脱敏：隐藏 user:password 段。"""
    if not url:
        return ''
    try:
        scheme, _, rest = url.partition('://')
        authority, _, tail = rest.partition('/')
        host = authority
        if '@' in authority:
            host = authority.split('@', 1)[1]
        return (
            f'{scheme}://••••@{host}/{tail}'
            if '@' in authority
            else f'{scheme}://{authority}/{tail}'
        )
    except Exception:  # noqa: BLE001
        return '（无法解析的数据库地址）'


def _advanced_items(settings) -> list[dict[str, Any]]:
    db_url = str(getattr(settings, 'database_url', '') or '')
    return [
        {
            'key': 'vector_store_backend',
            'label': '向量库后端',
            'value': str(getattr(settings, 'vector_store_backend', '')),
        },
        {
            'key': 'chroma_collection_name',
            'label': 'Chroma 集合名',
            'value': str(getattr(settings, 'chroma_collection_name', '')),
        },
        {'key': 'database_summary', 'label': '数据库', 'value': _mask_database_url(db_url)},
        {
            'key': 'server_api_port',
            'label': 'API 端口',
            'value': str(getattr(settings, 'api_port', '')),
        },
    ]


def _coerce(env_key: str, value: Any) -> Any:
    kind = FIELD_TYPES[env_key]
    if kind == 'int':
        return int(value)
    if kind == 'float':
        return float(value)
    if kind == 'bool':
        if isinstance(value, bool):
            return value
        if str(value).strip().lower() in ('true', '1', 'yes', 'on'):
            return True
        if str(value).strip().lower() in ('false', '0', 'no', 'off'):
            return False
        raise ValueError('需要 true/false')
    return str(value)


@router.get('', response_model=dict)
def get_settings_view() -> dict[str, Any]:
    settings = get_settings()
    groups: list[dict[str, Any]] = []
    for group in GROUPS:
        items = []
        for item in group['items']:
            key: str = item['key']
            payload = {k: v for k, v in item.items() if k not in ('key',)}
            payload['key'] = key
            if FIELD_TYPES[key] == 'secret':
                payload['state'] = _secret_state(settings, key)
                payload['value'] = ''
            else:
                payload['value'] = _settings_value(key)
            items.append(payload)
        groups.append({**{k: v for k, v in group.items() if k != 'items'}, 'items': items})
    groups.append(
        {
            'id': 'advanced',
            'title': '高级信息（只读）',
            'desc': '以下项目需修改源码或重启服务后生效，此处仅展示当前值。',
            'items': [
                {
                    'key': info['key'],
                    'label': info['label'],
                    'value': info['value'],
                    'type': 'readonly',
                }
                for info in _advanced_items(settings)
            ],
        }
    )
    return {'groups': groups}


@router.put('', response_model=dict)
def update_settings_view(request: UpdateRequest) -> dict[str, Any]:
    updates_raw: dict[str, Any] = request.updates or {}
    unknown = [key for key in updates_raw if key not in FIELD_TYPES]
    if unknown:
        from fastapi import HTTPException

        raise HTTPException(status_code=400, detail=f'未知配置项: {", ".join(unknown)}')

    # 校验 + 规范化（secret 保持原文；其余类型强转，早失败不落盘）
    normalized: dict[str, str | None] = {}
    try:
        for key, value in updates_raw.items():
            if FIELD_TYPES[key] == 'secret':
                if value == CLEAR_SENTINEL or value is None or str(value).strip() == '':
                    normalized[key] = None
                else:
                    normalized[key] = str(value).strip()
            elif FIELD_TYPES[key] == 'bool':
                normalized[key] = 'true' if _coerce(key, value) else 'false'
            else:
                normalized[key] = str(_coerce(key, value))
    except (ValueError, TypeError) as exc:
        from fastapi import HTTPException

        raise HTTPException(status_code=400, detail=f'配置值不合法: {exc}') from exc

    write_env(normalized)
    apply_to_os_env(normalized)
    _reload_runtime()
    return {
        'ok': True,
        'applied': sorted(k for k, v in normalized.items() if v is not None),
        'cleared': sorted(k for k, v in normalized.items() if v is None),
    }


def _reload_runtime() -> None:
    """清空各级 lru 缓存，让下一请求用新配置重建单例。"""
    from reading_assistant.api.deps import (
        get_embedding_model as deps_embedding,
    )
    from reading_assistant.api.deps import (
        get_llm,
        get_vector_store,
    )
    from reading_assistant.model.factory import (
        get_chat_model,
    )
    from reading_assistant.model.factory import (
        get_embedding_model as factory_embedding,
    )

    for fn in (
        get_settings,
        get_chat_model,
        factory_embedding,
        get_llm,
        deps_embedding,
        get_vector_store,
    ):
        try:
            fn.cache_clear()
        except AttributeError:
            pass
