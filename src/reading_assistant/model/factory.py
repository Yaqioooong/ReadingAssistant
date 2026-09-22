from abc import ABC, abstractmethod
from functools import lru_cache

from dotenv import load_dotenv
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_deepseek import ChatDeepSeek
from langchain_openai.chat_models.base import BaseChatOpenAI

from reading_assistant.config import get_settings

# ⚠️ 必须 override=False（dotenv 默认）——即「进程环境变量 > .env > 字段默认值」。
#
# 原来这里是 override=True，它把 .env 的值**强写进 os.environ**，于是
# 「用命令行环境变量覆盖 .env」在某些调用路径下静默失效：
# 本模块一旦被 import（构建 app 必走），os.environ 就被 .env 改写，
# 而 get_settings() 是 @lru_cache —— 谁先调用决定读到哪一份。
# 实测（2026-09-20）：`TOP_K=16 uv run python -c "get_settings().top_k"` 得到 16，
# 但同一环境变量在 `run_all --only e2e` 路径下被 .env 覆盖回 6，
# 于是 AB 两组跑的是同一配置、指标逐字相同 —— 实验静默失效。
# 这类「配置看着改了其实没改」的坑最难查，所以这里退回标准约定。
load_dotenv()


class BaseModelFactory(ABC):
    @abstractmethod
    def generate(self) -> Embeddings | BaseChatOpenAI:
        """创建并返回模型实例。"""


class ChatModelFactory(BaseModelFactory):
    def generate(self) -> BaseChatOpenAI:
        settings = get_settings()
        return ChatDeepSeek(
            model=settings.llm_model,
            api_key=settings.deepseek_api_key if settings.deepseek_api_key else None,
            extra_body={'thinking': {'type': 'disabled'}},
        )


class EmbeddingsFactory(BaseModelFactory):
    def generate(self) -> Embeddings:
        settings = get_settings()
        return DashScopeEmbeddings(model=settings.embedding_model)


@lru_cache
def get_chat_model() -> BaseChatOpenAI:
    """获取（缓存的）DeepSeek 对话模型。"""
    return ChatModelFactory().generate()


@lru_cache
def get_embedding_model() -> Embeddings:
    """获取（缓存的）DashScope Embedding 模型。"""
    return EmbeddingsFactory().generate()


if __name__ == '__main__':
    chat = get_chat_model()
    resp = chat.invoke("你好，请回复：'连接成功'")
    print(resp)

    emb = get_embedding_model()
    vec = emb.embed_query('测试文本')
    print('Embedding OK, 向量维度:', len(vec))
