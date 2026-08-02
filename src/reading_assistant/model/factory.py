from abc import ABC, abstractmethod
from functools import lru_cache

from dotenv import load_dotenv
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_deepseek import ChatDeepSeek
from langchain_openai.chat_models.base import BaseChatOpenAI

from reading_assistant.config import get_settings

load_dotenv(override=True)


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