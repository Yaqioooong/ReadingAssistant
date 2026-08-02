from abc import ABC, abstractmethod
from typing import Optional
from dotenv import load_dotenv
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_deepseek import ChatDeepSeek
from langchain_openai.chat_models.base import BaseChatOpenAI

from utils.config_handler import agent_config, model_config

load_dotenv(override=True)


class BaseModelFactory(ABC):

    @abstractmethod
    def generate(self) -> Optional[Embeddings | BaseChatOpenAI]:
        pass


class ChatModelFactory(BaseModelFactory):
    def generate(self) -> Optional[Embeddings | BaseChatOpenAI]:
        return ChatDeepSeek(
            model=model_config['chat_model_name'],
            extra_body={
                "thinking": {
                    "type": "disabled"
                }
            }
        )


class EmbeddingsFactory(BaseModelFactory):
    def generate(self) -> Optional[Embeddings | BaseChatOpenAI]:
        return DashScopeEmbeddings(model=model_config['embedding_model_name'])


chat_model = ChatModelFactory().generate()
embedding_model = EmbeddingsFactory().generate()
