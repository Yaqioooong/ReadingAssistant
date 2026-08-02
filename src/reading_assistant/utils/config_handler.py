import yaml
from utils.path_tools import get_abs_path


def get_model_config(config_path: str = get_abs_path("config/model.yml"), encoding="utf-8"):
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)

def get_chroma_config(config_path: str = get_abs_path("config/chroma.yml"), encoding: str = "utf-8"):
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def get_prompt_config(config_path: str = get_abs_path("config/prompt.yml"), encoding: str = "utf-8"):
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def get_agent_config(config_path: str = get_abs_path("config/agent.yml"), encoding: str = "utf-8"):
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)

model_config = get_model_config()
chroma_config = get_chroma_config()
prompt_config = get_prompt_config()
agent_config = get_agent_config()

if __name__ == '__main__':
    print(agent_config['llm_model'])