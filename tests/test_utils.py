"""工具模块冒烟测试：路径与日志。"""

import logging

from reading_assistant.utils.logger_handler import get_logger
from reading_assistant.utils.path_tools import get_abs_path, get_project_root


def test_project_root_and_abs_path() -> None:
    root = get_project_root()
    assert root.endswith('ReadingAssistant')
    assert get_abs_path('pyproject.toml').endswith('pyproject.toml')


def test_get_logger_writes_to_file(tmp_path) -> None:
    log_file = tmp_path / 'test.log'
    logger = get_logger(name='utils_test', log_file=str(log_file), console_level=logging.ERROR)

    logger.info('hello reading assistant')

    assert logger.handlers  # 不重复添加 handler
    assert 'hello reading assistant' in log_file.read_text(encoding='utf-8')
