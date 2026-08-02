import logging
import os
from datetime import datetime

from reading_assistant.utils.path_tools import get_abs_path

# 日志保存的根目录
LOG_ROOT = get_abs_path('logs')
# 确保日志目录存在
os.makedirs(LOG_ROOT, exist_ok=True)

# 日志模式配置
DEFAULT_LOG_FORMAT = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
)


def get_logger(
    name: str = 'ReadingAssistant',
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    log_file: str | None = None,
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # 避免重复添加handler
    if logger.handlers:
        return logger

    # 控制台handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(DEFAULT_LOG_FORMAT)

    logger.addHandler(console_handler)

    # 日志文件存放路径
    if not log_file:
        log_file = os.path.join(LOG_ROOT, f'{name}_{datetime.now().strftime("%Y%m%d")}.log')

    file_handle = logging.FileHandler(log_file, encoding='utf-8')
    file_handle.setLevel(file_level)
    file_handle.setFormatter(DEFAULT_LOG_FORMAT)
    logger.addHandler(file_handle)

    return logger


# 快捷获取日志器
logger = get_logger()

if __name__ == '__main__':
    logger.info('信息日志')
    logger.error('报错日志')
    logger.warning('警告日志')
