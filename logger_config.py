import logging
import os
from logging.handlers import RotatingFileHandler
import sys

# 日志级别映射
LOG_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL
}

# 默认日志配置
DEFAULT_LOG_LEVEL = "info"
DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DEFAULT_LOG_DIR = "logs"
DEFAULT_MAX_BYTES = 10 * 1024 * 1024  # 10MB
DEFAULT_BACKUP_COUNT = 5

def setup_logger(name=None, log_level=None, log_to_file=True, log_dir=None, max_bytes=None, backup_count=None):
    """
    配置并返回一个日志记录器
    
    参数:
        name: 日志记录器名称，默认为根记录器
        log_level: 日志级别，可以是字符串或日志级别常量
        log_to_file: 是否将日志写入文件
        log_dir: 日志文件目录
        max_bytes: 单个日志文件最大字节数
        backup_count: 保留的备份文件数量
    
    返回:
        配置好的日志记录器
    """
    # 获取日志记录器
    logger = logging.getLogger(name)
    
    # 如果已经配置过处理器，则不重复配置
    if logger.handlers:
        return logger
    
    # 设置日志级别
    if log_level is None:
        log_level = os.getenv("LOG_LEVEL", DEFAULT_LOG_LEVEL)
    
    if isinstance(log_level, str):
        log_level = LOG_LEVELS.get(log_level.lower(), logging.INFO)
    
    logger.setLevel(log_level)
    
    # 创建格式化器
    log_format = os.getenv("LOG_FORMAT", DEFAULT_LOG_FORMAT)
    formatter = logging.Formatter(log_format)
    
    # 添加控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 添加文件处理器（如果启用）
    if log_to_file:
        # 设置日志目录
        if log_dir is None:
            log_dir = os.getenv("LOG_DIR", DEFAULT_LOG_DIR)
        
        # 确保日志目录存在
        os.makedirs(log_dir, exist_ok=True)
        
        # 设置文件处理器参数
        if max_bytes is None:
            max_bytes = int(os.getenv("LOG_MAX_BYTES", DEFAULT_MAX_BYTES))
        
        if backup_count is None:
            backup_count = int(os.getenv("LOG_BACKUP_COUNT", DEFAULT_BACKUP_COUNT))
        
        # 日志文件路径
        log_file = os.path.join(log_dir, f"{name if name else 'app'}.log")
        
        # 创建文件处理器（支持日志轮转）
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

# 获取应用主日志记录器
def get_app_logger():
    """
    获取应用主日志记录器
    """
    return setup_logger("her_server")

# 获取模块日志记录器
def get_module_logger(module_name):
    """
    获取模块日志记录器
    
    参数:
        module_name: 模块名称
    
    返回:
        配置好的模块日志记录器
    """
    return setup_logger(f"her_server.{module_name}")