import os
from dotenv import load_dotenv
import redis
from time import sleep
from logger_config import get_module_logger

# 获取模块日志记录器
logger = get_module_logger("redis_utils")

load_dotenv()

def get_redis_connection(retries=3, delay=2):
    for attempt in range(retries):
        try:
            conn = redis.Redis(
                host=os.getenv("REDIS_HOST"),
                port=int(os.getenv("REDIS_PORT")),
                username="default",  # 显式使用默认用户名
                password=os.getenv("REDIS_PASSWORD"),
                ssl=True,  # 云服务通常需要SSL
                ssl_cert_reqs=None,  # 不验证证书
                decode_responses=True,
                socket_timeout=10,
                socket_connect_timeout=10
            )
            if conn.ping():
                logger.info("Redis连接成功")
                return conn
        except redis.AuthenticationError as e:
            logger.error(f"认证失败，请检查密码是否正确")
            raise
        except redis.ConnectionError as e:
            if attempt < retries - 1:
                logger.warning(f"连接失败，第{attempt+1}次重试...")
                sleep(delay)
                continue
            logger.error(f"Redis连接失败: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"未知错误: {str(e)}")
            raise
    return None

# 全局Redis连接
redis_conn = get_redis_connection()