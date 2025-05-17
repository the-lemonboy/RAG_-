from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import RedisChatMessageHistory
import hashlib
import redis
from logger_config import get_module_logger

# 获取模块日志记录器
logger = get_module_logger("memory_store")

def clear_all_redis_data(url: str):
    """清除Redis中所有会话数据"""
    try:
        r = redis.Redis.from_url(url)
        if not r.ping():
            logger.error("Redis连接测试失败")
            return False
            
        keys = r.keys('*')
        if keys:
            logger.info(f"正在删除{len(keys)}个Redis键")
            r.delete(*keys)
            logger.info("Redis数据清除完成")
            return True
        logger.info("没有找到需要删除的Redis键")
        return False
    except Exception as e:
        logger.error(f"清除Redis数据时出错: {str(e)}")
        return False

def test_redis_connection(url: str):
    """测试Redis连接是否成功"""
    try:
        r = redis.Redis.from_url(url)
        return r.ping()
    except Exception as e:
        return False

def get_memory(user_host: str, session_id: str, is_new_session: bool = False, clear_all: bool = False):
    """
    获取或创建对话内存
    :param clear_all: 是否清除所有Redis数据
    """
    try:
        redis_url = "redis://redis-17542.c323.us-east-1-2.ec2.redns.redis-cloud.com:17542"
        
        if clear_all:
            logger.info(f"清除用户 {user_host} 的所有Redis数据")
            clear_all_redis_data(redis_url)
            
        # 生成唯一会话key
        session_key = hashlib.md5(f"{user_host}_{session_id}".encode()).hexdigest()
        logger.info(f"为用户 {user_host} 创建会话 {session_id[:8]}...")
        
        if not test_redis_connection(redis_url):
            logger.error("无法连接到Redis服务器，将使用内存存储")
            from langchain_community.chat_message_histories import ChatMessageHistory
            message_history = ChatMessageHistory()
        else:
            # 使用Redis存储对话历史
            message_history = RedisChatMessageHistory(
                url=redis_url,
                ttl=3600,
                session_id=session_key
            )
            logger.info(f"成功创建Redis会话历史，TTL=3600秒")
        
        # 如果是新会话，清空历史记录
        if is_new_session:
            logger.info(f"清空会话 {session_id[:8]}... 的历史记录")
            message_history.clear()
            
            # 清除所有相关Redis键
            try:
                r = redis.Redis.from_url(redis_url)
                keys = r.keys(f"*{session_key}*")
                if keys:
                    r.delete(*keys)
                    logger.info(f"已删除 {len(keys)} 个相关Redis键")
            except Exception as e:
                logger.warning(f"清除Redis键时出错: {str(e)}")
    except Exception as e:
        logger.error(f"创建会话内存时出错: {str(e)}")
        # 出错时使用内存存储作为备选
        from langchain_community.chat_message_histories import ChatMessageHistory
        message_history = ChatMessageHistory()
        logger.info("已回退到内存存储模式")
        
    # 返回带有消息历史的内存对象
    
    return ConversationBufferMemory(
        chat_memory=message_history,
        return_messages=True
    )