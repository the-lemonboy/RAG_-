from fastapi import FastAPI, Request, UploadFile
from fastapi.responses import StreamingResponse
from langchain_openai import ChatOpenAI
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain.chains import ConversationChain, RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv
import redis
from vector_store import get_vectorstore, get_mmr_retriever, get_contextual_retriever
from logger_config import get_module_logger

# 获取模块日志记录器
logger = get_module_logger("main")

# 加载环境变量
load_dotenv()



# Redis连接池配置
REDIS_POOL = None
def get_redis_connection(retries=3, delay=2):
    """获取Redis连接，支持重试和超时设置"""
    global REDIS_POOL
    
    # 如果连接池已存在且有效，直接使用
    if REDIS_POOL and REDIS_POOL.connection_kwargs.get('password') == os.getenv("REDIS_PASSWORD"):
        conn = redis.Redis(connection_pool=REDIS_POOL)
        try:
            if conn.ping():
                return conn
        except:
            pass
            
    for attempt in range(retries):
        try:
            # 创建或更新连接池
            REDIS_POOL = redis.ConnectionPool(
                host='redis-17542.c323.us-east-1-2.ec2.redns.redis-cloud.com',
                port=17542,
                decode_responses=True,
                username="default",
                password=os.getenv("REDIS_PASSWORD"),  # 从环境变量获取Redis密码
                max_connections=50,
                socket_timeout=15,
                socket_connect_timeout=15,
                health_check_interval=30,
                retry_on_timeout=True
            )
            conn = redis.Redis(connection_pool=REDIS_POOL)
            # 测试连接
            if conn.ping():
                logger.info("Redis连接成功")
                return conn
        except redis.AuthenticationError as e:
            logger.error(f"Redis认证失败，请检查密码是否正确: {str(e)}")
            raise
        except redis.ConnectionError as e:
            if attempt < retries - 1:
                logger.warning(f"Redis连接失败，第{attempt+1}次重试...")
                from time import sleep
                sleep(delay)
                continue
            logger.error(f"Redis连接失败: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Redis连接未知错误: {str(e)}")
            return None
    return None

# 尝试连接Redis，如果失败则使用内存存储
try:
    redis_conn = get_redis_connection()
    if not redis_conn:
        logger.warning("Redis连接失败，将使用内存存储作为备选方案")
        # 设置一个标志，表示使用本地模式
        USE_LOCAL_MODE = True
    else:
        USE_LOCAL_MODE = False
except Exception as e:
    logger.error(f"Redis初始化错误: {str(e)}")
    USE_LOCAL_MODE = True
    redis_conn = None

# 文档处理配置
TEXT_SPLITTER_CONFIG = {
    "chunk_size": 800,  # 减小块大小以获得更精确的匹配
    "chunk_overlap": 150,  # 适当的重叠以保持上下文连贯性
    "separators": ["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""],  # 优先按自然段落和句子分割
}

# 初始化LLM
try:
    chat = ChatOpenAI(
    streaming=True,
    model="deepseek-chat",
    temperature=0.2,
    openai_api_base="https://api.deepseek.com/v1"
)
    logger.info("LLM初始化成功")
except Exception as e:
    if "Incorrect API key" in str(e):
        logger.error("API密钥无效，请检查.env文件中的OPENAI_API_KEY配置")
    elif "Connection error" in str(e):
        logger.error(f"无法连接到API服务，请检查网络或API地址: {os.getenv('OPENAI_API_BASE')}")
    else:
        logger.error(f"LLM初始化失败: {str(e)}")
    chat = None

# 定义优化的QA提示模板
QA_PROMPT_TEMPLATE = """
你是一个专业的PDF文档问答助手。请基于以下提供的文档内容，回答用户的问题。

文档内容：
{context}

用户问题：{query}

请注意：
1. 只回答与文档内容直接相关的问题
2. 如果文档中没有相关信息，请明确说明无法回答
3. 不要编造不在文档中的信息
4. 回答要简洁、准确、有条理
5. 如果合适，可以使用文档中的原文进行引用

你的回答：
"""

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    
    # 应用启动时初始化向量数据库
    try:
        logger.info("应用启动，初始化向量数据库")
        _ = get_vectorstore()
        logger.info("向量数据库初始化完成")
    except Exception as e:
        logger.error(f"向量数据库初始化失败: {str(e)}")
    yield
    # 应用关闭时的清理操作
    logger.info("应用关闭")

app = FastAPI(lifespan=lifespan)

@app.post("/chat")
async def chat_endpoint(request: Request):
    body = await request.json()
    user_input = body["message"]
    session_id = body.get("session_id", "default")
    user_host = body.get("userHost", "unknown")
    
    logger.info(f"收到用户请求: {user_input[:50]}...")

    # 获取向量数据库
    vectorstore = get_vectorstore()
    if not vectorstore:
        logger.error("向量数据库未初始化")
        return StreamingResponse(
            iter(["data: 请先上传PDF文档以初始化知识库。\n\n"]),
            media_type="text/event-stream"
        )
    
    # 使用上下文压缩检索器或MMR检索器
    try:
        # 优先使用上下文压缩检索器
        retriever = get_contextual_retriever()
        if not retriever:
            # 备选使用MMR检索器
            retriever = get_mmr_retriever(k=3, fetch_k=5, lambda_mult=0.7)
            logger.info("使用MMR检索器")
        else:
            logger.info("使用上下文压缩检索器")
    except Exception as e:
        logger.error(f"创建检索器失败: {str(e)}")
        # 出错时使用基础检索器
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        logger.info("使用基础检索器")
    
    # 创建检索链 - 使用自定义提示模板
    qa_prompt = PromptTemplate(
        template=QA_PROMPT_TEMPLATE,
        input_variables=["context", "query"]
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=chat,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,  # 返回源文档以便于调试
        chain_type_kwargs={"prompt": qa_prompt},  # 使用自定义提示模板
        verbose=True  # 启用详细日志输出
    )
    
    # 使用消息历史 - 根据Redis连接状态选择存储方式
    if not USE_LOCAL_MODE:
        try:
            # 使用Redis消息历史
            redis_url = f"redis://default:{os.getenv('REDIS_PASSWORD')}@{os.getenv('REDIS_HOST')}:{os.getenv('REDIS_PORT')}"
            message_history = RedisChatMessageHistory(
                url=redis_url,
                session_id=f"{user_host}_{session_id}"
            )
            logger.info(f"为用户 {user_host} 创建Redis会话 {session_id}")
        except Exception as e:
            logger.error(f"创建Redis会话历史失败: {str(e)}")
            # 失败时回退到内存存储
            from langchain_community.chat_message_histories import ChatMessageHistory
            message_history = ChatMessageHistory()
            logger.warning(f"为用户 {user_host} 创建内存会话 {session_id}")
    else:
        # 本地模式 - 使用内存存储
        from langchain_community.chat_message_histories import ChatMessageHistory
        message_history = ChatMessageHistory()
        logger.info(f"本地模式: 为用户 {user_host} 创建内存会话 {session_id}")

    async def event_stream():
        try:
            logger.info(f"开始处理用户输入: {user_input[:50]}...")
            # 检查LLM是否初始化成功
            if chat is None:
                logger.error("LLM模型初始化失败")
                yield f"data: 系统错误：语言模型初始化失败，请联系管理员。\n\n"
                return
                
            # 检查向量数据库是否已加载内容
            if not os.path.exists("./chroma_db"):
                logger.error("向量数据库未初始化")
                yield f"data: 请先上传PDF文档以初始化知识库。\n\n"
                return
            
            # 检查是否是询问PDF内容的请求 - 改进内容摘要识别
            summary_keywords = ["介绍", "内容", "摘要", "概述", "总结", "summary", "overview", "brief"]
            if any(keyword in user_input.lower() for keyword in summary_keywords):
                logger.info("检测到内容摘要请求")
                # 使用MMR检索器获取更多样化的文档样本
                mmr_retriever = get_mmr_retriever(k=5, fetch_k=8, lambda_mult=0.5)  # 增加多样性
                docs = mmr_retriever.get_relevant_documents("")
                
                # 生成更结构化的摘要
                summary_parts = []
                for i, doc in enumerate(docs, 1):
                    # 提取更短的摘要，避免过长
                    content = doc.page_content.strip()
                    summary = content[:150] + "..." if len(content) > 150 else content
                    summary_parts.append(f"• 片段{i}: {summary}")
                
                summary_text = "\n\n".join(summary_parts)
                yield f"data: 以下是PDF文档的主要内容摘要：\n\n{summary_text}\n\n"
                return
            
            # 使用改进的相关性评估方法
            # 1. 先获取相关文档
            # In the chat_endpoint function, replace:
            relevant_docs = retriever.invoke(user_input)
            # With:
            relevant_docs = retriever.invoke(user_input)
            logger.info(f"检索到{len(relevant_docs)}个相关文档")
            
            # 2. 检查文档是否为空
            if not relevant_docs or all(doc.page_content.strip() == "" for doc in relevant_docs):
                logger.warning("未找到相关文档")
                yield f"data: 抱歉，我只能回答与上传PDF文档内容相关的问题。\n\n"
                return
            
            # 3. 计算相似度分数 - 使用向量存储的相似度搜索
            try:
                scores = vectorstore.similarity_search_with_score(user_input, k=3)
                # 使用最佳匹配文档的相似度分数进行判断
                if scores:
                    # 分数越小表示相似度越高
                    best_score = min([score[1] for score in scores])
                    logger.info(f"最佳相似度分数: {best_score}")
                    
                    # 动态相似度阈值 - 根据查询长度调整
                    # 较短的查询需要更严格的相似度要求
                    query_length = len(user_input)
                    threshold = 0.65 if query_length < 10 else 0.75 if query_length < 20 else 0.85
                    
                    if best_score > threshold:
                        logger.warning(f"相似度低于阈值: {best_score} > {threshold}")
                        yield f"data: 抱歉，这个问题与PDF内容关联度不足，请询问与文档更相关的问题。\n\n"
                        return
            except Exception as e:
                logger.error(f"计算相似度时出错: {str(e)}")
                # 出错时继续处理，不中断流程
            
            # 4. 执行QA链获取回答
            logger.info("执行QA链获取回答")
            try:
                response = await qa_chain.acall({"query": user_input})
            except Exception as e:
                logger.error(f"执行QA链时出错: {str(e)}")
                # 检查是否是模型不存在的错误
                if "Model Not Exist" in str(e):
                    logger.error("模型不存在错误，请检查DeepSeek模型名称配置")
                    yield f"data: 系统错误：指定的AI模型不存在，请联系管理员更新配置。\n\n"
                else:
                    yield f"data: 处理您的请求时出现错误，请稍后再试。\n\n"
                return
            
            # 5. 处理响应
            if not response or not response.get('result'):
                logger.warning("QA链返回空响应")
                yield f"data: 抱歉，未能获取到有效回答。\n\n"
            else:
                # 记录源文档信息用于调试
                if 'source_documents' in response:
                    source_docs = response['source_documents']
                    logger.info(f"回答基于{len(source_docs)}个源文档")
                
                # 返回结果
                result = response['result']
                logger.info(f"生成回答: {result[:50]}...")
                yield f"data: {result}\n\n"
                
        except Exception as e:
            logger.error(f"处理请求时出错: {str(e)}")
            yield f"data: [ERROR] 处理您的请求时出现错误，请稍后再试。\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"}
    )

# 在应用启动时检查Redis连接状态
if not USE_LOCAL_MODE and redis_conn:
    try:
        if redis_conn.ping():
            print("Redis连接成功")
        else:
            print("Redis连接失败，但应用将继续在本地模式下运行")
    except Exception as e:
        print(f"Redis连接测试失败: {str(e)}")
        print("应用将在本地模式下运行")
else:
    print("应用将在本地模式下运行，不使用Redis")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)