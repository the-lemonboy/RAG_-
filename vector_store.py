from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_openai import ChatOpenAI
from pypdf import PdfReader
import os
from logger_config import get_module_logger

# 获取模块日志记录器
logger = get_module_logger("vector_store")

# 全局缓存向量数据库连接
vectorstore_cache = None

def load_pdf_to_vectorstore():
    """直接加载pdf_files/demo.pdf文件到向量数据库，使用优化的文本分割和嵌入策略"""
    try:
        pdf_path = "pdf_files/demo.pdf"
        if not os.path.exists(pdf_path):
            logger.error("默认PDF文件不存在")
            raise FileNotFoundError("默认PDF文件不存在")
            
        logger.info(f"开始处理PDF文件: {pdf_path}")
        pdf_reader = PdfReader(pdf_path)
        text = "\n".join([page.extract_text() for page in pdf_reader.pages])
        if not text.strip():
            logger.error("PDF内容为空或无法提取")
            raise ValueError("PDF内容为空或无法提取")
        
        logger.info(f"成功提取PDF文本，总长度: {len(text)}字符")
            
        # 使用更高级的文本分割器 - 优化块大小和重叠以提高检索质量
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,  # 减小块大小以获得更精确的匹配
            chunk_overlap=150,  # 适当的重叠以保持上下文连贯性
            separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""],  # 优先按自然段落和句子分割
            length_function=len  # 使用字符长度而非token长度
        )
        texts = text_splitter.split_text(text)
        logger.info(f"文本分割完成，共生成{len(texts)}个文本块")
        
        # 使用高质量的嵌入模型
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}  # 归一化嵌入以提高相似度计算质量
        )
        logger.info("嵌入模型初始化完成")
        
        # 存储到向量数据库，添加元数据
        vectorstore = Chroma.from_texts(
            texts=texts,
            embedding=embeddings,
            persist_directory="./chroma_db",
            collection_metadata={"hnsw:space": "cosine"}  # 使用余弦相似度空间
        )
        vectorstore.persist()  # 确保数据持久化
        logger.info("向量数据库创建并持久化完成")
        
        # 验证向量数据库是否存储成功
        test_docs = vectorstore.similarity_search("测试", k=1)
        if not test_docs or not test_docs[0].page_content.strip():
            logger.error("向量数据库存储失败，无法检索测试文档")
            raise RuntimeError("向量数据库存储失败")
        logger.info("向量数据库验证成功")
            
        return vectorstore
    except Exception as e:
        print(f"加载PDF失败: {str(e)}")
        return None

def get_vectorstore():
    """获取或初始化向量数据库，支持MMR检索"""
    global vectorstore_cache
    if vectorstore_cache is None:
        logger.info("初始化向量数据库连接")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        if not os.path.exists("./chroma_db"):
            logger.info("向量数据库不存在，开始创建新的向量数据库")
            vectorstore_cache = load_pdf_to_vectorstore()
        else:
            logger.info("加载现有向量数据库")
            vectorstore_cache = Chroma(
                persist_directory="./chroma_db",
                embedding_function=embeddings
            )
    return vectorstore_cache

def get_mmr_retriever(k=3, fetch_k=5, lambda_mult=0.7):
    """获取使用MMR(最大边际相关性)的检索器，平衡相关性和多样性
    
    参数:
        k: 返回的文档数量
        fetch_k: 初始检索的文档数量，应大于k
        lambda_mult: 权衡相关性(1.0)和多样性(0.0)的参数，默认0.7偏向相关性
    """
    vectorstore = get_vectorstore()
    if not vectorstore:
        logger.error("无法获取向量数据库，MMR检索器创建失败")
        return None
        
    # 使用MMR检索策略
    retriever = vectorstore.as_retriever(
        search_type="mmr",  # 使用MMR搜索而非简单相似度搜索
        search_kwargs={
            "k": k,  # 最终返回的文档数
            "fetch_k": fetch_k,  # 初始检索的文档数，应大于k
            "lambda_mult": lambda_mult  # 相关性与多样性的平衡参数
        }
    )
    logger.info(f"MMR检索器创建成功，参数: k={k}, fetch_k={fetch_k}, lambda_mult={lambda_mult}")
    return retriever

def get_contextual_retriever():
    """获取上下文压缩检索器，可以提取与查询最相关的文档片段"""
    try:
        # 基础检索器
        base_retriever = get_mmr_retriever(k=4, fetch_k=6)
        if not base_retriever:
            return None
            
        # 使用LLM提取器压缩文档
        llm = ChatOpenAI(temperature=0)
        compressor = LLMChainExtractor.from_llm(llm)
        
        # 创建上下文压缩检索器
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever
        )
        logger.info("上下文压缩检索器创建成功")
        return compression_retriever
    except Exception as e:
        logger.error(f"创建上下文压缩检索器失败: {str(e)}")
        # 出错时返回基础检索器
        return get_mmr_retriever()