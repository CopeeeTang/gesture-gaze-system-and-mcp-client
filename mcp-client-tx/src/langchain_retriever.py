import os
import asyncio
import nest_asyncio
from typing import List, Dict, Any, Optional, Union
from dotenv import load_dotenv
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema.document import Document

# 应用nest_asyncio以在Jupyter环境中支持asyncio
nest_asyncio.apply()

# 加载环境变量
load_dotenv()

class LangChainRetriever:
    """使用LangChain和FAISS进行检索的类"""
    
    def __init__(self, 
                 embedding_model: str = "text-embedding-3-small",
                 api_key: Optional[str] = None,
                 top_k: int = 3):
        """初始化LangChain检索器
        
        Args:
            embedding_model: 嵌入模型名称
            api_key: OpenAI API密钥
            top_k: 返回的最相关文档数量
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("没有找到OPENAI_API_KEY，请设置环境变量或在初始化时提供")
            
        self.embedding_model = embedding_model
        self.top_k = top_k
        
        # 初始化OpenAI嵌入
        self.embeddings = OpenAIEmbeddings(
            model=embedding_model,
            openai_api_key=self.api_key
        )
        
        # 初始化FAISS向量存储
        self.vectorstore = None
    
    async def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """将文档添加到检索器
        
        Args:
            documents: 文档列表，每个文档是一个字典，包含页面内容和元数据
        """
        # 转换为LangChain文档格式
        langchain_docs = []
        for doc in documents:
            langchain_docs.append(
                Document(
                    page_content=doc['page_content'],
                    metadata=doc['metadata']
                )
            )
        
        # 将文档添加到FAISS向量存储
        if self.vectorstore is None:
            # 第一次添加文档，创建向量存储
            loop = asyncio.get_event_loop()
            self.vectorstore = await loop.run_in_executor(
                None,
                lambda: FAISS.from_documents(langchain_docs, self.embeddings)
            )
        else:
            # 已有向量存储，添加到现有存储
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self.vectorstore.add_documents(langchain_docs)
            )
    
    async def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """检索与查询最相关的文档
        
        Args:
            query: 查询文本
            
        Returns:
            List[Dict[str, Any]]: 最相关的文档列表
        """
        if self.vectorstore is None:
            return []
        
        # 执行相似度搜索
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None,
            lambda: self.vectorstore.similarity_search_with_score(query, k=self.top_k)
        )
        
        # 格式化返回结果
        formatted_results = []
        for doc, score in results:
            formatted_results.append({
                'page_content': doc.page_content,
                'metadata': doc.metadata,
                'score': float(score)  # 将numpy.float64转换为Python float
            })
            
        return formatted_results
    
    # 以下是为了方便调试和查看索引信息的方法
    
    def get_index_stats(self) -> Dict[str, Any]:
        """获取索引统计信息
        
        Returns:
            Dict[str, Any]: 索引统计信息
        """
        if self.vectorstore is None:
            return {"status": "未初始化"}
            
        try:
            index = self.vectorstore.index
            return {
                "文档数量": index.ntotal,
                "维度": index.d,
                "索引类型": str(type(index))
            }
        except Exception as e:
            return {"错误": str(e)}
            
    async def search_by_vector(self, vector: List[float], top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """通过向量搜索文档
        
        Args:
            vector: 查询向量
            top_k: 返回的最相关文档数量
            
        Returns:
            List[Dict[str, Any]]: 最相关的文档列表
        """
        if self.vectorstore is None:
            return []
            
        k = top_k or self.top_k
        
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None,
            lambda: self.vectorstore.similarity_search_by_vector(vector, k=k)
        )
        
        # 格式化返回结果
        formatted_results = []
        for doc in results:
            formatted_results.append({
                'page_content': doc.page_content,
                'metadata': doc.metadata
            })
            
        return formatted_results