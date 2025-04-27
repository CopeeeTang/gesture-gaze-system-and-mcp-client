import os
import json
import asyncio
import numpy as np
from typing import List, Dict, Any, Optional, Union
from openai import AsyncOpenAI, OpenAI
from dotenv import load_dotenv
from vector_store import VectorStore

load_dotenv()  # load environment variables from .env

class Document:
    """表示文档的类，包含文本内容和元数据"""
    
    def __init__(self, page_content: str, metadata: Dict[str, Any] = None):
        """初始化文档
        
        Args:
            page_content: 文档内容
            metadata: 文档元数据
        """
        self.page_content = page_content
        self.metadata = metadata or {}
        self.embedding: Optional[List[float]] = None

class EmbeddingRetriever:
    """使用OpenAI Embeddings API进行文本检索的类"""
    
    def __init__(self, model: str = "text-embedding-3-small", 
                 api_key: Optional[str] = None,
                 vector_store: Optional[VectorStore] = None,
                 top_k: int = 3):
        """初始化检索器
        
        Args:
            model: 嵌入模型名称
            api_key: OpenAI API密钥
            vector_store: 向量存储实例
            top_k: 返回的最相关文档数量
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("没有找到OPENAI_API_KEY，请设置环境变量或在初始化时提供")
            
        self.model = model
        self.top_k = top_k
        self.vector_store = vector_store or VectorStore()
        
        # 初始化OpenAI客户端
        self.client = OpenAI(api_key=self.api_key)
        self.async_client = AsyncOpenAI(api_key=self.api_key)
    
    async def get_embedding(self, text: str) -> List[float]:
        """获取文本的嵌入向量
        
        Args:
            text: 输入文本
            
        Returns:
            List[float]: 嵌入向量
        """
        # 限制文本长度，避免超出API限制
        text = text[:8000]  # 根据模型的具体限制调整
        
        try:
            response = await self.async_client.embeddings.create(
                input=text,
                model=self.model
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"获取嵌入失败: {e}")
            # 返回一个全零向量作为后备
            return [0.0] * 1536  # 模型的向量维度

    async def add_documents(self, documents: List[Document]) -> None:
        """将文档添加到检索器
        
        Args:
            documents: 文档列表
        """
        for document in documents:
            # 获取文档的嵌入向量
            document.embedding = await self.get_embedding(document.page_content)
            # 将文档添加到向量存储
            self.vector_store.add_document(document)
    
    async def retrieve(self, query: str) -> List[Document]:
        """检索与查询最相关的文档
        
        Args:
            query: 查询文本
            
        Returns:
            List[Document]: 最相关的文档列表
        """
        # 获取查询的嵌入向量
        query_embedding = await self.get_embedding(query)
        
        # 从向量存储中检索最相关的文档
        results = self.vector_store.search(query_embedding, self.top_k)
        
        return results