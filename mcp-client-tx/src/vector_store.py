import os
import numpy as np
from typing import List, Dict, Any, Optional

class VectorStore:
    """简单的向量存储类，用于存储文档及其嵌入向量"""
    
    def __init__(self, embedding_dim: int = 1536):
        """初始化向量存储
        
        Args:
            embedding_dim: 嵌入向量的维度
        """
        self.embedding_dim = embedding_dim
        self.documents = []  # 存储所有文档
        self.vectors = []    # 存储所有嵌入向量
        
    def add_document(self, document) -> None:
        """将文档添加到向量存储
        
        Args:
            document: 包含页面内容、元数据和嵌入向量的文档对象
        """
        if not hasattr(document, 'embedding') or document.embedding is None:
            raise ValueError("文档必须包含嵌入向量才能添加到向量存储")
            
        self.documents.append(document)
        self.vectors.append(document.embedding)
        
    def add_documents(self, documents: List) -> None:
        """将多个文档添加到向量存储
        
        Args:
            documents: 文档列表
        """
        for document in documents:
            self.add_document(document)
            
    def cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """计算两个向量之间的余弦相似度
        
        Args:
            a: 第一个向量
            b: 第二个向量
            
        Returns:
            float: 余弦相似度
        """
        a = np.array(a)
        b = np.array(b)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def search(self, query_vector: List[float], top_k: int = 3) -> List:
        """搜索与查询向量最相似的文档
        
        Args:
            query_vector: 查询向量
            top_k: 返回的最相似文档数量
            
        Returns:
            List: 最相似的文档列表
        """
        if not self.documents:
            return []
            
        # 计算查询向量与所有文档向量的相似度
        similarities = []
        for vector in self.vectors:
            similarity = self.cosine_similarity(query_vector, vector)
            similarities.append(similarity)
            
        # 找到top_k个最相似的文档
        indices = np.argsort(similarities)[::-1][:top_k]
        
        # 返回最相似的文档
        results = []
        for idx in indices:
            results.append(self.documents[idx])
            
        return results
        
    def clear(self) -> None:
        """清空向量存储"""
        self.documents = []
        self.vectors = []
        
    def size(self) -> int:
        """获取向量存储中的文档数量
        
        Returns:
            int: 文档数量
        """
        return len(self.documents)