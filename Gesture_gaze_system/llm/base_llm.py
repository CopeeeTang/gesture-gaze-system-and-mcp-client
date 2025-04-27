from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union

class BaseLLM(ABC):
    """LLM基类，定义所有LLM实现必须支持的接口"""
    
    @abstractmethod
    def chat(self, 
             messages: List[Dict[str, str]], 
             functions: Optional[List[Dict[str, Any]]] = None, 
             temperature: float = 0.7,
             max_tokens: Optional[int] = None) -> Dict[str, Any]:
        """
        与模型进行对话
        
        Args:
            messages (List[Dict[str, str]]): 对话历史，格式为 [{"role": "user", "content": "..."}]
            functions (Optional[List[Dict[str, Any]]]): 函数调用定义，用于function calling
            temperature (float): 温度参数，控制随机性
            max_tokens (Optional[int]): 最大生成token数
            
        Returns:
            Dict[str, Any]: 模型响应，包括文本内容和可能的函数调用
        """
        pass
    
    @abstractmethod
    def function_call(self, 
                     messages: List[Dict[str, str]], 
                     functions: List[Dict[str, Any]],
                     temperature: float = 0.1) -> Dict[str, Any]:
        """
        使用函数调用能力与模型交互
        
        Args:
            messages (List[Dict[str, str]]): 对话历史
            functions (List[Dict[str, Any]]): 函数定义
            temperature (float): 温度参数
            
        Returns:
            Dict[str, Any]: 包含函数调用信息的响应
        """
        pass
    
    @abstractmethod
    def process_multimodal_input(self,
                               image: Optional[Union[str, bytes]] = None,
                               gesture: Optional[str] = None, 
                               gaze: Optional[tuple] = None,
                               text: Optional[str] = None,
                               functions: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        处理多模态输入(图像、手势、眼动、文本)
        
        Args:
            image (Optional[Union[str, bytes]]): 图像数据或路径
            gesture (Optional[str]): 手势信息
            gaze (Optional[tuple]): 眼动数据 (x, y, r)
            text (Optional[str]): 用户文本输入
            functions (Optional[List[Dict[str, Any]]]): 可用函数定义
            
        Returns:
            Dict[str, Any]: 模型响应
        """
        pass