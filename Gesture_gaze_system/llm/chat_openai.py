import os
import json
from typing import Dict, List, Optional, Any, Union
import openai

from .base_llm import BaseLLM
from .utils import create_multimodal_message

class OpenAILLM(BaseLLM):
    """OpenAI API调用实现"""
    
    def __init__(self, api_key=None, model="gpt-4o"):
        """
        初始化OpenAI客户端
        
        Args:
            api_key (str, optional): OpenAI API密钥，如果不指定则从环境变量获取
            model (str, optional): 模型名称，默认为gpt-4o
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("未提供OpenAI API密钥，请在.env文件中设置OPENAI_API_KEY或在初始化时提供")
            
        self.model = model
        self.client = openai.OpenAI(api_key=self.api_key,base_url=os.environ.get("OPENAI_BASE_URL"))
        print(f"OpenAI客户端初始化完成，使用模型: {model}")
        
    def chat(self, 
             messages: List[Dict[str, str]], 
             functions: Optional[List[Dict[str, Any]]] = None, 
             temperature: float = 0.7,
             max_tokens: Optional[int] = None) -> Dict[str, Any]:
        """
        基础对话功能实现
        
        Args:
            messages (List[Dict[str, str]]): 对话历史
            functions (Optional[List[Dict[str, Any]]]): 函数定义
            temperature (float): 温度参数
            max_tokens (Optional[int]): 最大生成token数
            
        Returns:
            Dict[str, Any]: 模型响应
        """
        # 参数准备
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        
        if max_tokens:
            kwargs["max_tokens"] = max_tokens
            
        # 如果提供了functions，添加到请求中
        if functions:
            kwargs["tools"] = [{"type": "function", "function": f} for f in functions]
            kwargs["tool_choice"] = "auto"
            
        # 发送请求
        response = self.client.chat.completions.create(**kwargs)
        
        # 处理响应
        message = response.choices[0].message
        result = {
            "role": "assistant",
            "content": message.content
        }
        
        # 如果有工具调用
        if hasattr(message, 'tool_calls') and message.tool_calls:
            tool_call = message.tool_calls[0]
            if tool_call.type == 'function':
                result["content"] = None
                result["function_call"] = {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments
                }
                
        return result
    
    def function_call(self, 
                     messages: List[Dict[str, str]], 
                     functions: List[Dict[str, Any]],
                     temperature: float = 0.2) -> Dict[str, Any]:
        """
        使用函数调用能力与模型交互
        
        Args:
            messages (List[Dict[str, str]]): 对话历史
            functions (List[Dict[str, Any]]): 函数定义
            temperature (float): 温度参数
            
        Returns:
            Dict[str, Any]: 包含函数调用信息的响应
        """
        return self.chat(messages, functions, temperature)
    
    def process_multimodal_input(self,
                               image: Optional[Union[str, bytes]] = None,
                               gesture: Optional[str] = None, 
                               gaze: Optional[tuple] = None,
                               text: Optional[str] = None,
                               functions: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        处理多模态输入
        
        Args:
            image (Optional[Union[str, bytes]]): 图像数据或路径
            gesture (Optional[str]): 手势信息
            gaze (Optional[tuple]): 眼动数据 (x, y, r)
            text (Optional[str]): 用户文本输入
            functions (Optional[List[Dict[str, Any]]]): 可用函数定义
            
        Returns:
            Dict[str, Any]: 模型响应
        """
        # 确保使用的是支持多模态的模型
        if not (self.model.startswith("gpt-4") and ("vision" in self.model or "-o" in self.model)):
            print(f"警告: 当前模型 {self.model} 可能不支持多模态输入，推荐使用gpt-4-vision或gpt-4o")
            
        # 创建多模态消息
        multimodal_message = create_multimodal_message(image, gesture, gaze, text)
        
        # 构建消息列表
        messages = []
        
        # 添加系统提示词
        system_message = """你是一个能够理解多模态输入的AI助手。你将接收图像、手势信息和眼动数据，
通过综合分析这些输入来理解用户意图并执行适当的操作。

手势可能是以下之一:
- pinch: 捏合手势，通常表示选择或确认
- double pinch: 双击捏合手势，通常表示执行特殊操作
- grip: 抓握手势，通常表示抓取或拖动
- twist left: 左扭手势，通常表示向左旋转或返回
- twist right: 右扭手势，通常表示向右旋转或前进
- thumb up: 竖起大拇指，通常表示肯定或赞同
- thumb down: 竖起大拇指，通常表示否定或不赞同

眼动数据表示为(x,y,r)，其中(x,y)是用户注视的坐标，r是注视区域的半径。"""

        messages.append({"role": "system", "content": system_message})
        messages.append(multimodal_message)
        
        # 调用chat方法
        return self.chat(messages, functions)