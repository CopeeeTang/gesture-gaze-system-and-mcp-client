import os
import json
import asyncio
from typing import List, Dict, Any, Optional, TypedDict, Union
from openai import OpenAI, AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()  # 加载.env文件中的环境变量

class ToolDefinition:
    def __init__(self, name: str, description: str, inputSchema: Dict[str, Any] = {}):
        """初始化一个工具定义
        Args:
            name: 工具名称
            description: 工具描述
            inputSchema: 工具输入模式
        """
        self.name = name
        self.description = description
        self.inputSchema = inputSchema
        
    def to_dict(self) -> Dict[str, Any]:
        """将工具定义转换为字典格式，用于OpenAI API"""
        tool_dict = {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.inputSchema
            }
        }
        return tool_dict

class ChatOpenAI:
    def __init__(self, model: str = "gpt-4o", api_key: Optional[str] = None, 
                 base_url: Optional[str] = None, system_prompt: Optional[str] = None,
                 tools: Optional[List[ToolDefinition]] = None, context: Optional[str] = None,
                 max_tokens: Optional[int] = None):
        """初始化ChatOpenAI类
        
        Args:
            model: 使用的模型，默认为"gpt-4o"
            api_key: OpenAI API密钥，默认从环境变量中获取
            base_url: OpenAI API基础URL
            system_prompt: 系统提示
            tools: 工具列表
            context: 上下文
            max_tokens: 最大生成token数
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("没有找到OPENAI_API_KEY，请设置环境变量或在初始化时提供")
        
        self.model = model
        self.base_url = base_url
        self.system_prompt = system_prompt
        self.tools = tools or []
        self.context = context
        self.max_tokens = max_tokens
        self.temperature = 0.7
        
        # 初始化消息历史
        self.messages: List[Dict[str, Any]] = []
        if system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})
        if context:
            self.messages.append({"role": "system", "content": context})
            
        # 初始化同步和异步客户端
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.async_client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
            
    def list_tools(self) -> None:
        """打印可用工具列表"""
        if not self.tools:
            print("没有可用的工具")
            return
        
        print(f"\n可用工具 ({len(self.tools)}):")
        for i, tool in enumerate(self.tools, 1):
            print(f"{i}. {tool.name}: {tool.description}")
            
    def get_tools_for_api(self) -> List[Dict[str, Any]]:
        """获取用于API的工具列表"""
        return [tool.to_dict() for tool in self.tools]
            
    def append_tool_result(self, tool_call_id: str, result: str) -> None:
        """添加工具调用结果到消息历史
        
        Args:
            tool_call_id: 工具调用ID
            result: 工具调用结果
        """
        self.messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": result
        })
            
    async def chat(self, prompt: Optional[str] = None, temperature: Optional[float] = None) -> Dict[str, Any]:
        """与模型进行对话
        
        Args:
            prompt: 用户提示
            temperature: 温度参数，控制随机性
            
        Returns:
            Dict[str, Any]: 模型响应
        """
        # 如果提供了提示，将其添加到消息历史
        if prompt:
            self.messages.append({"role": "user", "content": prompt})
            
        # 构建请求参数
        # 如果是gpt-4-turbo之前的老模型，用老参数
        request_args = {
            "model": self.model,
            "messages": self.messages,
            "temperature": temperature or self.temperature,
        }
        
        if self.max_tokens:
            request_args["max_tokens"] = self.max_tokens
            
        # 如果有工具，添加到请求参数
        if self.tools:
            request_args["tools"] = self.get_tools_for_api()
            
        try:
            # 调用API
            response = await self.async_client.chat.completions.create(**request_args)
            
            response_message = response.choices[0].message
            
            # 将助手的回复添加到消息历史
            self.messages.append({
                "role": "assistant",
                "content": response_message.content or "",
            })
            
            # 如果有工具调用，添加到消息历史
            if hasattr(response_message, 'tool_calls') and response_message.tool_calls:
                self.messages[-1]["tool_calls"] = response_message.tool_calls
                
            # 构建返回结果
            result = {
                "content": response_message.content,
                "tool_calls": []
            }
            
            # 如果有工具调用，格式化返回
            if hasattr(response_message, 'tool_calls') and response_message.tool_calls:
                for tool_call in response_message.tool_calls:
                    result["tool_calls"].append({
                        "id": tool_call.id,
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments
                        }
                    })
                    
            return result
        except Exception as e:
            print(f"OpenAI API调用失败: {e}")
            return {"content": f"Error: {e}", "tool_calls": []}