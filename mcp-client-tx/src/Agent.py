import json
import asyncio  # 确保导入 asyncio
from MCPClient import MCPClient
from chat_openai import ChatOpenAI,ToolDefinition
from utils import log_title
from typing import List, Dict, Any, Optional, TypedDict

class Agent:
    model:str
    system_prompt:str
    context:str
    llm:Optional[ChatOpenAI] = None
    
    def __init__(self,model:str,mcpClient:List[MCPClient],system_prompt:Optional[str]=None,context:Optional[str]=None):
        """初始化Agent实例"""
        self.mcpClient = mcpClient
        self.model = model
        self.system_prompt = system_prompt
        self.context = context
        self._is_initialized = False
        
    async def __aenter__(self):
        """异步上下文管理器的进入方法，初始化Agent。"""
        await self.init()
        return self # 必须返回 self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器的退出方法，关闭Agent。"""
        await self.close()
        
    async def init(self):
        """初始化Agent，包括MCP客户端和LLM。"""
        if self._is_initialized:
            return
            
        log_title("初始化mcp客户端")
        # MCP客户端将单独在外部由调用者管理初始化
        # 确保已经初始化
        
        tools=[]
        for mcpClient in self.mcpClient:
            tools_mcp = mcpClient.get_tools()
            for tool_mcp in tools_mcp:
                # 确保 inputSchema 存在且是字典，否则提供默认空字典
                input_schema = tool_mcp.inputSchema if hasattr(tool_mcp, 'inputSchema') and isinstance(tool_mcp.inputSchema, dict) else {}
                tools.append(ToolDefinition(name=tool_mcp.name,
                                            description=tool_mcp.description,
                                            inputSchema=input_schema))
            
        log_title("初始化llm")
        self.llm = ChatOpenAI(model=self.model,system_prompt=self.system_prompt,tools=tools,context=self.context)
        self.llm.list_tools()
        self._is_initialized = True

    async def close(self):
        """关闭Agent相关资源。MCP客户端将由外部调用者关闭。"""
        log_title("关闭agent资源")
        # MCP客户端的关闭由创建它们的上下文负责，这里不再关闭它们
        self._is_initialized = False
        
    async def invoke(self,prompt:str):
        """调用大模型进行对话，处理工具调用。"""
        if not self._is_initialized:
            await self.init()
            
        if not self.llm:
            raise ValueError("llm未初始化")
        
        log_title("调用llm")
        # 开始对话
        response = await self.llm.chat(prompt)
        
        while True:
            if len(response["tool_calls"]) > 0:
                for tool_call in response["tool_calls"]:
                    found_client = None
                    for client in self.mcpClient:
                        if any(t.name == tool_call["function"]["name"] for t in client.get_tools()):
                        #if client.get_tools().find(lambda t: t.name == tool_call.function.name):
                            found_client = client
                            break
                    if found_client:
                        log_title(f"工具使用: {tool_call['function']['name']}")
                        print(f"Calling tool: {tool_call['function']['name']} with arguments: {tool_call['function']['arguments']}")
                        try:
                            tool_result = await found_client.call_tool(
                                tool_call["function"]["name"],
                                json.loads(tool_call["function"]["arguments"])
                            )
                            print(f"Tool result: {tool_result}")
                            self.llm.append_tool_result(tool_call["id"],str(tool_result))
                        except json.JSONDecodeError as e:
                            print(f"Tool error: {e}")
                            self.llm.append_tool_result(tool_call["id"],f"Invalid JSON arguments: {e}")
                        except Exception as e:
                            print(f"Error calling tool {tool_call['function']['name']}: {e}")
                            self.llm.append_tool_result(tool_call["id"],f'Error: {e}')
                    else:
                        self.llm.append_tool_result(tool_call["id"],"Tool not found")
                response = await self.llm.chat()
            else:
                return response["content"]