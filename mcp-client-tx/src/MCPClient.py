import asyncio
from typing import Optional
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters,Tool
from mcp.client.stdio import stdio_client

from dotenv import load_dotenv
from typing import List, Dict, Any, Optional, TypedDict
load_dotenv()  # load environment variables from .env

class MCPClient:
    name: str
    mcp: Optional[ClientSession]
    command: str
    args: List[str]
    tools: List[Tool]
    exit_stack: AsyncExitStack

    def __init__(self,name:str,cmd:str,args:List[str],version:Optional[str]):
        '''
        初始化mcp客户端
        Args:
            name:str 客户端名称
            cmd:str 命令,npx/uv/pnpm/bun  调用的时候记得加run
            args:List[str] 参数
            version:Optional[str] 版本
        关闭客户端需要在类里面，否则会触发线程错误
        '''
        self.name = name
        self.mcp:Optional[ClientSession] = None
        self.command = cmd
        self.transport = None
        self.args = args
        self.tools = []
        # 延迟创建AsyncExitStack，确保它在同一个任务中创建和关闭
        self.exit_stack = None
    
    async def __aenter__(self):
        """实现异步上下文管理器协议"""
        await self.init()
        return self
    
    async def __aexit__(self, exc_type, exc_value, traceback):
        """实现异步上下文管理器的退出方法"""
        await self.close()
        return False  # 不抑制异常
    
    async def close(self):
        '''
        关闭mcp客户端
        '''
        if self.exit_stack:
            try:
                await self.exit_stack.aclose()
                print(f"关闭MCP客户端 {self.name} 成功")
                self.exit_stack = None
            except Exception as e:
                print(f"关闭MCP客户端 {self.name} 时出错: {e}")

    async def init(self):
        '''
        初始化mcp客户端
        '''
        # 确保每次init都创建新的exit_stack
        if self.exit_stack is not None:
            await self.close()
        self.exit_stack = AsyncExitStack()
        await self.connect_to_server()
    
    def get_tools(self):
        '''
        列出可用工具
        '''
        return self.tools

    async def call_tool(self,name:str,params:Dict[str,Any]):
        '''
        调用工具
        '''
        if not self.mcp:
            raise ValueError(f"MCP客户端 {self.name} 未初始化")
        return await self.mcp.call_tool(name,params)

    async def connect_to_server(self):
        """Connect to an MCP server
        优化：不需要拉代码到本地去跑，用uv/pnpm/bun/npx去跑
        Args:
            server_script_path: Path to the server script (.py or .js)
        """
        try:
            if not self.exit_stack:
                self.exit_stack = AsyncExitStack()
                
            server_params = StdioServerParameters(
                command=self.command,
                args=self.args,
                env=None
            )
            self.transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
            self.stdio, self.write = self.transport
            self.mcp = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))

            # 初始化mcp客户端
            await self.mcp.initialize()

            # 列出可用工具
            response = await self.mcp.list_tools()
            self.tools = response.tools
            print(f"\nConnected to server {self.name} with tools:", [tool.name for tool in self.tools])
        except Exception as e:
            print(f"Error connecting to server {self.name}: {e}")
            # 发生错误时，确保资源被清理
            if self.exit_stack:
                try:
                    await self.exit_stack.aclose()
                except Exception:
                    pass
                self.exit_stack = None
            raise e