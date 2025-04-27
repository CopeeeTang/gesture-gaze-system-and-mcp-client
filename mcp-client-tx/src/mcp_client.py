import json
import asyncio
import websockets
import nest_asyncio
from typing import Dict, List, Any, Optional, Callable, Awaitable, Union
from uuid import uuid4

# 应用nest_asyncio以便在Jupyter等环境中支持asyncio
nest_asyncio.apply()

class MCPClient:
    """Model Context Protocol客户端实现"""
    
    def __init__(self, websocket_uri: str = "ws://localhost:8765"):
        """初始化MCP客户端
        
        Args:
            websocket_uri: WebSocket服务器URI
        """
        self.websocket_uri = websocket_uri
        self.websocket = None
        self.is_connected = False
        self.message_handlers = {}
        self.response_futures = {}
        self.stream_handlers = {}
        self.heartbeat_task = None
    
    async def connect(self) -> bool:
        """连接到MCP服务器
        
        Returns:
            bool: 连接是否成功
        """
        try:
            self.websocket = await websockets.connect(self.websocket_uri)
            self.is_connected = True
            print(f"已连接到MCP服务器: {self.websocket_uri}")
            
            # 启动消息处理循环
            asyncio.create_task(self._message_handler())
            
            # 启动心跳
            self.heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            
            return True
        except Exception as e:
            print(f"连接MCP服务器失败: {e}")
            self.is_connected = False
            return False
    
    async def disconnect(self) -> None:
        """断开与MCP服务器的连接"""
        if self.is_connected and self.websocket:
            # 取消心跳任务
            if self.heartbeat_task:
                self.heartbeat_task.cancel()
                self.heartbeat_task = None
            
            await self.websocket.close()
            self.websocket = None
            self.is_connected = False
            print("已断开与MCP服务器的连接")
    
    async def call_tool(self, 
                         tool_name: str, 
                         parameters: Dict[str, Any],
                         timeout: float = 60) -> Dict[str, Any]:
        """调用MCP工具并等待结果
        
        Args:
            tool_name: 工具名称
            parameters: 工具参数
            timeout: 超时时间（秒）
            
        Returns:
            Dict[str, Any]: 工具调用结果
            
        Raises:
            TimeoutError: 如果调用超时
            ConnectionError: 如果未连接到服务器
        """
        if not self.is_connected or not self.websocket:
            raise ConnectionError("未连接到MCP服务器")
        
        # 生成请求ID
        request_id = str(uuid4())
        
        # 创建请求消息
        message = {
            "type": "tool_call",
            "request_id": request_id,
            "tool_name": tool_name,
            "parameters": parameters
        }
        
        # 创建Future对象用于等待响应
        future = asyncio.Future()
        self.response_futures[request_id] = future
        
        # 发送请求
        await self.websocket.send(json.dumps(message))
        
        try:
            # 等待响应，设置超时
            response = await asyncio.wait_for(future, timeout)
            return response
        except asyncio.TimeoutError:
            # 超时时从响应futures中移除
            self.response_futures.pop(request_id, None)
            raise TimeoutError(f"工具调用超时: {tool_name}")
        finally:
            # 确保从响应futures中移除
            self.response_futures.pop(request_id, None)
    
    async def stream_tool_calls(self, 
                              tool_name: str, 
                              parameters: Dict[str, Any],
                              handler: Callable[[str, Any], Awaitable[None]]) -> str:
        """流式调用MCP工具
        
        Args:
            tool_name: 工具名称
            parameters: 工具参数
            handler: 处理流式响应的回调函数，接收stream_id和数据
            
        Returns:
            str: 流ID，可用于取消流
            
        Raises:
            ConnectionError: 如果未连接到服务器
        """
        if not self.is_connected or not self.websocket:
            raise ConnectionError("未连接到MCP服务器")
        
        # 生成流ID
        stream_id = str(uuid4())
        
        # 创建请求消息
        message = {
            "type": "stream_tool_call",
            "stream_id": stream_id,
            "tool_name": tool_name,
            "parameters": parameters
        }
        
        # 注册流处理程序
        self.stream_handlers[stream_id] = handler
        
        # 发送请求
        await self.websocket.send(json.dumps(message))
        return stream_id
    
    async def cancel_stream(self, stream_id: str) -> bool:
        """取消流式工具调用
        
        Args:
            stream_id: 流ID
            
        Returns:
            bool: 取消是否成功
        """
        if not self.is_connected or not self.websocket:
            return False
        
        # 创建取消消息
        message = {
            "type": "cancel_stream",
            "stream_id": stream_id
        }
        
        # 发送请求
        await self.websocket.send(json.dumps(message))
        
        # 移除流处理程序
        self.stream_handlers.pop(stream_id, None)
        return True
    
    async def send_heartbeat(self) -> None:
        """发送心跳消息以保持连接"""
        if self.is_connected and self.websocket:
            message = {
                "type": "heartbeat"
            }
            await self.websocket.send(json.dumps(message))
    
    async def _heartbeat_loop(self) -> None:
        """心跳循环任务"""
        while self.is_connected and self.websocket:
            await self.send_heartbeat()
            await asyncio.sleep(30)  # 每30秒发送一次心跳
    
    async def _message_handler(self) -> None:
        """处理来自服务器的消息"""
        while self.is_connected and self.websocket:
            try:
                # 接收消息
                message_str = await self.websocket.recv()
                message = json.loads(message_str)
                
                # 根据消息类型处理
                message_type = message.get("type")
                
                if message_type == "tool_response":
                    # 处理工具响应
                    request_id = message.get("request_id")
                    if request_id in self.response_futures:
                        future = self.response_futures[request_id]
                        if not future.done():
                            future.set_result(message.get("result"))
                
                elif message_type == "stream_data":
                    # 处理流数据
                    stream_id = message.get("stream_id")
                    if stream_id in self.stream_handlers:
                        handler = self.stream_handlers[stream_id]
                        await handler(stream_id, message.get("data"))
                
                elif message_type == "stream_end":
                    # 处理流结束
                    stream_id = message.get("stream_id")
                    self.stream_handlers.pop(stream_id, None)
                
                elif message_type == "error":
                    # 处理错误
                    request_id = message.get("request_id")
                    if request_id in self.response_futures:
                        future = self.response_futures[request_id]
                        if not future.done():
                            future.set_exception(Exception(message.get("error")))
                    # 如果是流错误，可能还需要额外处理
                
                elif message_type == "heartbeat_ack":
                    # 心跳确认，可以记录但不需要特殊处理
                    pass
                
            except websockets.exceptions.ConnectionClosed:
                # 连接关闭，断开客户端
                self.is_connected = False
                self.websocket = None
                print("与MCP服务器的连接已关闭")
                break
            except Exception as e:
                print(f"处理消息时出错: {e}")
    
    # 实用方法
    def is_tool_available(self, tool_name: str) -> bool:
        """检查工具是否可用
        
        Args:
            tool_name: 工具名称
            
        Returns:
            bool: 工具是否可用
        """
        # 这里可以实现工具可用性检查，例如缓存可用工具列表
        # 简单实现，始终返回True
        return True
    
    async def get_available_tools(self) -> List[str]:
        """获取可用工具列表
        
        Returns:
            List[str]: 可用工具名称列表
        """
        try:
            result = await self.call_tool("mcp_system_list_tools", {})
            return result.get("tools", [])
        except Exception as e:
            print(f"获取可用工具列表失败: {e}")
            return []