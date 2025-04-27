import os
import json
import asyncio
from typing import Dict, List, Any, Optional, Union, Callable
from .mcp_client import MCPClient

class PromptTools:
    """MCP提示工具的实用工具和封装器类"""
    
    def __init__(self, mcp_client: MCPClient):
        """初始化PromptTools
        
        Args:
            mcp_client: MCP客户端实例
        """
        self.client = mcp_client
    
    async def user_input(self, prompt: str, timeout: float = 60) -> str:
        """通过MCP获取用户输入
        
        Args:
            prompt: 提示用户的消息
            timeout: 超时时间（秒）
            
        Returns:
            str: 用户输入的文本
        """
        try:
            result = await self.client.call_tool(
                "mcp_prompt_tools_prompt_user_for_input", 
                {"prompt": prompt},
                timeout=timeout
            )
            return result.get("response", "")
        except Exception as e:
            print(f"获取用户输入失败: {e}")
            return ""
    
    async def user_confirm(self, 
                           prompt: str, 
                           confirm_button_text: str = "确认",
                           cancel_button_text: str = "取消",
                           timeout: float = 60) -> bool:
        """向用户请求确认
        
        Args:
            prompt: 提示用户的消息
            confirm_button_text: 确认按钮文本
            cancel_button_text: 取消按钮文本
            timeout: 超时时间（秒）
            
        Returns:
            bool: 用户是否确认
        """
        try:
            result = await self.client.call_tool(
                "mcp_prompt_tools_prompt_user_for_confirmation", 
                {
                    "prompt": prompt,
                    "confirmButtonText": confirm_button_text,
                    "cancelButtonText": cancel_button_text
                },
                timeout=timeout
            )
            return result.get("confirmed", False)
        except Exception as e:
            print(f"获取用户确认失败: {e}")
            return False
    
    async def user_choice(self, 
                          prompt: str, 
                          choices: List[str],
                          timeout: float = 60) -> str:
        """提示用户从选项中选择
        
        Args:
            prompt: 提示用户的消息
            choices: 选项列表
            timeout: 超时时间（秒）
            
        Returns:
            str: 用户选择的选项
        """
        try:
            result = await self.client.call_tool(
                "mcp_prompt_tools_prompt_user_for_choice", 
                {
                    "prompt": prompt,
                    "choices": choices
                },
                timeout=timeout
            )
            return result.get("choice", "")
        except Exception as e:
            print(f"获取用户选择失败: {e}")
            return ""
    
    async def show_message(self, message: str, timeout: float = 10) -> bool:
        """向用户显示消息
        
        Args:
            message: 要显示的消息
            timeout: 超时时间（秒）
            
        Returns:
            bool: 消息是否成功显示
        """
        try:
            result = await self.client.call_tool(
                "mcp_prompt_tools_show_message", 
                {"message": message},
                timeout=timeout
            )
            return True
        except Exception as e:
            print(f"显示消息失败: {e}")
            return False
    
    async def show_panel(self, 
                       html_content: str, 
                       title: str = "信息面板",
                       timeout: float = 10) -> bool:
        """向用户显示自定义HTML面板
        
        Args:
            html_content: HTML内容
            title: 面板标题
            timeout: 超时时间（秒）
            
        Returns:
            bool: 面板是否成功显示
        """
        try:
            result = await self.client.call_tool(
                "mcp_prompt_tools_show_panel", 
                {
                    "title": title,
                    "htmlContent": html_content
                },
                timeout=timeout
            )
            return True
        except Exception as e:
            print(f"显示面板失败: {e}")
            return False
    
    async def upload_file(self, 
                         prompt: str = "请上传文件",
                         file_filter: str = "",
                         timeout: float = 300) -> Dict[str, Any]:
        """请求用户上传文件
        
        Args:
            prompt: 提示信息
            file_filter: 文件过滤器（例如 ".txt,.docx"）
            timeout: 超时时间（秒）
            
        Returns:
            Dict[str, Any]: 文件信息，包含路径、名称、大小等
        """
        try:
            result = await self.client.call_tool(
                "mcp_prompt_tools_prompt_user_for_file_upload", 
                {
                    "prompt": prompt,
                    "fileFilter": file_filter
                },
                timeout=timeout
            )
            return result or {}
        except Exception as e:
            print(f"文件上传失败: {e}")
            return {}
    
    async def save_file(self, 
                       content: Union[str, bytes], 
                       suggested_filename: str = "",
                       file_filter: str = "",
                       timeout: float = 60) -> bool:
        """让用户将内容保存为文件
        
        Args:
            content: 文件内容（字符串或字节）
            suggested_filename: 建议的文件名
            file_filter: 文件过滤器（例如 ".txt,.docx"）
            timeout: 超时时间（秒）
            
        Returns:
            bool: 文件是否成功保存
        """
        # 如果内容是字节，需要进行Base64编码
        content_type = "text"
        content_value = content
        
        if isinstance(content, bytes):
            import base64
            content_type = "base64"
            content_value = base64.b64encode(content).decode('utf-8')
        
        try:
            result = await self.client.call_tool(
                "mcp_prompt_tools_prompt_user_for_file_save", 
                {
                    "content": content_value,
                    "contentType": content_type,
                    "suggestedFilename": suggested_filename,
                    "fileFilter": file_filter
                },
                timeout=timeout
            )
            return result.get("saved", False)
        except Exception as e:
            print(f"文件保存失败: {e}")
            return False
    
    async def clipboard_read(self, timeout: float = 10) -> str:
        """读取用户剪贴板内容
        
        Args:
            timeout: 超时时间（秒）
            
        Returns:
            str: 剪贴板内容
        """
        try:
            result = await self.client.call_tool(
                "mcp_prompt_tools_read_clipboard", 
                {},
                timeout=timeout
            )
            return result.get("content", "")
        except Exception as e:
            print(f"读取剪贴板失败: {e}")
            return ""
    
    async def clipboard_write(self, content: str, timeout: float = 10) -> bool:
        """写入内容到用户剪贴板
        
        Args:
            content: 要写入的内容
            timeout: 超时时间（秒）
            
        Returns:
            bool: 是否成功写入
        """
        try:
            result = await self.client.call_tool(
                "mcp_prompt_tools_write_clipboard", 
                {"content": content},
                timeout=timeout
            )
            return True
        except Exception as e:
            print(f"写入剪贴板失败: {e}")
            return False