import json
import os
import time
from typing import Any, Dict, List, Optional, Union, Callable

import httpx

from .mcp_settings import MCPSettings, ModelSettings, SystemRole


class MCPClient:
    """Anthropic API客户端"""
    
    def __init__(self, settings: Optional[MCPSettings] = None):
        """
        初始化客户端
        
        Args:
            settings: 可选的MCPSettings实例。如果为None，将从配置文件加载
        """
        self.settings = settings or MCPSettings.from_config()
        self._client = httpx.Client(timeout=120.0)
        
        # 检查API密钥是否已设置
        if not self.settings.api_key:
            raise ValueError("API密钥未设置。请设置环境变量ANTHROPIC_API_KEY或在配置文件中提供api_key")
    
    def _prepare_headers(self) -> Dict[str, str]:
        """准备请求头"""
        return {
            "Content-Type": "application/json",
            "x-api-key": self.settings.api_key,
            "anthropic-version": "2023-06-01"
        }
    
    def _prepare_messages(self, 
                        prompt: str, 
                        system_prompt: Optional[str] = None,
                        history: Optional[List[Dict[str, str]]] = None) -> List[Dict[str, str]]:
        """
        准备消息列表
        
        Args:
            prompt: 主提示词
            system_prompt: 系统提示词，如果为None则使用模型默认设置
            history: 历史消息列表
            
        Returns:
            消息列表
        """
        messages = []
        
        # 添加系统消息
        if system_prompt:
            messages.append({
                "role": self.settings.system_role.value,
                "content": system_prompt
            })
        
        # 添加历史消息
        if history:
            messages.extend(history)
        
        # 添加用户消息
        messages.append({
            "role": "user",
            "content": prompt
        })
        
        return messages
    
    def _send_request(self, 
                    model_name: str, 
                    messages: List[Dict[str, str]], 
                    stream: bool = False,
                    temperature: Optional[float] = None,
                    top_p: Optional[float] = None,
                    max_tokens: Optional[int] = None) -> Dict[str, Any]:
        """
        发送请求到Anthropic API
        
        Args:
            model_name: 模型名称
            messages: 消息列表
            stream: 是否使用流式响应
            temperature: 温度
            top_p: top-p值
            max_tokens: 最大标记数
            
        Returns:
            API响应
        """
        model_settings = self.settings.models.get(model_name)
        if not model_settings:
            model_settings = ModelSettings(name=model_name)
        
        payload = {
            "model": model_name,
            "messages": messages,
            "stream": stream,
            "temperature": temperature if temperature is not None else model_settings.temperature,
            "top_p": top_p if top_p is not None else model_settings.top_p,
            "max_tokens": max_tokens if max_tokens is not None else model_settings.max_tokens
        }
        
        headers = self._prepare_headers()
        
        try:
            response = self._client.post(
                f"{self.settings.api_url}/v1/messages",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            error_info = ""
            try:
                error_info = e.response.json()
            except:
                error_info = e.response.text
            
            raise Exception(f"API请求失败: {e.response.status_code} - {error_info}")
        except Exception as e:
            raise Exception(f"请求出错: {str(e)}")
    
    def _handle_stream_response(self, 
                              response: httpx.Response, 
                              callback: Optional[Callable[[str], None]] = None) -> str:
        """
        处理流式响应
        
        Args:
            response: HTTP响应
            callback: 处理每个部分响应的回调函数
            
        Returns:
            完整响应文本
        """
        full_text = ""
        
        try:
            for line in response.iter_lines():
                if not line or line.startswith(b":"):
                    continue
                
                if line.startswith(b"data: "):
                    data_str = line[6:].decode("utf-8")
                    if data_str.strip() == "[DONE]":
                        break
                    
                    try:
                        data = json.loads(data_str)
                        if "content" in data and len(data["content"]) > 0:
                            delta = data["content"][0].get("text", "")
                            full_text += delta
                            
                            if callback:
                                callback(delta)
                    except json.JSONDecodeError:
                        pass
        except Exception as e:
            raise Exception(f"处理流式响应出错: {str(e)}")
        
        return full_text
    
    def generate(self,
               prompt: str,
               model_name: Optional[str] = None,
               system_prompt: Optional[str] = None,
               history: Optional[List[Dict[str, str]]] = None,
               stream: bool = False,
               temperature: Optional[float] = None,
               top_p: Optional[float] = None,
               max_tokens: Optional[int] = None,
               callback: Optional[Callable[[str], None]] = None) -> str:
        """
        生成文本响应
        
        Args:
            prompt: 提示词
            model_name: 模型名称，如果为None则使用默认模型
            system_prompt: 系统提示词，如果为None则使用模型默认设置
            history: 历史消息列表
            stream: 是否使用流式响应
            temperature: 温度
            top_p: top-p值
            max_tokens: 最大标记数
            callback: 处理流式响应的回调函数
            
        Returns:
            生成的文本响应
        """
        model_name = model_name or self.settings.default_model
        messages = self._prepare_messages(prompt, system_prompt, history)
        
        # 如果使用流式响应
        if stream:
            headers = self._prepare_headers()
            model_settings = self.settings.models.get(model_name, ModelSettings(name=model_name))
            
            payload = {
                "model": model_name,
                "messages": messages,
                "stream": True,
                "temperature": temperature if temperature is not None else model_settings.temperature,
                "top_p": top_p if top_p is not None else model_settings.top_p,
                "max_tokens": max_tokens if max_tokens is not None else model_settings.max_tokens
            }
            
            try:
                with self._client.stream(
                    "POST",
                    f"{self.settings.api_url}/v1/messages",
                    headers=headers,
                    json=payload,
                    timeout=300.0
                ) as response:
                    response.raise_for_status()
                    return self._handle_stream_response(response, callback)
            except httpx.HTTPStatusError as e:
                error_info = ""
                try:
                    error_info = e.response.json()
                except:
                    error_info = e.response.text
                
                raise Exception(f"API请求失败: {e.response.status_code} - {error_info}")
            except Exception as e:
                raise Exception(f"请求出错: {str(e)}")
        
        # 非流式响应
        response = self._send_request(
            model_name=model_name,
            messages=messages,
            stream=False,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens
        )
        
        return response["content"][0]["text"]
    
    def close(self):
        """关闭客户端"""
        if self._client:
            self._client.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()