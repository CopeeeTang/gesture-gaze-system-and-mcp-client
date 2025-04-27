import json
import time
from typing import Dict, List, Optional, Union, AsyncIterator, Iterator, Any

import httpx


class MCPClient:
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.anthropic.com",
        model: str = "claude-3-opus-20240229",
        timeout: float = 60.0,
        max_retries: int = 3,
    ):
        """
        初始化 MCP 客户端

        参数:
            api_key: Anthropic API 密钥
            base_url: API 基础 URL
            model: 使用的模型名称
            timeout: 请求超时时间（秒）
            max_retries: 最大重试次数
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.client = httpx.Client(
            timeout=timeout,
            headers={
                "Content-Type": "application/json",
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
            },
        )
        self.async_client = None

    async def atext(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 1024,
        system: Optional[str] = None,
        temperature: float = 0.7,
        top_p: float = 0.7,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        异步发送文本生成请求

        参数:
            messages: 聊天消息列表
            max_tokens: 生成的最大令牌数
            system: 系统提示
            temperature: 采样温度
            top_p: 核采样概率
            **kwargs: 其他参数传递给API

        返回:
            API 响应
        """
        if self.async_client is None:
            self.async_client = httpx.AsyncClient(
                timeout=self.timeout,
                headers={
                    "Content-Type": "application/json",
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                },
            )

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            **kwargs,
        }

        if system:
            payload["system"] = system

        for attempt in range(self.max_retries):
            try:
                response = await self.async_client.post(
                    f"{self.base_url}/v1/messages",
                    json=payload,
                )
                response.raise_for_status()
                return response.json()
            except httpx.HTTPError as e:
                if attempt == self.max_retries - 1:
                    raise e
                await asyncio.sleep(2 ** attempt)  # 指数退避

    def text(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 1024,
        system: Optional[str] = None,
        temperature: float = 0.7,
        top_p: float = 0.7,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        同步发送文本生成请求

        参数:
            messages: 聊天消息列表
            max_tokens: 生成的最大令牌数
            system: 系统提示
            temperature: 采样温度
            top_p: 核采样概率
            **kwargs: 其他参数传递给API

        返回:
            API 响应
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            **kwargs,
        }

        if system:
            payload["system"] = system

        for attempt in range(self.max_retries):
            try:
                response = self.client.post(
                    f"{self.base_url}/v1/messages",
                    json=payload,
                )
                response.raise_for_status()
                return response.json()
            except httpx.HTTPError as e:
                if attempt == self.max_retries - 1:
                    raise e
                time.sleep(2 ** attempt)  # 指数退避

    async def astream(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 1024,
        system: Optional[str] = None,
        temperature: float = 0.7,
        top_p: float = 0.7,
        **kwargs,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        异步流式生成文本

        参数:
            messages: 聊天消息列表
            max_tokens: 生成的最大令牌数
            system: 系统提示
            temperature: 采样温度
            top_p: 核采样概率
            **kwargs: 其他参数传递给API

        返回:
            生成文本的异步迭代器
        """
        if self.async_client is None:
            self.async_client = httpx.AsyncClient(
                timeout=self.timeout,
                headers={
                    "Content-Type": "application/json",
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                },
            )

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": True,
            **kwargs,
        }

        if system:
            payload["system"] = system

        for attempt in range(self.max_retries):
            try:
                async with self.async_client.stream(
                    "POST",
                    f"{self.base_url}/v1/messages",
                    json=payload,
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        line = line.strip()
                        if not line or line == "data: [DONE]":
                            continue
                        if line.startswith("data: "):
                            line = line[6:]
                            yield json.loads(line)
                return  # 正常退出流
            except httpx.HTTPError as e:
                if attempt == self.max_retries - 1:
                    raise e
                await asyncio.sleep(2 ** attempt)  # 指数退避

    def stream(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 1024,
        system: Optional[str] = None,
        temperature: float = 0.7,
        top_p: float = 0.7,
        **kwargs,
    ) -> Iterator[Dict[str, Any]]:
        """
        同步流式生成文本

        参数:
            messages: 聊天消息列表
            max_tokens: 生成的最大令牌数
            system: 系统提示
            temperature: 采样温度
            top_p: 核采样概率
            **kwargs: 其他参数传递给API

        返回:
            生成文本的迭代器
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": True,
            **kwargs,
        }

        if system:
            payload["system"] = system

        for attempt in range(self.max_retries):
            try:
                with self.client.stream(
                    "POST",
                    f"{self.base_url}/v1/messages",
                    json=payload,
                ) as response:
                    response.raise_for_status()
                    for line in response.iter_lines():
                        line = line.strip()
                        if not line or line == "data: [DONE]":
                            continue
                        if line.startswith("data: "):
                            line = line[6:]
                            yield json.loads(line)
                return  # 正常退出流
            except httpx.HTTPError as e:
                if attempt == self.max_retries - 1:
                    raise e
                time.sleep(2 ** attempt)  # 指数退避

    def close(self):
        """关闭客户端连接"""
        self.client.close()
        if self.async_client:
            self.async_client.aclose()