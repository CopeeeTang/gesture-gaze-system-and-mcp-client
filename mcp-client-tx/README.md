# MCP客户端

这是一个用于与Model Control Protocol (MCP)交互的Python客户端实现。

## 安装

```bash
pip install -e .
```

## 使用方法

### 初始化客户端

```python
from mcp_client import MCPClient

# 创建客户端实例
client = MCPClient(
    api_key="your_api_key_here",
    base_url="https://api.anthropic.com",
    model="claude-3-haiku-20240307"
)
```

### 生成文本

```python
# 同步方式调用
response = client.generate(
    messages=[
        {"role": "user", "content": "你好，请介绍一下自己。"}
    ],
    max_tokens=1000
)
print(response)

# 异步方式调用
async def generate_async():
    response = await client.generate_async(
        messages=[
            {"role": "user", "content": "你好，请介绍一下自己。"}
        ],
        max_tokens=1000
    )
    print(response)

# 流式响应
for chunk in client.generate_stream(
    messages=[
        {"role": "user", "content": "你好，请介绍一下自己。"}
    ],
    max_tokens=1000
):
    print(chunk, end="", flush=True)

# 异步流式响应
async def generate_stream_async():
    async for chunk in client.generate_stream_async(
        messages=[
            {"role": "user", "content": "你好，请介绍一下自己。"}
        ],
        max_tokens=1000
    ):
        print(chunk, end="", flush=True)
```

### 关闭客户端

```python
# 关闭客户端，释放资源
client.close()

# 在异步环境中关闭客户端
await client.aclose()
```

## 参数说明

- `api_key`: Anthropic API密钥
- `base_url`: API基础URL，默认为Anthropic API
- `model`: 使用的模型名称，默认为"claude-3-haiku-20240307"
- `timeout`: 请求超时时间（秒）
- `max_retries`: 最大重试次数