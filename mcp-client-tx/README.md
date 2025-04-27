# LLM + MCP + RAG (Python版本)
python

## 目标

本项目演示了如何使用 Python 构建一个结合了多个 MCP (Model Context Protocol) 客户端的 Agent，并在此基础上进行 RAG (Retrieval-Augmented Generation) 操作。

## **The augmented LLM**

- [Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents)

## **依赖**

```bash
# 使用uv进行环境隔离
pip install uv
uv venv
source .venv/bin/activate  # Linux/MacOS
# .venv\Scripts\activate    # Windows

# 安装依赖
uv pip install -r requirements.txt
```

## LLM

- [OpenAI API](https://platform.openai.com/docs/api-reference/chat)

## MCP

- [MCP 架构](https://modelcontextprotocol.io/docs/concepts/architecture)
- [MCP Client](https://modelcontextprotocol.io/quickstart/client)
- [Fetch MCP](https://github.com/modelcontextprotocol/servers/tree/main/src/fetch)
- [Filesystem MCP](https://github.com/modelcontextprotocol/servers/tree/main/src/filesystem)

## RAG

- [Retrieval Augmented Generation](https://scriv.ai/guides/retrieval-augmented-generation-overview/)
    - 译文: https://www.yuque.com/serviceup/misc/cn-retrieval-augmented-generation-overview
- 各种Loaders: https://python.langchain.com/docs/integrations/document_loaders/
- [硅基流动](https://cloud.siliconflow.cn/models)
    - 邀请码： **x771DtAF**
- [json数据](https://jsonplaceholder.typicode.com/)

## 向量

- 维度
- 模长
- 点乘 Dot Product
    - 对应位置元素的积，求和
- 余弦相似度 cos
    - 1 → 方向完全一致
    - 0 → 垂直
    - -1 → 完全想法

## 使用方法

1. 准备环境
   ```bash
   # 复制示例环境配置文件并修改为你的配置
   cp .env.example .env
   # 编辑.env文件，填入你的API密钥
   ```

2. 运行示例
   ```bash
   python src/main.py
   ```

## 项目结构

```
llm-mcp-rag-python/
├── .env.example        # 环境变量示例
├── requirements.txt    # 项目依赖
├── README.md           # 项目说明
├── src/                # 源代码
│   ├── main.py         # 入口文件
│   ├── agent.py        # Agent类
│   ├── chat_openai.py  # ChatOpenAI类
│   ├── embedding_retriever.py # EmbeddingRetriever类
│   ├── mcp_client.py   # MCPClient类
│   ├── vector_store.py # VectorStore类
│   └── utils.py        # 工具函数
├── knowledge/          # 知识库文件
├── output/             # 输出结果
└── images/             # 图片资源
```