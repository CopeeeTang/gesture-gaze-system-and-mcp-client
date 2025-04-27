# 手势凝视系统和MCP客户端

这个仓库包含了两个项目：
1. **Gesture_gaze_system**: 一个使用手势识别和眼动跟踪与LLM交互的系统
2. **mcp-client-tx**: 一个基于Model Context Protocol (MCP) 的客户端实现

## Gesture_gaze_system

一个多模态交互系统，能够通过捕捉手势、眼动和图像与LLM进行交互。

### 主要特性
- 支持多种LLM模型（Phi-4, QwenOmni, OpenAI等）
- 多模态输入（手势、眼动、图像、文本）
- MCP(Model Context Protocol)客户端集成

### 安装

```bash
# Linux/Mac
cd Gesture_gaze_system
bash run.sh

# Windows
cd Gesture_gaze_system
run.bat
```

## mcp-client-tx

一个基于MCP（Model Context Protocol）的客户端实现，可以集成多个MCP服务器并支持RAG（检索增强生成）。

### 主要特性
- MCP客户端集成
- 支持RAG检索
- 支持工具调用
- 支持多种检索方法（向量存储、嵌入检索）

### 安装

```bash
# Linux/Mac
cd mcp-client-tx
bash run.sh

# Windows
cd mcp-client-tx
run.bat
```

## 关于MCP (Model Context Protocol)

MCP是一个开放标准，用于模型、上下文和工具之间的通信。它允许LLM与外部工具和资源交互，扩展大语言模型的能力范围。

更多信息: [MCP官方网站](https://modelcontextprotocol.io/)

## 许可

这些项目使用MIT许可证。