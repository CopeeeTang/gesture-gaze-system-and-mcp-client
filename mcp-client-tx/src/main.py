import asyncio
import datetime
import os
from openai import OpenAI
from chat_openai import ChatOpenAI
from MCPClient import MCPClient
from Agent import Agent
from vector_store import VectorStore
from embedding_retriever import EmbeddingRetriever, Document
from langchain_retriever import LangChainRetriever

async def test_chat():
    """测试基本的聊天功能"""
    try:
        llm = ChatOpenAI(
            model="gpt-4o",
            system_prompt="你是一个AI助手，请根据用户的问题给出回答。"
        )
        response = await llm.chat(prompt="你好，向我详细的介绍一下三角函数")

        print("\n基础聊天测试成功:", response)
    except Exception as e:
        print("基础聊天测试失败:", str(e))

def get_current_time():
    """获取当前时间的工具函数"""
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"当前时间是: {current_time}"

async def test_tool_call():
    """测试工具调用功能"""
    try:
        llm = ChatOpenAI(
            model="gpt-4o",
            system_prompt="你是一个AI助手，请根据用户的问题给出回答。",
            tools=[{
                "name": "get_current_time",
                "description": "获取当前时间",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }]
        )
        response = await llm.chat(prompt="请获取当前时间")
        print("\n工具调用测试成功:", response)
        
        if response["tool_calls"]:
            for tool_call in response["tool_calls"]:
                if tool_call["function"]["name"] == "get_current_time":
                    # 执行工具函数
                    tool_result = get_current_time()
                    # 将工具结果添加到对话
                    llm.append_tool_result(tool_call["id"], tool_result)
                    # 获取助手对工具执行结果的回复
                    final_response = await llm.chat()
                    print("\n工具执行结果:", tool_result)
                    print("助手回复:", final_response)
    except Exception as e:
        print("工具调用测试失败:", str(e))

async def test_mcp():
    """测试mcp功能"""
    
    mcp = MCPClient('fetch', 'uv', ["run", "mcp-server-fetch"], None)
    await mcp.init()
    print(mcp.get_tools())

    await mcp.close()

async def test_agent_fetch_and_save():
    """测试使用Agent集成fetch和filesystem两个MCP客户端，使用 async with 管理生命周期"""
    try:
        # 使用async with创建和管理MCP客户端
        async with (
            MCPClient('fetch', 'uv', ["run", "mcp-server-fetch"], None) as fetch_client,
            MCPClient('filesystem', 'npx', ["-y", "@modelcontextprotocol/server-filesystem", "/home/lab/AgentExplore"], None) as filesystem_client,
        ):
            # 使用 async with 管理 Agent 的生命周期
            async with Agent(
                model="gpt-4o",
                mcpClient=[fetch_client, filesystem_client],
                system_prompt="""你是一个强大的助手，能够通过fetch工具获取网页内容，并使用filesystem工具将内容保存到文件中。
请仔细阅读用户的指令，并按照要求完成任务。""",
            ) as agent:
                
                # 发送指令
                prompt = """请执行以下任务：
        1. 使用fetch工具爬取https://modelcontextprotocol.io/llms-full.txt 和 https://zh.wikipedia.org/wiki/梁启超
        2. 获取到内容后，整理成一个格式良好的markdown文档，并用filesystem工具的write_file工具将整理好的markdown分别保存为'llms-full.txt'和'梁启超.md'文件
请在完成每个步骤后告知我进展情况。"""
                
                # 调用Agent
                response = await agent.invoke(prompt)
                
                print("\nAgent爬取和保存成功，响应:", response)
                
                # 返回文件路径，供后续RAG使用
                return ["/home/lab/AgentExplore/llms-full.txt", "/home/lab/AgentExplore/梁启超.md"]
                
    except Exception as e:
        print("Agent测试失败:", str(e))
        return []

async def rag_query_person(file_paths, query_name):
    """使用两种RAG方式对人名进行查询"""
    if not file_paths:
        print("没有文件可供查询")
        return
    
    print(f"\n\n===== 对 '{query_name}' 进行RAG查询 =====")
    
    # 准备文档数据
    documents = []
    for file_path in file_paths:
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                # 提取文件名作为元数据
                filename = os.path.basename(file_path)
                documents.append({
                    'page_content': content,
                    'metadata': {'source': filename, 'path': file_path}
                })
        except Exception as e:
            print(f"读取文件 {file_path} 失败: {e}")
    
    if not documents:
        print("没有可用的文档内容")
        return
    
    # 1. 使用原始检索器
    print("\n----- 原始检索器结果 -----")
    try:
        vector_store = VectorStore()
        retriever = EmbeddingRetriever(vector_store=vector_store, top_k=2)
        
        # 添加文档
        docs = [Document(page_content=doc["page_content"], metadata=doc["metadata"]) for doc in documents]
        await retriever.add_documents(docs)
        
        # 执行检索
        results = await retriever.retrieve(f"请详细介绍{query_name}的生平和成就")
        
        for i, doc in enumerate(results):
            print(f"\n结果 {i+1}: 来自 {doc.metadata.get('source', '未知来源')}")
            # 只显示内容的前200个字符，避免输出过长
            content_preview = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            print(f"内容预览: {content_preview}")
    except Exception as e:
        print(f"原始检索器失败: {e}")
    
    # 2. 使用LangChain检索器
    print("\n----- LangChain检索器结果 -----")
    try:
        retriever = LangChainRetriever(top_k=2)
        
        # 添加文档
        await retriever.add_documents(documents)
        
        # 执行检索
        results = await retriever.retrieve(f"请详细介绍{query_name}的生平和成就")
        
        for i, result in enumerate(results):
            print(f"\n结果 {i+1}: 来自 {result['metadata'].get('source', '未知来源')}")
            # 只显示内容的前200个字符，避免输出过长
            content_preview = result['page_content'][:200] + "..." if len(result['page_content']) > 200 else result['page_content']
            print(f"内容预览: {content_preview}")
            print(f"相关性分数: {result['score']}")
    except Exception as e:
        print(f"LangChain检索器失败: {e}")

async def main():
    """运行所有测试"""
    print("开始执行任务...")
    
    # 1. 调用agent爬取并保存文件
    file_paths = await test_agent_fetch_and_save()
    
    if file_paths:
        # 2. 使用两种RAG方式查询"刘慈欣"
        await rag_query_person(file_paths, "mcp")
        
        # 3. 使用两种RAG方式查询"梁启超"
        await rag_query_person(file_paths, "梁启超")
    
    print("\n任务完成")

if __name__ == "__main__":
    asyncio.run(main())