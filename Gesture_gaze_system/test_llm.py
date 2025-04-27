import os
import sys
import json
from dotenv import load_dotenv
from Gesture_gaze_system.llm import Phi4LLM, OpenAILLM

# 加载环境变量
load_dotenv()

# 定义测试用的工具函数
test_functions = [
    {
        "name": "get_weather",
        "description": "获取指定城市的天气信息",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "城市名称，如北京、上海等"
                },
                "date": {
                    "type": "string",
                    "description": "日期，格式为YYYY-MM-DD，如不指定则为今天"
                }
            },
            "required": ["city"]
        }
    },
    {
        "name": "set_alarm",
        "description": "设置闹钟提醒",
        "parameters": {
            "type": "object",
            "properties": {
                "time": {
                    "type": "string",
                    "description": "时间，格式为HH:MM，如08:30"
                },
                "label": {
                    "type": "string",
                    "description": "闹钟标签"
                }
            },
            "required": ["time"]
        }
    }
]

def test_chat(model_name):
    """测试基本聊天功能"""
    print(f"\n===== 测试 {model_name} 聊天功能 =====")
    
    if model_name == "phi4":
        model = Phi4LLM()
    elif model_name == "openai":
        model = OpenAILLM()
    else:
        print(f"不支持的模型: {model_name}")
        return
    
    test_messages = [
        {"role": "user", "content": "你好，请介绍一下自己"}
    ]
    
    print("发送消息:", test_messages[0]["content"])
    
    try:
        response = model.chat(test_messages)
        print(f"模型响应:\n{response['content']}")
        
        # 续对话测试
        test_messages.append({"role": "assistant", "content": response["content"]})
        test_messages.append({"role": "user", "content": "你能处理图像和手势输入吗？"})
        
        print("\n发送后续消息:", test_messages[-1]["content"])
        response = model.chat(test_messages)
        print(f"模型响应:\n{response['content']}")
        
        return True
    except Exception as e:
        print(f"聊天测试失败: {e}")
        return False

def test_function_call(model_name):
    """测试函数调用功能"""
    print(f"\n===== 测试 {model_name} 函数调用功能 =====")
    
    if model_name == "phi4":
        model = Phi4LLM()
    elif model_name == "openai":
        model = OpenAILLM()
    else:
        print(f"不支持的模型: {model_name}")
        return
    
    # 测试查询天气
    test_messages = [
        {"role": "user", "content": "我想知道明天北京的天气"}
    ]
    
    print("发送带函数意图的消息:", test_messages[0]["content"])
    
    try:
        response = model.function_call(test_messages, test_functions)
        
        if "function_call" in response:
            func_name = response["function_call"]["name"]
            func_args = json.loads(response["function_call"]["arguments"])
            print(f"函数调用: {func_name}")
            print(f"参数: {json.dumps(func_args, ensure_ascii=False, indent=2)}")
        else:
            print(f"未触发函数调用，模型响应:\n{response['content']}")
        
        # 测试设置闹钟
        test_messages = [
            {"role": "user", "content": "请帮我设置一个明天早上8点的闹钟，标记为'开会'"}
        ]
        
        print("\n发送带函数意图的消息:", test_messages[0]["content"])
        response = model.function_call(test_messages, test_functions)
        
        if "function_call" in response:
            func_name = response["function_call"]["name"]
            func_args = json.loads(response["function_call"]["arguments"])
            print(f"函数调用: {func_name}")
            print(f"参数: {json.dumps(func_args, ensure_ascii=False, indent=2)}")
        else:
            print(f"未触发函数调用，模型响应:\n{response['content']}")
        
        return True
    except Exception as e:
        print(f"函数调用测试失败: {e}")
        return False

def main():
    """主测试函数"""
    models = ["phi4", "openai"]
    
    for model_name in models:
        print(f"\n\n========== 开始测试模型: {model_name} ==========")
        
        # 跳过测试如果没有API密钥
        if model_name == "openai" and not os.environ.get("OPENAI_API_KEY"):
            print("跳过OpenAI测试，没有设置OPENAI_API_KEY环境变量")
            continue
            
        # 测试聊天功能
        chat_success = test_chat(model_name)
        
        # 测试函数调用功能
        function_call_success = test_function_call(model_name)
        
        # 打印测试结果
        print(f"\n{model_name} 测试结果:")
        print(f"  聊天功能: {'✓ 成功' if chat_success else '✗ 失败'}")
        print(f"  函数调用: {'✓ 成功' if function_call_success else '✗ 失败'}")

if __name__ == "__main__":
    main() 