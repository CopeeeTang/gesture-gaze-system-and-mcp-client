from .base_llm import BaseLLM
from .phi4 import Phi4LLM
from .qwen_omni import QwenOmniLLM
from .chat_openai import OpenAILLM

def get_llm(model_name):
    """
    根据模型名称获取对应的LLM实例
    
    Args:
        model_name (str): 模型名称，支持 'phi4', 'qwen', 'openai'
        
    Returns:
        BaseLLM: 对应的LLM实例
    """
    model_map = {
        'phi4': Phi4LLM,
        'qwen': QwenOmniLLM,
        'openai': OpenAILLM
    }
    
    if model_name.lower() not in model_map:
        raise ValueError(f"不支持的模型: {model_name}。支持的模型: {list(model_map.keys())}")
    
    return model_map[model_name.lower()]()