import os
import json
from typing import Dict, List, Optional, Any, Union
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base_llm import BaseLLM
from .utils import load_prompt_template, create_multimodal_message

class QwenOmniLLM(BaseLLM):
    """Qwen2.5 Omni模型的本地部署实现"""
    
    def __init__(self, model_path=None):
        """
        初始化Qwen2.5 Omni模型
        
        Args:
            model_path (str, optional): 模型路径，如果不指定则从环境变量获取
        """
        if model_path is None:
            model_path = os.environ.get("MODEL_PATH_QWEN", "Qwen/Qwen2.5-Omni-7B")
        
        print(f"正在加载Qwen2.5 Omni模型: {model_path}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        print("Qwen2.5 Omni模型加载完成")
        
    def chat(self, 
             messages: List[Dict[str, str]], 
             functions: Optional[List[Dict[str, Any]]] = None, 
             temperature: float = 0.7,
             max_tokens: Optional[int] = None) -> Dict[str, Any]:
        """
        基础对话功能实现
        
        Args:
            messages (List[Dict[str, str]]): 对话历史
            functions (Optional[List[Dict[str, Any]]]): 函数定义
            temperature (float): 温度参数
            max_tokens (Optional[int]): 最大生成token数
            
        Returns:
            Dict[str, Any]: 模型响应
        """
        if max_tokens is None:
            max_tokens = 1024
            
        # 如果有functions，则使用function_call方法
        if functions:
            return self.function_call(messages, functions, temperature)
            
        # 准备输入
        formatted_messages = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer(formatted_messages, return_tensors="pt").to(self.model.device)
        
        # 生成回复
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=temperature > 0.0,
        )
        
        # 解码并处理输出
        response_text = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        return {
            "content": response_text,
            "role": "assistant"
        }
    
    def function_call(self, 
                     messages: List[Dict[str, str]], 
                     functions: List[Dict[str, Any]],
                     temperature: float = 0.2) -> Dict[str, Any]:
        """
        使用函数调用能力与模型交互 (使用Qwen专有的函数调用接口)
        
        Args:
            messages (List[Dict[str, str]]): 对话历史
            functions (List[Dict[str, Any]]): 函数定义
            temperature (float): 温度参数
            
        Returns:
            Dict[str, Any]: 包含函数调用信息的响应
        """
        # 添加系统提示词
        system_prompt = load_prompt_template("qwen_system")
        if not any(msg.get("role") == "system" for msg in messages):
            messages = [{"role": "system", "content": system_prompt}] + messages
            
        # Qwen2.5支持原生函数调用
        formatted_messages = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer(formatted_messages, return_tensors="pt").to(self.model.device)
        
        # Qwen原生支持tools参数
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=temperature,
            do_sample=temperature > 0.0,
            tools=functions  # 直接传入tools定义
        )
        
        # 解码并处理输出
        response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        # 解析函数调用
        try:
            # Qwen的函数调用响应格式如下:
            # <tool_call>
            # {"name": "function_name", "arguments": {"param1": "value1"}}
            # </tool_call>
            if "<tool_call>" in response and "</tool_call>" in response:
                tool_call_content = response.split("<tool_call>")[1].split("</tool_call>")[0].strip()
                function_call_json = json.loads(tool_call_content)
                
                return {
                    "role": "assistant",
                    "content": None,
                    "function_call": {
                        "name": function_call_json["name"],
                        "arguments": json.dumps(function_call_json["arguments"])
                    }
                }
            else:
                # 如果没有函数调用，返回文本响应
                return {
                    "role": "assistant",
                    "content": response
                }
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            # 解析失败，返回原始响应
            print(f"函数调用解析失败: {e}")
            return {
                "role": "assistant",
                "content": response
            }
    
    def process_multimodal_input(self,
                               image: Optional[Union[str, bytes]] = None,
                               gesture: Optional[str] = None, 
                               gaze: Optional[tuple] = None,
                               text: Optional[str] = None,
                               functions: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        处理多模态输入
        
        Args:
            image (Optional[Union[str, bytes]]): 图像数据或路径
            gesture (Optional[str]): 手势信息
            gaze (Optional[tuple]): 眼动数据 (x, y, r)
            text (Optional[str]): 用户文本输入
            functions (Optional[List[Dict[str, Any]]]): 可用函数定义
            
        Returns:
            Dict[str, Any]: 模型响应
        """
        # 构建描述性提示词，包含多模态信息
        prompt = ""
        if text:
            prompt += text + "\n"
        
        if gesture:
            prompt += f"[手势: {gesture}]\n"
        
        if gaze:
            x, y, r = gaze
            prompt += f"[眼动: 位置({x}, {y}), 区域半径{r}]\n"
        
        # 构建消息列表
        messages = []
        
        # 添加系统提示词
        system_prompt = load_prompt_template("qwen_system")
        messages.append({"role": "system", "content": system_prompt})
        
        # Qwen支持多模态输入
        if image:
            # 处理图像
            from PIL import Image
            import base64
            from io import BytesIO
            
            if isinstance(image, str) and os.path.exists(image):
                img = Image.open(image)
            elif isinstance(image, bytes):
                img = Image.open(BytesIO(image))
            else:
                raise ValueError("图像格式不支持")
                
            # 转换为base64
            buffered = BytesIO()
            img.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            # 添加到消息中
            messages.append({
                "role": "user", 
                "content": [
                    {
                        "type": "image",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_str}"
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            })
        else:
            # 无图像，仅添加文本
            messages.append({"role": "user", "content": prompt})
        
        # 如果有functions，使用function_call，否则使用chat
        if functions:
            return self.function_call(messages, functions)
        else:
            return self.chat(messages)