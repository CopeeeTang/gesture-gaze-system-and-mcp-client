import os
import json
import re
from typing import Dict, List, Optional, Any, Union
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor

from .base_llm import BaseLLM
from .utils import load_prompt_template, format_functions_for_phi, create_multimodal_message

class Phi4LLM(BaseLLM):
    """Phi-4多模态模型的本地部署实现"""
    
    def __init__(self, model_path=None):
        """
        初始化Phi4模型
        
        Args:
            model_path (str, optional): 模型路径，如果不指定则从环境变量获取
        """
        if model_path is None:
            model_path = os.environ.get("MODEL_PATH_PHI4", "microsoft/Phi-4-multimodal-instruct")
        
        # 定义特殊标记用于函数调用
        self.system_prompt_start = '<|system|>'
        self.system_prompt_end = '<|end|>'
        self.user_prompt = '<|user|>'
        self.user_prompt_end = '<|end|>'
        self.assistant_prompt = '<|assistant|>'
        self.assistant_prompt_end = '<|end|>'
        self.tool_call_start = '<|tool_call|>'
        self.tool_call_end = '<|/tool_call|>'
        self.tool_def_start = '<|tool|>'
        self.tool_def_end = '<|/tool|>'
        
        print(f"正在加载Phi4模型: {model_path}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            _attn_implementation='flash_attention_2'  # 性能优化
        )
        
        # 使用正确的Tokenizer/Processor
        try:
            self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
            self.use_processor = True
            print("使用AutoProcessor进行输入处理")
        except:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            self.use_processor = False
            print("使用AutoTokenizer进行输入处理")
        
        print("Phi4模型加载完成")
        
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
        if self.use_processor:
            # 使用processor处理输入
            formatted_prompt = ""
            for msg in messages:
                if msg["role"] == "system":
                    formatted_prompt += f"{self.system_prompt_start}\n{msg['content']}\n{self.system_prompt_end}\n"
                elif msg["role"] == "user":
                    formatted_prompt += f"{self.user_prompt}\n{msg['content']}\n{self.user_prompt_end}\n"
                elif msg["role"] == "assistant":
                    formatted_prompt += f"{self.assistant_prompt}\n{msg['content']}\n{self.assistant_prompt_end}\n"
            
            formatted_prompt += f"{self.assistant_prompt}\n"
            
            inputs = self.processor(
                text=formatted_prompt,
                return_tensors="pt"
            ).to(self.model.device)
        else:
            # 使用tokenizer处理输入
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
        if self.use_processor:
            response_text = self.processor.batch_decode(
                outputs[:, inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )[0]
        else:
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
        使用函数调用能力与模型交互 (通过system prompt实现)
        
        Args:
            messages (List[Dict[str, str]]): 对话历史
            functions (List[Dict[str, Any]]): 函数定义
            temperature (float): 温度参数
            
        Returns:
            Dict[str, Any]: 包含函数调用信息的响应
        """
        # 构建增强的系统提示词
        tools_json = json.dumps(functions)
        system_prompt = f'''{self.system_prompt_start}
你是一个具备工具调用能力的AI助手，可以根据用户输入调用合适的工具函数。你只需要返回工具调用的具体格式。

可用函数：{self.tool_def_start}
{tools_json}
{self.tool_def_end}

函数调用规则:
1. 所有函数调用应以以下格式生成：{self.tool_call_start}[{{"name": "函数名", "arguments": {{"参数名": "参数值"}}}}]{self.tool_call_end}
2. 遵循提供的JSON架构，不要编造参数或值
3. 确保选择正确匹配用户意图的函数
4. 如果需要调用多个函数，请将它们放在同一个JSON数组中
{self.system_prompt_end}'''
        
        # 添加系统提示词
        has_system = False
        messages_with_system = []
        for msg in messages:
            if msg["role"] == "system":
                messages_with_system.append({"role": "system", "content": system_prompt + "\n" + msg["content"]})
                has_system = True
            else:
                messages_with_system.append(msg)
        
        if not has_system:
            messages_with_system = [{"role": "system", "content": system_prompt}] + messages
        
        # 执行常规聊天
        response = self.chat(messages_with_system, None, temperature)
        content = response["content"]
        
        # 解析函数调用
        tool_calls = self.parse_tool_calls(content)
        
        if tool_calls:
            # 找到第一个有效的函数调用
            tool_call = tool_calls[0]
            return {
                "role": "assistant",
                "content": None,
                "function_call": {
                    "name": tool_call["name"],
                    "arguments": json.dumps(tool_call.get("arguments", tool_call.get("parameters", {})))
                }
            }
        else:
            # 如果找不到函数调用，返回原始文本
            return {
                "role": "assistant",
                "content": content
            }
    
    def parse_tool_calls(self, response_text):
        """
        解析模型返回的工具调用，支持多种格式
        
        Args:
            response_text (str): 模型响应文本
            
        Returns:
            List[Dict]: 解析后的工具调用列表
        """
        tool_calls = []
        
        # 1. 尝试提取 <|tool_call|>[...]<|/tool_call|> 格式
        tool_call_pattern = rf'{re.escape(self.tool_call_start)}(.+?){re.escape(self.tool_call_end)}'
        tool_call_matches = re.findall(tool_call_pattern, response_text, re.DOTALL)
        
        if tool_call_matches:
            for match in tool_call_matches:
                try:
                    # 尝试解析JSON
                    parsed_json = json.loads(match)
                    if isinstance(parsed_json, list):
                        for call in parsed_json:
                            if isinstance(call, dict) and "name" in call:
                                # 统一参数字段名
                                if "arguments" in call and not "parameters" in call:
                                    call["parameters"] = call["arguments"]
                                tool_calls.append(call)
                    elif isinstance(parsed_json, dict) and "name" in parsed_json:
                        if "arguments" in parsed_json and not "parameters" in parsed_json:
                            parsed_json["parameters"] = parsed_json["arguments"]
                        tool_calls.append(parsed_json)
                except json.JSONDecodeError:
                    print(f"无法解析JSON: {match}")
        
        # 2. 如果上述方法失败，尝试查找常规的```json```格式
        if not tool_calls:
            json_pattern = r'```json\s*(.*?)\s*```'
            json_matches = re.findall(json_pattern, response_text, re.DOTALL)
            
            for match in json_matches:
                try:
                    parsed_json = json.loads(match)
                    # 处理直接的函数调用格式
                    if isinstance(parsed_json, dict) and "function" in parsed_json:
                        tool_calls.append({
                            "name": parsed_json["function"],
                            "parameters": parsed_json.get("parameters", {})
                        })
                    # 处理其他可能的JSON格式
                    elif isinstance(parsed_json, dict) and "name" in parsed_json:
                        if "arguments" in parsed_json:
                            parsed_json["parameters"] = parsed_json["arguments"]
                        tool_calls.append(parsed_json)
                except json.JSONDecodeError:
                    print(f"无法解析JSON代码块: {match}")
        
        # 3. 如果仍然失败，尝试查找可能的工具调用对象
        if not tool_calls:
            # 尝试匹配任何可能的JSON对象
            json_obj_pattern = r'{[^{}]*"(?:function|name)"[^{}]*}'
            obj_matches = re.findall(json_obj_pattern, response_text)
            
            for match in obj_matches:
                try:
                    parsed_json = json.loads(match)
                    if "function" in parsed_json:
                        tool_calls.append({
                            "name": parsed_json["function"],
                            "parameters": parsed_json.get("parameters", {})
                        })
                    elif "name" in parsed_json:
                        if "arguments" in parsed_json:
                            parsed_json["parameters"] = parsed_json["arguments"]
                        tool_calls.append(parsed_json)
                except json.JSONDecodeError:
                    continue
                    
        return tool_calls
    
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
        # 构建系统提示词
        system_template = load_prompt_template("phi4_system")
        
        # 如果有函数定义，替换模板中的占位符
        if functions:
            formatted_functions = format_functions_for_phi(functions)
            system_prompt = system_template.replace("{FUNCTION_DEFINITIONS}", formatted_functions)
        else:
            system_prompt = system_template.replace(
                "{FUNCTION_DEFINITIONS}", 
                "无可用函数。请直接回答用户问题，不要尝试调用函数。"
            )
        
        # 构建提示词文本
        user_prompt = ""
        if text:
            user_prompt += text + "\n"
        
        if gesture:
            user_prompt += f"用户手势: {gesture}\n"
        
        if gaze:
            x, y, r = gaze
            user_prompt += f"用户注视位置: ({x}, {y}), 半径: {r}\n"
        
        # 构建消息列表
        messages = [{"role": "system", "content": system_prompt}]
        
        # 处理图像
        if image and self.use_processor:
            # 如果使用processor，直接处理图像
            from PIL import Image as PILImage
            from io import BytesIO
            
            if isinstance(image, str) and os.path.exists(image):
                img = PILImage.open(image)
            elif isinstance(image, bytes):
                img = PILImage.open(BytesIO(image))
            else:
                raise ValueError("图像格式不支持")
            
            # 使用processor处理多模态输入
            inputs = self.processor(
                text=f"{self.user_prompt}\n{user_prompt}\n{self.user_prompt_end}\n{self.assistant_prompt}\n",
                images=img,
                return_tensors="pt"
            ).to(self.model.device)
            
            # 生成回复
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.7,
                do_sample=True
            )
            
            # 解码并处理输出
            response_text = self.processor.batch_decode(
                outputs[:, inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )[0]
            
            # 如果有函数，解析函数调用
            if functions:
                tool_calls = self.parse_tool_calls(response_text)
                if tool_calls:
                    tool_call = tool_calls[0]
                    return {
                        "role": "assistant",
                        "content": None,
                        "function_call": {
                            "name": tool_call["name"],
                            "arguments": json.dumps(tool_call.get("arguments", tool_call.get("parameters", {})))
                        }
                    }
            
            return {"role": "assistant", "content": response_text}
        else:
            # 使用标准消息格式
            from .utils import create_multimodal_message
            
            multimodal_message = create_multimodal_message(image, gesture, gaze, text)
            messages.append(multimodal_message)
            
            # 如果有函数，使用function_call，否则使用chat
            if functions:
                return self.function_call(messages, functions)
            else:
                return self.chat(messages)