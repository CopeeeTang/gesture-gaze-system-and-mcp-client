import os
import json
import base64
from typing import Dict, List, Any, Optional, Union

def load_prompt_template(template_name: str) -> str:
    """
    加载提示词模板
    
    Args:
        template_name (str): 模板名称
        
    Returns:
        str: 提示词模板内容
    """
    template_path = os.path.join(
        os.path.dirname(__file__), 
        "prompt_templates", 
        f"{template_name}.txt"
    )
    
    if not os.path.exists(template_path):
        raise FileNotFoundError(f"找不到提示词模板: {template_path}")
    
    with open(template_path, "r", encoding="utf-8") as f:
        return f.read()

def encode_image_to_base64(image_path: str) -> str:
    """
    将图像编码为base64字符串
    
    Args:
        image_path (str): 图像文件路径
        
    Returns:
        str: base64编码的图像
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def format_functions_for_phi(functions: List[Dict[str, Any]]) -> str:
    """
    为Phi4模型格式化函数调用定义
    
    Args:
        functions (List[Dict[str, Any]]): 函数定义列表
        
    Returns:
        str: 格式化后的函数定义文本，用于系统提示词
    """
    formatted_functions = []
    
    for func in functions:
        params = func.get("parameters", {}).get("properties", {})
        param_desc = []
        
        for param_name, param_props in params.items():
            param_type = param_props.get("type", "unknown")
            param_description = param_props.get("description", "")
            param_desc.append(f"- {param_name} ({param_type}): {param_description}")
        
        function_text = f"""函数名称: {func.get('name')}
描述: {func.get('description', '')}
参数:
{chr(10).join(param_desc)}
"""
        formatted_functions.append(function_text)
    
    return "\n\n".join(formatted_functions)

def create_multimodal_message(
    image: Optional[Union[str, bytes]] = None,
    gesture: Optional[str] = None,
    gaze: Optional[tuple] = None,
    text: Optional[str] = None
) -> Dict[str, Any]:
    """
    创建包含多模态内容的消息
    
    Args:
        image (Optional[Union[str, bytes]]): 图像数据或路径
        gesture (Optional[str]): 手势信息
        gaze (Optional[tuple]): 眼动数据 (x, y, r)
        text (Optional[str]): 用户文本输入
        
    Returns:
        Dict[str, Any]: 格式化的多模态消息
    """
    content = []
    
    # 添加文本部分
    message_parts = []
    if text:
        message_parts.append(f"用户文本输入: {text}")
    if gesture:
        message_parts.append(f"手势输入: {gesture}")
    if gaze:
        x, y, r = gaze
        message_parts.append(f"眼动输入: 注视位置({x}, {y})，区域半径{r}")
    
    if message_parts:
        content.append({"type": "text", "text": " ".join(message_parts)})
    
    # 添加图像部分
    if image:
        if isinstance(image, str):
            # 如果是路径，加载并编码图像
            if os.path.exists(image):
                image_data = encode_image_to_base64(image)
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}
                })
        else:
            # 假设已经是base64编码的图像
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image}"}
            })
    
    return {"role": "user", "content": content}