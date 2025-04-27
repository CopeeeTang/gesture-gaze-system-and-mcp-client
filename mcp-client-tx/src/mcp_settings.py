import json
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union


class SystemRole(str, Enum):
    """系统角色枚举"""
    DEFAULT = "default"
    ASSISTANT = "assistant"
    USER = "user"
    SYSTEM = "system"
    TOOL = "tool"


@dataclass
class ModelSettings:
    """模型设置"""
    name: str = "claude-3-5-sonnet-20240620"
    system_prompt: str = ""
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 4000


@dataclass
class MCPSettings:
    """MCP客户端设置"""
    api_key: str = ""
    api_url: str = "https://api.anthropic.com"
    models: Dict[str, ModelSettings] = None
    default_model: str = "claude-3-5-sonnet-20240620"
    system_role: SystemRole = SystemRole.SYSTEM
    config_file: Path = None
    
    def __post_init__(self):
        """初始化后处理"""
        if self.models is None:
            self.models = {
                "claude-3-5-sonnet-20240620": ModelSettings(),
                "claude-3-opus-20240229": ModelSettings(
                    name="claude-3-opus-20240229",
                    temperature=0.7,
                    top_p=0.95,
                    max_tokens=4000
                ),
                "claude-3-haiku-20240307": ModelSettings(
                    name="claude-3-haiku-20240307",
                    temperature=0.7,
                    top_p=0.95,
                    max_tokens=4000
                )
            }
    
    @classmethod
    def from_config(cls, config_file: Union[str, Path] = None) -> 'MCPSettings':
        """从配置文件加载设置"""
        if config_file is None:
            # 默认配置文件位置
            config_file = Path.home() / ".mcp" / "config.json"
        else:
            config_file = Path(config_file)
            
        settings = cls()
        settings.config_file = config_file
        
        if config_file.exists():
            try:
                with open(config_file, "r", encoding="utf-8") as f:
                    config = json.load(f)
                
                # 加载API密钥和URL
                settings.api_key = config.get("api_key", "")
                settings.api_url = config.get("api_url", "https://api.anthropic.com")
                settings.default_model = config.get("default_model", "claude-3-5-sonnet-20240620")
                settings.system_role = SystemRole(config.get("system_role", SystemRole.SYSTEM.value))
                
                # 加载模型设置
                if "models" in config:
                    for model_name, model_config in config["models"].items():
                        settings.models[model_name] = ModelSettings(
                            name=model_name,
                            system_prompt=model_config.get("system_prompt", ""),
                            temperature=model_config.get("temperature", 0.7),
                            top_p=model_config.get("top_p", 0.95),
                            max_tokens=model_config.get("max_tokens", 4000)
                        )
            except Exception as e:
                print(f"加载配置文件出错: {e}")
        
        # 尝试从环境变量加载API密钥
        env_api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not settings.api_key and env_api_key:
            settings.api_key = env_api_key
            
        return settings
    
    def save_config(self) -> None:
        """保存配置到文件"""
        if self.config_file is None:
            self.config_file = Path.home() / ".mcp" / "config.json"
            
        # 确保目录存在
        self.config_file.parent.mkdir(parents=True, exist_ok=True)
        
        config = {
            "api_key": self.api_key,
            "api_url": self.api_url,
            "default_model": self.default_model,
            "system_role": self.system_role.value,
            "models": {}
        }
        
        # 保存模型设置
        for model_name, model_settings in self.models.items():
            config["models"][model_name] = {
                "system_prompt": model_settings.system_prompt,
                "temperature": model_settings.temperature,
                "top_p": model_settings.top_p,
                "max_tokens": model_settings.max_tokens
            }
            
        with open(self.config_file, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)