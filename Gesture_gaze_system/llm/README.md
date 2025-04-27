# LLM模块说明

本模块实现了多种大语言模型的封装，支持本地部署模型(Phi-4、Qwen2.5 Omni)和API调用模型(OpenAI)。所有模型都支持多模态输入和函数调用功能，但实现方式有所不同。

## 模型概览

| 模型 | 类型 | 多模态支持 | 函数调用方式 |
|------|------|------------|------------|
| Phi-4 | 本地部署 | ✓ | 通过系统提示词+特殊标记实现 |
| Qwen2.5 Omni | 本地部署 | ✓ | 原生支持 |
| OpenAI (GPT-4o等) | API调用 | ✓ | 原生支持 |

## 函数调用实现差异

### Phi-4 (Phi4LLM)

Phi-4模型不原生支持函数调用API，但我们通过特殊标记和精心设计的系统提示词来实现稳定的函数调用。实现方式：

1. 使用特殊标记定义函数调用格式：
   ```python
   self.tool_call_start = '<|tool_call|>'
   self.tool_call_end = '<|/tool_call|>'
   self.tool_def_start = '<|tool|>'
   self.tool_def_end = '<|/tool|>'
   ```

2. 在系统提示词中清晰定义工具函数和调用规则：
   ```
   可用函数：<|tool|>
   {...函数定义JSON...}
   <|/tool|>

   函数调用规则:
   1. 所有函数调用应以以下格式生成：<|tool_call|>[{"name": "函数名", "arguments": {"参数名": "参数值"}}]<|/tool_call|>
   2. 遵循提供的JSON架构，不要编造参数或值
   3. 确保选择正确匹配用户意图的函数
   ```

3. 使用多层级解析策略解析模型输出：
   - 首先尝试提取特殊标记内的函数调用 (`<|tool_call|>...`)
   - 如果失败，尝试查找常规的JSON代码块 (```json```)
   - 最后尝试匹配可能的JSON结构

4. 对于多模态输入，使用`AutoProcessor`优化处理方式，提高性能和稳定性

**优点**：适配性更好，解析更健壮，可以处理多种可能的输出格式
**缺点**：仍然依赖模型遵循特定的输出格式指示

### Qwen2.5 Omni (QwenOmniLLM)

Qwen2.5 Omni原生支持函数调用功能，使用Transformers库中的专有接口实现：

1. 直接在`model.generate()`调用中通过`tools`参数传入函数定义
2. 模型生成的输出会包含特殊标记`<tool_call>`和`</tool_call>`
3. 通过解析这些标记提取函数调用信息

**优点**：原生支持，稳定性高
**缺点**：依赖特定版本的Transformers库

示例函数调用输出格式：
```
<tool_call>
{"name": "function_name", "arguments": {"param1": "value1"}}
</tool_call>
```

### OpenAI (OpenAILLM)

OpenAI模型(如GPT-4o)通过官方API直接支持函数调用：

1. 在API请求中通过`tools`参数传入函数定义
2. API响应中直接包含格式化的函数调用信息
3. 直接从响应对象中获取函数名和参数

**优点**：实现最简单，稳定性最高
**缺点**：依赖网络和API密钥，成本较高

API调用参数示例：
```python
response = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    tools=[{"type": "function", "function": f} for f in functions],
    tool_choice="auto"
)
```

## 多模态输入处理

所有模型都支持处理图像、手势和眼动数据等多模态输入，但处理方式略有不同：

1. **Phi-4**：支持两种处理方式：
   - 通过`AutoProcessor`处理图像和文本（性能更好）
   - 通过标准的多模态消息格式（兼容性更好）

2. **Qwen2.5 Omni**：原生支持图像输入，格式与OpenAI类似

3. **OpenAI**：GPT-4o和Vision系列模型支持多种格式的图像输入

## 使用建议

1. 对于需要稳定函数调用的生产环境，优先使用OpenAI API或Qwen2.5 Omni
2. 对于本地部署且资源有限的环境，可使用升级后的Phi-4实现，具有更好的函数调用稳定性
3. 对于多模态任务，推荐配置足够GPU内存的环境运行Phi-4，以启用flash attention优化