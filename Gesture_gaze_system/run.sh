#!/bin/bash

# 确保脚本在遇到错误时停止
set -e

# 创建环境变量文件（如果不存在）
if [ ! -f .env ]; then
    echo "创建.env文件..."
    cp .env.example .env
    echo "请编辑.env文件，填入你的API密钥"
fi

# 创建虚拟环境（如果不存在）
if [ ! -d .venv ]; then
    echo "创建虚拟环境..."
    uv venv
fi

# 激活虚拟环境
echo "激活虚拟环境..."
source .venv/bin/activate

uv pip install --upgrade pip

# 2. 单独安装 torch (如果你的环境还没有安装)
#    你需要根据你的 CUDA 版本选择合适的 torch 版本，
#    可以参考 PyTorch 官网: https://pytorch.org/get-started/locally/
#    例如，如果使用 CUDA 12.1:
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
#    如果不需要 GPU 支持 (CPU only):
#    uv pip install torch torchvision torchaudio

# 3. 使用 --no-build-isolation 安装 requirements.txt 中的所有包

echo "安装依赖..."
uv pip install -r requirements.txt --no-build-isolation
uv pip install -r requirements.txt

# 运行应用
echo "运行应用..."
python src/main.py