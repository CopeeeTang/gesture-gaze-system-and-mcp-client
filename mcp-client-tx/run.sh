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

# 安装依赖
echo "安装依赖..."
uv pip install -r requirements.txt

# 运行应用
echo "运行应用..."
python src/main.py