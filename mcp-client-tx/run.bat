@echo off
SETLOCAL

REM 创建环境变量文件（如果不存在）
IF NOT EXIST .env (
    echo 创建.env文件...
    copy .env.example .env
    echo 请编辑.env文件，填入你的API密钥
)

REM 创建虚拟环境（如果不存在）
IF NOT EXIST .venv (
    echo 创建虚拟环境...
    uv venv
)

REM 激活虚拟环境
echo 激活虚拟环境...
call .venv\Scripts\activate.bat

REM 安装依赖
echo 安装依赖...
uv pip install -r requirements.txt

REM 运行应用
echo 运行应用...
python src\main.py

ENDLOCAL