@echo off
REM ═══════════════════════════════════════════════════════════════
REM NexusAI 智能客服系统 - Windows启动脚本
REM ═══════════════════════════════════════════════════════════════

cd /d "%~dp0\.."

echo.
echo ═══════════════════════════════════════════════════════════
echo   NexusAI 智能客服系统 v0.1.0
echo ═══════════════════════════════════════════════════════════
echo.

REM 检查Python
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] 未找到Python,请安装Python 3.11+
    pause
    exit /b 1
)

echo [1/4] 检查运行环境...
python --version

echo [2/4] 检查依赖...
python -c "import fastapi, uvicorn, pydantic" 2>nul
if %errorlevel% neq 0 (
    echo 正在安装依赖...
    pip install fastapi uvicorn pydantic -q
)
echo   √ 依赖就绪

echo [3/4] 运行测试...
python -m tests.test_all
if %errorlevel% neq 0 (
    echo [ERROR] 测试未通过
    pause
    exit /b 1
)

echo [4/4] 启动服务...
echo.
echo ═══════════════════════════════════════════════════════════
echo   API:  http://localhost:8000
echo   Docs: http://localhost:8000/docs
echo   前端: 用浏览器打开 prototype\index.html
echo   停止: Ctrl+C
echo ═══════════════════════════════════════════════════════════
echo.

python -m uvicorn src.gateway.api:app --host 0.0.0.0 --port 8000 --reload
