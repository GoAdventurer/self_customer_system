#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# NexusAI 智能客服系统 - 一键启动脚本
# ═══════════════════════════════════════════════════════════════

set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

# 颜色定义
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'
BOLD='\033[1m'

echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}  NexusAI 智能客服系统 v0.1.0${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo ""

# ─── 检查Python环境 ───
echo -e "${YELLOW}[1/4]${NC} 检查运行环境..."

PYTHON=""
if command -v python3 &>/dev/null; then
    PYTHON="python3"
elif command -v python &>/dev/null; then
    PYTHON="python"
else
    echo -e "${RED}✗ 未找到Python,请安装Python 3.11+${NC}"
    exit 1
fi

PY_VERSION=$($PYTHON -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo -e "  Python: ${GREEN}$PY_VERSION${NC} ($($PYTHON --version))"

# ─── 检查依赖 ───
echo -e "${YELLOW}[2/4]${NC} 检查核心依赖..."

check_module() {
    if $PYTHON -c "import $1" 2>/dev/null; then
        echo -e "  ${GREEN}✓${NC} $1"
        return 0
    else
        echo -e "  ${RED}✗${NC} $1 未安装"
        return 1
    fi
}

MISSING=0
check_module fastapi || MISSING=1
check_module uvicorn || MISSING=1
check_module pydantic || MISSING=1

if [ $MISSING -eq 1 ]; then
    echo ""
    echo -e "${YELLOW}正在安装缺失依赖...${NC}"
    if command -v pip3 &>/dev/null; then
        pip3 install fastapi uvicorn pydantic -q
    elif command -v pip &>/dev/null; then
        pip install fastapi uvicorn pydantic -q
    else
        echo -e "${RED}✗ 未找到pip,请手动安装: pip install -r requirements.txt${NC}"
        exit 1
    fi
    echo -e "  ${GREEN}✓ 依赖安装完成${NC}"
fi

# ─── 运行测试 ───
echo -e "${YELLOW}[3/4]${NC} 运行回归测试..."

TEST_RESULT=$($PYTHON -m tests.test_all 2>&1 | tail -1)
if echo "$TEST_RESULT" | grep -q "全部测试通过"; then
    echo -e "  ${GREEN}✓ 27/27 测试通过${NC}"
else
    echo -e "  ${RED}✗ 测试未通过,请检查:${NC}"
    $PYTHON -m tests.test_all
    exit 1
fi

# ─── 启动服务 ───
echo -e "${YELLOW}[4/4]${NC} 启动API服务..."
echo ""
echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
echo -e "  ${BOLD}API 地址:${NC}  http://localhost:8000"
echo -e "  ${BOLD}API 文档:${NC}  http://localhost:8000/docs"
echo -e "  ${BOLD}前端界面:${NC}  file://$PROJECT_DIR/prototype/index.html"
echo -e ""
echo -e "  ${BOLD}快速体验:${NC}  用浏览器打开前端界面,输入问题即可对话"
echo -e "  ${BOLD}停止服务:${NC}  Ctrl+C"
echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
echo ""

$PYTHON -m uvicorn src.gateway.api:app --host 0.0.0.0 --port 8000 --reload
