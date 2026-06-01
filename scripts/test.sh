#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# NexusAI - 仅运行测试(不启动服务)
# ═══════════════════════════════════════════════════════════════

cd "$(dirname "$0")/.."

PYTHON=""
if command -v python3 &>/dev/null; then PYTHON="python3"
elif command -v python &>/dev/null; then PYTHON="python"
else echo "未找到Python"; exit 1; fi

echo ""
echo "运行全部测试..."
echo ""
$PYTHON -m tests.test_all
echo ""

echo "运行端到端演示..."
echo ""
$PYTHON scripts/demo.py
