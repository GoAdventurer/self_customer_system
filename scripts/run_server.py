#!/usr/bin/env python3
"""NexusAI 智能客服系统 - 快速启动脚本"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    """启动API服务"""
    try:
        import uvicorn
    except ImportError:
        print("❌ 请先安装依赖: pip install fastapi uvicorn")
        print("   或: pip install -e .")
        sys.exit(1)

    print("=" * 50)
    print("  NexusAI 智能客服系统 v0.1.0")
    print("  API: http://localhost:8000")
    print("  Docs: http://localhost:8000/docs")
    print("=" * 50)

    uvicorn.run(
        "src.gateway.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )


if __name__ == "__main__":
    main()
