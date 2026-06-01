#!/usr/bin/env python3
"""端到端场景演示 - 验证全链路可用"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import CustomerServicePipeline
from src.config.settings import SystemLevel


def demo():
    pipe = CustomerServicePipeline()

    print("=" * 60)
    print("  NexusAI 智能客服系统 · 端到端场景演示")
    print("=" * 60)

    # 场景1: 价保快速通道
    print("\n━━━ 场景1: 价保快速通道 ━━━")
    r = pipe.process("demo_1", "我前天买的东西降价了，能价保吗？")
    print(f"  用户: 我前天买的东西降价了，能价保吗？")
    print(f"  AI: {r.response_text}")
    print(f"  [意图={r.intent.intent} conf={r.confidence:.2f} 模型={r.model_used} 延迟={r.latency_ms:.1f}ms]")

    # 场景2: 退款流程
    print("\n━━━ 场景2: 退款流程 ━━━")
    r = pipe.process("demo_2", "我要退货退款")
    print(f"  用户: 我要退货退款")
    print(f"  AI: {r.response_text}")
    print(f"  [意图={r.intent.intent} conf={r.confidence:.2f} 模型={r.model_used} 延迟={r.latency_ms:.1f}ms]")

    # 场景3: 投诉升级
    print("\n━━━ 场景3: 投诉升级 ━━━")
    r = pipe.process("demo_3", "你们服务太差了！我要投诉到消协！必须赔偿！")
    print(f"  用户: 你们服务太差了！我要投诉到消协！必须赔偿！")
    print(f"  AI: {r.response_text}")
    print(f"  [意图={r.intent.intent} 转人工={r.suggest_transfer_human} 延迟={r.latency_ms:.1f}ms]")

    # 场景4: FAQ缓存命中
    print("\n━━━ 场景4: FAQ(第二次缓存命中) ━━━")
    r1 = pipe.process("demo_4", "7天无理由退货需要什么条件")
    print(f"  用户: 7天无理由退货需要什么条件")
    print(f"  AI: {r1.response_text}")
    print(f"  [缓存={r1.from_cache} 延迟={r1.latency_ms:.1f}ms]")

    r2 = pipe.process("demo_4b", "七天无理由退货条件是啥")
    print(f"  用户(换个问法): 七天无理由退货条件是啥")
    print(f"  AI: {r2.response_text}")
    print(f"  [缓存={r2.from_cache} 延迟={r2.latency_ms:.1f}ms]")

    # 场景5: 大促降级模式
    print("\n━━━ 场景5: 大促降级(L3 RED) ━━━")
    pipe.update_system_level(SystemLevel.RED)
    r = pipe.process("demo_5", "帮我分析一下复杂的退款政策问题")
    print(f"  用户: 帮我分析一下复杂的退款政策问题")
    print(f"  AI: {r.response_text}")
    print(f"  [模型={r.model_used} (降级模式,零LLM) 延迟={r.latency_ms:.1f}ms]")
    pipe.update_system_level(SystemLevel.GREEN)

    # 场景6: 多轮对话
    print("\n━━━ 场景6: 多轮对话 ━━━")
    r1 = pipe.process("demo_6", "查一下我的快递")
    print(f"  用户: 查一下我的快递")
    print(f"  AI: {r1.response_text}")
    r2 = pipe.process("demo_6", "太慢了催一下")
    print(f"  用户: 太慢了催一下")
    print(f"  AI: {r2.response_text}")
    ctx = pipe._sessions["demo_6"]
    print(f"  [轮次={ctx.turn_count} 情绪={ctx.emotion.value} 情绪分={ctx.emotion_score:.2f}]")

    print("\n" + "=" * 60)
    print("  ✅ 全部场景演示完成")
    print("=" * 60)


if __name__ == "__main__":
    demo()
