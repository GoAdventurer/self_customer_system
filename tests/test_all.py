"""测试套件 - 覆盖意图识别、DAG编排、模型路由、全链路管线"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.common.models import SessionContext, UserProfile, EmotionLevel, Channel
from src.config.settings import (
    SystemConfig, IntentConfig, ReasoningConfig, SystemLevel,
    ModelTier, create_default_config,
)
from src.intent.engine import IntentEngine, rule_match, classifier_predict, detect_emotion
from src.planning.dag_engine import DAGEngine, create_refund_dag, create_price_protect_dag, NodeStatus
from src.reasoning.model_router import ModelRouter, SemanticCache
from src.pipeline import CustomerServicePipeline


# ═══════════════════════════════════════════════
# 意图识别测试
# ═══════════════════════════════════════════════

def test_rule_match_refund():
    """规则匹配: 退款意图"""
    result = rule_match("我要退款")
    assert result is not None
    assert result.intent == "REFUND"
    assert result.confidence >= 0.9
    assert result.method == "rule"
    print("  ✓ test_rule_match_refund")


def test_rule_match_logistics():
    """规则匹配: 物流查询"""
    result = rule_match("我的快递到哪了")
    assert result is not None
    assert result.intent == "LOGISTICS"
    print("  ✓ test_rule_match_logistics")


def test_rule_match_price_protect():
    """规则匹配: 价保"""
    result = rule_match("刚买的就降价了能价保吗")
    assert result is not None
    assert result.intent == "PRICE_PROTECT"
    print("  ✓ test_rule_match_price_protect")


def test_rule_match_complaint():
    """规则匹配: 投诉"""
    result = rule_match("我要投诉你们")
    assert result is not None
    assert result.intent == "COMPLAINT"
    print("  ✓ test_rule_match_complaint")


def test_rule_no_match():
    """规则匹配: 无匹配"""
    result = rule_match("今天天气不错")
    assert result is None
    print("  ✓ test_rule_no_match")


def test_classifier_predict():
    """分类器: 退款意图"""
    result = classifier_predict("退款退货不想要了")
    assert result is not None
    assert result.intent == "REFUND"
    assert result.method == "classifier"
    print("  ✓ test_classifier_predict")


def test_emotion_detection_angry():
    """情绪检测: 愤怒"""
    level, score = detect_emotion("投诉到消协！你们太差了赔偿！")
    assert level == "angry"
    assert score > 0.7
    print("  ✓ test_emotion_detection_angry")


def test_emotion_detection_neutral():
    """情绪检测: 中性"""
    level, score = detect_emotion("查一下物流")
    assert level == "neutral"
    assert score < 0.3
    print("  ✓ test_emotion_detection_neutral")


def test_intent_engine_rule_path():
    """意图引擎: 规则路径(L1)"""
    engine = IntentEngine(config=IntentConfig())
    ctx = SessionContext()
    from src.common.models import Message, MessageRole
    ctx.messages.append(Message(role=MessageRole.USER, content="我要退货退款"))
    result = engine.recognize(ctx)
    assert result.intent == "REFUND"
    assert result.method == "rule"
    print("  ✓ test_intent_engine_rule_path")


def test_intent_engine_degraded_mode():
    """意图引擎: 降级模式(L3)仅走规则"""
    engine = IntentEngine(config=IntentConfig(), system_level=SystemLevel.RED)
    ctx = SessionContext()
    from src.common.models import Message, MessageRole
    ctx.messages.append(Message(role=MessageRole.USER, content="今天天气怎么样"))
    result = engine.recognize(ctx)
    assert result.method == "degraded"
    assert result.intent == "GENERAL_INQUIRY"
    print("  ✓ test_intent_engine_degraded_mode")


def test_escalation_user_request():
    """升级检测: 用户主动请求"""
    engine = IntentEngine(config=IntentConfig())
    ctx = SessionContext()
    from src.common.models import Message, MessageRole
    ctx.messages.append(Message(role=MessageRole.USER, content="转人工客服"))
    should, reason = engine.detect_escalation(ctx)
    assert should is True
    assert reason == "user_requested"
    print("  ✓ test_escalation_user_request")


def test_escalation_vip_negative():
    """升级检测: VIP+负面情绪"""
    engine = IntentEngine(config=IntentConfig())
    ctx = SessionContext(user=UserProfile(user_id="u1", tier="svip"))
    from src.common.models import Message, MessageRole
    ctx.messages.append(Message(role=MessageRole.USER, content="服务太差了太烂了"))
    ctx.emotion_score = 0.7
    should, reason = engine.detect_escalation(ctx)
    assert should is True
    assert reason == "vip_negative"
    print("  ✓ test_escalation_vip_negative")


# ═══════════════════════════════════════════════
# DAG编排测试
# ═══════════════════════════════════════════════

def test_dag_refund_creation():
    """DAG: 退款流程创建"""
    dag = create_refund_dag()
    assert dag.graph_id == "refund_flow"
    assert len(dag.nodes) == 7
    assert dag.entry_node == "query_order"
    print("  ✓ test_dag_refund_creation")


def test_dag_price_protect_creation():
    """DAG: 价保流程创建"""
    dag = create_price_protect_dag()
    assert dag.graph_id == "price_protect_flow"
    assert len(dag.nodes) == 4
    print("  ✓ test_dag_price_protect_creation")


def test_dag_engine_execution():
    """DAG引擎: 价保流程执行到完成"""
    engine = DAGEngine()
    execution = engine.run_to_completion_or_wait(
        intent="PRICE_PROTECT",
        session_id="test_session_1",
    )
    assert execution is not None
    assert execution.status == "completed"
    assert len(execution.completed_nodes) == 4
    print("  ✓ test_dag_engine_execution")


def test_dag_refund_hits_hitl():
    """DAG引擎: 退款流程执行到HITL等待"""
    engine = DAGEngine()
    execution = engine.run_to_completion_or_wait(
        intent="REFUND",
        session_id="test_session_2",
    )
    assert execution is not None
    assert execution.status == "waiting"
    # 应该在user_confirm节点等待
    user_confirm_state = execution.node_states.get("user_confirm")
    assert user_confirm_state is not None
    assert user_confirm_state.status == NodeStatus.WAITING
    print("  ✓ test_dag_refund_hits_hitl")


def test_dag_unknown_intent():
    """DAG引擎: 未注册意图返回None"""
    engine = DAGEngine()
    execution = engine.run_to_completion_or_wait(
        intent="UNKNOWN_INTENT",
        session_id="test_session_3",
    )
    assert execution is None
    print("  ✓ test_dag_unknown_intent")


# ═══════════════════════════════════════════════
# 模型路由测试
# ═══════════════════════════════════════════════

def test_semantic_cache():
    """语义缓存: 写入和命中"""
    cache = SemanticCache()
    cache.put("REFUND", {"order_id": "123"}, "退款已处理")
    entry = cache.get("REFUND", {"order_id": "123"})
    assert entry is not None
    assert entry.response == "退款已处理"
    assert entry.hit_count == 1
    print("  ✓ test_semantic_cache")


def test_semantic_cache_miss():
    """语义缓存: 不同slots未命中"""
    cache = SemanticCache()
    cache.put("REFUND", {"order_id": "123"}, "退款已处理")
    entry = cache.get("REFUND", {"order_id": "456"})
    assert entry is None
    print("  ✓ test_semantic_cache_miss")


def test_model_router_template_path():
    """模型路由: 高置信规则命中走模板"""
    config = create_default_config()
    router = ModelRouter(
        config=config.reasoning,
        models=config.models,
    )
    ctx = SessionContext()
    from src.common.models import IntentResult
    intent = IntentResult(intent="FAQ_RETURN_POLICY", confidence=0.95, method="rule")
    response = router.route(ctx, intent)
    assert response.model_tier == ModelTier.MICRO
    assert "7天" in response.text
    print("  ✓ test_model_router_template_path")


def test_model_router_degraded():
    """模型路由: L3降级全走模板"""
    config = create_default_config()
    router = ModelRouter(
        config=config.reasoning,
        models=config.models,
        system_level=SystemLevel.RED,
    )
    ctx = SessionContext()
    from src.common.models import IntentResult
    intent = IntentResult(intent="REFUND", confidence=0.6, method="classifier")
    response = router.route(ctx, intent)
    assert response.model_tier == ModelTier.MICRO
    assert response.model_id == "template"
    print("  ✓ test_model_router_degraded")


def test_model_router_cache_hit():
    """模型路由: 缓存命中"""
    config = create_default_config()
    router = ModelRouter(config=config.reasoning, models=config.models)
    ctx = SessionContext()
    from src.common.models import IntentResult, Message, MessageRole
    ctx.messages.append(Message(role=MessageRole.USER, content="退款"))
    intent = IntentResult(intent="REFUND", confidence=0.95, method="rule")

    # 第一次调用(写入缓存)
    r1 = router.route(ctx, intent)
    assert r1.from_cache is False

    # 第二次调用(缓存命中)
    r2 = router.route(ctx, intent)
    assert r2.from_cache is True
    print("  ✓ test_model_router_cache_hit")


# ═══════════════════════════════════════════════
# 全链路管线测试
# ═══════════════════════════════════════════════

def test_pipeline_simple_query():
    """管线: 简单查询端到端"""
    pipe = CustomerServicePipeline()
    result = pipe.process("s1", "我要退货")
    assert result.response_text != ""
    assert result.intent.intent == "REFUND"
    assert result.latency_ms > 0
    print("  ✓ test_pipeline_simple_query")


def test_pipeline_escalation():
    """管线: 情绪升级转人工"""
    pipe = CustomerServicePipeline()
    result = pipe.process("s2", "转人工客服")
    assert result.suggest_transfer_human is True
    assert "转接" in result.response_text or "人工" in result.response_text
    print("  ✓ test_pipeline_escalation")


def test_pipeline_multi_turn():
    """管线: 多轮对话保持上下文"""
    pipe = CustomerServicePipeline()
    r1 = pipe.process("s3", "查物流")
    r2 = pipe.process("s3", "具体到哪了")
    # 同一session应该保持
    assert r1.session_id == r2.session_id
    ctx = pipe._sessions["s3"]
    assert ctx.turn_count == 2
    print("  ✓ test_pipeline_multi_turn")


def test_pipeline_degradation():
    """管线: 系统降级模式"""
    pipe = CustomerServicePipeline()
    pipe.update_system_level(SystemLevel.RED)
    result = pipe.process("s4", "帮我分析一下这个问题")
    # 降级模式下不应调用LLM
    assert result.model_used == "template"
    print("  ✓ test_pipeline_degradation")


def test_pipeline_price_protect():
    """管线: 价保快速通道"""
    pipe = CustomerServicePipeline()
    result = pipe.process("s5", "刚买就降价了能价保吗")
    assert result.intent.intent == "PRICE_PROTECT"
    assert result.response_text != ""
    print("  ✓ test_pipeline_price_protect")


# ═══════════════════════════════════════════════
# 扩展能力测试 (话题栈/RAG/推荐/Saga补偿/情感画像/限流)
# ═══════════════════════════════════════════════

def test_topic_stack_switch_and_resume():
    """话题栈: 切换时挂起旧话题，完成后回切恢复"""
    from src.common.models import TopicStack
    ts = TopicStack()
    ts.switch_to("PRODUCT", focus_entity={"product_category": "耳机"})
    ts.switch_to("AFTER_SALE", focus_entity={"order_id": "O1"})
    assert ts.active_topic.topic_type == "AFTER_SALE"
    assert any(t.topic_type == "PRODUCT" for t in ts.suspended)
    # 模糊跟进不切换话题
    ts.switch_to("GENERAL")
    assert ts.active_topic.topic_type == "AFTER_SALE"
    # 完成当前话题 → 自动恢复被挂起的 PRODUCT
    ts.resolve_active()
    assert ts.active_topic.topic_type == "PRODUCT"
    print("  ✓ test_topic_stack_switch_and_resume")


def test_pipeline_rag_grounded():
    """管线: FAQ走RAG，返回有据可依的回答+来源引用"""
    pipe = CustomerServicePipeline()
    result = pipe.process("rag1", "7天无理由退货有什么条件")
    assert result.model_used == "rag"
    assert len(result.sources) >= 1
    print("  ✓ test_pipeline_rag_grounded")


def test_pipeline_recommendation():
    """管线: 导购推荐意图走推荐引擎"""
    pipe = CustomerServicePipeline()
    result = pipe.process("rec1", "推荐一款降噪耳机")
    assert result.intent.intent == "RECOMMEND"
    assert result.model_used == "recommend"
    assert "耳机" in result.response_text
    print("  ✓ test_pipeline_recommendation")


def test_dag_bind_gateway_calls_tools():
    """规划+执行: DAG节点经工具网关真正调用工具(留下审计)"""
    from src.execution.tool_gateway import create_default_gateway
    gw = create_default_gateway()
    engine = DAGEngine()
    engine.bind_gateway(gw, create_refund_dag())
    ex = engine.run_to_completion_or_wait(
        "REFUND", "sx", {"order_id": "O1", "session_id": "sx"})
    assert ex.status == "waiting"  # 停在HITL前完成了3个读节点
    assert len(gw.audit_log) >= 3  # query_order/validate_refund/risk_check
    print("  ✓ test_dag_bind_gateway_calls_tools")


def test_dag_saga_compensation():
    """规划: 节点失败触发Saga补偿，逆序撤销已成功的副作用节点"""
    from src.planning.dag_engine import (
        DAGDefinition, DAGNode, DAGEdge, NodeType, DAGExecution, NodeExecution,
    )
    engine = DAGEngine()
    compensated = []
    engine.register_handler("svc.do", lambda s: {"ok": True})

    def _boom(s):
        raise RuntimeError("boom")
    engine.register_handler("svc.fail", _boom)
    engine.register_handler("svc.undo", lambda s: compensated.append("undo"))

    dag = DAGDefinition(
        graph_id="t", name="t", entry_node="a",
        nodes=[
            DAGNode("a", NodeType.ACTION, "A", handler="svc.do",
                    compensate_handler="svc.undo"),
            DAGNode("b", NodeType.ACTION, "B", handler="svc.fail",
                    depends_on=["a"], retries=0),
        ],
        edges=[DAGEdge("a", "b")],
    )
    ex = DAGExecution(run_id="r", graph_id="t", session_id="s")
    for n in dag.nodes:
        ex.node_states[n.node_id] = NodeExecution(node_id=n.node_id)
    for _ in range(10):
        executed = engine.execute_step(dag, ex)
        if not executed or ex.status in ("completed", "failed"):
            break
    assert ex.status == "failed"
    engine.compensate(dag, ex)
    assert compensated == ["undo"]
    print("  ✓ test_dag_saga_compensation")


def test_emotion_profile_persist():
    """反馈: 长期情感画像增量更新与读取"""
    import tempfile
    from src.feedback.memory import LongTermMemory, MemoryConfig
    m = LongTermMemory(MemoryConfig(db_path=tempfile.mktemp(suffix=".db")))
    m.update_emotion_profile("u", 0.9, "angry", topic="COMPLAINT", escalated=True)
    p2 = m.update_emotion_profile("u", 0.8, "angry", topic="REFUND")
    assert p2["sample_count"] == 2
    assert p2["escalation_count"] == 1
    assert p2["churn_risk"] in ("MEDIUM", "HIGH")
    got = m.get_emotion_profile("u")
    assert "COMPLAINT" in got["sensitive_topics"]
    assert got["preferred_tone"] in ("EMPATHETIC", "EFFICIENT")
    print("  ✓ test_emotion_profile_persist")


def test_rate_limiter():
    """网关: 令牌桶限流 + VIP豁免"""
    from src.gateway.api import TokenBucketRateLimiter
    rl = TokenBucketRateLimiter(rate_per_sec=0.0, burst=2)
    assert rl.allow("t1", SystemLevel.GREEN) is True
    assert rl.allow("t1", SystemLevel.GREEN) is True
    assert rl.allow("t1", SystemLevel.GREEN) is False    # 桶空且无补充
    assert rl.allow("t1", SystemLevel.GREEN, is_vip=True) is True  # VIP豁免
    print("  ✓ test_rate_limiter")


def test_recommender_budget_filter():
    """推荐: 预算过滤 + 多目标排序"""
    from src.execution.recommend import Recommender, create_default_catalog
    rec = Recommender(create_default_catalog())
    resp = rec.recommend("给老人买手机", category="手机", budget=2000)
    assert resp.recommendations
    assert all(r.product.price <= 2000 for r in resp.recommendations)
    print("  ✓ test_recommender_budget_filter")


# ═══════════════════════════════════════════════
# 运行所有测试
# ═══════════════════════════════════════════════

def run_all_tests():
    """执行全部测试并输出报告"""
    test_groups = [
        ("意图识别", [
            test_rule_match_refund,
            test_rule_match_logistics,
            test_rule_match_price_protect,
            test_rule_match_complaint,
            test_rule_no_match,
            test_classifier_predict,
            test_emotion_detection_angry,
            test_emotion_detection_neutral,
            test_intent_engine_rule_path,
            test_intent_engine_degraded_mode,
            test_escalation_user_request,
            test_escalation_vip_negative,
        ]),
        ("DAG编排", [
            test_dag_refund_creation,
            test_dag_price_protect_creation,
            test_dag_engine_execution,
            test_dag_refund_hits_hitl,
            test_dag_unknown_intent,
        ]),
        ("模型路由", [
            test_semantic_cache,
            test_semantic_cache_miss,
            test_model_router_template_path,
            test_model_router_degraded,
            test_model_router_cache_hit,
        ]),
        ("全链路管线", [
            test_pipeline_simple_query,
            test_pipeline_escalation,
            test_pipeline_multi_turn,
            test_pipeline_degradation,
            test_pipeline_price_protect,
        ]),
        ("扩展能力", [
            test_topic_stack_switch_and_resume,
            test_pipeline_rag_grounded,
            test_pipeline_recommendation,
            test_dag_bind_gateway_calls_tools,
            test_dag_saga_compensation,
            test_emotion_profile_persist,
            test_rate_limiter,
            test_recommender_budget_filter,
        ]),
    ]

    total = 0
    passed = 0
    failed = 0
    failures = []

    print("\n" + "=" * 60)
    print("  NexusAI 智能客服系统 · 回归测试")
    print("=" * 60)

    for group_name, tests in test_groups:
        print(f"\n▶ {group_name} ({len(tests)} tests)")
        for test_fn in tests:
            total += 1
            try:
                test_fn()
                passed += 1
            except AssertionError as e:
                failed += 1
                failures.append((test_fn.__name__, str(e)))
                print(f"  ✗ {test_fn.__name__}: {e}")
            except Exception as e:
                failed += 1
                failures.append((test_fn.__name__, str(e)))
                print(f"  ✗ {test_fn.__name__}: ERROR - {e}")

    print("\n" + "=" * 60)
    print(f"  结果: {passed}/{total} passed, {failed} failed")
    print("=" * 60)

    if failures:
        print("\n失败详情:")
        for name, err in failures:
            print(f"  - {name}: {err}")
        return False

    print("\n✅ 全部测试通过!")
    return True


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
