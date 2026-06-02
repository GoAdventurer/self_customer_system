"""
NexusAI 智能客服系统 - 全链路处理管线

本模块是系统的"主干"，将六层架构串联为一条完整的处理链路。
每次用户输入都经过: 网关→输入层→意图层→规划层→执行层→推理层→反馈层。

架构定位:
    Pipeline 是系统的编排核心(不同于规划层的DAG编排)。
    规划层编排的是"业务流程步骤"(如退款7步)，
    Pipeline 编排的是"架构层级的串联顺序"。

设计原则:
    1. 单一入口: process() 是唯一的对外接口，屏蔽内部复杂性
    2. 上下文传递: SessionContext 在各层间流转，逐步丰富
    3. 快速失败: 升级检查在意图识别后立即执行，避免不必要的后续处理
    4. 可降级: 任何层级失败都有兜底(模板回复)，保证系统始终有响应

调用链路:
    外部请求 → process(session_id, user_input)
        → ① 输入层: 追加消息 + 情绪检测
        → ② 意图层: 三级识别 + 升级检查
        → ③ 规划层: DAG编排(仅复杂流程)
        → ④ 执行层: 调用外部API(通过DAG节点)
        → ⑤ 推理层: 模型路由 + 生成回复
        → ⑥ 反馈层: 记录日志
    ← PipelineResult

使用示例:
    >>> from src.pipeline import CustomerServicePipeline
    >>> pipe = CustomerServicePipeline()
    >>> result = pipe.process("session_001", "我要退款")
    >>> print(result.response_text)  # "好的，请提供订单号..."
    >>> print(result.intent.intent)  # "REFUND"
"""
import time
from typing import Optional

from .common.models import (
    Message, MessageRole, PipelineResult, SessionContext,
    EmotionLevel, SessionStatus, UserProfile,
)
from .config.settings import SystemConfig, SystemLevel, create_default_config
from .intent.engine import IntentEngine, detect_emotion
from .planning.dag_engine import DAGEngine
from .reasoning.model_router import ModelRouter
from .input.processor import InputProcessor
from .common.observability import Tracer, AuditLogger, MetricsCollector, AuditEntry
from .execution.tool_gateway import create_default_gateway
from .execution.rag import RAGPipeline, create_default_knowledge_base
from .execution.recommend import Recommender, create_default_catalog
from .feedback.memory import LongTermMemory, MemoryConfig


# 走"检索型问答(RAG)"的意图 — 信息查询类，用知识库生成有据可依的回答
KNOWLEDGE_INTENTS = {
    "FAQ", "FAQ_RETURN_POLICY", "INVOICE", "LOGISTICS",
    "GENERAL_INQUIRY", "COMPLEX_INQUIRY", "UNKNOWN",
}

# 意图 → 知识库分类(限定 RAG 检索范围，提升精度；None 表示全库检索)
INTENT_KB_CATEGORY = {
    "FAQ_RETURN_POLICY": "退货",
    "INVOICE": "发票",
    "LOGISTICS": "物流",
}

# 话题类型映射 — 将意图归并为更粗粒度的跨轮话题(对应 §3.3.1)
INTENT_TOPIC_TYPE = {
    "REFUND": "AFTER_SALE", "FAQ_RETURN_POLICY": "AFTER_SALE",
    "PRICE_PROTECT": "AFTER_SALE", "INVOICE": "AFTER_SALE",
    "LOGISTICS": "LOGISTICS", "COMPLAINT": "COMPLAINT",
    "RECOMMEND": "PRODUCT", "TRANSFER_HUMAN": "SERVICE",
}


class CustomerServicePipeline:
    """
    智能客服全链路处理管线

    系统的核心编排器，负责:
        1. 管理会话生命周期(创建/获取/过期)
        2. 按顺序调用六层架构的各引擎
        3. 处理层间数据传递和异常兜底
        4. 支持运行时动态降级(系统级别切换)

    线程安全说明:
        MVP版本使用内存字典存储会话，非线程安全。
        生产环境应替换为Redis，天然支持并发访问。

    Attributes:
        config: 全局系统配置
        intent_engine: 意图识别引擎(三级路由)
        dag_engine: DAG流程编排引擎
        model_router: 多模型路由引擎(含缓存)
    """

    def __init__(self, config: Optional[SystemConfig] = None):
        """
        初始化管线及各层引擎

        Args:
            config: 系统配置。传None则使用默认开发配置。
                    生产环境应从配置中心加载后传入。
        """
        self.config = config or create_default_config()

        # ═══ 初始化各层引擎 ═══
        # 输入层: PII脱敏 + 归一化 + 注入检测
        self.input_processor = InputProcessor()

        # 意图层: 三级混合识别(规则 → 分类器 → LLM)
        self.intent_engine = IntentEngine(
            config=self.config.intent,
            system_level=self.config.system_level,
        )

        # 执行层: 统一工具网关(订单/风控/支付/CRM/通知/物流/价保)
        self.tool_gateway = create_default_gateway(mock_mode=True)

        # 规划层: DAG流程编排(退款/价保等复杂流程)
        # 将工具网关绑定为各流程节点的 handler，使 DAG 节点真正调用执行层工具
        self.dag_engine = DAGEngine()
        for factory in DAGEngine.FLOW_REGISTRY.values():
            self.dag_engine.bind_gateway(self.tool_gateway, factory())

        # 执行层: RAG 知识检索(FAQ/政策/物流等检索型问答)
        self.rag = RAGPipeline(create_default_knowledge_base())

        # 执行层: 商品导购推荐引擎(召回→过滤→排序→理由)
        self.recommender = Recommender(create_default_catalog())

        # 推理层: 多模型路由 + 语义缓存 + Token预算
        self.model_router = ModelRouter(
            config=self.config.reasoning,
            models=self.config.models,
            system_level=self.config.system_level,
        )

        # 反馈层: 长期记忆(用户画像/事件/会话存档/情感画像，SQLite 持久化)
        self.memory = LongTermMemory(
            MemoryConfig(db_path=getattr(self.config, "memory_db_path", "data/memory.db"))
        )

        # 可观测性: Trace + 审计 + 指标
        self.audit_logger = AuditLogger()
        self.metrics = MetricsCollector()

        # 会话存储
        # MVP: 内存字典(开发/测试用)
        # 生产: 替换为Redis(支持TTL自动过期、分布式访问)
        self._sessions: dict[str, SessionContext] = {}

    def get_or_create_session(self, session_id: str = None, **kwargs) -> SessionContext:
        """
        获取已有会话或创建新会话

        实现会话的"断点续接"能力:
            - 用户30分钟内回来，恢复原会话上下文
            - 新用户或过期会话，创建全新上下文

        Args:
            session_id: 会话ID。传入已有ID则尝试恢复，传None则新建。
            **kwargs: 传递给SessionContext构造函数的额外参数。
                      常用: tenant_id, channel, user

        Returns:
            SessionContext实例(新建或已存在的)
        """
        # 尝试恢复已有会话
        if session_id and session_id in self._sessions:
            return self._sessions[session_id]

        # 从 kwargs 中分离用户标识(用于构建画像 + 加载长期记忆)
        user_id = kwargs.pop("user_id", None)
        user_tier = kwargs.pop("user_tier", "normal")

        # 创建新会话
        ctx = SessionContext(**kwargs)
        if session_id:
            ctx.session_id = session_id

        # 已识别用户: 构建画像并用长期记忆 + 情感画像注入(断点续接的认知层)
        if user_id:
            ctx.user = UserProfile(user_id=user_id, tier=user_tier)
            try:
                lt = self.memory.get_user_profile(user_id)
                ctx.user.history_complaints = lt.get("complaint_count", 0)
                ctx.metadata["long_term_profile"] = lt
                emo = self.memory.get_emotion_profile(user_id)
                if emo:
                    ctx.metadata["emotion_profile"] = emo
            except Exception:
                pass

        self._sessions[ctx.session_id] = ctx
        return ctx

    def process(self, session_id: str, user_input: str) -> PipelineResult:
        """
        处理单次用户输入 — 全链路核心方法

        这是系统对外暴露的唯一处理入口。
        一次调用完成从"用户输入"到"AI回复"的完整流程。

        处理流程:
            1. 获取/创建会话上下文
            2. [输入层] 追加消息 + 情绪检测
            3. [意图层] 三级意图识别
            4. [意图层] 升级检查(愤怒/VIP/超轮次→转人工)
            5. [规划层] DAG编排(仅注册了流程的意图)
            6. [推理层] 模型路由 + 生成回复
            7. [反馈层] 记录bot回复到消息历史
            8. 返回PipelineResult

        Args:
            session_id: 会话ID。同一用户多轮对话使用相同ID。
            user_input: 用户输入的文本内容(已由前端做基础校验)

        Returns:
            PipelineResult: 包含回复文本、意图、模型、延迟等完整信息

        Performance:
            - 简单FAQ(缓存命中): <1ms
            - 规则命中+模板回复: <5ms
            - 分类器+小模型: 50-200ms
            - LLM兜底+大模型: 1-5s (需流式返回优化首token)
        """
        start_time = time.time()
        tracer = Tracer()

        # ═══ Step 0: 获取会话上下文 ═══
        ctx = self.get_or_create_session(session_id)

        # ═══ 已升级会话: 不走AI，只存消息等坐席处理 ═══
        # 区分两个阶段(对应 §4.3 人机协同状态机):
        #   ESCALATED: 刚触发升级，坐席尚未接入 → 系统回复"请稍候"安抚用户
        #   WAITING_AGENT: 坐席已接手(已发过消息) → 系统静默，不插话干扰人机对话
        if ctx.status == SessionStatus.ESCALATED:
            ctx.messages.append(Message(role=MessageRole.USER, content=user_input))
            return PipelineResult(
                session_id=ctx.session_id,
                response_text="您的消息已收到，正在为您匹配人工客服，请稍候...",
                model_used="human_queue",
                latency_ms=(time.time() - start_time) * 1000,
                confidence=1.0,
                suggest_transfer_human=True,
            )
        if ctx.status == SessionStatus.WAITING_AGENT:
            # 坐席已接手：只存消息，不插入系统回复(用户与坐席直接对话)
            ctx.messages.append(Message(role=MessageRole.USER, content=user_input))
            return PipelineResult(
                session_id=ctx.session_id,
                response_text="",  # 空回复，前端不渲染系统气泡
                model_used="human_direct",
                latency_ms=(time.time() - start_time) * 1000,
                confidence=1.0,
                suggest_transfer_human=False,
            )

        # ═══ Step 1: 输入层 — PII脱敏 + 归一化 + 注入检测 ═══
        input_result = self.input_processor.process(user_input)

        # 提示注入阻断
        if input_result.is_blocked:
            response_text = "抱歉，您的消息包含不安全内容，无法处理。请重新描述您的问题。"
            ctx.messages.append(Message(role=MessageRole.USER, content="[blocked]"))
            ctx.messages.append(Message(role=MessageRole.BOT, content=response_text))
            self.metrics.record_request(
                latency_ms=(time.time()-start_time)*1000, success=False, intent="BLOCKED")
            return PipelineResult(
                session_id=ctx.session_id, response_text=response_text,
                latency_ms=(time.time()-start_time)*1000, confidence=0.0,
            )

        # 使用脱敏后文本
        cleaned_input = input_result.cleaned_text
        ctx.messages.append(Message(role=MessageRole.USER, content=cleaned_input))

        # 情绪检测
        emotion_level, emotion_score = detect_emotion(cleaned_input)
        ctx.emotion = EmotionLevel(emotion_level) if emotion_level != "neutral" else EmotionLevel.NEUTRAL
        ctx.emotion_score = emotion_score

        # ═══ Step 2: 意图层 — 三级混合识别 ═══
        intent_result = self.intent_engine.recognize(ctx)
        ctx.intent = intent_result
        ctx.slots.update(intent_result.slots)

        # ═══ Step 2.1: 话题层 — 维护话题栈(支持复杂话题切换/挂起/回切) ═══
        # 对应 §3.3.1: 把当前意图归并为跨轮话题，更新焦点实体与隔离槽位
        self._update_topic_stack(ctx, intent_result)

        # ═══ Step 2.5: 升级检查 ═══
        should_escalate, escalate_reason = self.intent_engine.detect_escalation(ctx)
        if should_escalate:
            ctx.status = SessionStatus.ESCALATED
            response_text = self._generate_escalation_response(escalate_reason)
            self.metrics.record_request(
                latency_ms=(time.time()-start_time)*1000, intent=intent_result.intent)
            # 反馈层: 记录升级事件 + 更新长期情感画像
            self._record_feedback(ctx, intent_result, escalated=True)
            return self._build_result(ctx, response_text, start_time, suggest_transfer=True)

        # ═══ Step 3: 规划层 + 执行层 — 按意图类型选择处理路径 ═══
        response_text = None
        model_used = ""
        sources: list[str] = []
        from_cache = False
        tokens_used = 0
        model_cost = 0.0

        degraded = self.config.system_level in (SystemLevel.RED, SystemLevel.BLACK)

        # 路径A: 事务型流程(退款/价保/物流查询) → DAG 编排 + 执行层工具
        # LOGISTICS 特殊处理: 仅当有 order_id 时走 DAG(无则继续走 RAG 知识路径)
        if intent_result.intent in DAGEngine.FLOW_REGISTRY:
            if intent_result.intent == "LOGISTICS" and not intent_result.slots.get("order_id"):
                pass  # 无 order_id 的物流查询，跳过 DAG，走后续 RAG 路径
            else:
                self._run_dag(ctx, intent_result, session_id)
                # LOGISTICS DAG 完成后直接用回填数据生成回复(跳过模板兜底)
                if intent_result.intent == "LOGISTICS" and ctx.slots.get("tracking_no"):
                    tracking = ctx.slots.get("tracking_no", "")
                    location = ctx.slots.get("current_location", "运输中")
                    eta = ctx.slots.get("estimated_delivery", "预计1-3天")
                    status = ctx.slots.get("shipping_status", "in_transit")
                    status_map = {"shipped": "已发货", "in_transit": "运输中",
                                  "delivered": "已签收", "pending": "待发货"}
                    response_text = (
                        f"已为您查到物流信息：\n"
                        f"📦 快递单号: {tracking}\n"
                        f"📍 当前位置: {location}\n"
                        f"🚚 状态: {status_map.get(status, status)}\n"
                        f"⏰ 预计送达: {eta}\n"
                        f"如需催促派送，我可以帮您联系快递员。"
                    )
                    model_used = "dag_logistics"

        # 路径B: 导购推荐 → 推荐引擎(召回→排序→理由)
        elif intent_result.intent == "RECOMMEND" and not degraded:
            rec = self.recommender.recommend(
                cleaned_input,
                category=intent_result.slots.get("category"),
                budget=self._parse_budget(cleaned_input),
            )
            response_text = rec.answer
            if rec.comparison:
                response_text += "\n\n对比:\n" + rec.comparison
            model_used = "recommend"
            sources = [r.product.product_id for r in rec.recommendations]

        # 路径C: 检索型问答(FAQ/政策/物流) → RAG 知识检索(有据可依 + 引用来源)
        elif intent_result.intent in KNOWLEDGE_INTENTS and not degraded:
            rag_resp = self.rag.query(
                cleaned_input,
                category=INTENT_KB_CATEGORY.get(intent_result.intent),
                context=[m.content for m in ctx.messages[-5:] if m.role == MessageRole.USER],
            )
            # 仅当 RAG 有足够置信度时采信；否则回落到模型路由
            if rag_resp.confidence >= self.rag.CONFIDENCE_MEDIUM and not rag_resp.warning == "no_knowledge_found":
                response_text = rag_resp.answer
                model_used = "rag"
                sources = [s["title"] for s in rag_resp.sources]

        # ═══ Step 4-5: 推理层 — 模型路由兜底(未被上面路径处理时) ═══
        if response_text is None:
            model_response = self.model_router.route(ctx, intent_result)
            response_text = model_response.text
            model_used = model_response.model_id
            from_cache = model_response.from_cache
            tokens_used = model_response.input_tokens + model_response.output_tokens
            model_cost = model_response.cost

        # ═══ Step 6: 反馈层 — 记录回复 + 长期记忆 ═══
        ctx.messages.append(Message(role=MessageRole.BOT, content=response_text))
        self._record_feedback(ctx, intent_result, escalated=False)

        # ═══ 指标+审计 ═══
        latency_ms = (time.time() - start_time) * 1000
        self.metrics.record_request(
            latency_ms=latency_ms, intent=intent_result.intent,
            model_cost=model_cost, cache_hit=from_cache)
        self.audit_logger.log_simple(
            session_id=ctx.session_id, action="QUERY",
            description=f"intent={intent_result.intent} model={model_used}",
            trace_id=tracer.trace_id)

        return PipelineResult(
            session_id=ctx.session_id,
            response_text=response_text,
            intent=intent_result,
            model_used=model_used,
            tokens_used=tokens_used,
            latency_ms=latency_ms,
            from_cache=from_cache,
            confidence=intent_result.confidence,
            sources=sources,
        )

    # ═══════════════════════════════════════════════════════════════
    # 内部辅助方法 (规划/执行/话题/反馈)
    # ═══════════════════════════════════════════════════════════════

    def _run_dag(self, ctx: SessionContext, intent_result, session_id: str):
        """运行事务型 DAG 流程，并把执行结果回填到会话上下文。

        DAG 节点通过绑定的 ToolGateway 真正调用执行层工具；遇 HITL 挂起则记录
        当前等待节点，供前端二次确认；失败则已由引擎触发 Saga 补偿。
        """
        initial_state = {
            "session_id": session_id,
            "user_id": ctx.user.user_id if ctx.user else "",
            **ctx.slots,
        }
        dag_execution = self.dag_engine.run_to_completion_or_wait(
            intent=intent_result.intent,
            session_id=session_id,
            initial_state=initial_state,
        )
        if not dag_execution:
            return
        # 将 DAG 执行实例缓存到会话(供 resume/审计)
        ctx.metadata["dag_run_id"] = dag_execution.run_id
        ctx.metadata["dag_status"] = dag_execution.status
        if dag_execution.status == "waiting":
            ctx.status = SessionStatus.WAITING_USER
            ctx.current_dag_node = next(
                (nid for nid, ns in dag_execution.node_states.items()
                 if ns.status.value == "waiting"), None
            )
        # 把工具返回的关键数据并入槽位(供后续轮次/回复使用)
        for key in ("order_detail", "refund_amount", "price_diff", "tracking_no",
                    "shipping_status", "estimated_delivery", "current_location"):
            if key in dag_execution.global_state:
                ctx.slots[key] = dag_execution.global_state[key]

    def _update_topic_stack(self, ctx: SessionContext, intent_result):
        """根据本轮意图更新话题栈(对应 §3.3.1)。"""
        topic_type = INTENT_TOPIC_TYPE.get(intent_result.intent, "GENERAL")
        # 焦点实体: 优先订单号，其次商品类目
        focus = {}
        if intent_result.slots.get("order_id"):
            focus["order_id"] = intent_result.slots["order_id"]
        if intent_result.slots.get("category"):
            focus["product_category"] = intent_result.slots["category"]
        ctx.topic_stack.switch_to(topic_type, focus_entity=focus, slots=dict(intent_result.slots))

    def _record_feedback(self, ctx: SessionContext, intent_result, escalated: bool):
        """反馈层: 长期记忆写回(仅对已识别用户)。

        · 升级事件 → user_events
        · 每轮情绪 → 长期情感画像(EmotionProfile)
        · 已识别用户的会话满足轮次门槛时归档
        生产: 应异步化(消息队列)，避免阻塞主链路。
        """
        if ctx.user is None:
            return
        uid = ctx.user.user_id
        try:
            # 更新长期情感画像
            profile = self.memory.update_emotion_profile(
                user_id=uid,
                emotion_score=ctx.emotion_score,
                emotion_level=ctx.emotion.value,
                topic=intent_result.intent,
                escalated=escalated,
            )
            ctx.metadata["emotion_profile"] = profile

            if escalated:
                self.memory.record_event(
                    uid, "escalation",
                    summary=f"升级转人工(intent={intent_result.intent})",
                    result=ctx.intent.intent if ctx.intent else "",
                )
            if intent_result.intent == "REFUND":
                self.memory.record_event(
                    uid, "refund", summary="发起退款流程",
                    metadata={"slots": ctx.slots})
            # 会话归档(达到轮次门槛)
            if ctx.turn_count >= self.memory.config.archive_session_min_turns:
                self.memory.archive_session(
                    session_id=ctx.session_id, user_id=uid,
                    intent=intent_result.intent,
                    summary=(ctx.last_user_message or "")[:200],
                    turn_count=ctx.turn_count,
                    escalated=escalated,
                    resolved=not escalated,
                )
        except Exception:
            # 长期记忆失败不得影响主链路可用性
            pass

    @staticmethod
    def _parse_budget(text: str) -> Optional[float]:
        """从需求文本中解析预算上限(支持"两千""2000以内""1k"等)。"""
        import re
        cn_num = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
                  "六": 6, "七": 7, "八": 8, "九": 9}
        m = re.search(r"(\d+(?:\.\d+)?)\s*([kK千万])?", text)
        if m:
            val = float(m.group(1))
            unit = m.group(2)
            if unit in ("k", "K", "千"):
                val *= 1000
            elif unit == "万":
                val *= 10000
            return val
        m2 = re.search(r"([一二两三四五六七八九])千", text)
        if m2:
            return cn_num[m2.group(1)] * 1000
        return None

    def resume_dag(self, session_id: str, confirmed: bool = True) -> Optional[str]:
        """恢复因 HITL 挂起的 DAG(用户在前端点击确认/取消后调用)。

        对应 §3.2 HITL 与 §8 断点恢复: 把等待中的 HITL 节点标记完成并续跑，
        或在用户取消时触发流程终止。当前 MVP 直接复跑流程到完成(网关幂等保证
        已成功的读节点不重复产生副作用)。

        Returns:
            DAG 最终状态字符串(completed/failed/None)
        """
        ctx = self._sessions.get(session_id)
        if not ctx or not ctx.intent:
            return None
        if not confirmed:
            ctx.status = SessionStatus.ACTIVE
            ctx.current_dag_node = None
            return "cancelled"
        # 复跑: 因 mock 工具幂等，直接重新执行到完成，跳过 HITL
        from .planning.dag_engine import NodeStatus
        execution = self.dag_engine.create_execution(ctx.intent.intent, session_id)
        if not execution:
            return None
        execution.global_state.update({"session_id": session_id, **ctx.slots})
        dag_def = DAGEngine.FLOW_REGISTRY[ctx.intent.intent]()
        # 预先把 HITL 节点标记完成(用户已确认)
        for node in dag_def.nodes:
            if node.node_type.value == "hitl":
                execution.node_states[node.node_id].status = NodeStatus.COMPLETED
        for _ in range(20):
            executed = self.dag_engine.execute_step(dag_def, execution)
            if not executed or execution.status in ("completed", "failed"):
                break
        if execution.status == "failed":
            self.dag_engine.compensate(dag_def, execution)
        ctx.current_dag_node = None
        ctx.status = SessionStatus.COMPLETED if execution.status == "completed" else ctx.status
        ctx.metadata["dag_status"] = execution.status
        return execution.status

    def _generate_escalation_response(self, reason: str) -> str:
        """
        根据升级原因生成对应的转人工过渡话术

        不同升级原因对应不同的安抚语气:
            - 用户主动请求: 直接确认转接
            - 情绪愤怒: 道歉 + 转接
            - 超轮次未解决: 承认问题复杂 + 转接
            - VIP负面: 尊称 + 专属转接

        Args:
            reason: 升级原因标识(来自 IntentEngine.detect_escalation)

        Returns:
            适配的过渡话术文本
        """
        responses = {
            "user_requested": "好的，正在为您转接人工客服，请稍候...",
            "emotion_angry": "非常抱歉给您带来不好的体验，正在为您转接专属客服，请稍候。",
            "unresolved_5_turns": "看来这个问题比较复杂，我为您转接人工客服来更好地帮助您。",
            "vip_negative": "尊敬的VIP会员，我为您转接专属客服处理，请稍候。",
            "service_quality_complaint": "非常抱歉我的回复没有帮到您。为确保您得到满意的解答，正在为您转接人工客服。",
        }
        # 如果reason是加权评分触发的(格式: "weighted_score_0.82")，使用通用话术
        return responses.get(reason, "正在为您转接人工客服，请稍候...")

    def _build_result(self, ctx: SessionContext, text: str,
                      start_time: float, suggest_transfer: bool = False) -> PipelineResult:
        """
        构建管线结果(快速通道版本)

        用于不经过推理层的快速返回场景(如升级转人工)。
        与正常流程的区别: 跳过模型路由，直接使用预设话术。

        Args:
            ctx: 当前会话上下文
            text: 要返回的回复文本
            start_time: 管线开始处理的时间戳(用于计算延迟)
            suggest_transfer: 是否标记"建议转人工"(前端据此展示转接UI)

        Returns:
            PipelineResult实例
        """
        # 记录bot回复到消息历史
        ctx.messages.append(Message(role=MessageRole.BOT, content=text))
        return PipelineResult(
            session_id=ctx.session_id,
            response_text=text,
            intent=ctx.intent,
            model_used="template",
            latency_ms=(time.time() - start_time) * 1000,
            confidence=ctx.intent.confidence if ctx.intent else 0.0,
            suggest_transfer_human=suggest_transfer,
        )

    def update_system_level(self, level: SystemLevel):
        """
        动态更新系统负载级别

        运维接口调用此方法实现运行时的限流/降级切换。
        级别变更会立即影响所有后续请求的处理行为:
            - 意图层: L3+关闭LLM兜底
            - 推理层: L3+零LLM调用，全走模板

        使用场景:
            - 大促开始前: GREEN → YELLOW (预热)
            - 流量爆发时: 自动 YELLOW → ORANGE → RED
            - 故障恢复后: RED → ORANGE → GREEN (逐步恢复)

        Args:
            level: 目标系统级别(SystemLevel枚举值)

        Side Effects:
            同时更新 intent_engine 和 model_router 的级别引用
        """
        self.config.system_level = level
        self.intent_engine.system_level = level
        self.model_router.system_level = level
