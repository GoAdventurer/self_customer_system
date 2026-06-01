"""
NexusAI 智能客服系统 - 公共数据模型

本模块定义了贯穿全链路的核心数据结构，是六层架构之间通信的"契约"。
所有层级通过这些统一的数据模型传递信息，确保数据结构一致性。

架构定位:
    公共模型层 (Common Models) 是所有业务层的基础依赖。
    任何层级的输入/输出都应使用此处定义的数据结构。

模块包含:
    - Channel: 接入渠道枚举
    - EmotionLevel: 情绪级别枚举
    - SessionStatus: 会话状态枚举
    - MessageRole: 消息角色枚举
    - UserProfile: 用户画像数据
    - Message: 单条消息
    - IntentResult: 意图识别结果
    - SessionContext: 会话上下文(核心，贯穿全链路)
    - PipelineResult: 管线处理结果(API层返回)

设计原则:
    1. 不可变优先: 使用 dataclass 的 frozen=False 但建议只追加不修改
    2. 可序列化: 所有字段类型支持 JSON 序列化(便于Redis存储和API传输)
    3. 自包含: SessionContext 包含该会话的所有上下文信息，避免跨模块查询
"""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional
import uuid


class Channel(Enum):
    """
    接入渠道枚举

    标识用户通过哪个渠道接入客服系统。
    不同渠道可能有不同的消息格式限制和交互能力。

    Values:
        APP: 移动App(支持富媒体、语音、文件上传)
        WEB: 网页端(支持文本、文件上传)
        WECHAT: 微信小程序/公众号(消息格式受限)
        PHONE: 电话(语音转文字，无法发送富媒体)
    """
    APP = "app"
    WEB = "web"
    WECHAT = "wechat"
    PHONE = "phone"


class EmotionLevel(Enum):
    """
    情绪级别枚举

    由输入层的情绪检测模块标注，用于:
        1. 升级决策(愤怒→转人工)
        2. 回复语气调整(负面→增加共情)
        3. 坐席工作台情绪曲线展示
        4. 质检评分参考

    Values:
        POSITIVE: 正面情绪(满意、感谢)
        NEUTRAL: 中性(正常咨询)
        NEGATIVE: 负面(不满、抱怨)
        ANGRY: 愤怒(投诉、威胁，需立即升级)
    """
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    ANGRY = "angry"


class SessionStatus(Enum):
    """
    会话状态枚举

    对应架构文档中的会话状态机(docs/02-scenario-deep-dive.md §2.6)。
    会话在生命周期内可经历多个状态转换。

    状态流转:
        ACTIVE → WAITING_USER (追问槽位时)
        ACTIVE → ESCALATED (触发升级条件)
        ACTIVE → COMPLETED (问题解决)
        WAITING_USER → ACTIVE (用户回复)
        WAITING_USER → ABANDONED (超时未回复)
        ESCALATED → WAITING_AGENT (排队中)
        WAITING_AGENT → ACTIVE (坐席接入后)

    Values:
        ACTIVE: 活跃对话中(AI或坐席正在处理)
        WAITING_USER: 等待用户回复(追问槽位/确认信息)
        WAITING_AGENT: 等待人工坐席接入(已升级)
        ESCALATED: 已触发升级(正在匹配坐席)
        COMPLETED: 会话已完成(问题已解决)
        ABANDONED: 已放弃(超时未回复)
    """
    ACTIVE = "active"
    WAITING_USER = "waiting_user"
    WAITING_AGENT = "waiting_agent"
    ESCALATED = "escalated"
    COMPLETED = "completed"
    ABANDONED = "abandoned"


class MessageRole(Enum):
    """
    消息角色枚举

    标识一条消息的发送者身份，用于:
        1. 对话渲染(气泡方向/颜色)
        2. 上下文构建(区分用户输入和系统回复)
        3. 审计日志(标记操作主体)

    Values:
        USER: 终端用户发送的消息
        BOT: AI机器人自动生成的回复
        AGENT: 人工坐席发送的消息
        SYSTEM: 系统通知(如"正在转接人工客服")
    """
    USER = "user"
    BOT = "bot"
    AGENT = "agent"
    SYSTEM = "system"


@dataclass
class UserProfile:
    """
    用户画像

    从CRM/用户系统获取的用户信息，注入到会话上下文中，
    用于个性化服务决策:
        - VIP用户优先升级、更高Token预算
        - 高LTV用户倾向性满足补偿请求
        - 历史投诉多的用户提前预警

    Attributes:
        user_id: 用户唯一标识(已脱敏)
        tier: 会员等级。"normal"普通 / "vip"VIP / "svip"超级VIP
        ltv: 用户生命周期价值(元)。用于判断补偿额度合理性
        history_complaints: 历史投诉次数。>2次的用户强制升级
        satisfaction_score: 历史满意度均分(1-5)
        recent_orders: 最近订单ID列表。用于自动补全槽位
    """
    user_id: str
    tier: str = "normal"          # normal / vip / svip
    ltv: float = 0.0
    history_complaints: int = 0
    satisfaction_score: float = 0.0
    recent_orders: list[str] = field(default_factory=list)


@dataclass
class Message:
    """
    单条消息

    会话中的一条完整消息记录，包含内容、角色、时间戳和扩展元数据。
    消息列表按时间顺序存储在 SessionContext.messages 中。

    Attributes:
        message_id: 消息唯一ID(UUID v4)，用于审计追溯
        role: 消息发送者角色(USER/BOT/AGENT/SYSTEM)
        content: 消息文本内容
        timestamp: 消息创建时间(UTC)
        metadata: 扩展元数据。可包含:
            - "emotion_score": 该条消息的情绪分数
            - "intent": 该条消息触发的意图
            - "attachments": 附件列表
            - "ai_confidence": AI回复的置信度
    """
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    role: MessageRole = MessageRole.USER
    content: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class IntentResult:
    """
    意图识别结果

    意图层三级识别引擎的输出结果，包含主意图、置信度、识别方法和抽取的槽位。
    此结果传递给规划层用于DAG选择，传递给推理层用于模型路由。

    Attributes:
        intent: 主意图标签。如 "REFUND"、"LOGISTICS"、"PRICE_PROTECT"
        confidence: 置信度(0-1)。
            >0.9 高置信(规则命中)
            0.8-0.9 中高置信(分类器)
            <0.7 低置信(需LLM兜底或澄清)
        method: 识别方法。标记是哪一级引擎命中的:
            "rule": L1规则/正则命中(零成本)
            "classifier": L2分类器命中
            "llm_fallback": L3 LLM兜底
            "degraded": 降级模式下的兜底返回
            "none": 未能识别
        slots: 抽取的结构化槽位。如 {"order_id": "12345", "reason": "质量"}
        sub_intents: 子意图列表(复合问题场景)。
            如用户同时问退货+退税，sub_intents=["RETURN", "TAX_REFUND"]
    """
    intent: str                     # 主意图标签
    confidence: float               # 置信度 0-1
    method: str = "rule"            # 识别方法: rule / classifier / llm_fallback / degraded
    slots: dict[str, Any] = field(default_factory=dict)
    sub_intents: list[str] = field(default_factory=list)


@dataclass
class TopicFrame:
    """
    话题帧 — 对应 docs/01-architecture-overview.md §3.3.1 TopicFrame

    一个话题是跨轮的语义焦点（区别于单轮的意图）。一个话题可包含多轮多意图
    （如"导购"话题内含介绍、比价、加购）。话题帧记录该话题的焦点实体、
    隔离的槽位、当前状态，使会话支持"挂起—恢复"的复杂话题切换。

    Attributes:
        topic_id: 话题唯一ID
        topic_type: 话题类型(PRODUCT/ORDER/AFTER_SALE/LOGISTICS/COMPLAINT/PROMOTION...)
        focus_entity: 话题焦点实体，如 {"product_id": "P1"} / {"order_id": "O1"}
        slots: 该话题隔离收集的槽位(不与其他话题串话)
        status: ACTIVE(当前焦点) / SUSPENDED(被挂起) / RESOLVED(已完成)
        last_active_at: 最近活跃时间(用于回切判断和过期回收)
    """
    topic_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    topic_type: str = "GENERAL"
    focus_entity: dict[str, Any] = field(default_factory=dict)
    slots: dict[str, Any] = field(default_factory=dict)
    status: str = "ACTIVE"        # ACTIVE / SUSPENDED / RESOLVED
    last_active_at: datetime = field(default_factory=datetime.now)


@dataclass
class TopicStack:
    """
    话题栈 — 对应 docs/01-architecture-overview.md §3.3.1 TopicStack

    解决"一句话含多话题"与"会话中来回切换话题"的复杂表达问题。
    栈顶为当前焦点话题，被挂起的话题压入栈中(后进先出)，已关闭话题进入历史
    (用于"刚才那个"的指代消解)。

    Attributes:
        active_topic: 当前焦点话题(栈顶)
        suspended: 被挂起的话题列表(后进先出)
        history: 已关闭话题列表(用于指代消解)
        max_depth: 最大嵌套深度，超出则归并最旧话题
    """
    active_topic: Optional[TopicFrame] = None
    suspended: list[TopicFrame] = field(default_factory=list)
    history: list[TopicFrame] = field(default_factory=list)
    max_depth: int = 5

    def switch_to(self, topic_type: str, focus_entity: dict = None,
                  slots: dict = None) -> TopicFrame:
        """
        根据新一轮的意图/实体更新话题栈，返回当前生效的话题帧。

        决策逻辑(对应 §3.3.1 话题切换决策表):
          - 无活跃话题 → 新建并置为活跃
          - 与栈顶同类型或同焦点实体 → 延续(更新槽位/焦点)
          - 能在挂起栈/历史中找到匹配旧话题 → 回切恢复
          - 否则 → 挂起当前话题，压栈，新建活跃话题
        """
        focus_entity = focus_entity or {}
        slots = slots or {}

        # 无活跃话题：新建
        if self.active_topic is None:
            self.active_topic = TopicFrame(
                topic_type=topic_type, focus_entity=focus_entity, slots=slots)
            return self.active_topic

        # 延续当前话题：同类型 / 同焦点实体 / 模糊跟进(GENERAL，如"那这个呢")
        same_type = self.active_topic.topic_type == topic_type
        same_entity = bool(focus_entity) and self._entity_match(self.active_topic.focus_entity, focus_entity)
        is_followup = topic_type == "GENERAL"  # 无明确新话题的承接性跟进
        if same_type or same_entity or is_followup:
            self.active_topic.last_active_at = datetime.now()
            if focus_entity:
                self.active_topic.focus_entity.update(focus_entity)
            self.active_topic.slots.update(slots)
            if same_type or same_entity:
                self.active_topic.topic_type = topic_type
            return self.active_topic

        # 回切：在挂起栈/历史中寻找匹配的旧话题
        recovered = self._find_recoverable(topic_type, focus_entity)
        if recovered is not None:
            # 当前话题挂起
            self.active_topic.status = "SUSPENDED"
            self.suspended.append(self.active_topic)
            recovered.status = "ACTIVE"
            recovered.last_active_at = datetime.now()
            recovered.slots.update(slots)
            self.active_topic = recovered
            return self.active_topic

        # 全新话题：挂起当前，压栈，新建
        self.active_topic.status = "SUSPENDED"
        self.suspended.append(self.active_topic)
        self._enforce_depth()
        self.active_topic = TopicFrame(
            topic_type=topic_type, focus_entity=focus_entity, slots=slots)
        return self.active_topic

    def resolve_active(self):
        """完成当前话题：弹出并自动恢复下一个被挂起的话题。"""
        if self.active_topic is None:
            return
        self.active_topic.status = "RESOLVED"
        self.history.append(self.active_topic)
        self.active_topic = self.suspended.pop() if self.suspended else None
        if self.active_topic is not None:
            self.active_topic.status = "ACTIVE"

    @staticmethod
    def _entity_match(a: dict, b: dict) -> bool:
        """两个焦点实体是否指向同一对象(任一同名键值相等即视为匹配)。"""
        return any(k in a and a[k] == v for k, v in b.items())

    def _find_recoverable(self, topic_type: str, focus_entity: dict) -> Optional[TopicFrame]:
        """在挂起栈与历史中查找可回切的旧话题(类型或焦点实体匹配)。"""
        for pool in (self.suspended, self.history):
            for frame in reversed(pool):
                if frame.topic_type == topic_type or self._entity_match(frame.focus_entity, focus_entity):
                    if frame in self.suspended:
                        self.suspended.remove(frame)
                    elif frame in self.history:
                        self.history.remove(frame)
                    return frame
        return None

    def _enforce_depth(self):
        """限制栈深：超出 max_depth 时归并最旧的挂起话题到历史。"""
        while len(self.suspended) >= self.max_depth:
            oldest = self.suspended.pop(0)
            oldest.status = "RESOLVED"
            self.history.append(oldest)


@dataclass
class EmotionProfile:
    """
    客户长期情感画像 — 对应 docs/01-architecture-overview.md §4.2.2

    情绪信号不止用于单会话升级，还沉淀为长期情感画像，影响后续服务策略
    (模型路由、话术风格、主动挽留)。

    Attributes:
        user_id: 用户ID
        baseline_sentiment: 历史基线情绪[0-1](识别"易怒型"客户)
        recent_trend: 近期趋势 IMPROVING / STABLE / DETERIORATING
        sensitive_topics: 历史触发负面情绪的话题(意图标签)
        escalation_count: 历史升级次数
        preferred_tone: 偏好服务语气 EMPATHETIC / EFFICIENT / FORMAL
        churn_risk: 流失风险 LOW / MEDIUM / HIGH
        sample_count: 累计样本数(用于增量更新基线)
    """
    user_id: str
    baseline_sentiment: float = 0.1
    recent_trend: str = "STABLE"
    sensitive_topics: list[str] = field(default_factory=list)
    escalation_count: int = 0
    preferred_tone: str = "EFFICIENT"
    churn_risk: str = "LOW"
    sample_count: int = 0


@dataclass
class SessionContext:
    """
    会话上下文 - 系统核心数据结构

    贯穿全链路的会话状态容器，包含该次会话的所有上下文信息。
    从网关层创建，经过每一层处理时不断丰富，最终用于生成回复。

    生命周期:
        1. 网关层创建(注入session_id, tenant_id, channel)
        2. 输入层丰富(追加消息, 标注情绪)
        3. 意图层标注(设置intent, 填充slots)
        4. 规划层推进(更新current_dag_node, status)
        5. 推理层消费(基于全部上下文生成回复)
        6. 反馈层记录(写入日志和长期记忆)

    持久化策略:
        - 短期(Redis): 活跃会话，TTL=30min
        - 长期(DB): 有业务价值的会话(投诉/退款)持久化

    Attributes:
        session_id: 会话唯一ID(UUID v4)
        tenant_id: 租户ID(多租户隔离)
        channel: 接入渠道
        user: 用户画像(可选，匿名用户无此信息)
        messages: 消息列表(按时间顺序)
        intent: 当前轮次的意图识别结果
        emotion: 当前情绪级别
        emotion_score: 情绪分数(0-1, 越高越激烈)
        status: 会话当前状态
        slots: 已收集的全部槽位(跨轮次累积)
        current_dag_node: 当前执行到的DAG节点ID(用于中断恢复)
        created_at: 会话创建时间
        metadata: 扩展元数据
    """
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str = "default"
    channel: Channel = Channel.APP
    user: Optional[UserProfile] = None
    messages: list[Message] = field(default_factory=list)
    intent: Optional[IntentResult] = None
    emotion: EmotionLevel = EmotionLevel.NEUTRAL
    emotion_score: float = 0.0
    status: SessionStatus = SessionStatus.ACTIVE
    slots: dict[str, Any] = field(default_factory=dict)
    current_dag_node: Optional[str] = None
    topic_stack: "TopicStack" = field(default_factory=lambda: TopicStack())
    created_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def last_user_message(self) -> Optional[str]:
        """
        获取最近一条用户消息的文本内容

        从消息列表末尾向前查找第一条 role=USER 的消息。
        用于意图识别和回复生成时获取当前用户输入。

        Returns:
            最近用户消息的文本，或None(无用户消息)
        """
        for msg in reversed(self.messages):
            if msg.role == MessageRole.USER:
                return msg.content
        return None

    @property
    def turn_count(self) -> int:
        """
        获取用户对话轮次数

        统计消息列表中 role=USER 的消息数量。
        用于升级决策(连续5轮未解决→强制升级)。

        Returns:
            用户消息总数(即对话轮次)
        """
        return sum(1 for m in self.messages if m.role == MessageRole.USER)

    @property
    def is_vip(self) -> bool:
        """
        判断当前用户是否为VIP

        VIP用户享有:
            - 不限流(大促期间也保障)
            - 更高Token预算
            - 负面情绪时优先升级
            - 专属坐席组

        Returns:
            True if 用户等级为vip或svip
        """
        return self.user is not None and self.user.tier in ("vip", "svip")


@dataclass
class PipelineResult:
    """
    全链路处理结果

    CustomerServicePipeline.process() 的返回值，包含:
        - 给用户的回复文本
        - 处理过程中的元信息(意图、模型、延迟等)
        - 后续动作建议(是否转人工)

    此结构直接映射为API响应(ChatResponse)返回给前端。

    Attributes:
        session_id: 会话ID(前端用于后续请求关联)
        response_text: AI生成的回复文本(直接展示给用户)
        intent: 意图识别结果(供前端展示意图标签/调试)
        model_used: 实际使用的模型ID。
            "template": 模板回复(零LLM)
            "cache": 缓存命中
            其他: 具体模型ID如"qwen2.5-7b"
        tokens_used: 本次请求消耗的总Token数(输入+输出)
        latency_ms: 全链路处理耗时(毫秒)
        from_cache: 是否来自语义缓存命中
        confidence: 意图识别置信度(用于前端展示确定性)
        sources: 回答引用的知识来源列表(防幻觉设计)
        suggest_transfer_human: 是否建议转人工。
            True时前端应展示转人工按钮/自动触发转接
    """
    session_id: str
    response_text: str
    intent: Optional[IntentResult] = None
    model_used: str = ""
    tokens_used: int = 0
    latency_ms: float = 0.0
    from_cache: bool = False
    confidence: float = 0.0
    sources: list[str] = field(default_factory=list)
    suggest_transfer_human: bool = False
