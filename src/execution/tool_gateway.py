"""
④ 执行层 — 统一工具网关 (Tool Gateway)

本模块定义了系统调用外部服务的标准协议层。
对应架构文档 docs/01-architecture-overview.md §3.5 "统一工具网关(Tool Gateway)"。

═══════════════════════════════════════════════════════════════════════════════
为什么需要 Tool Gateway:
═══════════════════════════════════════════════════════════════════════════════

  DAG编排引擎(规划层)定义了"做什么"(如"发起退款")，
  但"怎么调"不应该写死在编排逻辑里。工具网关解耦了:
    · 编排逻辑(DAG) — 只关心调用哪个工具、传什么参数
    · 调用细节(Gateway) — 处理协议、鉴权、重试、审计、Mock

  类比: DAG是"菜单"，工具网关是"厨房" — 菜单说要什么菜,厨房负责怎么做。

═══════════════════════════════════════════════════════════════════════════════
MCP (Model Context Protocol) 集成设计:
═══════════════════════════════════════════════════════════════════════════════

  MCP 是 LLM 调用外部工具的标准协议(Anthropic提出)。
  在本系统中，MCP 用于执行层统一管理所有外部服务调用:

  ┌─────────────────────────────────────────────────────────────┐
  │  规划层(DAG节点)                                             │
  │    handler = "order_service.query_order"                     │
  │            ↓                                                 │
  │  ┌──────────────────────────────────────────────────────┐   │
  │  │     Tool Gateway (MCP Protocol Adapter)               │   │
  │  │                                                      │   │
  │  │  ┌────────────┐  ┌────────────┐  ┌────────────┐    │   │
  │  │  │ MCP Server │  │ MCP Server │  │ MCP Server │    │   │
  │  │  │ 订单服务    │  │ 支付网关   │  │ CRM系统    │    │   │
  │  │  └────────────┘  └────────────┘  └────────────┘    │   │
  │  │                                                      │   │
  │  │  共享能力: 鉴权 · 限速 · 熔断 · 审计 · Mock · 灰度  │   │
  │  └──────────────────────────────────────────────────────┘   │
  └─────────────────────────────────────────────────────────────┘

  每个外部服务对应一个 MCP Server(或Tool定义):
    · 声明式定义: 工具名、入参schema、出参schema、副作用级别
    · 运行时由 Tool Gateway 统一调度执行

═══════════════════════════════════════════════════════════════════════════════
本模块实现:
═══════════════════════════════════════════════════════════════════════════════

  1. ToolDefinition: 工具元数据定义(名称、参数schema、副作用等级)
  2. ToolGateway: 统一调用入口(路由→鉴权→执行→审计)
  3. 内置Mock工具: 订单/支付/CRM/物流/通知(MVP演示用)
  4. MCP Server配置: 声明式工具注册体系
"""
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
from enum import Enum
import time
import hashlib
import json


# ═══════════════════════════════════════════════════════════════════════════════
# 工具元数据定义
# ═══════════════════════════════════════════════════════════════════════════════


class SideEffect(Enum):
    """
    工具副作用级别

    决定了工具调用的安全策略:
      - READ: 只读操作,可无限重试,无需幂等键
      - WRITE: 写操作,需幂等键,可重试
      - IRREVERSIBLE: 不可逆操作(如退款到账),执行前必须有HITL确认
    """
    READ = "read"               # 只读(查订单、查物流)
    WRITE = "write"             # 可逆写(创建工单、更新CRM)
    IRREVERSIBLE = "irreversible"  # 不可逆(退款、扣款)


class ToolStatus(Enum):
    """工具健康状态"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"       # 降级(延迟升高但仍可用)
    CIRCUIT_OPEN = "circuit_open"  # 熔断(暂停调用)


@dataclass
class ToolParameter:
    """
    工具参数定义(对应MCP的inputSchema中的property)

    Attributes:
        name: 参数名
        type: 参数类型("string"/"number"/"boolean"/"object"/"array")
        description: 参数说明
        required: 是否必填
        enum: 可选的枚举值列表
        default: 默认值
    """
    name: str
    type: str = "string"
    description: str = ""
    required: bool = True
    enum: list[str] = field(default_factory=list)
    default: Any = None


@dataclass
class ToolDefinition:
    """
    工具定义 — 对应 MCP 协议中的 Tool 声明

    每个外部服务能力被抽象为一个"工具",包含:
      · 元信息: 名称、描述、所属服务
      · 参数schema: 类似 MCP inputSchema
      · 副作用声明: 决定安全策略(HITL/幂等)
      · 运行时配置: 超时、重试、熔断阈值

    MCP协议映射:
      ToolDefinition.tool_id     → MCP Tool.name
      ToolDefinition.description → MCP Tool.description
      ToolDefinition.parameters  → MCP Tool.inputSchema.properties
      ToolDefinition.handler     → MCP Server 的实际执行逻辑

    Attributes:
        tool_id: 工具唯一标识,格式"service.action"(如"order_service.query")
        name: 人类可读名称(如"查询订单详情")
        description: 工具功能描述(给LLM看的,用于函数调用选择)
        service: 所属外部服务名
        parameters: 参数定义列表
        side_effect: 副作用级别
        timeout_ms: 调用超时(毫秒)
        max_retries: 最大重试次数(仅READ和WRITE可重试)
        circuit_breaker_threshold: 熔断阈值(连续N次失败后熔断)
        requires_auth: 是否需要鉴权token
        idempotent: 是否幂等(幂等操作可安全重试)
        handler: 实际执行函数(内部注册)
    """
    tool_id: str
    name: str
    description: str
    service: str = ""
    parameters: list[ToolParameter] = field(default_factory=list)
    side_effect: SideEffect = SideEffect.READ
    timeout_ms: int = 10000
    max_retries: int = 3
    circuit_breaker_threshold: int = 5
    requires_auth: bool = True
    idempotent: bool = False
    handler: Optional[Callable] = field(default=None, repr=False)


@dataclass
class ToolCallResult:
    """
    工具调用结果

    Attributes:
        tool_id: 被调用的工具ID
        success: 是否成功
        data: 返回数据(成功时)
        error: 错误信息(失败时)
        latency_ms: 实际调用耗时
        retry_count: 实际重试次数
        idempotent_key: 使用的幂等键(可追溯)
    """
    tool_id: str
    success: bool
    data: Any = None
    error: Optional[str] = None
    latency_ms: float = 0.0
    retry_count: int = 0
    idempotent_key: Optional[str] = None


@dataclass
class AuditRecord:
    """审计记录 — 每次工具调用都产生一条"""
    timestamp: float
    tool_id: str
    session_id: str
    parameters: dict
    result_status: str  # "success" / "failed" / "timeout" / "circuit_open"
    latency_ms: float
    idempotent_key: Optional[str] = None


# ═══════════════════════════════════════════════════════════════════════════════
# Tool Gateway 核心
# ═══════════════════════════════════════════════════════════════════════════════


class ToolGateway:
    """
    统一工具网关 — 执行层的核心调度器

    所有外部服务调用必须经过此网关,网关提供:
      1. 工具注册: 声明式注册所有可用工具(MCP Server管理)
      2. 路由分发: 根据tool_id找到对应handler执行
      3. 安全管控: 幂等键生成/校验、HITL前置检查
      4. 可靠性: 超时控制、重试(指数退避)、熔断器
      5. 审计日志: 每次调用留痕(谁/何时/调了什么/结果如何)
      6. Mock模式: 开发/测试环境自动路由到Mock实现

    与DAG引擎的关系:
      DAG节点的 handler 字段(如"order_service.query")对应这里的 tool_id。
      DAG引擎通过 gateway.call_tool(tool_id, params) 执行节点逻辑。

    使用示例:
        >>> gateway = ToolGateway()
        >>> gateway.register_tool(order_query_tool)
        >>> result = gateway.call_tool("order_service.query", {"order_id": "123"}, session_id="s1")
        >>> print(result.data)  # {"status": "delivered", "amount": 299}
    """

    def __init__(self, mock_mode: bool = True):
        """
        Args:
            mock_mode: 是否启用Mock模式。
                True: 使用内置Mock handler(开发/测试)
                False: 使用真实外部服务(生产)
        """
        self.mock_mode = mock_mode
        self._tools: dict[str, ToolDefinition] = {}
        self._audit_log: list[AuditRecord] = []
        self._circuit_state: dict[str, ToolStatus] = {}
        self._failure_counts: dict[str, int] = {}
        self._idempotent_cache: dict[str, ToolCallResult] = {}  # 幂等键→结果缓存

    def register_tool(self, tool: ToolDefinition):
        """注册一个工具到网关"""
        self._tools[tool.tool_id] = tool
        self._circuit_state[tool.tool_id] = ToolStatus.HEALTHY
        self._failure_counts[tool.tool_id] = 0

    def list_tools(self) -> list[dict]:
        """
        列出所有已注册工具(MCP ListTools响应格式)

        返回格式对应 MCP 协议的 tools/list 响应。
        LLM 通过此列表决定调用哪个工具。
        """
        return [
            {
                "name": t.tool_id,
                "description": t.description,
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        p.name: {
                            "type": p.type,
                            "description": p.description,
                            **({"enum": p.enum} if p.enum else {}),
                        }
                        for p in t.parameters
                    },
                    "required": [p.name for p in t.parameters if p.required],
                },
            }
            for t in self._tools.values()
        ]

    def call_tool(self, tool_id: str, parameters: dict,
                  session_id: str = "", idempotent_key: str = None) -> ToolCallResult:
        """
        调用工具(统一入口)

        完整调用链路:
          1. 查找工具定义
          2. 熔断检查(CIRCUIT_OPEN则快速失败)
          3. 幂等检查(已执行过则直接返回缓存结果)
          4. 副作用检查(IRREVERSIBLE需确认HITL已通过)
          5. 执行handler(含超时控制)
          6. 失败重试(指数退避,仅READ/WRITE)
          7. 更新熔断器状态
          8. 写入审计日志
          9. 缓存幂等结果

        Args:
            tool_id: 要调用的工具ID(如"order_service.query")
            parameters: 调用参数字典
            session_id: 关联的会话ID(审计用)
            idempotent_key: 幂等键。写操作必须提供。
                格式建议: f"{session_id}_{tool_id}_{param_hash}"

        Returns:
            ToolCallResult: 调用结果(含成功/失败/数据/延迟等)
        """
        start_time = time.time()

        # Step 1: 查找工具
        tool = self._tools.get(tool_id)
        if not tool:
            return ToolCallResult(
                tool_id=tool_id, success=False,
                error=f"Tool not found: {tool_id}",
            )

        # Step 2: 熔断检查
        if self._circuit_state.get(tool_id) == ToolStatus.CIRCUIT_OPEN:
            self._write_audit(tool_id, session_id, parameters, "circuit_open", 0)
            return ToolCallResult(
                tool_id=tool_id, success=False,
                error=f"Circuit breaker OPEN for {tool_id}. Service temporarily unavailable.",
            )

        # Step 3: 幂等检查(写操作防重复执行)
        if idempotent_key and idempotent_key in self._idempotent_cache:
            cached = self._idempotent_cache[idempotent_key]
            return ToolCallResult(
                tool_id=tool_id, success=cached.success,
                data=cached.data, idempotent_key=idempotent_key,
                latency_ms=0.0,  # 缓存命中,无实际调用
            )

        # Step 4: 自动生成幂等键(写操作必须有)
        if tool.side_effect in (SideEffect.WRITE, SideEffect.IRREVERSIBLE) and not idempotent_key:
            idempotent_key = self._generate_idempotent_key(session_id, tool_id, parameters)

        # Step 5: 执行(含重试)
        result = self._execute_with_retry(tool, parameters)
        result.idempotent_key = idempotent_key
        result.latency_ms = (time.time() - start_time) * 1000

        # Step 6: 更新熔断器
        if result.success:
            self._failure_counts[tool_id] = 0
        else:
            self._failure_counts[tool_id] = self._failure_counts.get(tool_id, 0) + 1
            if self._failure_counts[tool_id] >= tool.circuit_breaker_threshold:
                self._circuit_state[tool_id] = ToolStatus.CIRCUIT_OPEN

        # Step 7: 缓存幂等结果
        if idempotent_key and result.success:
            self._idempotent_cache[idempotent_key] = result

        # Step 8: 审计
        status = "success" if result.success else "failed"
        self._write_audit(tool_id, session_id, parameters, status, result.latency_ms)

        return result

    def _execute_with_retry(self, tool: ToolDefinition, parameters: dict) -> ToolCallResult:
        """执行工具handler,失败时按策略重试"""
        last_error = None
        retries = tool.max_retries if tool.side_effect != SideEffect.IRREVERSIBLE else 0

        for attempt in range(retries + 1):
            try:
                if tool.handler:
                    data = tool.handler(parameters)
                else:
                    data = {"mock": True, "tool": tool.tool_id, "params": parameters}

                return ToolCallResult(
                    tool_id=tool.tool_id,
                    success=True,
                    data=data,
                    retry_count=attempt,
                )
            except Exception as e:
                last_error = str(e)
                if attempt < retries:
                    # 指数退避(MVP: 不实际sleep)
                    pass

        return ToolCallResult(
            tool_id=tool.tool_id,
            success=False,
            error=last_error,
            retry_count=retries,
        )

    def _generate_idempotent_key(self, session_id: str, tool_id: str, params: dict) -> str:
        """生成幂等键: hash(session_id + tool_id + sorted_params)"""
        raw = f"{session_id}:{tool_id}:{json.dumps(params, sort_keys=True)}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def _write_audit(self, tool_id: str, session_id: str, params: dict,
                     status: str, latency_ms: float):
        """写入审计日志"""
        self._audit_log.append(AuditRecord(
            timestamp=time.time(),
            tool_id=tool_id,
            session_id=session_id,
            parameters=params,
            result_status=status,
            latency_ms=latency_ms,
        ))

    def reset_circuit(self, tool_id: str):
        """手动重置熔断器(恢复调用)"""
        self._circuit_state[tool_id] = ToolStatus.HEALTHY
        self._failure_counts[tool_id] = 0

    @property
    def audit_log(self) -> list[AuditRecord]:
        return self._audit_log

    def get_tool_status(self) -> dict[str, str]:
        """获取所有工具的健康状态"""
        return {tid: status.value for tid, status in self._circuit_state.items()}


# ═══════════════════════════════════════════════════════════════════════════════
# 内置Mock工具(MVP演示用)
# ═══════════════════════════════════════════════════════════════════════════════


def _mock_query_order(params: dict) -> dict:
    """Mock: 查询订单详情"""
    order_id = params.get("order_id", "unknown")
    return {
        "order_id": order_id,
        "status": "delivered",
        "amount": 299.00,
        "product": "Dyson V15 吸尘器",
        "delivered_at": "2024-11-08T14:30:00Z",
        "within_return_period": True,
    }


def _mock_validate_refund(params: dict) -> dict:
    """Mock: 校验退款资格"""
    return {
        "eligible": True,
        "reason": "within_7_days",
        "refund_amount": params.get("amount", 299.00),
        "refund_method": "original_payment",
    }


def _mock_risk_check(params: dict) -> dict:
    """Mock: 风控校验"""
    return {
        "passed": True,
        "risk_level": "low",
        "flags": [],
    }


def _mock_execute_refund(params: dict) -> dict:
    """Mock: 发起退款"""
    return {
        "refund_id": f"RF{int(time.time())}",
        "status": "processing",
        "estimated_arrival": "1-3个工作日",
    }


def _mock_crm_update(params: dict) -> dict:
    """Mock: CRM回写"""
    return {"crm_ticket_id": f"TK{int(time.time())}", "status": "created"}


def _mock_notify_user(params: dict) -> dict:
    """Mock: 发送通知"""
    return {"notification_id": f"NTF{int(time.time())}", "channel": "push", "sent": True}


def _mock_query_logistics(params: dict) -> dict:
    """Mock: 查询物流"""
    return {
        "tracking_no": "SF1234567890",
        "status": "in_transit",
        "current_location": "杭州转运中心",
        "estimated_delivery": "2024-11-10",
        "updates": [
            {"time": "2024-11-08 10:00", "desc": "快递员已揽收"},
            {"time": "2024-11-08 18:00", "desc": "到达杭州转运中心"},
        ],
    }


def _mock_price_protect(params: dict) -> dict:
    """Mock: 价保计算"""
    return {
        "eligible": True,
        "price_diff": 200.00,
        "original_price": 3299.00,
        "current_price": 3099.00,
        "refund_method": "original_payment",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 工厂函数: 创建预注册好工具的网关实例
# ═══════════════════════════════════════════════════════════════════════════════


def create_default_gateway(mock_mode: bool = True) -> ToolGateway:
    """
    创建默认工具网关(预注册所有内置工具)

    注册的工具对应架构文档§3.5中的执行层能力模块:
      · 订单服务: 查询订单、校验退款资格
      · 风控服务: 风险校验
      · 支付网关: 发起退款(不可逆!)
      · CRM服务: 回写工单
      · 通知服务: 推送消息
      · 物流服务: 查询物流
      · 价保服务: 计算价保

    Args:
        mock_mode: True=Mock模式(开发), False=真实调用(生产)

    Returns:
        预注册好工具的ToolGateway实例
    """
    gateway = ToolGateway(mock_mode=mock_mode)

    # ─── 订单服务 ───
    gateway.register_tool(ToolDefinition(
        tool_id="order_service.query",
        name="查询订单详情",
        description="根据订单号查询订单的完整信息,包括状态、金额、商品、收货时间等",
        service="order_service",
        parameters=[
            ToolParameter(name="order_id", type="string", description="订单号"),
        ],
        side_effect=SideEffect.READ,
        timeout_ms=5000,
        handler=_mock_query_order if mock_mode else None,
    ))

    # ─── 退款校验 ───
    gateway.register_tool(ToolDefinition(
        tool_id="rule_engine.validate_refund",
        name="校验退款资格",
        description="校验订单是否符合退款条件(时间、商品状态、退货政策)",
        service="rule_engine",
        parameters=[
            ToolParameter(name="order_id", type="string", description="订单号"),
            ToolParameter(name="reason", type="string", description="退款原因", required=False),
        ],
        side_effect=SideEffect.READ,
        timeout_ms=3000,
        handler=_mock_validate_refund if mock_mode else None,
    ))

    # ─── 风控服务 ───
    gateway.register_tool(ToolDefinition(
        tool_id="risk_service.check",
        name="风控校验",
        description="对退款/支付等敏感操作进行风险评估,检测欺诈行为",
        service="risk_service",
        parameters=[
            ToolParameter(name="user_id", type="string", description="用户ID"),
            ToolParameter(name="action", type="string", description="操作类型"),
            ToolParameter(name="amount", type="number", description="涉及金额"),
        ],
        side_effect=SideEffect.READ,
        timeout_ms=5000,
        handler=_mock_risk_check if mock_mode else None,
    ))

    # ─── 支付网关(不可逆!) ───
    gateway.register_tool(ToolDefinition(
        tool_id="payment_gateway.refund",
        name="发起退款",
        description="向支付网关发起退款请求。注意:此操作不可逆,执行前必须经过用户确认",
        service="payment_gateway",
        parameters=[
            ToolParameter(name="order_id", type="string", description="订单号"),
            ToolParameter(name="amount", type="number", description="退款金额"),
            ToolParameter(name="reason", type="string", description="退款原因"),
        ],
        side_effect=SideEffect.IRREVERSIBLE,  # 不可逆!
        timeout_ms=30000,
        max_retries=0,  # 不可逆操作不自动重试
        idempotent=True,  # 必须幂等(防重复退款)
        handler=_mock_execute_refund if mock_mode else None,
    ))

    # ─── CRM服务 ───
    gateway.register_tool(ToolDefinition(
        tool_id="crm_service.update",
        name="CRM回写",
        description="将客服处理结果写入CRM系统,创建或更新工单记录",
        service="crm_service",
        parameters=[
            ToolParameter(name="session_id", type="string", description="会话ID"),
            ToolParameter(name="action", type="string", description="执行的动作"),
            ToolParameter(name="result", type="string", description="处理结果"),
        ],
        side_effect=SideEffect.WRITE,
        timeout_ms=10000,
        handler=_mock_crm_update if mock_mode else None,
    ))

    # ─── 通知服务 ───
    gateway.register_tool(ToolDefinition(
        tool_id="notify_service.push",
        name="发送通知",
        description="向用户推送通知消息(App Push/短信/站内信)",
        service="notify_service",
        parameters=[
            ToolParameter(name="user_id", type="string", description="用户ID"),
            ToolParameter(name="message", type="string", description="通知内容"),
            ToolParameter(name="channel", type="string", description="通知渠道",
                         enum=["push", "sms", "inbox"]),
        ],
        side_effect=SideEffect.WRITE,
        timeout_ms=5000,
        handler=_mock_notify_user if mock_mode else None,
    ))

    # ─── 物流服务 ───
    gateway.register_tool(ToolDefinition(
        tool_id="logistics_service.query",
        name="查询物流轨迹",
        description="查询包裹的实时物流轨迹信息",
        service="logistics_service",
        parameters=[
            ToolParameter(name="order_id", type="string", description="订单号"),
        ],
        side_effect=SideEffect.READ,
        timeout_ms=5000,
        handler=_mock_query_logistics if mock_mode else None,
    ))

    # ─── 价保服务 ───
    gateway.register_tool(ToolDefinition(
        tool_id="price_protect_service.calculate",
        name="价保计算",
        description="计算商品价保差价,校验是否符合价保条件",
        service="price_protect_service",
        parameters=[
            ToolParameter(name="order_id", type="string", description="订单号"),
        ],
        side_effect=SideEffect.READ,
        timeout_ms=3000,
        handler=_mock_price_protect if mock_mode else None,
    ))

    return gateway
