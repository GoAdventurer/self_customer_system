"""
NexusAI 智能客服系统 - API网关层 (FastAPI)

本模块实现系统的HTTP API接入层，是外部世界与内部处理管线之间的唯一入口。
对应架构文档 docs/01-architecture-overview.md §3.1 接入网关层。

架构定位:
    网关层负责:
        1. 协议适配: HTTP/WebSocket 请求转为内部调用
        2. 请求校验: Pydantic模型自动校验请求参数
        3. 鉴权: (MVP阶段跳过，生产环境加入OAuth2/JWT)
        4. 限流: (MVP阶段跳过，生产环境加入令牌桶)
        5. 路由: 将请求分发到正确的处理管线
        6. 响应序列化: 内部结果转为标准API响应

API清单:
    GET  /health          - 健康检查(负载均衡器探活)
    GET  /status          - 系统状态(运维大盘拉取)
    POST /chat            - 对话接口(核心API)
    PUT  /system/level    - 更新系统级别(运维限流/降级)
    GET  /sessions/{id}   - 查询会话详情(调试/审计)

启动方式:
    开发环境: python scripts/run_server.py
    生产环境: uvicorn src.gateway.api:app --workers 4 --host 0.0.0.0

技术选型:
    FastAPI: 性能(Starlette+异步)、类型安全(Pydantic)、自动文档(OpenAPI)
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import time
import threading

from ..pipeline import CustomerServicePipeline
from ..config.settings import SystemLevel, create_default_config


# ═══════════════════════════════════════════════════════════════
# 限流器 — 令牌桶(对应 §3.1 分级限流 / 大促保护)
# ═══════════════════════════════════════════════════════════════

class TokenBucketRateLimiter:
    """按租户的令牌桶限流器。

    对应架构 §3.1: "分级限流(按租户/渠道/接口)、排队与降级"。
    · 每个租户独立桶，按 refill_rate 持续补充令牌
    · VIP 会话可豁免(高峰优先放行VIP，§3.1)
    · 系统负载等级越高，可用速率越低(降级保护)
    线程安全: 用单锁保护(MVP；生产应用 Redis + Lua 原子计数实现分布式限流)。
    """

    # 不同系统级别的速率折扣(对正常速率的乘数)
    LEVEL_FACTOR = {
        SystemLevel.GREEN: 1.0, SystemLevel.YELLOW: 0.8,
        SystemLevel.ORANGE: 0.5, SystemLevel.RED: 0.25, SystemLevel.BLACK: 0.1,
    }

    def __init__(self, rate_per_sec: float = 50.0, burst: int = 100):
        self.rate = rate_per_sec
        self.burst = burst
        self._buckets: dict[str, tuple[float, float]] = {}  # tenant → (tokens, last_ts)
        self._lock = threading.Lock()

    def allow(self, tenant_id: str, level: SystemLevel, is_vip: bool = False) -> bool:
        """判断是否放行一次请求(VIP 豁免)。"""
        if is_vip:
            return True
        now = time.time()
        factor = self.LEVEL_FACTOR.get(level, 1.0)
        effective_rate = self.rate * factor
        with self._lock:
            tokens, last = self._buckets.get(tenant_id, (float(self.burst), now))
            # 补充令牌
            tokens = min(self.burst, tokens + (now - last) * effective_rate)
            if tokens < 1.0:
                self._buckets[tenant_id] = (tokens, now)
                return False
            self._buckets[tenant_id] = (tokens - 1.0, now)
            return True


# ═══════════════════════════════════════════════════════════════
# API 请求/响应模型 (Pydantic)
#
# 使用 Pydantic BaseModel 定义 API 契约:
#   - 自动参数校验(类型、长度、格式)
#   - 自动生成 OpenAPI Schema
#   - IDE 类型提示支持
# ═══════════════════════════════════════════════════════════════

class ChatRequest(BaseModel):
    """
    对话请求体

    前端/客户端发送对话消息时的请求结构。

    Attributes:
        session_id: 会话ID(可选)。
            - 传入: 继续已有会话(多轮对话)
            - 不传: 系统自动创建新会话并在响应中返回
        message: 用户消息文本。
            长度限制: 1-2000字符(防止超长输入攻击)
        tenant_id: 租户ID。多租户场景下标识所属租户。
            默认"default"(单租户模式)
        channel: 接入渠道标识。用于差异化处理逻辑。
            可选值: "app" / "web" / "wechat" / "phone"
        user_id: 用户ID(可选)。传入后系统加载用户画像。
        user_tier: 用户等级。影响限流、Token预算、升级策略。
            可选值: "normal" / "vip" / "svip"
    """
    session_id: Optional[str] = None
    message: str = Field(..., min_length=1, max_length=2000)
    tenant_id: str = "default"
    channel: str = "web"
    user_id: Optional[str] = None
    user_tier: str = "normal"


class ChatResponse(BaseModel):
    """
    对话响应体

    系统处理完毕后返回给前端的完整响应。
    前端据此渲染AI回复气泡、展示意图标签、决定是否转人工。

    Attributes:
        session_id: 会话ID(前端需保存，后续请求携带以保持上下文)
        response: AI回复文本(直接展示给用户)
        intent: 识别的意图标签(前端可展示为标签/用于埋点)
        confidence: 意图置信度(0-1, 前端可据此展示"不确定"提示)
        model_used: 使用的模型ID(调试信息，生产环境可隐藏)
        latency_ms: 处理耗时(毫秒, 用于性能监控)
        from_cache: 是否命中缓存(用于成本分析)
        suggest_transfer_human: 是否建议转人工。
            True时前端应展示"转人工"按钮或自动触发转接流程
    """
    session_id: str
    response: str
    intent: Optional[str] = None
    confidence: float = 0.0
    model_used: str = ""
    latency_ms: float = 0.0
    from_cache: bool = False
    suggest_transfer_human: bool = False


class SystemStatus(BaseModel):
    """
    系统状态响应

    运营大盘拉取系统健康状态时的响应结构。
    对应前端 prototype/index.html 的"技术运行大盘"。

    Attributes:
        level: 当前系统负载级别(GREEN/YELLOW/ORANGE/RED/BLACK)
        active_sessions: 活跃会话数(反映当前负载)
        cache_size: 语义缓存条目数(反映缓存效率)
        uptime_seconds: 系统运行时长(秒)
    """
    level: str
    active_sessions: int
    cache_size: int
    uptime_seconds: float


class SystemLevelUpdate(BaseModel):
    """
    系统级别更新请求

    运维通过此接口手动切换系统负载级别(限流/降级)。
    通常由自动化监控触发，或运维人员在大促期间手动操作。

    Attributes:
        level: 目标级别。
            必须是合法枚举值: GREEN / YELLOW / ORANGE / RED / BLACK
            使用正则校验确保输入安全
    """
    level: str = Field(..., pattern="^(GREEN|YELLOW|ORANGE|RED|BLACK)$")


# ═══════════════════════════════════════════════════════════════
# FastAPI 应用实例
# ═══════════════════════════════════════════════════════════════

app = FastAPI(
    title="NexusAI 智能客服系统",
    version="0.1.0",
    description="企业级智能客服系统 API — 六层架构 MVP 实现",
)

# 全局管线实例(整个进程共享)
# 生产环境: 使用依赖注入(Depends)替代全局变量
pipeline = CustomerServicePipeline(config=create_default_config())

# 全局限流器(令牌桶)
rate_limiter = TokenBucketRateLimiter(
    rate_per_sec=pipeline.config.rate_limit.per_tenant_qps
    if hasattr(pipeline.config, "rate_limit") else 50.0,
    burst=100,
)

# 服务启动时间(用于计算uptime)
_start_time = time.time()


# ═══════════════════════════════════════════════════════════════
# API 路由定义
# ═══════════════════════════════════════════════════════════════

@app.get("/health")
def health_check():
    """
    健康检查接口

    供负载均衡器(如K8s liveness probe)定期探活。
    返回200表示服务正常，返回非200触发重启。

    响应示例:
        {"status": "ok", "level": "L0"}
    """
    return {"status": "ok", "level": pipeline.config.system_level.value}


@app.get("/status", response_model=SystemStatus)
def system_status():
    """
    系统状态查询

    运营大盘每5秒拉取一次，展示在"技术运行大盘"上。
    包含: 系统级别、活跃会话数、缓存大小、运行时长。

    响应示例:
        {"level": "GREEN", "active_sessions": 42, "cache_size": 1205, "uptime_seconds": 3600.5}
    """
    return SystemStatus(
        level=pipeline.config.system_level.name,
        active_sessions=len(pipeline._sessions),
        cache_size=pipeline.model_router.cache.size,
        uptime_seconds=time.time() - _start_time,
    )


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    对话接口 — 系统核心API

    处理用户消息并返回AI回复。
    单次调用完成: 输入预处理 → 意图识别 → 流程编排 → 模型调用 → 返回回复。

    调用流程:
        1. 获取或创建会话(基于session_id)
        2. 调用 Pipeline.process() 执行全链路处理
        3. 将内部PipelineResult映射为API响应

    错误处理:
        - 参数校验失败: 422 (Pydantic自动处理)
        - 处理异常: 500 + 错误详情

    性能指标(P95):
        - 缓存命中: <5ms
        - 规则+模板: <10ms
        - 小模型: <500ms
        - 大模型: <3000ms (需配合流式返回优化)

    请求示例:
        POST /chat
        {"session_id": "abc123", "message": "我要退款", "channel": "app"}

    响应示例:
        {"session_id": "abc123", "response": "好的,请提供订单号...",
         "intent": "REFUND", "confidence": 0.95, "latency_ms": 2.3}
    """
    try:
        # ── 分级限流(VIP 豁免，按系统负载等级动态收紧) ──
        is_vip = request.user_tier in ("vip", "svip")
        if not rate_limiter.allow(request.tenant_id, pipeline.config.system_level, is_vip):
            raise HTTPException(
                status_code=429,
                detail="当前咨询量较大，请稍后重试(系统限流保护)",
            )

        # 获取或创建会话
        session_id = request.session_id or None
        if not session_id:
            # 新会话: 创建上下文，注入租户/渠道/用户信息(用户信息用于加载长期画像)
            ctx = pipeline.get_or_create_session(
                tenant_id=request.tenant_id,
                channel=request.channel,
                user_id=request.user_id,
                user_tier=request.user_tier,
            )
            session_id = ctx.session_id

        # 调用全链路管线处理
        result = pipeline.process(session_id, request.message)

        # 映射为API响应
        return ChatResponse(
            session_id=result.session_id,
            response=result.response_text,
            intent=result.intent.intent if result.intent else None,
            confidence=result.confidence,
            model_used=result.model_used,
            latency_ms=round(result.latency_ms, 2),
            from_cache=result.from_cache,
            suggest_transfer_human=result.suggest_transfer_human,
        )

    except HTTPException:
        # 限流(429)等已明确的 HTTP 错误原样抛出，不要被下方 500 吞掉
        raise
    except Exception as e:
        # 生产环境: 应记录详细错误日志 + 告警
        # 此处返回500但不暴露内部栈信息
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")


@app.put("/system/level")
def update_system_level(req: SystemLevelUpdate):
    """
    更新系统负载级别(运维接口)

    用于大促期间的限流/降级切换。
    调用后立即生效，影响所有后续请求。

    使用场景:
        - 大促开始前: PUT /system/level {"level": "YELLOW"} (预热)
        - 流量突增时: PUT /system/level {"level": "RED"} (降级)
        - 恢复正常后: PUT /system/level {"level": "GREEN"} (恢复)

    安全说明:
        生产环境此接口需要鉴权(仅运维角色可调用)。
        建议加入操作审计日志记录。

    请求示例:
        PUT /system/level
        {"level": "RED"}

    响应示例:
        {"status": "ok", "new_level": "RED"}
    """
    level = SystemLevel[req.level]
    pipeline.update_system_level(level)
    return {"status": "ok", "new_level": level.name}


@app.get("/metrics")
def metrics():
    """运营指标接口 — 供监控大盘/告警拉取(对应 §4.5.3 监控大盘)。

    返回: 请求量/延迟分位/缓存命中率/意图分布 + 工具网关健康 + 长期记忆统计。
    """
    m = pipeline.metrics.get_metrics()
    return {
        "pipeline": m,
        "system_level": pipeline.config.system_level.name,
        "tool_gateway": pipeline.tool_gateway.get_tool_status(),
        "memory": pipeline.memory.stats(),
        "active_sessions": len(pipeline._sessions),
    }


class ResumeRequest(BaseModel):
    """HITL 恢复请求(用户对挂起流程的二次确认结果)。"""
    confirmed: bool = True


@app.post("/sessions/{session_id}/resume")
def resume_session(session_id: str, req: ResumeRequest):
    """恢复因 HITL(人工确认)挂起的流程 — 对应 §3.2 HITL / §8 断点恢复。

    退款等不可逆操作在用户二次确认前会挂起；前端确认后调用本接口续跑 DAG。
    """
    status = pipeline.resume_dag(session_id, confirmed=req.confirmed)
    if status is None:
        raise HTTPException(status_code=404, detail="无可恢复的会话或流程")
    return {"session_id": session_id, "dag_status": status}


@app.get("/sessions/{session_id}")
def get_session(session_id: str):
    """
    查询会话详情(调试/审计接口)

    返回指定会话的当前状态信息，用于:
        - 开发调试: 查看会话上下文是否正确
        - 坐席工作台: 查看AI处理过的会话摘要
        - 审计追溯: 回溯某次对话的完整处理过程

    Args:
        session_id: 要查询的会话ID(URL路径参数)

    Returns:
        会话摘要信息(不返回完整消息历史,避免数据泄露)

    Raises:
        404: 会话不存在(已过期或ID无效)
    """
    if session_id not in pipeline._sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    ctx = pipeline._sessions[session_id]
    return {
        "session_id": ctx.session_id,
        "status": ctx.status.value,
        "turn_count": ctx.turn_count,
        "intent": ctx.intent.intent if ctx.intent else None,
        "emotion": ctx.emotion.value,
        "emotion_score": ctx.emotion_score,
        "message_count": len(ctx.messages),
    }
