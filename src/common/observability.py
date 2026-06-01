"""
可观测性 - 全链路Trace + 结构化审计日志

对应架构文档 docs/01-architecture-overview.md §4.5 (§4.5.1-4.5.3)。

提供:
  1. TraceContext: 全链路追踪上下文(OpenTelemetry兼容)
  2. Span: 单个操作的计时和元数据记录
  3. AuditLogger: 结构化审计日志(所有动作可追溯)
  4. MetricsCollector: 核心指标采集(QPS/延迟/错误率/成本)

数据存储:
  · Trace/日志: 写入 data/logs/ 目录(JSON Lines格式)
  · 指标: 内存聚合 + 定期刷写
  · 生产环境: 对接 OpenTelemetry Collector → Jaeger/Grafana
"""
import time
import json
import os
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional
from contextlib import contextmanager
from enum import Enum


# ═══════════════════════════════════════════════════════════════════════════════
# Trace 全链路追踪
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Span:
    """
    Span - 单个操作的追踪记录

    对应架构文档 §4.5.2 中的 Trace 结构。
    每个层级的处理是一个Span,Span可嵌套(parent-child)。

    Attributes:
        span_id: Span唯一ID
        trace_id: 所属Trace ID(同一请求的所有Span共享)
        parent_id: 父Span ID(None=根Span)
        name: 操作名(如 "intent_recognition", "model_call")
        service: 所属服务/层(如 "intent_layer", "reasoning_layer")
        start_time: 开始时间戳
        end_time: 结束时间戳
        duration_ms: 耗时(毫秒)
        status: 状态(ok/error)
        attributes: 附加属性(key-value)
        events: 事件列表(时间点事件)
    """
    span_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    trace_id: str = ""
    parent_id: Optional[str] = None
    name: str = ""
    service: str = ""
    start_time: float = field(default_factory=time.time)
    end_time: float = 0.0
    duration_ms: float = 0.0
    status: str = "ok"
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[dict] = field(default_factory=list)

    def finish(self, status: str = "ok"):
        """结束Span并计算耗时"""
        self.end_time = time.time()
        self.duration_ms = (self.end_time - self.start_time) * 1000
        self.status = status

    def add_event(self, name: str, attributes: dict = None):
        """添加时间点事件"""
        self.events.append({
            "name": name,
            "timestamp": time.time(),
            "attributes": attributes or {},
        })

    def set_attribute(self, key: str, value: Any):
        """设置属性"""
        self.attributes[key] = value

    def to_dict(self) -> dict:
        """序列化为字典(JSON Lines输出用)"""
        return {
            "span_id": self.span_id,
            "trace_id": self.trace_id,
            "parent_id": self.parent_id,
            "name": self.name,
            "service": self.service,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": round(self.duration_ms, 2),
            "status": self.status,
            "attributes": self.attributes,
            "events": self.events,
        }


class Tracer:
    """
    全链路追踪器

    管理一次请求中所有Span的创建、嵌套、完成和输出。
    对应架构文档 §4.5.2 Trace结构。

    使用方式:
        >>> tracer = Tracer()
        >>> with tracer.start_span("process_request", service="gateway") as root:
        ...     root.set_attribute("session_id", "abc")
        ...     with tracer.start_span("intent_recognition", service="intent") as s:
        ...         s.set_attribute("method", "rule")
        ...         s.set_attribute("confidence", 0.95)
        >>> tracer.export()  # 输出所有Span
    """

    def __init__(self, trace_id: str = None):
        self.trace_id = trace_id or str(uuid.uuid4())[:12]
        self.spans: list[Span] = []
        self._current_span: Optional[Span] = None

    @contextmanager
    def start_span(self, name: str, service: str = ""):
        """
        创建并进入一个新Span(上下文管理器)

        自动处理:
          · 设置 trace_id 和 parent_id
          · 退出时自动 finish() 计算耗时
          · 异常时标记 status="error"
        """
        span = Span(
            trace_id=self.trace_id,
            parent_id=self._current_span.span_id if self._current_span else None,
            name=name,
            service=service,
        )

        parent = self._current_span
        self._current_span = span

        try:
            yield span
            span.finish("ok")
        except Exception as e:
            span.finish("error")
            span.set_attribute("error.message", str(e))
            raise
        finally:
            self._current_span = parent
            self.spans.append(span)

    def export(self) -> list[dict]:
        """导出所有Span为字典列表"""
        return [s.to_dict() for s in self.spans]

    @property
    def total_duration_ms(self) -> float:
        """总耗时(根Span的duration)"""
        root_spans = [s for s in self.spans if s.parent_id is None]
        return root_spans[0].duration_ms if root_spans else 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# 结构化审计日志
# ═══════════════════════════════════════════════════════════════════════════════

class AuditAction(Enum):
    """审计动作类型"""
    QUERY = "QUERY"
    CREATE = "CREATE"
    UPDATE = "UPDATE"
    DELETE = "DELETE"
    APPROVE = "APPROVE"
    REJECT = "REJECT"
    ESCALATE = "ESCALATE"
    MODEL_CALL = "MODEL_CALL"
    TOOL_CALL = "TOOL_CALL"


@dataclass
class AuditEntry:
    """
    审计日志条目

    对应架构文档 §4.5.1 审计日志Schema。
    每条记录包含: 谁(actor) + 何时(timestamp) + 对什么(target) + 做了什么(action) + 结果(result)。
    """
    timestamp: float = field(default_factory=time.time)
    trace_id: str = ""
    session_id: str = ""
    actor_type: str = "AI"          # AI / HUMAN_AGENT / SYSTEM / USER
    actor_id: str = ""
    action: str = ""                # AuditAction value
    category: str = ""              # REFUND / TICKET / MODEL_CALL etc.
    target_type: str = ""           # ORDER / TICKET / USER / SESSION
    target_id: str = ""
    description: str = ""
    result_status: str = "SUCCESS"  # SUCCESS / FAILED / TIMEOUT
    duration_ms: float = 0.0
    metadata: dict = field(default_factory=dict)


class AuditLogger:
    """
    结构化审计日志记录器

    写入 data/logs/audit_YYYY-MM-DD.jsonl (每天一个文件)。
    生产环境可对接 Elasticsearch/OLAP 做长期存储和查询。
    """

    def __init__(self, log_dir: str = "data/logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

    def log(self, entry: AuditEntry):
        """写入一条审计日志"""
        filename = time.strftime("audit_%Y-%m-%d.jsonl")
        filepath = os.path.join(self.log_dir, filename)

        record = {
            "timestamp": entry.timestamp,
            "trace_id": entry.trace_id,
            "session_id": entry.session_id,
            "actor": {"type": entry.actor_type, "id": entry.actor_id},
            "action": entry.action,
            "category": entry.category,
            "target": {"type": entry.target_type, "id": entry.target_id},
            "description": entry.description,
            "result_status": entry.result_status,
            "duration_ms": round(entry.duration_ms, 2),
            "metadata": entry.metadata,
        }

        with open(filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def log_simple(self, session_id: str, action: str, description: str,
                   target_id: str = "", trace_id: str = "", **kwargs):
        """简化版日志记录"""
        self.log(AuditEntry(
            trace_id=trace_id,
            session_id=session_id,
            action=action,
            description=description,
            target_id=target_id,
            **kwargs,
        ))


# ═══════════════════════════════════════════════════════════════════════════════
# 指标采集
# ═══════════════════════════════════════════════════════════════════════════════

class MetricsCollector:
    """
    核心指标采集器

    采集架构文档 §4.5.3 中定义的关键指标:
      · QPS (请求/秒)
      · 延迟分布 (P50/P95/P99)
      · 错误率
      · 模型调用成本
      · 缓存命中率
      · 意图分布
    """

    def __init__(self):
        self._request_count: int = 0
        self._error_count: int = 0
        self._latencies: list[float] = []  # 最近1000个请求的延迟
        self._model_costs: float = 0.0
        self._cache_hits: int = 0
        self._cache_misses: int = 0
        self._intent_counts: dict[str, int] = {}
        self._start_time: float = time.time()

    def record_request(self, latency_ms: float, success: bool = True,
                       intent: str = "", model_cost: float = 0.0,
                       cache_hit: bool = False):
        """记录一次请求的指标"""
        self._request_count += 1
        if not success:
            self._error_count += 1

        # 延迟(保留最近1000个)
        self._latencies.append(latency_ms)
        if len(self._latencies) > 1000:
            self._latencies = self._latencies[-1000:]

        # 成本
        self._model_costs += model_cost

        # 缓存
        if cache_hit:
            self._cache_hits += 1
        else:
            self._cache_misses += 1

        # 意图分布
        if intent:
            self._intent_counts[intent] = self._intent_counts.get(intent, 0) + 1

    def get_metrics(self) -> dict:
        """获取当前指标快照"""
        uptime = time.time() - self._start_time
        sorted_latencies = sorted(self._latencies) if self._latencies else [0]

        return {
            "uptime_seconds": round(uptime, 1),
            "total_requests": self._request_count,
            "qps": round(self._request_count / max(uptime, 1), 2),
            "error_rate": round(self._error_count / max(self._request_count, 1) * 100, 2),
            "latency": {
                "p50": round(sorted_latencies[len(sorted_latencies)//2], 2) if sorted_latencies else 0,
                "p95": round(sorted_latencies[int(len(sorted_latencies)*0.95)] if sorted_latencies else 0, 2),
                "p99": round(sorted_latencies[int(len(sorted_latencies)*0.99)] if sorted_latencies else 0, 2),
            },
            "cache_hit_rate": round(
                self._cache_hits / max(self._cache_hits + self._cache_misses, 1) * 100, 1
            ),
            "total_model_cost": round(self._model_costs, 4),
            "intent_distribution": dict(sorted(
                self._intent_counts.items(), key=lambda x: x[1], reverse=True
            )[:10]),
        }
