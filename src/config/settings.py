"""
NexusAI 智能客服系统 - 核心配置模块

本模块定义了系统运行所需的全部配置项，采用 dataclass 实现类型安全的配置管理。
配置按职责划分为多个子配置类，统一由 SystemConfig 聚合。

架构定位:
    本模块是全局配置中心，被六层架构中的每一层引用。
    配置支持运行时动态切换（如系统负载级别），便于大促场景的限流/降级操作。

设计原则:
    1. 类型安全: 使用 dataclass + Enum，IDE 可自动补全，避免字符串硬编码
    2. 默认合理: create_default_config() 提供开发环境开箱即用的默认值
    3. 分层隔离: 每个子模块的配置独立，修改互不影响
    4. 可扩展: 新增配置项只需添加字段 + 默认值，不破坏现有代码

使用示例:
    >>> from src.config.settings import create_default_config, SystemLevel
    >>> config = create_default_config()
    >>> config.system_level = SystemLevel.RED  # 动态切换到降级模式
    >>> model = config.get_model(ModelTier.SMALL)  # 获取7B小模型配置
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class SystemLevel(Enum):
    """
    系统负载级别枚举

    对应架构文档中的分级限流/降级策略（docs/02-scenario-deep-dive.md §4.2）。
    各级别对应的行为差异:
        - GREEN (L0): 全功能正常运行，无限制
        - YELLOW (L1): 预警状态，触发预扩容和缓存预热
        - ORANGE (L2): 限流状态，关闭LLM兜底，分类器降精度
        - RED (L3): 降级状态，零LLM调用，全部走模板回复
        - BLACK (L4): 熔断保护，仅放行VIP，静态页兜底

    切换时机:
        - 自动切换: 基于 QPS/延迟/错误率 阈值自动触发
        - 手动切换: 运维通过 PUT /system/level 接口强制切换
    """
    GREEN = "L0"    # 正常 (QPS ≤ 800)
    YELLOW = "L1"   # 预警 (QPS 800-3000)
    ORANGE = "L2"   # 限流 (QPS 3000-8000)
    RED = "L3"      # 降级 (QPS 8000-15000)
    BLACK = "L4"    # 熔断 (QPS > 15000)


class ModelTier(Enum):
    """
    模型分级枚举

    按能力/成本/延迟将模型分为5个等级，供推理层路由引擎选择。
    对应架构文档 docs/01-architecture-overview.md §3.6.1 多模型路由选型矩阵。

    分级逻辑:
        - MICRO:  分类器/BERT级别，本地CPU推理，延迟<20ms，成本≈0
        - SMALL:  7B参数级别，本地GPU推理，延迟50-150ms
        - MEDIUM: 72B参数级别，私有化部署，延迟300-600ms
        - LARGE:  云端大模型(如Claude Sonnet)，延迟400-800ms
        - XLARGE: 超大模型(如Claude Opus)，仅用于极复杂推理
    """
    MICRO = "micro"     # BERT/分类器, 本地CPU, ≈0成本
    SMALL = "small"     # 7B参数, 本地GPU, 低成本
    MEDIUM = "medium"   # 72B参数, 私有化部署, 中等成本
    LARGE = "large"     # 云端大模型, 按token计费
    XLARGE = "xlarge"   # 超大模型, 最高成本(仅复杂推理)


@dataclass
class ModelConfig:
    """
    单个模型的完整配置

    每个模型实例包含连接信息、性能参数和成本信息。
    系统启动时加载所有模型配置到模型池，推理层按需选取。

    Attributes:
        model_id: 模型唯一标识符，如 "qwen2.5-7b"、"claude-sonnet"
        tier: 模型所属等级(MICRO/SMALL/MEDIUM/LARGE/XLARGE)
        endpoint: 模型API端点URL。本地模型格式 "http://localhost:port/v1/chat/completions"
        api_key: API密钥(云端模型需要，本地模型为空)
        max_tokens: 单次请求最大生成token数
        temperature: 采样温度(0-1)。客服场景建议0.3，保证回复稳定性
        timeout_ms: 请求超时时间(毫秒)。超时后触发Fallback降级
        cost_per_1k_input: 每1000个输入token的成本(元)
        cost_per_1k_output: 每1000个输出token的成本(元)
        supports_function_calling: 是否支持函数调用/工具使用
        is_local: 是否为本地部署(影响隐私路由决策)
    """
    model_id: str
    tier: ModelTier
    endpoint: str
    api_key: str = ""
    max_tokens: int = 4096
    temperature: float = 0.3
    timeout_ms: int = 30000
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0
    supports_function_calling: bool = False
    is_local: bool = False


@dataclass
class RedisConfig:
    """
    Redis 连接配置

    Redis 在系统中承担两个角色:
        1. 会话短期状态存储(上下文、槽位、当前DAG节点)
        2. 语义缓存的L1精确缓存层

    Attributes:
        host: Redis服务器地址
        port: Redis端口
        db: 数据库编号(建议会话用db=0，缓存用db=1)
        password: 认证密码(生产环境必须设置)
        session_ttl_seconds: 会话过期时间。默认30分钟，超时后会话标记为ABANDONED
    """
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: str = ""
    session_ttl_seconds: int = 1800  # 30min，对应架构文档中"超过30分钟标记ABANDONED"


@dataclass
class RateLimitConfig:
    """
    限流配置

    实现架构文档中的分级限流策略(docs/02-scenario-deep-dive.md §4.4)。
    限流采用令牌桶算法，按全局/租户/用户三级维度控制。

    Attributes:
        global_qps: 系统全局QPS上限。超出后触发排队机制
        per_tenant_qps: 单租户QPS上限(SLA合同约定)
        per_user_rpm: 单用户每分钟请求上限(防刷)
        queue_max_depth: 排队队列最大深度。超出直接拒绝+引导自助
        queue_ttl_seconds: 排队请求的存活时间。超时释放+推送通知
    """
    global_qps: int = 3000
    per_tenant_qps: int = 500
    per_user_rpm: int = 60
    queue_max_depth: int = 10000
    queue_ttl_seconds: int = 120


@dataclass
class IntentConfig:
    """
    意图识别层配置

    控制三级识别引擎(规则/分类器/LLM)的路由行为。
    对应架构文档 docs/01-architecture-overview.md §3.3。

    Attributes:
        rule_confidence_threshold: 规则匹配置信度阈值。
            ≥此值直接返回，跳过分类器和LLM。默认0.9(高确定性)
        classifier_confidence_threshold: 分类器置信度阈值。
            ≥此值返回分类器结果。默认0.8
        llm_fallback_enabled: 是否启用LLM兜底。
            降级模式(L2+)时应设为False以节省成本
        max_slot_retries: 槽位追问最大次数。
            超过此次数仍未获取完整槽位，建议转人工
    """
    rule_confidence_threshold: float = 0.9
    classifier_confidence_threshold: float = 0.8
    llm_fallback_enabled: bool = True
    max_slot_retries: int = 3


@dataclass
class PlanningConfig:
    """
    规划层(DAG编排引擎)配置

    控制任务图的执行行为，包括超时、重试、检查点策略等。
    对应架构文档 docs/03-orchestration-design.md。

    Attributes:
        max_dag_steps: DAG最大执行步数(防失控的安全阀)。
            超过此步数强制终止，避免无限循环
        default_node_timeout_ms: 单节点默认超时时间(毫秒)。
            每个节点可覆盖此值
        max_retries: 节点执行失败时的最大重试次数。
            采用指数退避策略(1s, 2s, 4s...)
        checkpoint_strategy: 检查点策略。
            "every_node": 每个节点完成后持久化(最安全)
            "on_io": 仅在IO节点后持久化(性能优)
            "manual": 手动触发(最高性能)
        hitl_timeout_seconds: HITL(Human-in-the-Loop)等待超时。
            用户确认节点的最大等待时间，超时标记为ABANDONED
    """
    max_dag_steps: int = 20
    default_node_timeout_ms: int = 30000
    max_retries: int = 3
    checkpoint_strategy: str = "every_node"
    hitl_timeout_seconds: int = 300  # 5分钟用户确认超时


@dataclass
class ReasoningConfig:
    """
    推理层配置

    控制模型路由、语义缓存、Token预算等核心行为。
    对应架构文档 docs/01-architecture-overview.md §3.6.1。

    Attributes:
        semantic_cache_enabled: 是否启用语义缓存。
            开启后高频FAQ可实现零LLM调用(成本节省40-60%)
        cache_similarity_threshold: 语义缓存L2层的相似度阈值。
            >0.95直接返回，0.90-0.95返回但标注"可能不精确"
        session_token_budget_normal: 普通用户单会话Token预算。
            超预算后降级到小模型或模板回复
        session_token_budget_vip: VIP用户单会话Token预算。
            VIP享受更高预算，保障服务质量
        fallback_to_template: 模型不可用时是否降级到模板回复。
            设为True保证系统始终有响应(可用性优先)
    """
    semantic_cache_enabled: bool = True
    cache_similarity_threshold: float = 0.95
    session_token_budget_normal: int = 4000
    session_token_budget_vip: int = 8000
    fallback_to_template: bool = True


@dataclass
class SystemConfig:
    """
    系统全局配置(顶层聚合)

    聚合所有子模块配置，作为系统唯一的配置入口点。
    各层引擎初始化时接收此配置，按需提取所需子配置。

    Attributes:
        system_level: 当前系统负载级别(可运行时动态修改)
        redis: Redis连接配置
        rate_limit: 限流配置
        intent: 意图识别配置
        planning: 规划层配置
        reasoning: 推理层配置
        models: 已注册的模型配置列表(模型池)
        debug: 调试模式开关(开启后输出详细日志)
    """
    system_level: SystemLevel = SystemLevel.GREEN
    redis: RedisConfig = field(default_factory=RedisConfig)
    rate_limit: RateLimitConfig = field(default_factory=RateLimitConfig)
    intent: IntentConfig = field(default_factory=IntentConfig)
    planning: PlanningConfig = field(default_factory=PlanningConfig)
    reasoning: ReasoningConfig = field(default_factory=ReasoningConfig)
    models: list[ModelConfig] = field(default_factory=list)
    debug: bool = False

    def get_model(self, tier: ModelTier) -> Optional[ModelConfig]:
        """
        获取指定等级的模型配置

        从模型池中查找第一个匹配指定等级的模型。
        如果该等级无可用模型，返回None（调用方需处理降级逻辑）。

        Args:
            tier: 目标模型等级(ModelTier枚举值)

        Returns:
            匹配的ModelConfig实例，或None(无匹配)

        Example:
            >>> config = create_default_config()
            >>> small_model = config.get_model(ModelTier.SMALL)
            >>> print(small_model.model_id)  # "qwen2.5-7b"
        """
        for m in self.models:
            if m.tier == tier:
                return m
        return None


# ═══════════════════════════════════════════════════════════════
# 配置工厂函数
# ═══════════════════════════════════════════════════════════════

def create_default_config() -> SystemConfig:
    """
    创建默认系统配置(对接真实环境)

    模型池配置(对接当前环境的实际服务):
        - intent-classifier (MICRO): 本地规则/分类器，无需LLM，延迟<5ms
        - qwen2.5:7b (SMALL): Ollama本地千问7B，隐私安全，延迟<3s
        - claude-sonnet-4.6 (MEDIUM): CodeBuddy代理，标准对话，延迟<10s
        - claude-opus-4.7 (LARGE): CodeBuddy代理，复杂推理，延迟<30s

    大模型调用链路:
        本系统 → CodeBuddy Proxy(localhost:8765) → 实际模型API
        本系统 → Ollama(localhost:11434) → 本地模型推理

    Embedding:
        Ollama qwen3-embedding:8b (首选) / text-embedding-v4 (备选)

    Returns:
        配置好的SystemConfig实例
    """
    return SystemConfig(
        models=[
            # MICRO: 本地规则引擎(无需模型调用)
            ModelConfig(
                model_id="intent-classifier",
                tier=ModelTier.MICRO,
                endpoint="local://classifier",
                is_local=True,
                timeout_ms=100,
            ),
            # SMALL: Ollama本地千问7B(隐私安全/低延迟/零外部依赖)
            ModelConfig(
                model_id="qwen2.5:7b",
                tier=ModelTier.SMALL,
                endpoint="http://localhost:11434/v1/chat/completions",
                is_local=True,
                max_tokens=2048,
                timeout_ms=10000,
                cost_per_1k_input=0.0,    # 本地模型零调用费用
                cost_per_1k_output=0.0,
            ),
            # MEDIUM: CodeBuddy代理→claude-sonnet(标准对话/工具调用)
            ModelConfig(
                model_id="claude-sonnet-4.6",
                tier=ModelTier.MEDIUM,
                endpoint="http://localhost:8765/v1/chat/completions",
                max_tokens=4096,
                timeout_ms=30000,
                cost_per_1k_input=0.012,
                cost_per_1k_output=0.048,
                supports_function_calling=True,
                is_local=False,
            ),
            # LARGE: CodeBuddy代理→claude-opus(复杂推理/多意图分析)
            ModelConfig(
                model_id="claude-opus-4.7",
                tier=ModelTier.LARGE,
                endpoint="http://localhost:8765/v1/chat/completions",
                max_tokens=4096,
                timeout_ms=45000,
                cost_per_1k_input=0.060,
                cost_per_1k_output=0.240,
                supports_function_calling=True,
                is_local=False,
            ),
        ]
    )
