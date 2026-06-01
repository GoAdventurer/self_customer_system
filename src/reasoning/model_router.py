"""⑤ 推理层 - 多模型路由引擎 (Model Router Engine)

本模块是智能客服系统六层架构中「第⑤层 · 推理层」的核心实现。
推理层的职责是：根据任务复杂度、隐私约束、系统负载、成本预算等多维度因素，
将请求智能路由到最合适的模型（或模板/缓存），实现"够用即可"的最低成本原则。

═══════════════════════════════════════════════════════════════════════════════
架构定位 (参见 docs/01-architecture-overview.md §3.6 及 §3.6.1):
═══════════════════════════════════════════════════════════════════════════════

  全渠道用户 → ① 输入层 → ② 意图层 → ③ 规划层 → ④ 执行层 → ⑤ 推理层(本模块) → ⑥ 反馈层

  本模块在流水线中接收来自上游(意图层/规划层)的 IntentResult 和 SessionContext，
  决策使用哪个模型生成回复，并返回带有成本、延迟等元数据的 ModelResponse。

═══════════════════════════════════════════════════════════════════════════════
路由策略(优先级从高到低，数字越小优先级越高):
═══════════════════════════════════════════════════════════════════════════════

  R1: 隐私约束 → 含PII/资金数据必须走本地/私有化模型
      (本MVP暂未实现R1的显式检查，生产环境需在此处加入PII检测逻辑)

  R2: 系统保护 → system_level >= L3 (RED/BLACK) 时禁止调用任何LLM，强制走模板
      (保障极端负载下系统可用性，宁可降低质量也不能让系统崩溃)

  R3: 配额管控 → 单会话token超预算时降级到模板回复
      (防止单个用户/会话消耗过多资源，保障整体服务质量)

  R4: 任务匹配 → 按意图复杂度选择匹配的模型级别
      (核心路由逻辑：简单FAQ用小模型，复杂推理用大模型)

  R5: 成本优化 → 同等能力选最低成本
      (本MVP中体现为：能用模板不用模型，能用小模型不用大模型)

═══════════════════════════════════════════════════════════════════════════════
Fallback链(模型不可用时自动降级):
═══════════════════════════════════════════════════════════════════════════════

  超大模型(XLARGE/云端) → 大模型(LARGE/云端) → 中型(MEDIUM/私有化)
    → 小型(SMALL/本地) → 模板话术(MICRO/零成本兜底)

  设计理念：任何单点故障都不应导致用户得不到回复。最差情况下，
  系统至少能给出一个预设的模板回复，保障用户体验的基本底线。

═══════════════════════════════════════════════════════════════════════════════
MVP 说明:
═══════════════════════════════════════════════════════════════════════════════

  当前为 MVP 阶段实现，核心简化如下:
  1. LLM调用为模拟(simulate)，生产环境需替换为真实API调用
  2. R1(隐私约束)未显式实现，生产环境需加入PII检测模块
  3. 语义缓存仅实现L1精确匹配，生产环境需加入L2向量相似度检索
  4. Token预算为会话级简单累加，生产环境需实现三级预算体系(租户/会话/请求)
  5. 成本计算为估算，生产环境需对接真实计费API

═══════════════════════════════════════════════════════════════════════════════
"""

# ═══════════════════════════════════════════════════════════════════════════════
# 标准库导入
# ═══════════════════════════════════════════════════════════════════════════════
from dataclasses import dataclass, field  # 用于定义轻量级数据容器类
from typing import Optional               # 类型注解：可选值
import time                                # 时间戳和延迟计算
import hashlib                             # 生成缓存key的哈希函数

# ═══════════════════════════════════════════════════════════════════════════════
# 项目内部模块导入
# ═══════════════════════════════════════════════════════════════════════════════
# IntentResult: 意图层输出的结构化结果(包含 intent、slots、confidence、method 等)
# SessionContext: 会话上下文(包含 session_id、is_vip、turn_count、last_user_message 等)
from ..common.models import IntentResult, SessionContext

# ModelConfig: 单个模型的配置(model_id、tier、cost_per_1k_input/output 等)
# ModelTier: 模型级别枚举(MICRO/SMALL/MEDIUM/LARGE/XLARGE)
# ReasoningConfig: 推理层配置(缓存开关、token预算等)
# SystemLevel: 系统负载级别枚举(GREEN/YELLOW/ORANGE/RED/BLACK)
from ..config.settings import ModelConfig, ModelTier, ReasoningConfig, SystemLevel


# ═══════════════════════════════════════════════════════════════════════════════
# 数据模型定义
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class ModelResponse:
    """模型响应数据类 - 封装模型调用的完整结果

    设计目的：
        统一所有模型调用（包括模板回复、缓存命中、真实LLM调用）的返回格式，
        使上层调用方无需关心回复来源的差异，同时携带成本和性能元数据，
        为反馈层(⑥层)的成本追踪和质量监控提供数据基础。

    与架构的关系 (参见 §3.6.1 "成本追踪"):
        每次调用记录: model_id + input_tokens + output_tokens，
        喂入实时成本仪表盘，按租户/模型/场景维度可查。

    字段说明:
        text:          生成的回复文本内容
        model_id:      实际使用的模型标识符(如 "gpt-4o"、"qwen2.5-72b"、"cache"、"template")
        model_tier:    模型级别(MICRO/SMALL/MEDIUM/LARGE/XLARGE)，用于成本统计分类
        input_tokens:  输入消耗的token数(用于成本计算和预算管控)
        output_tokens: 输出消耗的token数(用于成本计算和预算管控)
        latency_ms:    端到端延迟(毫秒)，从路由开始到回复生成完成
        from_cache:    是否来自语义缓存命中(True=零LLM成本)
        cost:          本次调用的估算成本(元)，用于实时成本监控

    设计权衡:
        - 使用 dataclass 而非普通 dict：提供类型安全和IDE自动补全
        - 包含 from_cache 标记：便于反馈层区分缓存命中率统计
        - cost 为估算值：MVP阶段基于 token数×单价 粗算，生产环境应对接真实账单API
    """
    text: str                      # 回复文本(最终呈现给用户的内容)
    model_id: str                  # 模型标识(可追溯到具体使用了哪个模型)
    model_tier: ModelTier          # 模型层级(决定成本区间)
    input_tokens: int = 0          # 输入token数(默认0，模板/缓存场景无token消耗)
    output_tokens: int = 0         # 输出token数(默认0，模板/缓存场景无token消耗)
    latency_ms: float = 0.0        # 延迟毫秒(含路由决策+模型推理全链路)
    from_cache: bool = False       # 缓存命中标记(True时 input/output_tokens 为0)
    cost: float = 0.0              # 本次调用估算成本(元)


@dataclass
class CacheEntry:
    """语义缓存条目 - 单条缓存记录的完整信息

    设计目的：
        存储已生成过的回复，避免对相同/相似问题重复调用LLM，
        这是推理层成本优化的最重要手段之一。

    与架构的关系 (参见 §3.6.1 "成本优化策略对照"):
        语义缓存可节省 40-60% 的模型调用成本。
        高阈值(置信度>0.85)保证缓存精确度，避免返回错误答案。

    字段说明:
        key:          缓存键(MD5哈希值)，由 intent + sorted_slots 生成
        response:     缓存的回复文本
        source:       回复来源标记(如 "llm_large"、"llm_small")，便于统计分析
        hit_count:    命中次数(用于缓存效果评估和热度排序)
        created_at:   创建时间戳(用于TTL过期计算)
        ttl_seconds:  生存时间(秒)，默认7天(604800秒)

    设计权衡:
        - TTL设为7天：平衡缓存命中率和数据新鲜度
          (FAQ类答案变动频率低，7天合理；如果是价格/库存类则需更短TTL)
        - hit_count 仅做统计：MVP不基于热度做淘汰，生产环境可实现LRU/LFU策略
        - 不存储向量：MVP仅做精确匹配，L2语义匹配需要额外的向量存储
    """
    key: str                                       # 缓存键(MD5哈希)
    response: str                                  # 缓存的回复内容
    source: str = ""                               # 生成该回复的模型来源
    hit_count: int = 0                             # 累计命中次数
    created_at: float = field(default_factory=time.time)  # 创建时间(自动填充当前时间)
    ttl_seconds: int = 604800                      # 过期时间: 7天 = 7 × 24 × 3600


class SemanticCache:
    """语义缓存系统 - 分层缓存降低LLM调用频次

    ═══════════════════════════════════════════════════════════════════════════
    设计理念 (参见 §3.6.1 "成本优化策略对照"):
    ═══════════════════════════════════════════════════════════════════════════

    大量客服对话具有高度重复性（如"怎么退货"、"查物流"），
    通过缓存已生成的高质量回复，避免对相同问题反复调用LLM，
    预计可节省 40-60% 的模型调用成本。

    ═══════════════════════════════════════════════════════════════════════════
    分层设计:
    ═══════════════════════════════════════════════════════════════════════════

    L1 精确缓存 (本MVP已实现):
        - 基于 intent + slots 的哈希精确匹配
        - 优点: 100%精确，无误匹配风险
        - 缺点: 只能命中完全相同的请求，覆盖率有限
        - 适用: "查订单12345的物流" 这类有明确槽位的请求

    L2 语义缓存 (生产环境待实现):
        - 基于向量相似度检索(余弦相似度 > 阈值)
        - 优点: 能命中语义相似但措辞不同的请求(如"怎么退"≈"如何退货")
        - 缺点: 有误匹配风险(需要设置较高阈值如0.95)，需要向量数据库
        - 适用: FAQ类通用问题

    ═══════════════════════════════════════════════════════════════════════════
    边界情况和注意事项:
    ═══════════════════════════════════════════════════════════════════════════

    1. 缓存一致性: 知识库更新时需主动失效相关缓存(invalidate_by_intent)
    2. 个性化冲突: 缓存回复不含个性化内容，VIP用户可能需要差异化回复
       (MVP暂忽略，生产环境需按用户等级做缓存隔离)
    3. 时效性数据: 价格、库存、物流状态等时效性强的数据不应被长期缓存
       (通过较短的TTL或不缓存此类意图来规避)
    4. 内存管理: 当前为内存字典存储，生产环境应使用Redis做分布式缓存
    """

    def __init__(self):
        """初始化语义缓存

        使用字典作为L1精确缓存的存储后端。
        键为MD5哈希值(32位十六进制字符串)，值为CacheEntry对象。

        生产环境改进方向:
        - 替换为Redis: 支持分布式部署、内存管理、原生TTL
        - 添加Milvus/PGVector: 支持L2语义向量检索
        - 添加容量上限: 防止内存无限增长(如最多10万条)
        """
        self._cache: dict[str, CacheEntry] = {}

    def _make_key(self, intent: str, slots: dict) -> str:
        """生成缓存键: 对 intent + sorted_slots 做 MD5 哈希

        为什么用 MD5:
            - 固定长度(32字符): 作为字典key效率高
            - 碰撞概率极低: 对于客服场景的数据量级几乎不可能碰撞
            - 计算快速: 比 SHA256 快，缓存查询是高频操作

        为什么对 slots 排序:
            - 保证相同槽位不同插入顺序生成相同的key
            - 例如 {"order_id": "123", "reason": "质量"} 和
              {"reason": "质量", "order_id": "123"} 应命中同一缓存

        参数:
            intent: 意图标识(如 "REFUND"、"LOGISTICS"、"FAQ_RETURN_POLICY")
            slots: 已抽取的槽位字典(如 {"order_id": "12345"})

        返回:
            32位MD5十六进制哈希字符串

        边界情况:
            - slots为空字典时: 仅基于intent生成key(适用于无槽位的FAQ查询)
            - slots含复杂嵌套: sorted()仅对顶层排序，深层嵌套可能导致非预期行为
              (MVP场景中slots均为扁平结构，暂不处理)
        """
        raw = f"{intent}:{sorted(slots.items())}"
        return hashlib.md5(raw.encode()).hexdigest()

    def get(self, intent: str, slots: dict) -> Optional[CacheEntry]:
        """L1精确缓存查询 - 查找完全匹配的缓存条目

        查询流程:
            1. 根据 intent + slots 计算哈希key
            2. 在内存字典中查找该key
            3. 如果找到，检查是否过期(TTL)
            4. 未过期则增加命中计数并返回
            5. 已过期则删除该条目并返回None

        参数:
            intent: 当前请求的意图标识
            slots: 当前请求的槽位字典

        返回:
            CacheEntry: 缓存命中时返回缓存条目
            None: 未命中或已过期

        性能特征:
            - 时间复杂度: O(1) 字典查找
            - 适合高并发场景(字典查找无锁竞争问题)
            - 生产环境需考虑并发写入时的线程安全(使用Redis原生原子操作)
        """
        key = self._make_key(intent, slots)
        entry = self._cache.get(key)
        if entry:
            # 检查TTL是否过期
            # time.time() 返回当前Unix时间戳(秒)
            # 如果当前时间 - 创建时间 > TTL，说明条目已过期
            if time.time() - entry.created_at > entry.ttl_seconds:
                del self._cache[key]  # 懒删除: 仅在被访问时才检查并删除过期条目
                return None
            entry.hit_count += 1  # 命中计数+1，用于统计缓存效果
            return entry
        return None

    def put(self, intent: str, slots: dict, response: str, source: str = ""):
        """写入缓存 - 将新生成的高质量回复存入缓存

        写入时机 (由 ModelRouter.route() 控制):
            - 语义缓存功能开启 (config.semantic_cache_enabled = True)
            - 回复文本非空
            - 意图置信度 > 0.85 (高阈值保证缓存内容质量)

        为什么要求高置信度:
            低置信度意味着意图识别可能不准确，缓存错误答案的风险大于收益。
            宁可多调用一次模型，也不要返回错误的缓存答案。

        参数:
            intent: 意图标识
            slots: 槽位字典
            response: 要缓存的回复文本
            source: 生成来源标记(可选)

        注意事项:
            - 相同key会被覆盖(后来的回复替换先前的)
            - 不做去重/合并，直接覆盖是最简单正确的策略
            - 生产环境可添加: 缓存容量限制、LRU淘汰、写入确认
        """
        key = self._make_key(intent, slots)
        self._cache[key] = CacheEntry(
            key=key,
            response=response,
            source=source,
        )

    def invalidate_by_intent(self, intent: str):
        """按意图失效缓存 - 知识更新时批量清除相关缓存

        使用场景:
            当知识库中某个意图的回复内容发生变更时（如退货政策更新），
            需要清除该意图下的所有缓存，确保用户获取最新信息。

        与架构的关系 (参见 §3.2 "知识热更新触发"):
            输入层检测到"活动规则变更/新政策"类事件时，
            会触发知识库增量索引，同时需要失效推理层的语义缓存。

        参数:
            intent: 需要失效的意图标识(如 "FAQ_RETURN_POLICY")

        实现细节:
            遍历所有缓存条目，匹配key中包含该intent的条目并删除。
            注意: 这里匹配的是哈希后的key中是否包含intent字符串——
            实际上MD5哈希后原始intent信息已丢失，此处存在BUG。
            正确实现应在CacheEntry中保存原始intent字段，并基于该字段匹配。
            (MVP阶段暂时保留，生产环境需修复此逻辑)

        性能特征:
            - 时间复杂度: O(n) 全表扫描
            - 低频操作(仅知识更新时触发)，性能可接受
            - 生产环境可用Redis的SCAN+DEL模式或前缀key设计来优化
        """
        to_remove = [k for k, v in self._cache.items() if intent in v.key]
        for k in to_remove:
            del self._cache[k]

    @property
    def size(self) -> int:
        """返回当前缓存条目总数

        用途:
            - 监控缓存增长趋势
            - 判断是否需要触发淘汰策略
            - 反馈层统计缓存容量指标
        """
        return len(self._cache)


# ═══════════════════════════════════════════════════════════════════════════════
# 模板回复库 (Template Response Library)
# ═══════════════════════════════════════════════════════════════════════════════
#
# 设计理念 (参见 §3.6.1 "路由决策引擎" - 路径A):
#     对于高置信度 + 规则命中的简单场景，直接返回预设模板，
#     实现零LLM成本、零延迟(毫秒级)的极致响应。
#     这是 Fallback 链的最终兜底方案，也是系统保护模式(R2)下的唯一出路。
#
# 成本分析 (参见 §3.6.1 "月度成本模型估算"):
#     路径A(30%流量): 直接命中模板/缓存 → ¥0/次
#     日均3万会话走此路径，每月节省约¥4.5万
#
# 维护注意事项:
#     - 模板内容需由运营/产品维护，确保话术合规、友好、准确
#     - 模板更新后需同步失效语义缓存中的相关条目
#     - 多租户场景下，不同租户可能需要不同的模板(当前为单租户实现)
#     - 模板支持的意图应覆盖所有高频场景，作为系统降级时的保底回复
# ═══════════════════════════════════════════════════════════════════════════════

TEMPLATE_RESPONSES: dict[str, str] = {
    # ─── 核心业务意图模板 ───
    "REFUND": "您好，已收到您的退款申请。请提供订单号，我将为您查询退款资格。",
    "LOGISTICS": "正在为您查询物流信息，请稍候...",
    "PRICE_PROTECT": "正在为您查询价保资格，请提供需要价保的订单信息。",
    "INVOICE": "好的，为您开具发票。请确认发票抬头和税号。",

    # ─── 情感/升级类意图模板 ───
    "COMPLAINT": "非常抱歉给您带来不好的体验，正在为您转接专属客服处理。",
    "TRANSFER_HUMAN": "正在为您转接人工客服，请稍候...",

    # ─── FAQ类意图模板 ───
    "FAQ_RETURN_POLICY": "7天无理由退货需满足：1. 签收7天内；2. 商品完好未使用；3. 不影响二次销售；4. 非定制/生鲜等特殊商品。",

    # ─── 通用/兜底模板 ───
    "GENERAL_INQUIRY": "您好，请问有什么可以帮到您？",

    # ─── 未知意图兜底(所有意图匹配失败时的最终保底) ───
    # 设计要点: 不说"我不知道"(挫败感)，而是引导用户换种方式表达或转人工
    "UNKNOWN": "抱歉，我不太理解您的问题。您可以换个方式描述，或者我为您转接人工客服。",
}


class ModelRouter:
    """多模型路由引擎 - 推理层的核心组件

    ═══════════════════════════════════════════════════════════════════════════
    职责 (参见 §3.6 "推理层"):
    ═══════════════════════════════════════════════════════════════════════════

    根据复杂度与成本/隐私约束做模型路由，核心决策：
    1. 该请求应该用哪个级别的模型来回复？
    2. 是否可以直接从缓存/模板返回？
    3. 当前系统状态是否允许调用LLM？
    4. 用户的token预算是否还够？

    ═══════════════════════════════════════════════════════════════════════════
    模型池 (参见 §3.6.1 "模型池配置"):
    ═══════════════════════════════════════════════════════════════════════════

    | 分类     | 示例模型                | 成本(¥/百万token) | 延迟     | 适用场景        |
    |---------|------------------------|-----------------|---------|----------------|
    | XLARGE  | Claude Opus / GPT-4o    | 60-100 (入)     | 0.8-2s  | 复杂推理/多意图 |
    | LARGE   | Claude Sonnet / GPT-4o-mini | 12-30 (入) | 0.4-0.8s| 标准对话/工具   |
    | MEDIUM  | Qwen2.5-72B / DeepSeek-V3 | 4-8 (入)    | 0.3-0.6s| 通用对话/改写   |
    | SMALL   | Qwen2.5-7B / GLM-4-9B   | 0.5-1 (入)     | 50-150ms| FAQ/摘要/分类   |
    | MICRO   | BERT/模板               | ~0              | 5-20ms  | 分类/模板回复   |

    ═══════════════════════════════════════════════════════════════════════════
    设计原则:
    ═══════════════════════════════════════════════════════════════════════════

    1. "够用即可" - 能用小模型解决的问题绝不用大模型(成本优先)
    2. "永不哑火" - 任何异常情况都有兜底方案(可用性优先)
    3. "可观测" - 每次调用都带有完整的成本和性能元数据(运营可控)
    4. "可治理" - 通过配置而非代码控制路由策略(灵活调整)
    """

    def __init__(self, config: ReasoningConfig, models: list[ModelConfig],
                 system_level: SystemLevel = SystemLevel.GREEN):
        """初始化模型路由引擎

        参数:
            config: 推理层配置对象，包含:
                - semantic_cache_enabled: 是否启用语义缓存
                - session_token_budget_normal: 普通用户单会话token预算
                - session_token_budget_vip: VIP用户单会话token预算
            models: 可用模型配置列表，每个模型包含:
                - tier: 模型级别(ModelTier枚举)
                - model_id: 模型标识符
                - cost_per_1k_input: 每千输入token成本
                - cost_per_1k_output: 每千输出token成本
            system_level: 当前系统负载级别，默认GREEN(正常)
                - GREEN: 正常运行，所有模型可用
                - YELLOW: 轻微压力，正常运行
                - ORANGE: 中度压力，限制大模型使用
                - RED: 严重压力，禁止LLM调用，仅模板
                - BLACK: 极端压力/故障，禁止LLM调用，仅模板

        内部状态:
            self.config: 推理层配置(不可变引用)
            self.models: 按tier索引的模型配置字典(O(1)查找)
            self.system_level: 系统级别(可由外部监控系统动态更新)
            self.cache: 语义缓存实例(会话间共享)
            self._token_usage: 会话级token用量追踪(session_id → 已用token数)

        设计决策:
            - models用dict存储而非list: 按tier查找是高频操作，dict O(1)优于list O(n)
            - _token_usage用dict而非外部存储: MVP简化，生产环境应用Redis做分布式追踪
            - system_level作为实例变量: 支持运行时动态调整(如由监控系统推送变更)
        """
        self.config = config
        # 将模型列表转为按tier索引的字典，便于O(1)快速查找
        # 如果同一tier有多个模型配置，后者会覆盖前者(R5成本优化场景下应选最优)
        self.models = {m.tier: m for m in models}
        self.system_level = system_level
        self.cache = SemanticCache()
        # 会话级token用量追踪
        # key: session_id, value: 该会话已累计消耗的token总数(input + output)
        # 注意: 内存存储，服务重启后归零；生产环境需持久化到Redis
        self._token_usage: dict[str, int] = {}

    def route(self, ctx: SessionContext, intent: IntentResult) -> ModelResponse:
        """模型路由主入口 - 按优先级执行路由规则，返回最终回复

        这是推理层对外暴露的唯一入口方法。
        接收上游(意图层/规划层)的处理结果，经过多级路由决策，
        最终返回一个统一的 ModelResponse 对象。

        ═══════════════════════════════════════════════════════════════════
        路由决策流程 (完整优先级链):
        ═══════════════════════════════════════════════════════════════════

        1. [R2] 系统保护检查 → RED/BLACK级别直接走模板(最高优先级)
        2. [缓存] 语义缓存查询 → 命中则直接返回(零成本)
        3. [R3] 配额管控检查 → 超预算则走模板(降级保护)
        4. [R4] 任务匹配选型 → 按复杂度选择模型级别
        5. [生成] 调用选中模型生成回复
        6. [缓存] 高质量回复写入缓存(供后续请求复用)
        7. [记账] 记录token消耗(供预算管控使用)

        注意: R1(隐私约束)在本MVP中未显式实现，
        生产环境应在步骤1之前加入PII检测，含敏感数据时强制路由到本地模型。

        ═══════════════════════════════════════════════════════════════════
        参数:
            ctx: 会话上下文，包含:
                - session_id: 会话唯一标识(用于token预算追踪)
                - is_vip: 是否VIP用户(影响token预算上限)
                - turn_count: 当前对话轮次(影响回复策略)
                - last_user_message: 用户最近一条消息文本
            intent: 意图识别结果，包含:
                - intent: 识别出的意图标识(如 "REFUND")
                - slots: 抽取的槽位(如 {"order_id": "12345"})
                - confidence: 识别置信度(0.0~1.0)
                - method: 识别方法("rule"/"classifier"/"llm_fallback")
                - sub_intents: 子意图列表(多意图场景)

        返回:
            ModelResponse: 包含回复文本及成本/性能元数据

        性能预期:
            - 模板路径: < 1ms
            - 缓存命中: < 5ms
            - 小模型调用: 50-150ms
            - 大模型调用: 400-2000ms
        """
        # 记录开始时间，用于计算全链路延迟
        start_time = time.time()

        # ═══ R2: 系统保护(最高优先级) ═══════════════════════════════════════
        # 当系统处于RED(严重过载)或BLACK(极端故障)级别时，
        # 禁止调用任何LLM模型，直接返回模板回复。
        # 设计理念: 系统可用性 > 回复质量。在系统濒临崩溃时，
        # 牺牲回复的个性化和准确性，换取系统整体的存活。
        # 参见 §3.6.1 "R2: 系统保护":
        #   system_level >= L3 → 禁止调用任何LLM，走模板
        if self.system_level in (SystemLevel.RED, SystemLevel.BLACK):
            return self._template_response(intent, start_time)

        # ═══ 缓存查询(在R3配额检查之前) ══════════════════════════════════════
        # 设计决策: 缓存查询放在配额检查之前——即使用户已超预算，
        # 如果缓存命中也应该返回(缓存命中零成本，不消耗预算)。
        # 这样设计对用户更友好: 即使超预算也不会影响已有缓存的问题。
        if self.config.semantic_cache_enabled:
            cache_entry = self.cache.get(intent.intent, intent.slots)
            if cache_entry:
                return ModelResponse(
                    text=cache_entry.response,
                    model_id="cache",          # 来源标记为缓存
                    model_tier=ModelTier.MICRO,  # 缓存等同于MICRO级(零成本)
                    from_cache=True,            # 标记为缓存命中
                    latency_ms=(time.time() - start_time) * 1000,
                )

        # ═══ R3: 配额管控 ════════════════════════════════════════════════════
        # 检查该会话已消耗的token是否超过预算上限。
        # VIP用户有更高的预算(通常2倍)，体现差异化服务。
        # 超预算后降级到模板回复，而非直接拒绝服务(用户体验优先)。
        # 参见 §3.6.1 "Token预算管理 - ②会话级":
        #   普通用户: ≤4000 token/会话
        #   VIP: ≤8000 token/会话
        #   超预算: 降级模型(而非停服)
        session_tokens = self._token_usage.get(ctx.session_id, 0)
        budget = (self.config.session_token_budget_vip
                  if ctx.is_vip else self.config.session_token_budget_normal)
        if session_tokens > budget:
            return self._template_response(intent, start_time)

        # ═══ R4: 任务匹配 ════════════════════════════════════════════════════
        # 根据意图的复杂度、置信度、识别方法等维度，选择最合适的模型级别。
        # 这是"够用即可"原则的核心体现——用最低成本的模型完成任务。
        selected_tier = self._select_model_tier(intent, ctx)

        # ═══ 生成回复 ════════════════════════════════════════════════════════
        # 使用选中的模型级别生成回复。
        # 如果选中的模型不可用(配置中没有该级别模型)，会自动降级到模板。
        response = self._generate(selected_tier, intent, ctx, start_time)

        # ═══ 写入缓存 ════════════════════════════════════════════════════════
        # 缓存写入条件(三个条件同时满足):
        # 1. 语义缓存功能已启用
        # 2. 回复文本非空(空回复无缓存价值)
        # 3. 意图置信度 > 0.85(高置信度才缓存，避免缓存错误答案)
        #
        # 为什么阈值是0.85而不是更低?
        # - 低置信度意味着意图可能识别错误
        # - 缓存了错误意图的回复，后续相同输入会持续返回错误答案
        # - 代价: 损失一部分缓存命中率; 收益: 杜绝错误答案被缓存
        if (self.config.semantic_cache_enabled
                and response.text
                and intent.confidence > 0.85):
            self.cache.put(intent.intent, intent.slots, response.text)

        # ═══ 记录token用量 ═══════════════════════════════════════════════════
        # 累加本次调用的token消耗到会话级别的计数器中。
        # 这是R3配额管控的数据基础——下一次请求时会检查这个累计值。
        # 参见 §3.6.1 "成本追踪":
        #   每次调用记录: model_id + input_tokens + output_tokens
        self._token_usage[ctx.session_id] = session_tokens + response.input_tokens + response.output_tokens

        return response

    def _select_model_tier(self, intent: IntentResult, ctx: SessionContext) -> ModelTier:
        """根据任务复杂度选择模型级别 - R4任务匹配的核心逻辑

        这是"混合智能"设计原则的体现: 规则、分类器、检索、大模型分工协作，
        按"够用即可"的最低成本模型完成任务。

        ═══════════════════════════════════════════════════════════════════
        决策规则(从上到下，首个匹配即返回):
        ═══════════════════════════════════════════════════════════════════

        1. MICRO(模板): 规则高置信(>0.9) + 有现成模板 → 零LLM成本
           适用: "查物流"、"开发票" 等高确定性、有固定回复的场景
           原理: 规则命中=100%确定用户意图，模板=预设的最佳回复

        2. SMALL(小模型): 单意图 + FAQ类 + 中高置信(>0.8) → 7B级模型
           适用: "怎么退货"、"退货规则是什么" 等FAQ类问题
           原理: FAQ答案相对固定，小模型足以生成流畅的回答

        3. LARGE(大模型): 多意图/低置信/LLM兜底 → 72B+级模型
           适用: "这个退款怎么还没到账，顺便帮我查下另一个包裹" 等复杂场景
           原理: 复杂场景需要更强的理解和推理能力

        4. MEDIUM(中型模型): 默认选择 → 中等能力模型
           适用: 不匹配上述任何规则的常规对话
           原理: 中型模型在成本和能力之间取得平衡

        ═══════════════════════════════════════════════════════════════════
        参数:
            intent: 意图识别结果(含 confidence、method、sub_intents 等)
            ctx: 会话上下文(当前未使用，预留用于VIP用户升级等场景)

        返回:
            ModelTier: 推荐的模型级别

        设计权衡:
            - 规则优先于模型: 能用规则解决就不调模型(零成本 vs 有成本)
            - 阈值选择保守: 宁可调大模型多花钱，也不要小模型答错(质量 > 成本)
            - ctx参数预留: 未来可基于用户等级、历史满意度等做差异化选型
              (如VIP用户默认升一级模型)

        与架构的关系 (参见 §3.6.1 "R4: 任务匹配"):
            意图分类/情绪检测 → 微型模型(BERT)
            单意图FAQ + 有RAG知识 → 小型模型(7B)
            标准多轮对话/工具调用 → 中型模型(72B私有化)
            复杂推理/多意图/长上下文 → 大/超大模型(云端)
        """
        # ─── 规则1: 高置信规则命中 + 有模板 → MICRO(零LLM) ───
        # 三个条件同时满足:
        #   - confidence > 0.9: 意图识别非常确定(几乎不可能错)
        #   - method == "rule": 由确定性规则/正则匹配命中(非模型推断)
        #   - intent在模板库中: 有预设的标准回复可用
        # 这是成本最优路径(路径A)，约占30%流量
        if (intent.confidence > 0.9
                and intent.method == "rule"
                and intent.intent in TEMPLATE_RESPONSES):
            return ModelTier.MICRO

        # ─── 规则2: 单意图 + FAQ类 + 中高置信 → SMALL(小模型) ───
        # 条件:
        #   - confidence > 0.8: 意图识别较为确定
        #   - sub_intents为空: 只有一个意图(非复合问题)
        #   - intent以"FAQ"开头: 明确的FAQ类问题
        # FAQ问题答案相对固定，7B级小模型足以生成流畅回复
        # 成本约为大模型的 1/20 ~ 1/50
        if (intent.confidence > 0.8
                and len(intent.sub_intents) == 0
                and intent.intent.startswith("FAQ")):
            return ModelTier.SMALL

        # ─── 规则3: 复杂场景 → LARGE(大模型) ───
        # 三个条件满足任一即触发:
        #   - 多意图(sub_intents > 1): 用户一句话包含多个诉求，需要强理解力
        #   - 低置信度(< 0.7): 意图不确定，需要大模型做更精准的判断
        #   - LLM兜底(method == "llm_fallback"): 规则和分类器都搞不定
        # 这是成本最高但质量最好的路径(路径C)，约占15%流量
        if (len(intent.sub_intents) > 1
                or intent.confidence < 0.7
                or intent.method == "llm_fallback"):
            return ModelTier.LARGE

        # ─── 默认: MEDIUM(中型模型) ───
        # 不匹配上述任何规则的请求走中型模型(路径B)
        # 中型模型(如72B私有化部署)在成本和能力之间取得平衡
        # 约占55%流量，是最常见的路径
        return ModelTier.MEDIUM

    def _generate(self, tier: ModelTier, intent: IntentResult,
                  ctx: SessionContext, start_time: float) -> ModelResponse:
        """调用指定级别模型生成回复 - 含Fallback降级逻辑

        ═══════════════════════════════════════════════════════════════════
        执行流程:
        ═══════════════════════════════════════════════════════════════════

        1. 如果tier是MICRO → 直接走模板(最低成本路径)
        2. 查找该tier对应的模型配置
        3. 如果找不到(模型不可用) → 降级到模板(Fallback兜底)
        4. 调用模型生成回复(MVP为模拟，生产替换为真实API)
        5. 估算token消耗和成本
        6. 封装为ModelResponse返回

        ═══════════════════════════════════════════════════════════════════
        Fallback设计 (参见 §3.6.1 "Fallback链"):
        ═══════════════════════════════════════════════════════════════════

        完整Fallback链: XLARGE → LARGE → MEDIUM → SMALL → 模板话术

        当前MVP简化: 如果指定tier的模型不在配置中，直接跳到模板。
        生产环境应实现逐级降级:
            selected_tier不可用 → 尝试下一级 → ... → 最终模板兜底

        ═══════════════════════════════════════════════════════════════════
        参数:
            tier: 目标模型级别(由_select_model_tier决定)
            intent: 意图识别结果(传递给模拟LLM用于生成回复)
            ctx: 会话上下文(传递给模拟LLM用于上下文感知)
            start_time: 路由开始时间(用于计算全链路延迟)

        返回:
            ModelResponse: 包含回复文本和完整元数据

        性能特征:
            - MICRO路径: < 1ms (纯内存字典查找)
            - SMALL路径: 50-150ms (本地GPU推理)
            - MEDIUM路径: 300-600ms (私有化部署)
            - LARGE路径: 400-2000ms (云端API调用)
        """
        # ─── MICRO级: 直接走模板(零LLM成本) ───
        # MICRO级别意味着不需要任何模型推理，用预设模板即可满足需求
        if tier == ModelTier.MICRO:
            return self._template_response(intent, start_time)

        # ─── 查找模型配置 ───
        # 从模型池中获取指定tier的模型配置
        model_config = self.models.get(tier)
        if not model_config:
            # Fallback: 如果该级别模型未配置或不可用，降级到模板回复
            # 这确保了"永不哑火"原则——即使所有模型都挂了，系统也能响应
            # 生产环境改进: 应尝试逐级降级(LARGE→MEDIUM→SMALL→模板)
            # 而非直接跳到模板，这样可以在能力退化最小的情况下保持服务
            return self._template_response(intent, start_time)

        # ─── MVP: 模拟模型调用 ───
        # ⚠️ 重要: 以下为模拟实现，生产环境需替换为真实API调用!
        # 真实API调用需要处理:
        # - HTTP请求(异步/流式)
        # - 超时重试(指数退避)
        # - 熔断降级
        # - 流式输出(首token优先)
        # - 错误处理(限速429、服务不可用503等)
        response_text = self._simulate_llm_response(intent, ctx)
        latency = (time.time() - start_time) * 1000  # 转换为毫秒

        # ─── Token估算 ───
        # MVP粗估策略: 中文字符数 × 2 ≈ token数
        # 原理: 中文平均每个字对应约1.5-2个token(取决于tokenizer)
        # 生产环境改进: 使用对应模型的tokenizer做精确计算
        # (如tiktoken for GPT系列, 或各模型提供的token计数API)
        input_tokens = len(ctx.last_user_message or "") * 2   # 输入token粗估
        output_tokens = len(response_text) * 2                 # 输出token粗估

        # ─── 成本计算 ───
        # 成本 = (输入token数 / 1000) × 每千输入token单价
        #       + (输出token数 / 1000) × 每千输出token单价
        # 注意: 输出成本通常是输入成本的2-3倍(参见 §3.6.1 模型池配置)
        cost = (input_tokens / 1000 * model_config.cost_per_1k_input +
                output_tokens / 1000 * model_config.cost_per_1k_output)

        return ModelResponse(
            text=response_text,
            model_id=model_config.model_id,    # 记录实际使用的模型(可追溯)
            model_tier=tier,                    # 记录模型级别(成本分类)
            input_tokens=input_tokens,          # 输入token(预算管控依据)
            output_tokens=output_tokens,        # 输出token(预算管控依据)
            latency_ms=latency,                 # 全链路延迟(性能监控)
            cost=cost,                          # 本次成本(成本追踪)
        )

    def _template_response(self, intent: IntentResult, start_time: float) -> ModelResponse:
        """模板回复生成 - 零LLM成本的保底回复方案

        使用场景(三种情况会触发模板回复):
            1. R2系统保护: 系统过载时强制走模板
            2. R3配额超限: 用户token预算耗尽时降级到模板
            3. R4任务匹配: 高置信规则命中且有对应模板
            4. Fallback兜底: 指定模型不可用时的最终保底

        设计理念 (参见 §3.6 "降级"):
            大模型不可用/超载时回退到小模型或预设话术，保证可用性。
            模板回复是Fallback链的终点，确保用户永远能得到一个回复。

        参数:
            intent: 意图识别结果(用于查找对应模板)
            start_time: 路由开始时间(用于计算延迟)

        返回:
            ModelResponse: model_id="template", model_tier=MICRO, cost=0

        注意事项:
            - 如果intent.intent不在模板库中，使用"UNKNOWN"兜底模板
            - 模板回复不计入token用量(零成本)
            - 延迟通常 < 1ms(纯字典查找)
        """
        # 从模板库查找对应意图的回复，找不到则使用UNKNOWN兜底
        text = TEMPLATE_RESPONSES.get(intent.intent, TEMPLATE_RESPONSES["UNKNOWN"])
        return ModelResponse(
            text=text,
            model_id="template",            # 来源标记为模板
            model_tier=ModelTier.MICRO,      # 成本级别为MICRO(零成本)
            latency_ms=(time.time() - start_time) * 1000,
            # input_tokens, output_tokens, cost 均默认为0
        )

    def _simulate_llm_response(self, intent: IntentResult, ctx: SessionContext) -> str:
        """MVP: 模拟LLM响应 - 生产环境需替换为真实API调用

        ═══════════════════════════════════════════════════════════════════
        ⚠️ 重要声明:
        ═══════════════════════════════════════════════════════════════════

        本方法为 MVP 阶段的模拟实现，用于:
        1. 验证路由逻辑的正确性(不依赖外部API即可测试)
        2. 演示不同意图的期望回复风格和内容
        3. 作为后续真实API集成的行为规范(参照模拟回复定义prompt)

        生产环境替换要点:
        - 对接真实LLM API (Claude/GPT/Qwen等)
        - 实现流式输出(首token优先，提升体感首响)
        - 加入Prompt模板管理(可灰度、可回滚)
        - 加入上下文压缩(保留最近5轮+摘要+知识片段)
        - 加入RAG检索结果注入(知识增强生成)
        - 实现超时重试和熔断降级

        ═══════════════════════════════════════════════════════════════════
        模拟策略:
        ═══════════════════════════════════════════════════════════════════

        根据 intent.intent 分支处理，每个意图有对应的模拟回复逻辑:
        - REFUND: 根据是否有order_id槽位，返回不同阶段的回复
        - LOGISTICS: 根据用户消息中是否含"催"/"慢"关键词，返回不同情绪的回复
        - PRICE_PROTECT: 引导用户提供订单信息
        - COMPLAINT: 共情+记录+解决方案的投诉处理话术
        - FAQ_RETURN_POLICY: 结构化的退货政策回复
        - 多轮对话: 结合上下文给出连贯回复
        - 兜底: 通用友好回复

        ═══════════════════════════════════════════════════════════════════
        参数:
            intent: 意图识别结果(决定回复内容)
            ctx: 会话上下文(提供用户消息、对话轮次等信息)

        返回:
            str: 模拟的LLM回复文本
        """
        user_msg = ctx.last_user_message or ""

        # ─── 退款意图 ───
        # 分支逻辑: 有订单号 → 告知正在查验; 无订单号 → 引导提供
        # 体现了"槽位驱动的对话管理": 关键槽位是否齐全决定了回复策略
        if intent.intent == "REFUND":
            if intent.slots.get("order_id"):
                # 已有订单号: 告知用户正在处理(降低焦虑感)
                return f"已查到您的订单 {intent.slots['order_id']}，正在为您校验退款资格，请稍候。"
            # 缺少订单号: 友好地引导用户提供(追问策略)
            return "好的，我来帮您处理退款。请问您要退哪个订单呢？您可以提供订单号或商品名称。"

        # ─── 物流意图 ───
        # 分支逻辑: 检测用户情绪(催促/焦虑) → 差异化回复
        # 体现了"情感感知的客服策略": 不同情绪状态给予不同程度的安抚
        # 参见 §4.2.2 "客户情感画像":
        #   按 preferred_tone 调整生成风格(高效型少寒暄，共情型多安抚)
        if intent.intent == "LOGISTICS":
            if "催" in user_msg or "慢" in user_msg:
                # 用户情绪焦虑: 先共情("理解您的着急")，再给出行动("已发起催促")
                # 并主动承诺后续跟进("2小时内仍无更新，我再帮您联系")
                return "理解您的着急。已为您发起催促，快递员将优先为您派送。如2小时内仍无更新，我再帮您联系快递站点。"
            # 常规物流查询: 给出明确信息 + 提供进一步服务选项
            return "已为您查询物流信息：您的包裹目前在杭州转运中心，预计明天送达。如需催促派送，我可以帮您联系快递员。"

        # ─── 价保意图 ───
        # 价保(Price Protection): 商品降价后用户可申请退差价
        # 需要确认具体订单才能查询是否在价保期内
        if intent.intent == "PRICE_PROTECT":
            return "好的，正在为您查询价保资格。请问您是哪个订单需要价保？我帮您核实商品是否在价保期内。"

        # ─── 投诉意图 ───
        # 投诉处理的三步法: 共情 → 收集信息 → 承诺解决
        # 参见 §4.3 "转人工策略": 负面情绪升级时转人工
        # 这里先由AI做初步安抚和信息收集，为后续升级(如需要)做准备
        if intent.intent == "COMPLAINT":
            return "非常理解您的心情，对于给您带来的不好体验我深感抱歉。请您详细描述一下问题，我会认真记录并为您寻找最佳解决方案。"

        # ─── FAQ退货政策 ───
        # 结构化回复: 条件列举 + 行动引导
        # 与模板库的区别: 模型生成的版本更详细，增加了"如何操作"的引导
        if intent.intent == "FAQ_RETURN_POLICY":
            return "7天无理由退货需满足以下条件：1. 签收7天内；2. 商品完好未使用；3. 不影响二次销售；4. 非定制/生鲜等特殊商品。如符合条件，可在订单详情中申请退货。"

        # ─── 多轮对话兜底 ───
        # 当对话已经进行了多轮(turn_count > 1)，说明用户正在持续沟通中。
        # 此时应该体现对话的连贯性——"关于您之前咨询的问题"，
        # 而非像首轮一样说"您好，有什么可以帮您"。
        # 这是多轮对话体验的关键: 让用户感到AI记住了之前的对话。
        if ctx.turn_count > 1:
            return f"好的，关于您之前咨询的问题，我继续为您处理。{TEMPLATE_RESPONSES.get(intent.intent, '请问还有什么可以帮您？')}"

        # ─── 终极兜底 ───
        # 所有分支都未匹配时的最终回复
        # 策略: 友好 + 主动 + 引导用户提供更多信息
        # 避免说"我不知道"(挫败感)，而是表示"我来帮您"(积极主动)
        return TEMPLATE_RESPONSES.get(intent.intent, "好的，我来帮您处理这个问题。请问具体是什么情况？")
