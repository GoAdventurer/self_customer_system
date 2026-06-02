"""③ 规划层 - DAG流程编排引擎
================================================================================

【模块定位 — 参考 docs/03-orchestration-design.md §1 设计目标与定位】
================================================================================

本模块是整个智能客服系统「规划层」的核心执行引擎。在系统分层架构中处于承上启下的位置：

  ┌──────────┐      ┌──────────────────┐      ┌──────────────┐
  │  意图层   │ ───▶ │  规划层(本模块)    │ ───▶ │   执行层      │
  │ NLU/分类  │      │  DAG编排引擎      │      │ Tool/API调用  │
  └──────────┘      └──────────────────┘      └──────────────┘
                           ↕
                    ┌──────────────┐
                    │   反馈层      │
                    │ Trace/质检   │
                    └──────────────┘

规划层的核心职责（设计文档 §1）：
  将"用户意图 + 槽位"映射为一张**可执行、可持久化、可补偿、可人工介入**的
  有向无环任务图(DAG)，并驱动其逐步执行直到完成或遇到外部等待点。

【为什么需要 DAG 编排而非让 LLM 自由调用工具？】
  设计文档 §1 核心原则：
  > "确定性可控：跨系统副作用（退款/改单/写CRM）由图显式编排，而非模型自由调用。"

  在企业级客服系统中，涉及资金变动（退款、价保退差价）的操作如果由 LLM 自主决策，
  存在以下不可接受的风险：
  - LLM 幻觉可能导致未经用户确认就执行不可逆操作
  - 无法提供审计追踪和合规保障
  - 失败时缺乏确定性的补偿路径
  - 无法保证操作的幂等性和一致性

  DAG 编排把这些关键决策从"概率性的 LLM 输出"变为"确定性的图结构约束"，
  同时通过 HITL 节点保留了人在回路中的控制权。

【核心能力清单】
  1. 意图→DAG 映射：将分类后的意图映射为对应的任务依赖图模板
  2. 条件分支：通过条件边(DAGEdge.condition)实现路径分流
  3. 并行扇出：一个节点多条出边，后继节点可并行执行
  4. HITL 人工确认：流程在关键决策点挂起等待用户/人工输入
  5. 状态持久化 & 断点恢复：执行状态可序列化，支持跨时间/跨进程恢复
  6. 重试容错：节点执行失败时按配置的次数重试
  7. Saga 补偿链：不可逆操作失败时按逆序触发补偿

【设计文档中有但本 MVP 暂未实现的能力（后续迭代补充）】
  - §2.2 Channel+Reducer 状态管理（当前用简单 dict 代替）
  - §4.2 LOOP 回环边（有界循环）
  - §4.2 ESCALATE/INTERRUPT 边（升级/中断）
  - §7 图静态校验（发布前的完备性/安全性检查）
  - §8 持久化到 Redis/PostgreSQL（当前纯内存）
  - §9 版本灰度发布

【文件结构概览】
  Part 1: 枚举定义 — NodeType(5种节点类型), NodeStatus(7种节点状态)
  Part 2: 静态数据结构 — DAGNode(点), DAGEdge(边), DAGDefinition(图模板)
  Part 3: 运行态数据结构 — NodeExecution(节点执行状态), DAGExecution(图执行实例)
  Part 4: 预置流程模板 — 退款流程(7步含HITL), 价保流程(4步快速通道)
  Part 5: 执行引擎 — DAGEngine(核心调度类)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional
import time


# ═══════════════════════════════════════════════════════════════════════════════════
# Part 1: 枚举定义 — 节点类型与节点状态
# ═══════════════════════════════════════════════════════════════════════════════════


class NodeType(Enum):
    """DAG节点类型枚举 — 定义节点的行为类别

    对应设计文档 docs/03-orchestration-design.md §3.2 节点类型分类表。

    【设计文档定义了12种节点类型，本实现精简为5种核心类型】

    映射关系表：
    ┌──────────────────────┬─────────────────┬────────────────────────────┐
    │ 设计文档类型          │ 本实现类型       │ 简化原因                    │
    ├──────────────────────┼─────────────────┼────────────────────────────┤
    │ START / END          │ (隐式)          │ 由 entry_node + 叶子节点表达 │
    │ TOOL (read/write)    │ ACTION          │ 统一为"执行动作"             │
    │ LLM                  │ ACTION          │ handler 路由到推理层即可     │
    │ ROUTER / DECISION    │ CONDITION       │ 纯计算分支                  │
    │ HITL                 │ HITL            │ 1:1 对应                   │
    │ PARALLEL_FORK        │ PARALLEL        │ 1:1 对应                   │
    │ (无对应)             │ TEMPLATE        │ 新增：无LLM的模板回复        │
    │ JOIN / BARRIER       │ (隐式)          │ 由 depends_on 多依赖表达    │
    │ COMPENSATION         │ (字段)          │ 由 compensate_handler 挂载  │
    │ SUBGRAPH             │ 未实现          │ 当前无子图嵌套需求           │
    │ MAP                  │ 未实现          │ 当前无批量动态扇出需求       │
    │ WAIT / TIMER         │ 未实现          │ 当前无定时等待需求           │
    └──────────────────────┴─────────────────┴────────────────────────────┘

    【为什么精简为5种而不是完整实现12种？】
    - 当前业务场景（退款/价保）仅需5种类型即可覆盖 >95% 用例
    - 更少的类型 = 更少的代码路径 = 更低的测试复杂度 = 更快的交付
    - 遵循 YAGNI 原则，待真实需求出现时再扩展
    - 扩展是向后兼容的：新增 Enum 值 + 在 execute_step 中添加分支即可

    【为什么用字符串值 Enum 而不是 int Enum？】
    - 持久化（JSON序列化）时人类可读："action" vs 1
    - 运营编排台前端展示友好
    - 调试日志直观
    - 性能差异在此场景可忽略
    """

    ACTION = "action"
    """执行动作节点 — 调用外部API/服务

    对应设计文档 §3.2 的 TOOL 类型（含 read/write/irreversible 三种副作用级别）。
    这是最常用的节点类型，覆盖了大部分"做事"的场景。

    副作用级别（由 metadata 中的 side_effect 字段区分）：
      - read: 只读查询，安全重试（如查询订单）
      - write: 有写入但可补偿（如回写CRM）
      - irreversible: 不可逆操作（如退款），前置必须有 HITL 确认

    执行引擎处理逻辑：
      1. 查找 handler 对应的注册函数
      2. 调用函数，传入 global_state
      3. 成功 → COMPLETED，失败 → 重试或 FAILED

    示例 handler：
      "order_service.query"     (read)
      "crm_service.update"      (write)
      "payment_gateway.refund"  (irreversible)
    """

    CONDITION = "condition"
    """条件分支节点 — 纯计算决策，根据 state 数据选择出边路径

    对应设计文档 §3.2 的 ROUTER/DECISION 类型。
    特点：side_effect=none，无副作用，可安全重试。

    执行逻辑：
      - 评估 node.condition 表达式（对应设计文档 §4.3 Predicate）
      - 根据结果激活对应的条件出边
      - 当前 MVP 简化：条件边评估由 execute_step 外部逻辑处理

    设计文档 §4.3 的 Predicate 结构化表达式（本实现暂用字符串简化）：
      { op: "eq", left: "state.risk_result", right: "PASS" }
    """

    HITL = "hitl"
    """人工确认节点 (Human-In-The-Loop) — 挂起流程等待外部输入

    对应设计文档 §3.2 的 HITL 类型，是实现"可中断可恢复"的关键机制。

    【为什么需要 HITL？— 设计文档 §3.2 关键约束】
    > "side_effect=irreversible 的节点前置必须有 HITL 确认节点"

    在客服场景中，退款是不可逆操作（钱退出去了不能自动收回）。
    系统必须在执行退款前获得用户的明确确认，HITL 节点就是这个"确认门槛"。

    执行引擎处理逻辑：
      1. 遇到 HITL 节点 → 标记状态为 WAITING
      2. 整个 execution 状态变为 "waiting"
      3. 引擎停止推进，将控制权交回上层
      4. 上层持久化 execution → 向前端发送确认请求
      5. 用户确认后 → 恢复 execution → 将 HITL 节点标记 COMPLETED → 继续推进

    超时处理（设计文档 §5.3）：
      "N4 → N_ABANDON: 30min 无回应 / 用户取消"
      当前 MVP 的超时处理由上层会话管理器负责，引擎本身不做超时计时。

    典型场景：
      - 退款二次确认（"确认退款 ¥199.00 到原支付方式？"）
      - 人工审批（高金额退款需主管审批）
      - 身份核验（要求用户提供验证码）
    """

    PARALLEL = "parallel"
    """并行扇出节点 — 同时激活多个后继分支

    对应设计文档 §3.2 的 PARALLEL_FORK 类型。

    当前实现中，并行扇出是通过"一个节点有多条出边"隐式表达的
    （如 execute_refund 同时连向 write_crm 和 notify_user），
    不需要显式的 PARALLEL 类型节点。

    此类型预留给未来需要显式扇出控制的场景：
      - 动态确定并行分支数量
      - 需要记录扇出时间戳
      - 需要限制并行度（如最多3个分支同时执行）

    设计文档 §2.2 的并行安全约束：
    > "并行分支不得对同一 last 通道写不同值"
    """

    TEMPLATE = "template"
    """模板回复节点 — 使用预定义模板生成回复，无需LLM调用

    本实现新增的类型（设计文档中由 LLM 节点 + template fallback 覆盖）。

    适用场景：结果格式固定、内容确定性强、不需要 LLM 创造性生成。

    优势（相比 LLM 节点）：
      - 零延迟：无需等待模型推理（~0ms vs ~500ms-2s）
      - 零成本：无 token 消耗
      - 完全可控：输出 100% 确定，无幻觉风险
      - 高并发：无模型服务容量限制

    典型场景：
      - 价保结果通知："您的订单符合价保条件，差价 ¥XX 将在3个工作日退回"
      - 状态通知："退款已成功，预计1-3个工作日到账"
      - 错误提示："抱歉，您的订单已超出退款期限"
    """


class NodeStatus(Enum):
    """节点执行状态枚举 — 描述节点在运行时的生命周期状态

    对应设计文档 docs/03-orchestration-design.md §2.3 运行态枚举。

    【节点状态机（对应设计文档中 NodeRun 状态转移图）】

    本实现的状态流转：
    ┌─────────────────────────────────────────────────────────────┐
    │                                                             │
    │  ┌─────────┐     ┌─────────┐     ┌───────────┐            │
    │  │ PENDING │────▶│ RUNNING │────▶│ COMPLETED │            │
    │  └─────────┘     └────┬────┘     └───────────┘            │
    │       ▲                │                                    │
    │       │ (重试)         ├────▶ ┌─────────┐                  │
    │       │                │      │ WAITING │ (HITL专用)       │
    │       │                │      └─────────┘                  │
    │       │                │                                    │
    │       └────────────────┼────▶ ┌────────┐                   │
    │       (retry_count<max)│      │ FAILED │ (重试耗尽)        │
    │                        │      └───┬────┘                   │
    │                        │          │                        │
    │                        │          ▼                        │
    │                        │    ┌──────────────┐              │
    │                        │    │ COMPENSATING │ (Saga回滚)    │
    │                        │    └──────────────┘              │
    │                        │                                    │
    │                        └────▶ ┌─────────┐                  │
    │                               │ SKIPPED │ (条件不满足)     │
    │                               └─────────┘                  │
    └─────────────────────────────────────────────────────────────┘

    【与设计文档的差异说明】
    设计文档 §2.3 中有 READY 和 RETRYING 两个额外状态：
      - READY（依赖满足等待调度）：本实现合并入 PENDING
        原因：当前单线程同步模型中不需要区分"等待依赖"和"等待调度"
      - RETRYING（正在重试中）：本实现通过 PENDING + retry_count > 0 表达
        原因：减少状态数，简化状态机转移逻辑
    这两个简化不影响功能正确性，仅损失了一些可观测性粒度。
    """

    PENDING = "pending"
    """待执行状态 — 节点已创建，等待依赖满足或调度

    这是所有节点的初始状态（DAGExecution 创建时）。
    当所有 depends_on 的节点都 COMPLETED 后，本节点变为"就绪"可被执行。
    重试时也会重置回此状态，由 execute_step 下一轮迭代重新拾取。
    """

    RUNNING = "running"
    """执行中状态 — handler 正在被调用

    进入此状态的时刻记录 started_at 时间戳。
    正常情况下此状态是瞬态的（同步执行完立即转为 COMPLETED/FAILED）。
    如果是异步执行（未来扩展），此状态可能持续较长时间。
    超过 timeout_ms 未完成应触发超时失败（当前 MVP 未实现超时中断）。
    """

    COMPLETED = "completed"
    """执行成功状态 — handler 返回且结果有效

    终态之一。进入此状态时记录 completed_at 时间戳和 result。
    后继节点的依赖满足计算只看此状态（FAILED/SKIPPED 不算"满足"）。
    对应设计文档 §3.3 中的 post_condition 校验通过。
    """

    FAILED = "failed"
    """执行失败状态 — handler 抛异常且重试次数耗尽

    终态之一。进入此状态时记录 error 信息。
    可能触发的后续动作（对应设计文档 §3.3 on_failure）：
      - "fail": 整图标记失败（当前实现的行为）
      - "skip": 跳过此节点继续执行（本 MVP 未实现）
      - "fallback": 走降级边到备选路径（本 MVP 未实现）
      - "escalate": 升级到人工处理（本 MVP 未实现）
    还可能触发 Saga 补偿链（设计文档 §6）。
    """

    WAITING = "waiting"
    """等待外部输入状态 — HITL 节点专用

    节点进入此状态后，整个 DAGExecution.status 也变为 "waiting"。
    上层需要：
      1. 持久化整个 execution（checkpoint）
      2. 向用户发送确认请求
      3. 等待回调

    超时设计（设计文档 §5.3）：
    HITL 节点有 timeout_ms 配置（如5分钟/30分钟），
    超时未回应应转为 ABANDONED（当前 MVP 由上层管理器处理）。
    """

    SKIPPED = "skipped"
    """已跳过状态 — 条件分支中未命中的路径

    当条件边评估后，未被选中路径上的节点标记为 SKIPPED。
    SKIPPED 节点不算作"完成"，不会解锁依赖它的下游节点。
    （当前 MVP 简化：尚未实现条件边评估和节点跳过逻辑）
    """

    COMPENSATING = "compensating"
    """补偿执行中状态 — Saga 回滚时正在执行补偿操作

    对应设计文档 §6 Saga 补偿与图回滚语义。

    触发条件：DAG 进入 COMPENSATING 状态时，按"已成功且有 compensate_handler
    的节点"逆序执行补偿。

    设计文档 §6 补偿原则：
      1. read/none 节点无需补偿（无副作用）
      2. write 节点走最终一致（对账修复）
      3. irreversible 节点需显式反向业务（如撤销退款）
      4. 补偿函数本身必须幂等，失败则升级人工
    """


# ═══════════════════════════════════════════════════════════════════════════════════
# Part 2: 静态数据结构 — DAG图模板的构成元素
# ═══════════════════════════════════════════════════════════════════════════════════
# 设计文档 §2 核心原则："定义与运行分离"
#   - DAGDefinition / DAGNode / DAGEdge = 静态模板（可版本化、可灰度）
#   - DAGExecution / NodeExecution = 运行实例（有状态、可中断恢复）
# ═══════════════════════════════════════════════════════════════════════════════════


@dataclass
class DAGNode:
    """DAG节点定义 — 图模板中一个步骤的静态规格说明

    对应设计文档 docs/03-orchestration-design.md §3.1 节点通用结构(NodeSpec)。

    这是节点的「静态定义」（模板层面），描述：
      - 这个节点「是什么」（身份：id, type, name）
      - 这个节点「怎么执行」（契约：handler, timeout, retries）
      - 这个节点「和谁有关」（依赖：depends_on）
      - 这个节点「失败了怎么办」（补偿：compensate_handler）

    而不是描述"当前执行到哪了"——那是 NodeExecution 的职责。

    【设计文档 §3.1 完整 NodeSpec 对比（本实现的简化取舍）】

    设计文档包含但本实现简化/省略的字段：
      - inputs/outputs Binding: 设计文档要求显式声明从 state 取哪些通道值、写回哪些通道
        本实现简化为：handler 直接接收整个 global_state dict，自行读写
        trade-off: 简单但失去了通道级的并发安全保障和数据流追踪
      - idempotency_key: 幂等键模板（如 "${session_id}:${node_id}:${order_id}"）
        本实现暂未包含，生产环境对 irreversible 节点必须补充
        trade-off: 缺失幂等键 = 重试时可能重复执行副作用
      - side_effect 声明: read/write/irreversible
        本实现放入 metadata（可选），未在引擎中强制校验
        trade-off: 引擎无法自动执行"irreversible 前必有 HITL"的合规检查
      - risk_level / audit / requires_approval: 治理字段
        本实现放入 metadata
      - on_error 策略: fail/skip/fallback/escalate
        本实现固定为 "fail"（重试耗尽即标记整图失败）

    【为什么 handler 是 Optional[str] 而非直接存储 Callable？】
    三个核心原因（对应设计文档"可版本化"的设计目标）：
      1. 可序列化：DAGDefinition 需要 JSON 持久化到数据库，Callable 不可序列化
      2. 可跨服务：handler_id 可以路由到远程 RPC/HTTP 调用（微服务架构）
      3. 可测试：测试时注入 mock handler，无需修改 DAGDefinition

    【为什么 depends_on 放在 Node 上（与 Edge 冗余）？】
    严格来说依赖关系由 Edge 定义。Node.depends_on 是一种「冗余索引」。
    好处：get_ready_nodes() 计算时 O(N) 即可判断就绪，无需遍历边集合
    代价：需保证 edges 和 depends_on 一致性（图校验 §7 应检查）
    """

    node_id: str
    """节点唯一标识 — 图内不可重复

    命名规范：snake_case，动词_名词 或 名词 形式
    示例："query_order", "validate_refund", "user_confirm", "execute_refund"
    用途：日志追踪、状态查询、depends_on 引用、前端编排台展示
    """

    node_type: NodeType
    """节点类型 — 决定执行引擎如何调度此节点

    不同类型的调度行为差异：
      - ACTION: 查找 handler → 调用 → 记录结果
      - HITL: 标记 WAITING → 挂起整图 → 等待外部恢复
      - TEMPLATE: 可以不调用 handler（用模板直接生成）
      - CONDITION: 评估 condition 表达式 → 决定走哪条出边
      - PARALLEL: 激活所有出边对应的后继节点
    """

    name: str
    """人类可读名称 — 展示在日志、监控面板、运营编排台

    应该用简洁的中文描述该步骤的业务动作。
    示例："查询订单", "校验退款资格", "用户二次确认", "发起退款"
    """

    handler: Optional[str] = None
    """处理函数标识 — 字符串形式，运行时映射到实际 Callable

    格式规范："service_name.method_name"
    示例：
      "order_service.query"           → 查询订单详情
      "rule_engine.validate_refund"   → 规则引擎校验退款资格
      "risk_service.check"            → 风控系统检查
      "payment_gateway.refund"        → 支付网关退款接口
      "crm_service.update"            → CRM系统回写
      "notify_service.push"           → 消息通知推送

    为 None 的场景：
      - HITL 节点不执行 handler（只做状态流转）
      - TEMPLATE 节点可能通过 metadata 中的模板配置生成回复

    运行时解析：
      engine._handlers[node.handler](execution.global_state) → result
    """

    timeout_ms: int = 30000
    """节点执行超时(毫秒) — 默认30秒

    对应设计文档 §2.1 policies.default_timeout_ms 的节点级覆盖。

    典型配置：
      - 查询接口(read): 5000-10000ms（网络 + DB查询）
      - 写入接口(write): 10000-30000ms（含可能的分布式事务）
      - HITL 用户确认: 300000ms (5分钟) 或更长
      - 批量操作: 可能需要 60000-120000ms

    超时后的行为（当前 MVP 未实现超时中断）：
      理想情况应 cancel 执行中的请求并标记为 FAILED，触发重试或失败处理。
    """

    retries: int = 3
    """最大重试次数 — 默认3次

    对应设计文档 §3.1 NodeSpec.retry。

    设计文档的关键约束：「仅幂等节点可重试」
      - read 节点：天然幂等，可安全重试
      - write 节点：需要 idempotency_key 保证幂等性
      - irreversible 节点：必须有 idempotency_key，否则重试可能导致重复退款

    当前 MVP 的重试策略简化：
      - 立即重试，无退避间隔（生产环境应加指数退避 + 随机抖动）
      - 不区分异常类型（如网络超时 vs 业务拒绝应区分处理）
      - 未实现熔断器模式（连续失败应停止重试并降级）
    """

    depends_on: list[str] = field(default_factory=list)
    """前驱依赖节点ID列表 — 所有依赖 COMPLETED 后本节点才可执行

    这是 DAG 调度的核心约束：通过声明依赖关系构建偏序。
    空列表 = 无前驱依赖 = 入口节点（如 "query_order"）。

    示例：
      depends_on=["query_order"]         → 订单查询完成后才能校验资格
      depends_on=["execute_refund"]      → 退款完成后才能通知用户
      depends_on=["risk_check"]          → 风控通过后才能让用户确认

    与 edges 的关系（冗余设计的一致性要求）：
      如果 edge(A→B) 存在，则 B.depends_on 应包含 A
      图校验（§7）应检查这种一致性
    """

    condition: Optional[str] = None
    """条件表达式 — 用于 CONDITION 类型节点的分支判断

    对应设计文档 §4.3 条件谓词(Predicate)的简化表示。
    当前为字符串形式（如 "risk_result == 'PASS'"），
    生产环境应扩展为结构化 Predicate 对象以支持安全计算。

    设计文档 §4.3 完整 Predicate 结构：
      { op: "and", clauses: [
          { op: "gt", left: "slots.amount", right: 5000 },
          { op: "ne", left: "tool_results.risk.decision", right: "PASS" }
      ]}
    """

    compensate_handler: Optional[str] = None
    """Saga补偿函数标识 — 失败时用于撤销此节点产生的副作用

    对应设计文档 §3.1 NodeSpec.compensation 和 §6 Saga补偿语义。

    只有 side_effect=write/irreversible 的节点需要配置此字段。
    read/none 节点无副作用，无需补偿。

    补偿触发时机（设计文档 §6）：
      当 DAG 进入 COMPENSATING 状态时，按"已成功且有 compensate_handler
      的节点"逆序执行补偿。

    补偿函数要求：
      1. 必须幂等（重复调用结果一致）
      2. 失败时应升级人工而非无限重试
      3. 可能涉及线下介入（如退款补偿需要人工审批追回）

    示例：
      "payment_gateway.cancel_refund" — 撤销已发起的退款单
    """

    metadata: dict[str, Any] = field(default_factory=dict)
    """扩展元数据字典 — 存放业务自定义和治理信息

    对应设计文档 §3.1 中的治理字段和自定义信息：
      - risk_level: "LOW"/"MEDIUM"/"HIGH"/"CRITICAL"
      - audit: bool (是否写审计日志)
      - requires_approval: bool (是否强制人工审批)
      - side_effect: "none"/"read"/"write"/"irreversible"
      - idempotency_key: 幂等键模板字符串
      - description: 节点详细描述
      - owner: 负责人
      - x/y: 编排台可视化坐标
    """


@dataclass
class DAGEdge:
    """DAG边定义 — 图模板中连接两个节点的有向边

    对应设计文档 docs/03-orchestration-design.md §4.1 边通用结构(EdgeSpec)。

    边的核心语义：「从 from_node 完成后，可以(在满足 condition 时)到达 to_node」

    【设计文档定义了10种边类型（§4.2），本实现统一为一种结构】

    通过字段组合表达不同语义：
      - 顺序边(SEQUENTIAL): condition=None, 前驱完成无条件进入后继
      - 条件边(CONDITIONAL): condition="<expr>", 满足条件才走
      - 并行扇出(FORK): 同一 from_node 连出多条边到不同 to_node
      - 汇聚(JOIN): 多条边指向同一 to_node（通过 to_node.depends_on 表达）

    本 MVP 暂未覆盖的边类型：
      - DEFAULT(兜底): 所有条件边都不命中时走（else分支）
      - LOOP(回环): 指向已访问节点的回边（需 max_iterations 上界）
      - FALLBACK(降级): on_error=fallback 时走
      - ESCALATE(升级): 转人工/升级
      - COMPENSATE(补偿): Saga 回滚时的反向边
      - INTERRUPT(中断): 全局中断（超时/用户取消）

    【互斥与完备性 — 设计文档 §4.3 & §7 校验规则】
    同源的 CONDITIONAL 边 + 一条 DEFAULT 边必须覆盖所有取值，
    否则可能出现"无路可走"的悬挂状态。图校验期应强制检查。
    """

    from_node: str
    """边的起始节点ID — 当此节点 COMPLETED 后，边有可能被激活"""

    to_node: str
    """边的目标节点ID — 边被激活后，此节点成为可执行候选"""

    condition: Optional[str] = None
    """条件表达式 — 满足此条件时才激活此边

    None = 无条件边(顺序边)，前驱完成即激活。
    非 None = 条件边，需评估表达式后决定。

    对应设计文档 §4.3 Predicate。当前为字符串简化表示。
    设计文档要求同源条件边互斥且完备（通过 priority 字段择优）。
    """

    label: str = ""
    """人类可读标签 — 用于编排台可视化和日志追踪

    不参与执行逻辑，纯展示用途。
    示例："风控通过", "金额>5000", "用户已确认", "超时未响应"
    """


@dataclass
class DAGDefinition:
    """DAG图定义（静态模板）— 一个完整业务流程的结构化蓝图

    对应设计文档 docs/03-orchestration-design.md §2.1 GraphDefinition。

    这是流程的「蓝图/模板」——描述一个业务流程包含哪些步骤、步骤间有什么
    依赖和条件关系，但不包含任何运行时状态。

    类比关系：
      DAGDefinition : DAGExecution ≈ Class : Instance ≈ 图纸 : 建筑

    同一个 DAGDefinition 可以被多个 DAGExecution 引用
    （如退款流程模板同时服务1000个用户的退款请求）。

    【版本化与灰度 — 设计文档 §9】
    - graph_id + version 构成不可变快照标识
    - 修改流程逻辑时发新 version（不修改旧版本）
    - 已在执行中的 run 锁定创建时的版本（不受新版本影响）
    - 支持灰度放量：新版本按 10%→50%→100% 逐步放开
    - 支持一键回滚：指标异常时回退到上一稳定版本

    【图校验 — 设计文档 §7（发布前静态检查）】
    DAGDefinition 发布前应通过以下校验（本 MVP 尚未实现）：
      ┌────────┬─────────────────────────────────────────────────────┐
      │ 类别   │ 规则                                                │
      ├────────┼─────────────────────────────────────────────────────┤
      │ 结构   │ 唯一 entry_node；所有节点从入口可达；无死胡同        │
      │ 无环   │ 除显式 LOOP 边外不得有环                            │
      │ 分支   │ 同源条件边 + DEFAULT 必须完备                       │
      │ 互斥   │ 同源条件边任一状态下命中集合互斥                     │
      │ 并行   │ 并行分支不对同一 'last' 通道写不同值                 │
      │ 合规   │ irreversible 节点前驱必含 HITL，且 audit=true       │
      │ 补偿   │ write/irreversible 节点应声明 compensation          │
      │ 预算   │ 存在 max_total_steps 和整图 sla_ms                  │
      └────────┴─────────────────────────────────────────────────────┘
    """

    graph_id: str
    """业务流程唯一标识

    全局唯一，如 "refund_flow", "price_protect_flow"。
    配合 version 构成完整的快照标识（graph_id@version 不可变）。
    """

    name: str
    """流程人类可读名称

    展示在运营编排台、监控面板、审计日志中。
    如 "退款全流程", "价保快速通道"。
    """

    version: str = "1.0.0"
    """语义化版本号(SemVer)

    版本变更规则建议：
      - patch (0.0.x): 修复bug，不改变拓扑
      - minor (0.x.0): 新增节点/边，向后兼容
      - major (x.0.0): 拓扑大改，不向后兼容
    """

    entry_node: str = ""
    """图的唯一入口节点ID — 执行从此节点开始

    设计文档 §7 校验规则：entry_node 必须唯一。
    入口节点的 depends_on 应为空列表（无前驱依赖）。
    """

    nodes: list[DAGNode] = field(default_factory=list)
    """图中所有节点的列表（点集合）

    顺序不影响执行逻辑（执行顺序由 edges + depends_on 决定）。
    但建议按拓扑序排列以提高可读性。
    """

    edges: list[DAGEdge] = field(default_factory=list)
    """图中所有边的列表（边集合）

    定义节点间的连接关系和条件约束。
    与 nodes.depends_on 应保持一致（冗余设计）。
    """

    def get_node(self, node_id: str) -> Optional[DAGNode]:
        """根据 ID 查找节点定义

        【实现方式】线性遍历，时间复杂度 O(N)。

        【为什么不用 dict 索引？】
        当前业务流程节点数 <10，线性遍历的绝对耗时微不足道（<1μs）。
        如果后续出现节点数 >50 的复杂流程，应在 __post_init__ 中
        构建 {node_id: DAGNode} 字典提供 O(1) 查找。
        YAGNI: 过早优化是万恶之源。

        Args:
            node_id: 要查找的节点唯一标识

        Returns:
            DAGNode 实例（找到时），或 None（未找到时）
        """
        for n in self.nodes:
            if n.node_id == node_id:
                return n
        return None

    def get_next_nodes(self, node_id: str) -> list[str]:
        """获取指定节点的所有直接后继节点ID

        遍历边集合，找出所有以 node_id 为起点的边的终点。

        【返回的是"拓扑上可能的"后继，不考虑条件过滤】
        条件边的过滤应在执行引擎层面处理（evaluate condition 后决定是否激活）。
        本方法是纯粹的图结构查询。

        【并行扇出时的行为】
        如果 node_id 有多条出边（如 execute_refund → write_crm 和 notify_user），
        返回多个后继节点ID。引擎可以决定并行执行它们。

        【空返回值的含义】
        返回空列表 = 该节点没有后继 = 图的叶子节点（终止点）。

        Args:
            node_id: 起始节点ID

        Returns:
            后继节点ID列表（可能为空 = 终止节点，可能多个 = 并行/分支）
        """
        return [e.to_node for e in self.edges if e.from_node == node_id]

    def get_ready_nodes(self, completed: set[str]) -> list[str]:
        """获取所有依赖已满足的可执行节点 — DAG调度的核心算法

        这是图调度引擎的核心方法。它实现了经典的 DAG 拓扑排序调度逻辑
        （类似 Kahn's 算法的在线增量版本）：

        算法：
          对于每个节点，如果：
            1. 它尚未完成（不在 completed 集合中）
            2. 它的所有前驱依赖都已完成（depends_on ⊆ completed）
          则该节点"就绪"，可以被执行。

        【时间复杂度】O(N × D)
        N = 节点总数，D = 最大依赖列表长度
        对于当前规模（N<10, D<3），单次调用 <1μs。

        【并行调度含义】
        返回的 ready 列表可能包含多个节点。在退款流程中：
          execute_refund COMPLETED 后 →
          write_crm 和 notify_user 同时满足依赖 →
          两者都在 ready 列表中 →
          引擎可以并行执行它们

        【边界情况】
        - completed = ∅ 且入口节点 depends_on = []
          → 入口节点就绪（all([]) = True）
        - 所有节点都在 completed 中
          → 返回空列表，表示图执行完毕
        - 某节点依赖了一个 FAILED 的节点
          → FAILED 不在 completed 中，该节点永远不会就绪
          → 这实际上实现了"失败传播"语义

        Args:
            completed: 已完成节点ID的集合（通常来自 DAGExecution.completed_nodes）

        Returns:
            就绪可执行的节点ID列表。空列表表示无法继续推进。
        """
        ready = []
        for node in self.nodes:
            # 跳过已完成的节点（不重复执行）
            if node.node_id in completed:
                continue
            # 检查所有前驱依赖是否都已完成
            # Python 的 all() 对空迭代器返回 True：
            #   all([]) = True → 无依赖的入口节点天然满足条件
            if all(dep in completed for dep in node.depends_on):
                ready.append(node.node_id)
        return ready


# ═══════════════════════════════════════════════════════════════════════════════════
# Part 3: 运行态数据结构 — DAG 执行实例
# ═══════════════════════════════════════════════════════════════════════════════════


@dataclass
class NodeExecution:
    """节点执行状态 — 记录单个节点在一次 DAG 运行中的动态信息

    对应设计文档 docs/03-orchestration-design.md §8 Checkpoint 中 node_runs 的一个元素。

    这是节点的「运行态」——记录"这个节点当前执行到哪了"。
    每个节点在每次 DAGExecution 中有且仅有一个 NodeExecution 实例。

    【与 DAGNode 的职责分离】
    ┌─────────────────────────────────────────────────────────┐
    │ DAGNode (静态定义)        │ NodeExecution (运行态)       │
    ├─────────────────────────────────────────────────────────┤
    │ node_id, type, name      │ node_id (关联到定义)         │
    │ handler, timeout, retries│ status (当前状态)            │
    │ depends_on, condition    │ result (执行结果)            │
    │ compensate_handler       │ error (错误信息)             │
    │ metadata                 │ started_at, completed_at     │
    │                          │ retry_count (已重试次数)      │
    └─────────────────────────────────────────────────────────┘

    【持久化 — 设计文档 §8】
    断点恢复时，此结构的所有字段需要被完整保存：
      - 已 COMPLETED 的节点：恢复后凭 idempotency_key 跳过不重复执行
      - RUNNING 的节点：恢复后需重新执行（可能导致重复 — 需幂等保护）
      - WAITING 的节点：恢复后继续等待外部输入
    """

    node_id: str
    """对应的节点定义ID（关联 DAGNode.node_id）"""

    status: NodeStatus = NodeStatus.PENDING
    """当前执行状态 — 初始为 PENDING"""

    result: Any = None
    """执行结果 — handler 的返回值

    成功时存储 handler 返回的数据。类型为 Any 以适配不同 handler 的返回格式。
    后续节点可通过 global_state 或直接查询此字段获取前驱结果。
    """

    error: Optional[str] = None
    """错误信息 — 仅 FAILED 状态时有值

    存储异常的 str(e) 表示。用于：
      - 错误日志记录
      - 运维告警消息
      - 人工排查问题定位
      - 反馈层质检分析
    """

    started_at: Optional[float] = None
    """开始执行的时间戳(Unix epoch, float, 秒级精度)

    进入 RUNNING 状态时记录。用于：
      - 计算执行耗时（duration_ms 属性）
      - SLA 监控（单节点耗时 vs timeout_ms）
      - 性能分析（哪个节点最慢）
    """

    completed_at: Optional[float] = None
    """完成/失败的时间戳

    进入 COMPLETED 或 FAILED 状态时记录。
    与 started_at 配合计算 duration_ms。
    """

    retry_count: int = 0
    """已重试次数 — 每次执行失败且允许重试时 +1

    当 retry_count >= DAGNode.retries 时，停止重试，节点进入终态 FAILED。

    重试语义（当前实现）：
      失败 → retry_count++ → 状态重置为 PENDING → 下一轮 execute_step 重新执行
    """

    @property
    def duration_ms(self) -> float:
        """计算节点执行耗时(毫秒)

        仅在节点已完成（有 started_at 和 completed_at）时返回有效值。
        用于性能监控和 SLA 统计。

        对应设计文档 §10：反馈层收集 "node_started/succeeded/failed" 事件流，
        其中 duration 是关键度量指标。

        Returns:
            执行耗时(毫秒)。节点未完成时返回 0。
        """
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at) * 1000
        return 0


@dataclass
class DAGExecution:
    """DAG执行实例（运行态）— 一次完整 DAG 流程执行的全部状态快照

    对应设计文档 docs/03-orchestration-design.md §8 Checkpoint 结构 和 §2.1 中的
    "定义与运行分离"原则中的运行侧。

    【核心概念】
    DAGDefinition(模板) + DAGExecution(实例) = 完整的可恢复执行上下文
    - 模板提供图结构（节点、边、拓扑）
    - 实例记录运行进度（哪些节点完成了、全局状态是什么）

    【状态机 — 设计文档 §2.3 图实例(GraphRun)状态】
    ┌─────────┐     ┌─────────┐     ┌───────────┐
    │ running │────▶│ waiting │────▶│  running  │──▶ completed
    └─────────┘     └────┬────┘     └───────────┘       │
         │               │                              ├──▶ failed
         │          (用户/人工确认)                       └──▶ compensating
         │
         └───────────────────────────────────────────────────▶ failed

    【全局状态 vs 设计文档的 Channel+Reducer（§2.2）】
    设计文档定义了精细的状态管理机制：
      - 每个状态字段是一个"通道(channel)"
      - 每个通道有"归并规则(reducer)"：last/merge/append/union/accumulate
      - 并行节点写入时按 reducer 合并，避免竞态

    本实现简化为一个平坦的 dict (global_state)：
      trade-off:
        + 简单直观，开发/调试方便
        + 当前并行节点（write_crm/notify_user）写不同 key，无冲突
        - 失去了并行安全保障（如果两个并行节点写同一 key，后者覆盖前者）
        - 失去了类型安全和schema校验

    【持久化策略 — 设计文档 §8】
    - 持久化时机：policies.checkpoint 控制（every_node/on_io/manual）
    - 短期 state → Redis（低延迟，支持高并发热恢复）
    - 长流程 Checkpoint → PostgreSQL（持久可靠，支持查询和审计）
    - 恢复语义：从 cursor(即当前进度) 续跑；已完成节点凭幂等键跳过
    - 本 MVP 为纯内存实现，生产部署需接入持久化适配器
    """

    run_id: str
    """执行实例唯一ID

    格式: "{graph_id}_{session_id}_{timestamp}"
    示例: "refund_flow_sess_abc123_1717200000"

    用途：
      - 日志关联（所有相关日志打此 run_id）
      - Checkpoint 主键（持久化/恢复的唯一标识）
      - 幂等去重（防止同一请求创建重复实例）

    生产环境建议：用 UUID v4 或雪花算法替代 timestamp，
    避免高并发下的 timestamp 碰撞。
    """

    graph_id: str
    """关联的 DAG 定义 ID — 指向 DAGDefinition.graph_id

    恢复时需配合 version 锁定具体版本的图模板（设计文档 §9）。
    """

    session_id: str
    """所属会话 ID — 关联用户会话上下文

    一个 session 可能触发多个 DAGExecution：
      - 用户先查价保 → 创建 price_protect_flow execution
      - 发现不符合 → 改为申请退款 → 创建 refund_flow execution
    """

    node_states: dict[str, NodeExecution] = field(default_factory=dict)
    """所有节点的执行状态映射 {node_id: NodeExecution}

    在 create_execution() 时初始化，覆盖图中每个节点。
    引擎执行过程中原地修改各节点的状态。
    """

    global_state: dict[str, Any] = field(default_factory=dict)
    """全局共享状态字典 — 节点间的数据传递通道

    对应设计文档 §2.2 GraphState 的简化实现。
    handler 函数接收此 dict 作为输入，可以读取前驱节点的结果或原始参数。

    典型内容：
      {
          "order_id": "ORD_12345",        # 意图层提取的槽位
          "user_id": "USR_67890",         # 用户标识
          "amount": 199.00,               # 退款金额
          "reason": "质量问题",            # 退款原因
          "order_detail": {...},           # query_order 节点的结果
          "risk_result": "PASS",          # risk_check 节点的结果
      }
    """

    status: str = "running"  # running / completed / failed / waiting
    """DAG 整体执行状态

    可能的取值和含义：
      - "running": 正在执行中，有节点在处理
      - "completed": 所有节点执行完毕，流程正常结束
      - "failed": 有节点失败且重试耗尽，流程异常终止
      - "waiting": 遇到 HITL 节点，等待外部输入（可挂起持久化）
    """

    created_at: float = field(default_factory=time.time)
    """执行实例创建时间戳(Unix epoch)

    用于 SLA 监控：当前时间 - created_at > sla_ms 时触发告警/降级。
    对应设计文档 §2.1 policies.sla_ms。
    """

    @property
    def completed_nodes(self) -> set[str]:
        """获取所有已成功完成的节点ID集合

        这是 DAG 调度的核心输入——get_ready_nodes() 以此判断后继节点的依赖是否满足。

        只有状态为 COMPLETED 的节点才被视为"已完成"。
        FAILED/WAITING/SKIPPED 的节点不算完成，不会解锁下游依赖。
        这意味着：一个节点失败后，所有依赖它的后续节点将永远无法执行（失败传播）。

        Returns:
            已完成节点ID的集合
        """
        return {nid for nid, ns in self.node_states.items()
                if ns.status == NodeStatus.COMPLETED}

    @property
    def failed_nodes(self) -> set[str]:
        """获取所有已失败的节点ID集合

        用途：
          1. 判断是否需要触发 Saga 补偿链（设计文档 §6）
          2. 生成错误报告/告警
          3. 决策是否升级人工
          4. 反馈层质检分析

        Returns:
            已失败节点ID的集合
        """
        return {nid for nid, ns in self.node_states.items()
                if ns.status == NodeStatus.FAILED}


# ═══════════════════════════════════════════════════════════════════════════════════
# Part 4: 预置流程模板（DAG定义工厂函数）
# ═══════════════════════════════════════════════════════════════════════════════════
# 对应设计文档 §5 "完整范例：退款流程图(refund_flow)" 和业务需求中的价保场景。
#
# 这些工厂函数生成静态 DAGDefinition 模板。生产环境中：
#   - 模板定义存储在数据库中（支持版本化和灰度 §9）
#   - 运营在低代码编排台维护（增删节点/边/条件 → 保存触发 §7 校验 → 发布）
#   - 当前硬编码为代码中的工厂函数（MVP 阶段足够）
# ═══════════════════════════════════════════════════════════════════════════════════


def create_refund_dag() -> DAGDefinition:
    """创建退款流程 DAG 模板 — 7步完整链路，含 HITL 人工确认

    对应设计文档 docs/03-orchestration-design.md §5 完整范例：退款流程图。
    这是系统最复杂、最重要的核心流程。

    【拓扑结构 — 对应设计文档 §5.1】

    ┌─────────────┐
    │ query_order │  ← 入口：查询订单详情
    └──────┬──────┘
           │ (顺序边)
    ┌──────▼──────────┐
    │ validate_refund │  ← 规则引擎：校验退款资格
    └──────┬──────────┘
           │ (顺序边)
    ┌──────▼──────┐
    │ risk_check  │  ← 风控系统：检查欺诈风险
    └──────┬──────┘
           │ (顺序边)
    ┌──────▼──────────┐
    │ user_confirm    │  ← ★ HITL节点：流程在此挂起等待用户确认
    │ (HITL, 5min超时) │
    └──────┬──────────┘
           │ (用户确认后恢复)
    ┌──────▼──────────┐
    │ execute_refund  │  ← ★ 核心不可逆操作：调支付网关退款
    │ (compensate:    │     配有 Saga 补偿函数
    │  cancel_refund) │
    └──────┬──────────┘
           │ (并行扇出: FORK)
     ┌─────┴──────┐
     │            │
    ┌▼────────┐  ┌▼────────────┐
    │write_crm│  │ notify_user │  ← 两个并行分支
    └─────────┘  └─────────────┘    CRM回写 + 用户通知

    【设计文档 §5.4 覆盖的关键设计点】
    1. 不可逆节点(execute_refund)前置 HITL(user_confirm)
       → 满足 §7 合规校验："side_effect=irreversible 前驱必含 HITL"
    2. write_crm / notify_user 并行且 on_error=fallback
       → CRM失败不阻塞用户通知（最终一致性）
    3. execute_refund 有 compensate_handler
       → Saga 补偿挂钩（§6），异常时可撤销退款
    4. execute_refund 应有 idempotency_key
       → 防重复退款（key="${session_id}:N5:${order_id}:${amount}"）

    【与设计文档完整版的差异】
    设计文档 §5.1 还包含以下节点/边，本 MVP 未实现：
      - N_ESC(转人工/澄清): 订单不存在时的异常分支
      - N_HUMAN(人工审核): 风控拦截时升级到人工审核组
      - N_ABANDON(放弃): HITL 超时/用户取消
      - N8(生成回复): LLM 生成最终回复文案
      - JOIN(汇聚): 等待 write_crm + notify_user 都完成（或任一完成）
      - DEFAULT/ESCALATE/INTERRUPT 边类型
    这些将在后续迭代中按需补充。

    Returns:
        退款流程的 DAGDefinition 静态模板实例
    """
    return DAGDefinition(
        graph_id="refund_flow",
        name="退款全流程",
        version="1.0.0",
        entry_node="query_order",  # 执行从"查询订单"开始
        nodes=[
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # Step 1: 查询订单
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # 设计文档 §5.2: N1, type=TOOL, side_effect=read, 可重试=是
            # 目的: 获取订单详情（金额/状态/商品/时间等），作为后续判断的数据基础
            # 入口节点: depends_on=[] → 无前驱依赖，引擎启动时第一个执行
            # 幂等性: read 操作天然幂等，失败可安全重试（默认3次）
            # 输出到 global_state: order_detail, order_status, amount, ...
            DAGNode(node_id="query_order", node_type=NodeType.ACTION,
                    name="查询订单", handler="order_service.query"),

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # Step 2: 校验退款资格
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # 设计文档 §5.2: N2, type=ROUTER, side_effect=none, 可重试=是
            # 目的: 通过规则引擎判断该订单是否满足退款条件
            # 校验规则（由 rule_engine 维护，可动态配置）：
            #   - 是否在退款期限内（通常7-30天）
            #   - 是否已退过款（防重复退款）
            #   - 商品类型是否支持退款（如虚拟商品可能不可退）
            #   - 订单状态是否允许退款（如已发货的订单需先退货）
            # 无副作用，纯规则计算
            DAGNode(node_id="validate_refund", node_type=NodeType.ACTION,
                    name="校验退款资格", handler="rule_engine.validate_refund",
                    depends_on=["query_order"]),

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # Step 3: 风控校验
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # 设计文档 §5.2: N3, type=TOOL, side_effect=read, on_error=escalate
            # 目的: 调用风控系统检查本次退款是否有欺诈风险
            # 检查维度:
            #   - 用户历史退款频率（异常高频 → 可能薅羊毛）
            #   - 退款金额异常（远超正常范围）
            #   - 设备指纹/IP地址/地理位置异常
            #   - 黑名单匹配
            # 输出: risk_decision (PASS/REJECT/REVIEW)
            # 设计文档中: 风控 REJECT → ESCALATE 边 → 转人工审核组（本MVP简化省略）
            DAGNode(node_id="risk_check", node_type=NodeType.ACTION,
                    name="风控校验", handler="risk_service.check",
                    depends_on=["validate_refund"]),

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # Step 4: 用户二次确认 (HITL - Human In The Loop)
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # 设计文档 §5.2: N4, type=HITL, requires_approval=True
            #
            # ★ 这是流程中的关键控制节点 ★
            #
            # 执行引擎遇到 HITL 节点时的行为：
            #   1. 将节点状态标记为 WAITING
            #   2. 将整图状态设为 "waiting"
            #   3. 停止继续推进后续节点
            #   4. 将控制权返回给上层调用方
            #
            # 上层（会话管理器）收到 waiting 后：
            #   1. 持久化 execution 到存储（Redis/PG）
            #   2. 向前端发送确认 UI："确认退款 ¥199.00 到原支付方式？[确认/取消]"
            #   3. 释放服务器资源（不占用线程等待）
            #
            # 用户确认后的恢复流程：
            #   1. 从存储恢复 execution
            #   2. 将 user_confirm 节点标记为 COMPLETED
            #   3. 将 execution.status 设回 "running"
            #   4. 继续调用 execute_step 推进
            #
            # 超时设计: timeout_ms=300000 (5分钟)
            # 设计文档 §5.3: "N4 → N_ABANDON: 30min 无回应 / 用户取消"
            # 当前 MVP: 超时处理由上层会话管理器的定时器负责
            #
            # ★ 为什么必须有这个节点？★
            # 设计文档 §7 合规校验规则:
            # > "side_effect=irreversible 节点前驱路径必含 HITL"
            # 下一步 execute_refund 是不可逆操作（退钱），
            # 法规和平台规则要求必须获得用户明确确认。
            DAGNode(node_id="user_confirm", node_type=NodeType.HITL,
                    name="用户二次确认", timeout_ms=300000,
                    depends_on=["risk_check"]),

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # Step 5: 发起退款 — 核心不可逆操作
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # 设计文档 §5.2: N5, type=TOOL, side_effect=irreversible
            #   compensation=C5(撤销退款), 可重试=是(幂等)
            #
            # ★ 这是整个流程中唯一的不可逆操作 ★
            # 调用支付网关实际执行退款 → 钱退到用户账户 → 不可自动收回
            #
            # 幂等性保障（设计文档 §5.4）:
            #   idempotency_key = "${session_id}:execute_refund:${order_id}:${amount}"
            #   作用: 网络超时重试时，支付网关凭此 key 识别重复请求并返回缓存结果
            #   缺失风险: 没有幂等键 → 重试 = 可能退两次钱 → 严重资损！
            #   当前 MVP 简化: 幂等键未实现，依赖 handler 层面保障
            #
            # Saga 补偿（设计文档 §6）:
            #   compensate_handler = "payment_gateway.cancel_refund"
            #   触发场景: 退款成功但后续流程出现严重不一致需要整体回滚
            #   实际情况: 退款补偿（追回已退的钱）通常需要人工介入
            #   补偿函数要求: 幂等、有审计日志、失败升级人工
            DAGNode(node_id="execute_refund", node_type=NodeType.ACTION,
                    name="发起退款", handler="payment_gateway.refund",
                    depends_on=["user_confirm"],
                    compensate_handler="payment_gateway.cancel_refund"),

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # Step 6: 回写CRM — 退款后更新客户关系管理系统
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # 设计文档 §5.2: N6, type=TOOL, side_effect=write
            #   on_error=fallback(补偿队列，不阻塞主流程)
            #
            # 目的: 在CRM中记录退款事件（客服工单、退款金额、原因等）
            # 这确保后续人工客服查看用户记录时能看到退款历史。
            #
            # ★ 并行执行 ★
            # depends_on=["execute_refund"] — 与 notify_user 共同依赖 execute_refund
            # 当 execute_refund COMPLETED 后，两者同时就绪，引擎可并行执行
            # 对应设计文档 §5.1: "N5 → {N6, N7} | FORK | 并行扇出"
            #
            # ★ 容错策略 ★
            # 设计文档: on_error=fallback(补偿队列)
            # 含义: CRM 写入失败不应阻塞用户侧体验（退款已成功了！）
            # 处理方式: 失败 → 进入异步补偿队列 → 定时重试 → 最终对账修复
            # 本 MVP: 简化为重试3次后标记 FAILED（未实现 fallback 降级路径）
            DAGNode(node_id="write_crm", node_type=NodeType.ACTION,
                    name="回写CRM", handler="crm_service.update",
                    depends_on=["execute_refund"]),

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # Step 7: 通知用户 — 退款后推送通知
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # 设计文档 §5.2: N7, type=TOOL, side_effect=write
            #   on_error=fallback(多通道降级)
            #
            # 目的: 通过消息通道告知用户退款结果（"退款 ¥199.00 已成功"）
            #
            # ★ 并行执行 ★ — 与 write_crm 同时进行
            #
            # ★ 多通道降级策略（设计文档提及）★
            # 优先级从高到低:
            #   1. App Push（即时推送，用户可能在 App 内）
            #   2. 短信（用户不在线时的可靠通知）
            #   3. 邮件（最终兜底通道）
            # 全部失败 → 人工跟进队列（确保用户知道退款结果）
            # 本 MVP: 简化为单一 notify_service.push 调用
            DAGNode(node_id="notify_user", node_type=NodeType.ACTION,
                    name="通知用户", handler="notify_service.push",
                    depends_on=["execute_refund"]),
        ],
        edges=[
            # ── 主链路: 顺序边，构建严格的前置条件递进关系 ──
            # query_order → validate_refund:
            #   订单存在 → 才有资格可验
            DAGEdge(from_node="query_order", to_node="validate_refund"),

            # validate_refund → risk_check:
            #   资格通过 → 才值得做风控（不合格的直接拒绝，节省风控资源）
            DAGEdge(from_node="validate_refund", to_node="risk_check"),

            # risk_check → user_confirm:
            #   风控通过 → 才让用户确认（风控拒绝的不给确认机会）
            DAGEdge(from_node="risk_check", to_node="user_confirm"),

            # user_confirm → execute_refund:
            #   用户确认 → 才执行退款（这是合规强制要求）
            DAGEdge(from_node="user_confirm", to_node="execute_refund"),

            # ── 并行扇出边: execute_refund 完成后同时走两条路 ──
            # 对应设计文档 §5.3: "N5 → {N6,N7} | FORK | 并行扇出"
            # execute_refund → write_crm:  退款成功 → 记录到CRM
            DAGEdge(from_node="execute_refund", to_node="write_crm"),
            # execute_refund → notify_user: 退款成功 → 通知用户
            DAGEdge(from_node="execute_refund", to_node="notify_user"),
        ],
    )


def create_price_protect_dag() -> DAGDefinition:
    """创建价保（价格保护）流程 DAG 模板 — 4步快速通道，全自动无 HITL

    价保(Price Protection)：用户购买商品后，在保价期内商品降价了，
    系统计算差价并告知用户是否可以退还差额。

    【为什么价保是"快速通道"？— 与退款流程的核心差异】

    ┌────────────────┬──────────────────┬─────────────────────┐
    │ 维度           │ 退款流程          │ 价保流程             │
    ├────────────────┼──────────────────┼─────────────────────┤
    │ 节点数         │ 7                │ 4                   │
    │ HITL 确认      │ 有（必须）        │ 无（全自动）         │
    │ 不可逆操作     │ 有(退款)          │ 无（仅查询+计算）    │
    │ Saga 补偿      │ 有               │ 无                  │
    │ 风控校验       │ 有               │ 无                  │
    │ 并行分支       │ 有(CRM+通知)     │ 无（纯线性）         │
    │ 典型耗时       │ 秒级(含等待分钟级)│ <3秒               │
    │ SLA 要求       │ 宽松(有HITL等待) │ 严格(<3s 端到端)    │
    └────────────────┴──────────────────┴─────────────────────┘

    价保不需要 HITL 的原因：
      - 本阶段只是"查询+计算+告知"，不产生资金变动
      - 没有不可逆操作（不需要用户确认"我同意查价格"）
      - 如果后续需要自动退差价，应该创建一个新的退款类 DAG

    【拓扑结构 — 纯线性链（最简 DAG 形态）】

    ┌─────────────┐    ┌───────────┐    ┌───────────────┐    ┌──────────────┐
    │ query_order │───▶│ calc_diff │───▶│ validate_rule │───▶│ reply_result │
    └─────────────┘    └───────────┘    └───────────────┘    └──────────────┘
      查询订单          计算差价          校验规则             模板回复

    Returns:
        价保流程的 DAGDefinition 静态模板实例
    """
    return DAGDefinition(
        graph_id="price_protect_flow",
        name="价保快速通道",
        version="1.0.0",
        entry_node="query_order",  # 同样从查询订单开始（获取购买价格和时间）
        nodes=[
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # Step 1: 查询订单
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # handler 与退款流程共用 "order_service.query"（服务复用）
            # 获取: 订单ID、商品SKU、购买价格、购买时间、订单状态
            # 这些信息是后续计算差价和校验规则的基础
            DAGNode(node_id="query_order", node_type=NodeType.ACTION,
                    name="查询订单", handler="order_service.query"),

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # Step 2: 计算差价
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # 核心计算逻辑：当前售价 - 购买价格 = 差价（应退金额）
            # 复杂性在于：
            #   - 多种优惠叠加（优惠券、满减、积分抵扣）需要还原"真实购买价格"
            #   - 当前价格可能也有活动（需取"最低可比价格"）
            #   - 多规格商品需要精确匹配到SKU级别
            # 输出: price_diff（差价金额）
            DAGNode(node_id="calc_diff", node_type=NodeType.ACTION,
                    name="计算差价", handler="price_protect_service.calculate",
                    depends_on=["query_order"]),

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # Step 3: 校验规则
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # 判断是否满足价保条件（与退款的 validate_refund 类似但规则不同）
            # 价保规则:
            #   - 保价期限: 购买后7天/15天/30天内（取决于商品类别）
            #   - 品类限制: 部分品类不支持价保（如限时特价品）
            #   - 申请次数: 同一订单每次价保期内限申请1次
            #   - 最低差额: 差价需 >= 阈值（如 ¥1.00）才可申请
            #   - 库存变化: 降价原因若为清仓，可能不适用价保
            # 输出: eligible(bool), reason(str), refund_amount(float)
            DAGNode(node_id="validate_rule", node_type=NodeType.ACTION,
                    name="校验规则", handler="price_protect_service.validate",
                    depends_on=["calc_diff"]),

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # Step 4: 返回结果 (TEMPLATE 节点)
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # 使用预定义模板直接生成回复，不调用 LLM
            #
            # 模板示例（根据 validate_rule 结果选择）:
            #   符合: "您的订单{order_id}符合价保条件，差价¥{diff}将在1-3个工作日退回原支付方式"
            #   不符合: "很抱歉，{reason}，暂时无法为您申请价保"
            #   已过期: "您的订单已超出{days}天保价期限"
            #
            # 为什么用 TEMPLATE 而不是 LLM?
            #   1. 确定性: 价保结果是明确的是/否 + 金额，无需创造性生成
            #   2. 速度: 零推理延迟（~0ms vs LLM ~500ms-2s）
            #   3. 成本: 零 token 消耗
            #   4. 风控: 无幻觉风险（不会编造虚假金额或承诺）
            #   5. 合规: 输出内容完全可审计、可预测
            DAGNode(node_id="reply_result", node_type=NodeType.TEMPLATE,
                    name="返回结果", depends_on=["validate_rule"]),
        ],
        edges=[
            # 纯线性顺序链: 每步严格依赖上一步结果
            # 无分支、无并行、无回环 — 最简 DAG 形态
            # 这也意味着总耗时 = 各步耗时之和（无法通过并行优化）
            DAGEdge(from_node="query_order", to_node="calc_diff"),
            DAGEdge(from_node="calc_diff", to_node="validate_rule"),
            DAGEdge(from_node="validate_rule", to_node="reply_result"),
        ],
    )


def create_logistics_query_dag() -> DAGDefinition:
    """创建物流查询流程 DAG 模板 — 2步快速通道，纯只读无副作用

    【场景说明】
    用户携带订单号查询物流状态（如"订单ORD12345到哪了"），
    系统先查订单确认存在，再查物流轨迹返回最新状态。

    【拓扑结构 — 纯线性链】

    ┌─────────────┐    ┌──────────────────┐
    │ query_order │───▶│ query_logistics  │
    └─────────────┘    └──────────────────┘
      查询订单           查询物流轨迹

    【与退款/价保 DAG 的区别】
    - 全部为 READ 操作，无 HITL / 无 Saga / 无不可逆
    - 耗时 < 1s（两次只读 API 调用）
    - 失败影响: 仅无法展示物流详情，不涉及资金
    """
    return DAGDefinition(
        graph_id="logistics_query_flow",
        name="物流查询快速通道",
        version="1.0.0",
        entry_node="query_order",
        nodes=[
            # Step 1: 查询订单（确认订单存在 + 获取发货状态）
            DAGNode(node_id="query_order", node_type=NodeType.ACTION,
                    name="查询订单", handler="order_service.query"),
            # Step 2: 查询物流轨迹（快递单号、位置、ETA）
            DAGNode(node_id="query_logistics", node_type=NodeType.ACTION,
                    name="查询物流轨迹", handler="logistics_service.query"),
        ],
        edges=[
            DAGEdge(from_node="query_order", to_node="query_logistics"),
        ],
    )


# ═══════════════════════════════════════════════════════════════════════════════════
# Part 5: DAG 执行引擎 — 核心调度与执行逻辑
# ═══════════════════════════════════════════════════════════════════════════════════


class DAGEngine:
    """DAG编排执行引擎 — 驱动 DAG 图的创建、调度、执行、中断与恢复

    这是规划层的核心组件，将静态的 DAGDefinition（蓝图）"跑起来"。

    【在系统架构中的角色 — 设计文档 §10 接口契约】

    ┌────────────────────────────────────────────────────────────────┐
    │                        DAGEngine                               │
    │                                                                │
    │  从意图层接收:                                                  │
    │    {intent, slots, context, user_profile}                      │
    │    → 选定 graph_id@version，初始化 execution                   │
    │                                                                │
    │  向执行层分发（每个 ACTION 节点）:                               │
    │    {handler, inputs(global_state), timeout, retry}             │
    │    ← {status, result, duration, error?}                       │
    │                                                                │
    │  向反馈层发送:                                                  │
    │    node_started / node_succeeded / node_failed 事件流          │
    │    (当前 MVP 未实现事件发射，后续通过 Observer 模式补充)          │
    └────────────────────────────────────────────────────────────────┘

    【核心设计决策】

    1. 步进式执行 (Step-by-step Execution):
       不是一次性执行到底，而是每次 execute_step() 推进一个"波次"(wave)。
       好处:
         - HITL 节点自然形成执行断点
         - 每步后可做 checkpoint 持久化
         - 便于超时/SLA 检查
         - 便于调试和可观测
       代价:
         - 调用方需要维护执行循环（run_to_completion_or_wait 封装了此循环）

    2. Handler 注册制 (Registry Pattern):
       通过 register_handler(id, func) 将执行逻辑绑定到字符串标识。
       好处:
         - DAGDefinition 可序列化/持久化（不包含 Callable）
         - 同一 handler_id 在不同环境可绑定不同实现（生产/测试/mock）
         - 支持热更新（不修改图定义即可替换实现）
       代价:
         - 需要初始化时手动注册所有 handler（增加 setup 代码）
         - 运行时可能出现"handler 未注册"的错误（需防御性处理）

    3. 简单重试 (Naive Retry):
       失败时立即重置为 PENDING 等待下一轮执行。
       好处: 实现简单，能应对临时性故障
       代价: 无退避间隔、不区分异常类型、无熔断机制
       生产环境需要: 指数退避 + 随机抖动 + 异常分类 + 熔断器

    【线程安全性】
    本实现为单线程同步模型，不保证线程安全。
    如果需要:
      - 多个 DAGExecution 并发执行: 每个用独立 DAGEngine 实例
      - 单个 DAG 内节点真正并行: 需要改用 asyncio 或线程池

    【扩展路线图（按优先级排序）】
    P0 (必须): Checkpoint 持久化（Redis/PG 适配器）
    P1 (应该): 条件边评估、超时中断、事件发射
    P2 (可以): 真正并行执行、Saga 补偿链触发、图校验
    P3 (未来): 子图嵌套、动态扇出(MAP)、回环(LOOP)
    """

    # ─────────────────────────────────────────────────────────────────────────────
    # 流程注册表 (Class Variable)
    # ─────────────────────────────────────────────────────────────────────────────
    # 意图(intent) → DAG定义工厂函数 的映射
    #
    # 这是"意图层 → 规划层"的路由表：
    #   intent_router 识别出 intent="REFUND" → 规划层查表 → 创建退款 DAG
    #
    # 对应设计文档 §10:
    #   "意图层→规划层: {intent, ...} → 选定 graph_id@version，初始化 GraphState"
    #
    # 为什么用 Callable[[], DAGDefinition] 而不是直接存 DAGDefinition?
    #   - 工厂函数每次调用返回新实例，避免多个 execution 共享同一定义对象
    #   - 支持未来的动态模板（如根据用户等级返回不同流程版本）
    #   - 如果直接存实例，修改其中一个的 nodes/edges 会影响所有引用者（虽然当前不应修改）
    #
    # 扩展方式:
    #   - 新增业务流程: 编写 create_xxx_dag() 函数，注册到此 dict
    #   - 动态加载: 从数据库读取配置，动态生成 DAGDefinition，注册到 FLOW_REGISTRY
    #   - 多版本: value 可改为 dict[version, factory] 实现版本选择
    FLOW_REGISTRY: dict[str, Callable[[], DAGDefinition]] = {
        "REFUND": create_refund_dag,              # 退款意图 → 7步退款全流程(含HITL)
        "PRICE_PROTECT": create_price_protect_dag,  # 价保意图 → 4步价保快速通道
        "LOGISTICS": create_logistics_query_dag,    # 物流意图 → 2步快速查询(有order_id时)
    }

    def __init__(self):
        """初始化 DAG 引擎

        创建空的 handler 注册表。引擎本身是无状态的（状态都在 DAGExecution 中），
        可以被多个执行实例复用。

        典型初始化代码:
        ```python
        engine = DAGEngine()

        # 注册各服务的 handler 实现
        engine.register_handler("order_service.query", order_svc.query_order)
        engine.register_handler("rule_engine.validate_refund", rule_svc.validate)
        engine.register_handler("risk_service.check", risk_svc.check)
        engine.register_handler("payment_gateway.refund", payment_svc.refund)
        engine.register_handler("crm_service.update", crm_svc.update)
        engine.register_handler("notify_service.push", notify_svc.push)
        engine.register_handler("price_protect_service.calculate", pp_svc.calc)
        engine.register_handler("price_protect_service.validate", pp_svc.validate)

        # 执行流程
        result = engine.run_to_completion_or_wait("REFUND", "session_123",
                                                  {"order_id": "ORD_456"})
        ```
        """
        # handler 注册表: {handler_id_string: actual_callable_function}
        # 这实现了设计文档中"规划层→执行层"的服务路由功能
        # handler 函数签名约定: func(global_state: dict[str, Any]) -> Any
        self._handlers: dict[str, Callable] = {}

    def register_handler(self, handler_id: str, func: Callable):
        """注册节点处理函数 — 将 handler 字符串标识绑定到实际 Callable

        当执行引擎遇到一个 ACTION 节点时，它会：
          1. 读取 node.handler（如 "order_service.query"）
          2. 在 self._handlers 中查找对应的函数
          3. 调用函数: func(execution.global_state) → result

        【设计文档对应 — §10 规划层→执行层接口】
        每个 TOOL 节点向执行层传递:
          {handler, inputs(已绑定), idempotency_key, timeout, retry}
        本实现简化: 只传 global_state 作为统一输入

        【handler 函数签名约定】
        func(global_state: dict[str, Any]) -> Any
          - 输入: 包含所有前驱节点结果和初始参数的全局状态字典
          - 输出: 任意返回值，存入 NodeExecution.result
          - 异常: 抛出 Exception 表示执行失败，引擎会处理重试

        Args:
            handler_id: 处理函数唯一标识，与 DAGNode.handler 字段值对应
                        命名规范: "service_name.method_name"
            func: 实际处理函数（签名见上）
        """
        self._handlers[handler_id] = func

    def create_execution(self, intent: str, session_id: str) -> Optional[DAGExecution]:
        """根据意图创建 DAG 执行实例 — DAG 生命周期的第一步

        流程: intent → 查注册表 → 获取 DAG 模板 → 创建运行态实例 → 初始化节点状态

        对应设计文档 §10 "意图层→规划层"接口:
          输入: {intent, intent_confidence, slots, context, user_profile}
          输出: 选定 graph_id@version，初始化 GraphState

        【run_id 生成策略】
        格式: "{graph_id}_{session_id}_{timestamp}"
        示例: "refund_flow_sess_abc123_1717200000"
        优点:
          - 包含流程类型（便于日志过滤: grep "refund_flow"）
          - 包含会话ID（便于用户维度查询）
          - 时间戳保证唯一性
        缺点:
          - 时钟回拨可能导致重复（生产环境建议改用 UUID/雪花ID）
          - 长度较长（存储效率略低）

        Args:
            intent: 用户意图标识（如 "REFUND", "PRICE_PROTECT"）
                    必须存在于 FLOW_REGISTRY 中
            session_id: 会话ID，关联用户上下文

        Returns:
            DAGExecution 实例（所有节点为 PENDING 状态，ready to execute）
            如果 intent 未在 FLOW_REGISTRY 中注册，返回 None
        """
        # 在注册表中查找对应的 DAG 模板工厂函数
        factory = self.FLOW_REGISTRY.get(intent)
        if not factory:
            # 未知意图 → 规划层无法处理
            # 上层应该: 走闲聊/FAQ路径，或要求意图层重新分类
            return None

        # 调用工厂函数生成 DAG 静态定义
        dag_def = factory()

        # 生成唯一运行实例ID
        run_id = f"{dag_def.graph_id}_{session_id}_{int(time.time())}"

        # 创建运行态容器
        execution = DAGExecution(
            run_id=run_id,
            graph_id=dag_def.graph_id,
            session_id=session_id,
        )

        # 初始化所有节点的执行状态
        # 每个节点一开始都是 PENDING（等待依赖满足后被调度）
        # 这一步确保 node_states 字典覆盖图中所有节点
        for node in dag_def.nodes:
            execution.node_states[node.node_id] = NodeExecution(node_id=node.node_id)

        return execution

    def execute_step(self, dag_def: DAGDefinition, execution: DAGExecution) -> list[str]:
        """执行一步: 找到所有可执行节点并执行 — 引擎的核心调度单元

        每次调用推进 DAG 执行一个"波次"(wave)：
        同一波次内所有就绪节点被依次处理（当前串行，未来可并行）。

        【调度算法】
        1. 获取就绪节点: get_ready_nodes(completed) → ready_list
        2. 过滤: 只取 status=PENDING 的（跳过已处理的）
        3. 对每个就绪节点按类型分派:
           ┌──────────────┬──────────────────────────────────────────┐
           │ HITL 类型    │ 标记 WAITING → 整图 "waiting" → 不执行    │
           │ 其他类型     │ RUNNING → 调用 handler → COMPLETED/FAILED │
           └──────────────┴──────────────────────────────────────────┘
        4. 完成检查: 所有节点 COMPLETED → 整图 "completed"

        【步进式设计的价值 — 对应设计文档 §8 Checkpoint】
        - HITL 节点自然形成断点: 挂起后可持久化 execution，释放资源
        - 每步后可做 checkpoint: 对应 policies.checkpoint="every_node"
        - 可观测: 外部可在步骤间插入监控/告警逻辑
        - SLA 检查: 每步后检查总耗时是否超过 sla_ms

        【并行节点的处理 — 当前实现 vs 理想实现】
        当前: for 循环顺序执行（简化实现）
          - 功能正确: 结果与并行执行相同（只是慢一些）
          - 适合 MVP: 不需要处理并发控制
        理想: asyncio.gather 或线程池真正并行
          - 需要: 并发安全的 global_state（channel+reducer §2.2）
          - 需要: 异步 handler 接口
          - 复杂度显著增加，按需引入

        【重试机制详解】
        当 handler 抛异常时:
          if retry_count < node.retries:
            → retry_count += 1
            → status = PENDING (下轮 execute_step 重新执行)
          else:
            → status = FAILED
            → execution.status = "failed" (整图失败)

        当前简化的 trade-off:
          - 无退避间隔: 立即重试，可能加重下游负载
          - 不区分异常: 4xx(业务拒绝) 和 5xx(临时故障) 一律重试
          - 无熔断: 连续失败不会停止尝试（直到重试次数耗尽）
        生产环境补充:
          - 指数退避: delay = base * 2^count + jitter
          - 异常分类: 只对 retriable_errors 重试
          - 熔断器: 连续失败 N 次后短路后续请求

        Args:
            dag_def: DAG 图定义（静态模板，提供拓扑和节点配置信息）
            execution: DAG 执行实例（运行态，会被原地修改）

        Returns:
            本步处理的节点ID列表（含成功执行和标记 WAITING 的）
            空列表表示无法继续推进（全部完成/全部阻塞/遇到 HITL）
        """
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # Step 1: 获取所有依赖已满足的候选节点
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        ready_nodes = dag_def.get_ready_nodes(execution.completed_nodes)
        executed = []

        for node_id in ready_nodes:
            # 获取节点静态定义（类型、handler、超时配置等）
            node = dag_def.get_node(node_id)
            if not node:
                # 防御性编程: ready_nodes 返回了 nodes 中不存在的 ID
                # 理论上不应发生（除非 edges/depends_on 与 nodes 不一致）
                continue

            # 获取节点运行态（当前 status、retry_count 等）
            node_state = execution.node_states[node_id]

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # Step 2: 只处理 PENDING 状态的节点
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # 非 PENDING 的情况说明:
            #   - COMPLETED: 之前已成功执行，无需重复
            #   - FAILED: 已失败且重试耗尽，不再尝试
            #   - WAITING: HITL 节点正在等待外部输入
            #   - RUNNING: 理论上不应出现（同步模型中不会有并发 RUNNING）
            #   - SKIPPED: 已被条件分支跳过
            if node_state.status != NodeStatus.PENDING:
                continue

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # Step 3a: HITL 节点特殊处理 — 挂起等待，不执行 handler
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # 对应设计文档 §3.2 HITL 节点和 §2.3 状态机中的 WAITING 状态
            #
            # 语义: 流程在此节点"暂停"，等待外部事件（用户点确认/人工审批通过）
            # 实现: 只修改状态，不调用任何函数
            #
            # 上层调用方收到 execution.status="waiting" 后的典型处理:
            #   1. checkpoint: 持久化 execution 到 Redis/PostgreSQL
            #   2. notify: 向前端/用户发送确认请求
            #   3. release: 释放服务器线程/连接资源
            #   4. wait: 等待回调（webhook/WebSocket/轮询）
            #   5. resume: 收到确认后恢复执行
            #
            # 注意: HITL 节点不需要 handler（没有处理逻辑）
            # 它的 "完成" 由外部直接修改 node_state.status = COMPLETED 触发
            if node.node_type == NodeType.HITL:
                node_state.status = NodeStatus.WAITING
                execution.status = "waiting"
                executed.append(node_id)
                # continue 而非 break:
                # 理论上 HITL 后续节点不在 ready 中（依赖 HITL 未 COMPLETED）
                # 但同一波次中如果有其他独立的就绪节点，也应该被处理
                # （在退款流程中不会出现此情况，但保持通用性）
                continue

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # Step 3b: 普通节点执行 — 调用 handler 并处理结果
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # 标记开始执行: PENDING → RUNNING
            node_state.status = NodeStatus.RUNNING
            node_state.started_at = time.time()

            try:
                # 尝试查找并调用注册的 handler 函数
                if node.handler and node.handler in self._handlers:
                    # ── handler 已注册: 调用实际业务逻辑 ──
                    # 传入 global_state 让 handler 读取输入参数和前驱节点结果
                    # handler 也可以写入 global_state 供后续节点使用
                    #
                    # 对应设计文档 §10 "规划层→执行层" 接口:
                    #   传递: {handler, inputs(已绑定), idempotency_key, timeout, retry}
                    #   本实现简化为: 只传 global_state dict
                    result = self._handlers[node.handler](execution.global_state)
                    node_state.result = result
                else:
                    # ── handler 未注册: 模拟执行成功（开发/测试阶段的 fallback）──
                    # 这个设计允许:
                    #   - DAG 定义先行（先定义流程拓扑）
                    #   - handler 后接（逐步接入真实服务）
                    #   - 测试驱动（验证图拓扑正确性，无需真实服务）
                    #
                    # 生产环境的 trade-off:
                    #   应改为: handler 未注册 → 直接 raise HandlerNotFoundError
                    #   避免: 线上误跑到未注册的节点却静默"成功"
                    node_state.result = {"status": "ok", "handler": node.handler}

                # ── 执行成功: RUNNING → COMPLETED ──
                node_state.status = NodeStatus.COMPLETED
                node_state.completed_at = time.time()
                executed.append(node_id)

            except Exception as e:
                # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                # Step 4: 执行失败处理 — 记录错误，决定重试或标记失败
                # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                node_state.status = NodeStatus.FAILED
                node_state.error = str(e)
                node_state.completed_at = time.time()

                # ── 重试判断 ──
                # 对应设计文档 §3.1: "retry: RetryPolicy (仅幂等节点可重试)"
                #
                # 重试决策逻辑:
                #   retry_count < node.retries → 还有重试机会
                #     → 计数器+1
                #     → 重置为 PENDING（下一轮 execute_step 会重新拾取执行）
                #   retry_count >= node.retries → 重试次数耗尽
                #     → 保持 FAILED
                #     → 整图标记 "failed"
                #
                # 【当前实现的局限和生产环境改进方向】
                # 1. 退避策略: 当前立即重试 → 应加指数退避
                #    delay_ms = min(base_delay * 2^count + random(0, jitter), max_delay)
                # 2. 异常分类: 当前所有异常都重试 → 应区分:
                #    - Retriable: TimeoutError, ConnectionError, 5xx
                #    - Non-retriable: ValueError, 4xx(业务拒绝), AuthError
                # 3. 熔断器: 连续 N 次失败 → 短路后续请求，快速失败
                # 4. 死信队列: 最终失败的节点 → 进入人工处理队列
                if node_state.retry_count < node.retries:
                    node_state.retry_count += 1
                    # 重置为 PENDING: 下一轮 execute_step 会重新检测到此节点就绪
                    # 并再次尝试执行（因为依赖仍满足，且状态回到了 PENDING）
                    node_state.status = NodeStatus.PENDING
                else:
                    # 重试耗尽: 节点最终失败
                    # 整图标记为 "failed"
                    #
                    # 后续应触发（当前 MVP 未实现）:
                    #   1. Saga 补偿链: 逆序补偿已执行的副作用节点（§6）
                    #   2. 告警: 通知运维/运营/on-call
                    #   3. 升级: 如果 on_error=escalate，转人工客服
                    #   4. 降级: 如果 on_error=fallback，走备选路径
                    #
                    # 对应设计文档 §3.3 on_failure 策略:
                    #   当前固定为 "fail"（整图失败）
                    #   设计文档支持: fail/skip/fallback/escalate 四种
                    execution.status = "failed"

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # Step 5: 完成检查 — 所有节点 COMPLETED → 整图完成
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 对应设计文档 §2.3 图实例状态: RUNNING → COMPLETED
        #
        # 判断条件: ALL 节点状态为 COMPLETED
        # 这意味着:
        #   - 并行分支的所有节点都必须完成（类似 JOIN(all) 策略）
        #   - 如果任何节点 FAILED/WAITING，此条件不满足
        #
        # 与设计文档 §5.3 的差异:
        #   设计文档中 JOIN 支持三种策略: all/any/quorum=k
        #   当前实现固定为 "all"（最严格）
        #   "any"策略（任一完成即继续）需要额外实现：
        #     - 标记未完成的并行分支为 SKIPPED
        #     - 或引入 terminal_nodes 概念
        if all(ns.status == NodeStatus.COMPLETED
               for ns in execution.node_states.values()):
            execution.status = "completed"

        return executed

    def run_to_completion_or_wait(self, intent: str, session_id: str,
                                   initial_state: dict = None) -> Optional[DAGExecution]:
        """执行 DAG 直到完成或遇到 HITL 等待 — 一站式高层 API

        封装了完整的"创建实例 → 注入状态 → 循环调度 → 返回结果"流程。
        调用方无需手动管理 execute_step 循环。

        【两种典型使用场景】

        场景1 — 价保查询（快速通道，无 HITL）:
        ```python
        result = engine.run_to_completion_or_wait(
            "PRICE_PROTECT", "sess_123", {"order_id": "ORD_456"}
        )
        assert result.status == "completed"  # 全程无阻塞，直接跑完
        # result.node_states["reply_result"].result 包含最终回复
        ```

        场景2 — 退款申请（有 HITL，需要两段执行）:
        ```python
        # 第一段: 执行到 HITL 挂起
        result = engine.run_to_completion_or_wait(
            "REFUND", "sess_789", {"order_id": "ORD_012", "amount": 199.0}
        )
        assert result.status == "waiting"  # 停在 user_confirm 节点

        # 持久化 + 等待用户确认...
        save_to_redis(result)
        send_confirm_ui_to_user(result)

        # 第二段: 用户确认后恢复执行
        # ⚠️ 不能再调 run_to_completion_or_wait（它会创建新实例）
        # 需要手动恢复并继续:
        result = load_from_redis(run_id)
        result.node_states["user_confirm"].status = NodeStatus.COMPLETED
        result.status = "running"
        dag_def = create_refund_dag()
        while result.status == "running":
            engine.execute_step(dag_def, result)
        assert result.status == "completed"  # 退款+CRM+通知 全部完成
        ```

        【max_iterations = 20 的设计考量】
        对应设计文档 §2.1 policies.max_total_steps: "防失控的总步数上限"

        为什么是 20?
          - 退款流程: 7节点，线性链最多7次迭代
          - 价保流程: 4节点，最多4次迭代
          - 加上重试场景: 每个节点最多重试3次 = 最坏 7*3 = 21 次
          - 设20次: 覆盖正常流程绰绰有余，又不会让异常情况无限循环
          - 如果超过20次仍未结束: 大概率是逻辑bug，应该强制退出并告警

        防护的异常场景:
          - 图定义存在环（不应出现在 DAG 中，但防御性编程）
          - 重试风暴（某节点一直失败一直重试）
          - 引擎bug导致节点状态流转异常

        Args:
            intent: 用户意图标识（如 "REFUND", "PRICE_PROTECT"）
                    必须存在于 FLOW_REGISTRY 中
            session_id: 会话ID，关联用户上下文
            initial_state: 初始全局状态（可选）
                          通常包含意图层提取的槽位信息:
                          {"order_id": "ORD_123", "user_id": "USR_456",
                           "amount": 199.00, "reason": "质量问题"}
                          对应设计文档 §10: 意图层传入的 slots/context

        Returns:
            DAGExecution 实例，可能处于以下状态:
              - "completed": 所有节点执行完毕（如价保快速通道）
              - "waiting": 遇到 HITL 节点挂起（如退款的用户确认）
              - "failed": 某节点失败且重试耗尽
              - "running": 达到 max_iterations 但未结束（异常情况）
            如果 intent 未注册，返回 None
        """
        # Step 1: 创建执行实例（选模板 + 初始化节点状态）
        execution = self.create_execution(intent, session_id)
        if not execution:
            return None

        # Step 2: 注入初始全局状态（意图层提取的槽位、用户信息等）
        # 这些数据通过 global_state 字典在节点间传递
        # handler 函数通过 global_state["order_id"] 等方式读取输入参数
        if initial_state:
            execution.global_state.update(initial_state)

        # Step 3: 获取 DAG 定义（用于 execute_step 的调度依据）
        # 注意: 这里再次调用 factory() 生成 DAGDefinition
        # 与 create_execution 中使用的是相同模板的不同实例（防御性设计）
        # 优化空间: 可以缓存 DAGDefinition（它是不可变的）
        factory = self.FLOW_REGISTRY.get(intent)
        if not factory:
            return None
        dag_def = factory()

        # Step 4: 主执行循环 — 循环推进直到无法继续
        # max_iterations: 防失控安全阀（对应设计文档 §2.1 max_total_steps）
        max_iterations = 20
        for _ in range(max_iterations):
            # 推进一步: 执行所有当前就绪的节点
            executed = self.execute_step(dag_def, execution)

            # 退出条件判断:
            #   条件1: executed 为空 → 没有节点被处理
            #     可能原因: 全部完成 / 全部依赖阻塞 / 全部失败
            #   条件2: status 变为终态或等待态
            #     "completed": 全部节点完成 → 正常结束
            #     "failed": 有节点最终失败 → 异常结束
            #     "waiting": 遇到 HITL → 挂起等待外部输入
            if not executed or execution.status in ("completed", "failed", "waiting"):
                break

        # 失败时触发 Saga 补偿链(设计文档 §6)
        if execution.status == "failed":
            self.compensate(dag_def, execution)

        return execution

    def bind_gateway(self, gateway, dag_def: "DAGDefinition" = None):
        """将 ToolGateway 中的工具批量绑定为 DAG handler(执行层接入)。

        对应设计文档 §3.5/§10: 规划层的 ACTION 节点 handler 字符串(如
        "order_service.query")对应执行层 ToolGateway 中的同名 tool_id。
        本方法把每个 tool_id 注册为一个闭包: 接收 global_state，调用
        gateway.call_tool，成功则把返回数据并入 global_state 并返回，失败则
        抛异常以触发引擎重试/补偿。

        Args:
            gateway: ToolGateway 实例(execution.tool_gateway)
            dag_def: 可选，仅绑定该图用到的 handler；不传则绑定网关全部工具
        """
        from ..execution.tool_gateway import ToolDefinition  # 局部导入避免循环依赖

        tool_ids = list(gateway._tools.keys())
        for tool_id in tool_ids:
            self.register_handler(tool_id, self._make_gateway_handler(gateway, tool_id))

        # 补偿用的反向工具(如撤销退款)若网关未提供，注册一个安全的占位实现
        for node in (dag_def.nodes if dag_def else []):
            ch = node.compensate_handler
            if ch and ch not in self._handlers:
                self.register_handler(ch, self._make_compensation_handler(gateway, ch))

    @staticmethod
    def _make_gateway_handler(gateway, tool_id: str) -> Callable:
        """构造单个 tool_id 的 DAG handler 闭包。"""
        def _handler(global_state: dict) -> Any:
            session_id = global_state.get("session_id", "")
            result = gateway.call_tool(tool_id, dict(global_state), session_id=session_id)
            if not result.success:
                raise RuntimeError(f"tool {tool_id} failed: {result.error}")
            # 将工具返回数据并入全局状态，供后续节点读取
            if isinstance(result.data, dict):
                global_state.update(result.data)
                global_state[f"_result.{tool_id}"] = result.data
            return result.data
        return _handler

    @staticmethod
    def _make_compensation_handler(gateway, handler_id: str) -> Callable:
        """构造补偿 handler 闭包(若网关注册了同名补偿工具则调用，否则记录待人工对账)。"""
        def _handler(global_state: dict) -> Any:
            session_id = global_state.get("session_id", "")
            if handler_id in gateway._tools:
                result = gateway.call_tool(handler_id, dict(global_state), session_id=session_id)
                return result.data if result.success else {"compensated": False, "need_manual": True}
            # 网关未提供补偿工具：标记需人工对账(设计文档 §6 最终一致/人工介入)
            return {"compensated": False, "need_manual": True, "handler": handler_id}
        return _handler

    def compensate(self, dag_def: DAGDefinition, execution: DAGExecution) -> list[str]:
        """执行 Saga 补偿链 — 逆序补偿已成功且声明了补偿函数的副作用节点

        对应设计文档 §6 Saga 补偿与图回滚语义:
          1. read/none 节点无补偿(无副作用)
          2. write/irreversible 节点若声明 compensate_handler 则逆序执行
          3. 补偿函数应幂等，失败标记需人工介入

        触发时机: execution.status == "failed" 时由 run_to_completion_or_wait 调用。
        逆序依据: 节点完成时间戳(completed_at)降序，先补偿最后成功的节点。

        Args:
            dag_def: DAG 静态定义(用于查 compensate_handler)
            execution: 运行实例(读取已完成节点、写补偿状态)

        Returns:
            被补偿的节点ID列表(按补偿执行顺序)
        """
        execution.status = "compensating"

        # 收集已成功且声明了补偿函数的节点，按完成时间逆序
        compensable = []
        for node_id in execution.completed_nodes:
            node = dag_def.get_node(node_id)
            if node and node.compensate_handler:
                ns = execution.node_states[node_id]
                compensable.append((ns.completed_at or 0.0, node_id, node.compensate_handler))
        compensable.sort(key=lambda x: x[0], reverse=True)

        compensated = []
        for _, node_id, comp_handler in compensable:
            ns = execution.node_states[node_id]
            ns.status = NodeStatus.COMPENSATING
            try:
                if comp_handler in self._handlers:
                    self._handlers[comp_handler](execution.global_state)
                compensated.append(node_id)
            except Exception as e:  # 补偿失败 → 记录，标记需人工(设计文档 §6)
                ns.error = f"compensation failed: {e}"
                execution.global_state.setdefault("_compensation_failures", []).append(node_id)

        execution.global_state["_compensated_nodes"] = compensated
        return compensated
