# 企业级智能客服系统 · 编排图（点边）结构设计

> 本文是主架构方案（`readme.md` 第 3.4 节 规划层）的深化设计，聚焦**编排图的结构定义与点（节点）边（边）设计**。所有结构以 schema/契约形式给出，属方案设计，不含实现代码。

---

## 1. 设计目标与定位

规划层把"用户意图 + 槽位"映射为一张**可执行、可持久化、可补偿、可人工介入**的任务图（Task Graph）。它是连接"理解（意图层）"与"行动（执行层）"的中枢。

设计须满足：
- **确定性可控**：跨系统副作用（退款/改单/写 CRM）由图显式编排，而非模型自由调用。
- **可中断可恢复**：长流程在任意节点可挂起（等待用户/人工/外部回调），断点续跑。
- **可补偿**：失败时按依赖逆序触发补偿（Saga）。
- **可分支可并行可回环**：支持条件分流、并行扇出/汇聚、有界循环。
- **可版本可灰度**：图定义版本化，可灰度、可回滚，运营可在低代码编排台维护。

> 图模型选型：**有向图 + 显式状态机**。结构上是有向图（DAG 为主，循环需显式声明并有界），运行时每个节点是一台小状态机；全局会话流程也由一台状态机托管，承载中断/恢复。

---

## 2. 核心结构定义（Structure Definitions）

四个一等公民：`GraphDefinition`（图定义/模板）、`GraphState`（运行态全局状态）、`Node`（点）、`Edge`（边）。定义与运行分离：`GraphDefinition` 是静态模板（可版本化），`GraphRun` 是一次执行实例。

### 2.1 GraphDefinition —— 图定义（静态模板）

```jsonc
GraphDefinition {
  graph_id: string,             // 业务流程标识, 如 "refund_flow"
  version: string,              // 语义化版本, 如 "v2.3.1", 支持灰度/回滚
  tenant_scope: string[],       // 适用租户(支持多租户差异化), ["*"] 表示全部
  entry_node: NodeId,           // 唯一入口节点
  terminal_nodes: NodeId[],     // 可终止节点集合(成功/失败/放弃)
  nodes: Node[],                // 点集合 (见 §3)
  edges: Edge[],                // 边集合 (见 §4)
  state_schema: StateSchema,    // 全局状态字段与归并规则 (见 §2.2)
  policies: {                   // 图级默认策略(节点可覆盖)
    default_timeout_ms: number,
    default_retry: RetryPolicy,
    checkpoint: "every_node" | "on_io" | "manual",
    max_total_steps: number,    // 防失控的总步数上限
    sla_ms: number              // 整图 SLA, 超时触发降级/升级
  },
  metadata: { owner, description, created_at, changelog }
}
```

### 2.2 GraphState —— 全局运行态（通道 + 归并器）

借鉴 LangGraph 的 **channel + reducer** 思想：状态由若干"通道"组成，每个通道声明**归并规则（reducer）**，决定多个节点（尤其并行节点）写入时如何合并，避免竞态。

```jsonc
StateSchema {
  channels: {
    // 通道名: { 类型, 归并规则, 是否持久化, 敏感级别 }
    session_id:    { type: "string", reduce: "last",        persist: true,  pii: false },
    intent:        { type: "string", reduce: "last",        persist: true },
    slots:         { type: "map",    reduce: "merge",       persist: true,  pii: true },   // 槽位增量合并
    context:       { type: "list",   reduce: "append",      persist: true },               // 会话上下文追加
    tool_results:  { type: "map",    reduce: "merge_by_key", persist: true },              // 各节点产出按 key 合并
    risk_flags:    { type: "set",    reduce: "union",        persist: true },               // 风险标记取并集
    cursor:        { type: "NodeId", reduce: "last",        persist: true },               // 当前所处节点
    status:        { type: "enum",   reduce: "last",        persist: true },               // 见 §2.3
    pending_hitl:  { type: "object", reduce: "last",        persist: true },               // 待人工/用户确认项
    error:         { type: "object", reduce: "last",        persist: true },
    budget:        { type: "object", reduce: "accumulate",  persist: true }                // token/成本累计
  }
}
```

归并规则（reduce）取值说明：

| reduce | 语义 | 典型通道 |
|--------|------|---------|
| `last` | 后写覆盖（最新值生效） | intent、cursor、status |
| `merge` / `merge_by_key` | 字典浅合并 / 按 key 合并 | slots、tool_results |
| `append` | 列表追加（保序） | context（对话历史） |
| `union` | 集合并集（幂等去重） | risk_flags |
| `accumulate` | 数值累加 | budget（token/成本） |

> **设计要点**：并行节点只能写**不同通道**或写带 `merge_by_key`/`union` 的通道，禁止两个并行节点对 `last` 通道写不同值（图校验期拦截，见 §7）。

### 2.3 运行态枚举（State Machine 状态）

```
图实例(GraphRun)状态:
  CREATED → RUNNING → (WAITING) → RUNNING → COMPLETED
                          ↑↓                    │
                    (用户/人工/回调)            ├→ FAILED
                                               ├→ COMPENSATING → COMPENSATED
                                               └→ ABANDONED (超时未回应)

节点(NodeRun)状态:
  PENDING → READY → RUNNING → SUCCEEDED
                       │           
                       ├→ WAITING (HITL/外部回调)
                       ├→ RETRYING → RUNNING
                       ├→ FAILED → COMPENSATING → COMPENSATED
                       └→ SKIPPED (条件未命中/被剪枝)
```

---

## 3. 点（节点）设计

### 3.1 节点通用结构（NodeSpec）

```jsonc
Node {
  id: NodeId,                   // 图内唯一
  type: NodeType,               // 见 §3.2
  name: string,                 // 人类可读
  // ---- 执行契约 ----
  handler: string,              // 绑定的能力标识(工具ID/模型ID/子图ID/规则集ID)
  inputs:  Binding[],           // 从 state 通道取值的绑定, 如 slots.order_id
  outputs: Binding[],           // 写回 state 哪些通道(及 reduce 由通道决定)
  // ---- 可靠性 ----
  timeout_ms: number,           // 覆盖图级默认
  retry: RetryPolicy,           // 重试策略(仅幂等节点可重试)
  idempotency_key: string,      // 模板表达式, 如 "${session_id}:${id}:${slots.order_id}"
  side_effect: "none" | "read" | "write" | "irreversible",
  compensation: NodeId | null,  // 失败回滚时触发的补偿节点(见 §3.3 / §6)
  // ---- 治理 ----
  risk_level: "LOW"|"MEDIUM"|"HIGH"|"CRITICAL",
  requires_approval: boolean,   // 是否强制人工/用户确认
  audit: boolean,               // 是否写结构化审计日志(资金/写操作=true)
  on_error: "fail" | "skip" | "fallback" | "escalate"
}
```

### 3.2 节点类型（NodeType 分类）

| 节点类型 | 作用 | side_effect | 可重试 | 典型示例 |
|---------|------|------------|--------|---------|
| `START` | 图入口，初始化 state | none | - | 流程起点 |
| `TOOL` | 调外部能力(执行层) | read/write/irreversible | 取决幂等 | 查订单、发起退款、写 CRM |
| `LLM` | 模型生成/抽取/判断 | read | 是 | 答案生成、槽位补抽、意图澄清 |
| `ROUTER` / `DECISION` | 纯计算分支决策，不产副作用 | none | 是 | 风控结果分流、复杂度路由 |
| `HITL` | 挂起等待用户/人工输入 | none | - | 退款二次确认、人工审批 |
| `PARALLEL_FORK` | 扇出，触发多分支并行 | none | - | 同时回写 CRM + 通知用户 |
| `JOIN` / `BARRIER` | 汇聚，等待并行分支(全部/任一/法定数) | none | - | 等待 CRM+通知 都完成 |
| `COMPENSATION` | 反向补偿(Saga) | write | 是(幂等) | 撤销已发起的退款单 |
| `SUBGRAPH` | 嵌套子图(复用流程) | 取决子图 | - | 退款内复用"身份核验"子图 |
| `WAIT` / `TIMER` | 定时/延时/等待外部回调 | none | - | 等支付异步回调、SLA 计时 |
| `MAP` | 对集合逐项执行(动态扇出) | 取决 | - | 批量订单逐单处理 |
| `END` | 终止(成功/失败/放弃) | none | - | 流程结束 |

> **关键约束**：`side_effect = irreversible` 的节点（如退款）**前置必须有 `HITL` 确认节点**，且 `audit=true`；图校验期强制检查（见 §7）。

### 3.3 节点执行契约（前置/后置/补偿）

每个节点遵循统一生命周期钩子（设计契约，非代码）：

```
pre_condition  : 入边条件 + inputs 完整性校验(缺槽→转 HITL 追问)
execute        : 调 handler, 受 timeout/retry/idempotency 约束
post_condition : 校验 outputs 合法性, 写回 state(按通道 reduce)
on_failure     : 按 on_error 策略(fail/skip/fallback/escalate); 若 side_effect=write 且已部分提交 → 标记需补偿
checkpoint     : 按 policies.checkpoint 持久化 state 快照(可恢复点)
```

---

## 4. 边设计

### 4.1 边通用结构（EdgeSpec）

```jsonc
Edge {
  id: EdgeId,
  from: NodeId,
  to: NodeId,
  type: EdgeType,               // 见 §4.2
  // ---- 条件边 ----
  condition: Predicate | null,  // 命中才走此边; 见 §4.3
  priority: number,             // 多条候选边时的择优顺序(数值小优先)
  // ---- 循环边 ----
  loop: { max_iterations: number, counter_channel: string } | null,
  // ---- 元信息 ----
  label: string,                // "风控通过" / "金额>5000" 等可读标签
  guard: Predicate | null       // 守卫(权限/合规前置), 不满足直接拒绝该跃迁
}
```

### 4.2 边类型（EdgeType 分类）

| 边类型 | 语义 | 说明 |
|--------|------|------|
| `SEQUENTIAL` | 顺序边 | 默认依赖：前驱成功后无条件进入后继 |
| `CONDITIONAL` | 条件边 | 命中 `condition` 才走；同源多条按 `priority` 评估，互斥 |
| `DEFAULT` | 兜底边 | 同源条件边都不命中时走（else 分支），最多一条 |
| `FORK` | 并行扇出边 | 从 `PARALLEL_FORK` 出发，多条同时激活 |
| `JOIN` | 汇聚边 | 指向 `JOIN` 节点，携带汇聚策略(`all`/`any`/`quorum=k`) |
| `LOOP` | 回环边 | 指向已访问节点，**必须**带 `loop.max_iterations` 上界 |
| `FALLBACK` | 降级边 | 节点 `on_error=fallback` 时走，指向降级处理 |
| `ESCALATE` | 升级边 | 触发转人工/升级（如风控拒绝、情绪超阈） |
| `COMPENSATE` | 补偿边 | 反向边，连接节点与其补偿节点，仅在回滚链激活 |
| `INTERRUPT` | 中断边 | 全局中断(用户主动取消/超时)，跳转到清理或 END |

### 4.3 条件谓词（Predicate）

条件边/守卫的判定表达式，引用 state 通道，纯函数无副作用：

```jsonc
Predicate =
  | { op: "eq"|"ne"|"gt"|"gte"|"lt"|"lte", left: "state.path", right: value }
  | { op: "in"|"not_in", left: "state.path", right: value[] }
  | { op: "exists"|"absent", left: "state.path" }
  | { op: "regex", left: "state.path", pattern: string }
  | { op: "and"|"or", clauses: Predicate[] }
  | { op: "not", clause: Predicate }

// 示例: 金额>5000 且 风控未通过 → 走人工审批边
{ op: "and", clauses: [
   { op: "gt", left: "slots.amount", right: 5000 },
   { op: "ne", left: "tool_results.risk.decision", right: "PASS" }
]}
```

> **互斥与完备性**：同一源节点的 `CONDITIONAL` 边集合 + 一条 `DEFAULT` 边必须覆盖所有取值（图校验期做完备性检查，防止"无路可走"的悬挂状态）。

---

## 5. 完整范例：退款流程图（refund_flow）

将主架构与深化推演中的退款场景，落为完整的点边定义。

### 5.1 拓扑结构

```
              ┌─────────┐
              │ START   │
              └────┬────┘
                   │ SEQUENTIAL
              ┌────▼─────────┐
              │ N1 查订单     │ TOOL/read  (OrderService)
              └────┬─────────┘
       CONDITIONAL │ 订单存在? ──── DEFAULT(不存在) ──▶ N_ESC(转人工/澄清)
              ┌────▼─────────┐
              │ N2 退款资格校验│ ROUTER  (RuleEngine)
              └────┬─────────┘
       CONDITIONAL │ 资格通过? ──── DEFAULT(超期) ─────▶ N_ESC
              ┌────▼─────────┐
              │ N3 风控校验   │ TOOL/read (RiskService)
              └────┬─────────┘
       CONDITIONAL │ 风控PASS? ──── DEFAULT(拦截) ─ESCALATE▶ N_HUMAN(人工审核)
              ┌────▼─────────┐
              │ N4 用户二次确认│ HITL  (requires_approval, 不可逆前置)
              └────┬─────────┘
        INTERRUPT  │ (超时30min/取消) ───────────────▶ N_ABANDON
              ┌────▼─────────┐
              │ N5 发起退款   │ TOOL/irreversible  (PaymentGateway)
              └────┬─────────┘  idem_key=${session}:N5:${order_id}:${amount}
                   │ comp = C5(撤销退款)
            FORK   ├───────────────┬────────────────┐
          ┌────────▼───┐     ┌─────▼────────┐        │
          │ N6 回写CRM  │     │ N7 通知用户   │        │
          │ TOOL/write │     │ TOOL/write   │        │
          │ on_error=  │     │ on_error=    │        │
          │  fallback  │     │  fallback    │        │
          └────────┬───┘     └─────┬────────┘        │
              JOIN └───────┬───────┘ (policy: any)   │  N6 失败→补偿队列(不阻塞)
                      ┌────▼────┐                     │  N7 失败→多通道降级
                      │ N8 生成 │ LLM/template        │
                      │  回复   │                     │
                      └────┬────┘                     │
                      ┌────▼────┐                      │
                      │  END    │◀─────────────────────┘
                      └─────────┘
```

### 5.2 节点定义表

| ID | type | side_effect | 可重试 | requires_approval | compensation | on_error |
|----|------|------------|--------|-------------------|--------------|----------|
| N1 查订单 | TOOL | read | 是 | 否 | — | fail |
| N2 资格校验 | ROUTER | none | 是 | 否 | — | fail |
| N3 风控校验 | TOOL | read | 是 | 否 | — | escalate |
| N4 用户确认 | HITL | none | — | **是** | — | escalate |
| N5 发起退款 | TOOL | **irreversible** | 是(幂等) | — | **C5** | escalate |
| N6 回写CRM | TOOL | write | 是(幂等) | 否 | — | fallback(补偿队列) |
| N7 通知用户 | TOOL | write | 是 | 否 | — | fallback(多通道降级) |
| N8 生成回复 | LLM | read | 是 | 否 | — | fallback(纯模板) |
| C5 撤销退款 | COMPENSATION | write | 是(幂等) | — | — | escalate |

### 5.3 关键边定义表

| from → to | type | condition / 说明 |
|-----------|------|------------------|
| N1 → N2 | CONDITIONAL | `exists(tool_results.order)` |
| N1 → N_ESC | DEFAULT | 订单不存在 → 澄清/转人工 |
| N2 → N3 | CONDITIONAL | `eq(tool_results.eligible, true)` |
| N3 → N4 | CONDITIONAL | `eq(tool_results.risk.decision, "PASS")` |
| N3 → N_HUMAN | ESCALATE | 风控拦截 → 人工审核组 |
| N4 → N5 | CONDITIONAL | `eq(pending_hitl.confirmed, true)` |
| N4 → N_ABANDON | INTERRUPT | 30min 无回应 / 用户取消 |
| N5 → {N6,N7} | FORK | 并行扇出 |
| {N6,N7} → N8 | JOIN(any) | 任一完成即继续，失败项进补偿/降级队列 |
| N5 ⇠ C5 | COMPENSATE | 仅回滚链激活（欺诈追回等） |

### 5.4 该范例覆盖的设计点

- 顺序 + 条件 + 默认（兜底）+ 并行扇出/汇聚 + 升级 + 中断 + 补偿边——边类型全覆盖。
- 不可逆节点 N5 前置 HITL（N4），满足合规强制约束。
- N6/N7 并行且 `on_error=fallback`，CRM 失败不阻塞用户通知（最终一致）。
- N5 幂等键防重复退款；C5 提供 Saga 补偿挂钩。

---

## 6. Saga 补偿与图回滚语义

```
正向链:  N1 → N2 → N3 → N4 → N5 → (N6 ∥ N7) → N8
副作用节点: 仅 N5(irreversible)、N6/N7(write)

回滚触发: 图进入 COMPENSATING 时, 沿"已成功且有 compensation 的节点"逆序执行补偿
补偿原则:
  1. read/none 节点无补偿(无副作用)
  2. write 节点(N6/N7)走最终一致, 通常以"对账修复"而非强补偿
  3. irreversible 节点(N5)补偿=显式反向业务(C5), 可能需人工/线下介入
  4. 补偿节点本身必须幂等, 失败则升级人工 + 告警

补偿链表示: 每个节点的 compensation 字段构成"正向边的镜像反向边(COMPENSATE)"
```

---

## 7. 图校验规则（编译期 / 发布前）

`GraphDefinition` 发布前必须通过静态校验，保证运行期不出现悬挂/死锁/竞态：

| 类别 | 规则 |
|------|------|
| 结构 | 唯一 `entry_node`；所有节点从入口可达；所有路径可达某 `END`（无死胡同）。 |
| 无环 | 除显式 `LOOP` 边外不得有环；每条 `LOOP` 边必须声明 `max_iterations`。 |
| 分支完备 | 同源 `CONDITIONAL` 边 + 至多一条 `DEFAULT` 必须覆盖全部取值（无"无路可走"）。 |
| 互斥 | 同源条件边在任一状态下命中集合互斥（或由 `priority` 唯一决断）。 |
| 并行安全 | 并行分支不得对同一 `last` 通道写不同值；只能写不同通道或 `merge/union/accumulate` 通道。 |
| 汇聚 | 每个 `FORK` 必须有对应 `JOIN`；`JOIN` 策略 ∈ {all, any, quorum=k}。 |
| 合规 | `side_effect=irreversible` 节点前驱路径必含 `HITL`，且 `audit=true`、`requires_approval=true`。 |
| 补偿 | 所有 `write/irreversible` 节点应声明 `compensation` 或显式标注"走最终一致对账"。 |
| 预算 | 存在 `max_total_steps` 与整图 `sla_ms`，防失控/防长流程饿死。 |

---

## 8. 持久化与断点恢复（Checkpoint）

```jsonc
Checkpoint {
  run_id: string,
  graph_id: string, graph_version: string,
  cursor: NodeId,                 // 恢复点
  state_snapshot: GraphState,     // 各通道当前值(脱敏存储)
  node_runs: NodeRun[],           // 各节点状态/重试次数/产出/幂等键
  pending_hitl: object | null,    // 等待中的人工/用户确认(含 TTL)
  created_at, updated_at
}
```

- **持久化时机**：由 `policies.checkpoint` 控制（`every_node` 最稳，`on_io` 平衡，`manual` 最省）。
- **恢复语义**：从 `cursor` 续跑；已成功的幂等节点凭 `idempotency_key` 命中缓存直接跳过，不重复执行副作用。
- **中断恢复**：`WAITING`（HITL/回调）状态可挂起数分钟至数天；TTL 到期转 `ABANDONED`。
- 与主架构呼应：短期 state 存 Redis，长流程 Checkpoint 落 PostgreSQL（见 `readme.md` §4.1 / §5）。

---

## 9. 图定义的版本化与灰度

| 维度 | 设计 |
|------|------|
| 版本 | `graph_id + version` 不可变；新逻辑发新版本，老 run 跑在创建时锁定的版本上（避免"半路换图"）。 |
| 灰度 | 新版本按租户 / 流量比例灰度放量（如 10%→50%→100%）；指标达标再全量。 |
| 回滚 | 一键回滚到上一稳定版本；进行中的 run 不受影响（版本锁定）。 |
| 编排台 | 运营在低代码可视化编排台增删点边、配置条件，保存即触发 §7 校验后才可发布。 |
| 兼容 | 状态 `StateSchema` 变更需向后兼容（新增通道带默认值；删除通道需迁移脚本）。 |

---

## 10. 与各层的接口契约（摘要）

| 上/下游 | 输入到规划层 | 规划层输出 |
|---------|-------------|-----------|
| 意图层 → 规划层 | `{intent, intent_confidence, slots, context, user_profile}` | 选定 `graph_id@version`，初始化 `GraphState` |
| 规划层 → 执行层 | 每个 `TOOL` 节点：`{handler, inputs(已绑定), idempotency_key, timeout, retry}` | 执行层回 `{status, outputs, duration, error?}` |
| 规划层 → 推理层 | 每个 `LLM` 节点：`{handler(模型路由提示), prompt 上下文, 预算约束}` | 回生成结果（受 §推理层路由与预算约束） |
| 规划层 → 反馈层 | 节点级事件流：`node_started/succeeded/failed/compensated`、状态机迁移 | 反馈层用于 trace、质检、SLA 计时 |
| 规划层 → 安全/审计 | `audit=true` 节点的请求/响应/风险标记 | 写结构化审计日志（见 `readme.md` §4.5.1） |

---

## 11. 小结

本设计把规划层的"任务依赖图"形式化为可校验、可执行、可恢复、可补偿的结构：

1. **结构定义**：`GraphDefinition`（模板）/`GraphState`（通道+归并器）/`Node`/`Edge` 四件套，定义与运行分离、可版本化。
2. **点设计**：12 类节点，统一执行契约（前置/执行/后置/失败/补偿/检查点），不可逆节点强制 HITL + 审计。
3. **边设计**：10 类边（顺序/条件/兜底/扇出/汇聚/回环/降级/升级/补偿/中断），条件谓词纯函数化，分支完备且互斥。
4. **运行保障**：状态机驱动中断恢复，Saga 补偿，编译期图校验，Checkpoint 持久化，版本灰度回滚。

> 关联文档：总体架构与各层职责见 `readme.md`；端到端场景时序（退款/投诉/大促/FAQ）的运行推演见 `architecture-deep-dive.md`。本图结构是这些场景在规划层的统一底座。
