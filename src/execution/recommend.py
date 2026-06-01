"""④ 执行层 - 商品导购与推荐引擎 (Product Recommendation Engine)

对应 docs/01-architecture-overview.md §4.2.1 商品导购与推荐能力。

"货品功能介绍"是检索型问答(走 RAG)，但电商场景常延伸到**导购、推荐、比价、
搭配、引导下单**，这超出 FAQ/RAG 范畴，需独立能力模块。本模块实现设计文档中
定义的推荐链路:

    用户需求(画像 + 约束如预算/人群)
      → 召回 (标签过滤 + 关键词匹配, Top-N)
      → 过滤 (库存/上下架/合规)
      → 排序 (转化率 × 利润 × 个性化, 可配权重)
      → 生成 (推荐理由, 强制基于真实商品属性, 防夸大)
      → 输出 (商品卡片 + 理由 + 比价)

设计约束(§4.2.1 导购防风险):
  · 推荐理由只能基于商品库真实属性(防虚假宣传/夸大功效)
  · 价格实时取数(本 MVP 用目录内价格)
  · 特殊品类(医疗/食品)加合规话术(本 MVP 预留 compliance_note 字段)

本实现为 MVP: 内存目录 + 规则化召回排序。生产环境应替换为:
  · 召回: 向量检索(商品 embedding) + 协同过滤(用户行为)
  · 排序: CTR/CVR 预估模型 + 多目标融合
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Product:
    """商品记录(商品库的一条)。

    Attributes:
        product_id: 商品ID
        name: 商品名
        category: 类目(如 "手机"/"吸尘器"/"耳机")
        price: 现价(元)
        tags: 标签(如 ["快充","老人","大字","长续航"])，用于召回匹配
        features: 真实属性键值(推荐理由只能引用这些，防夸大)
        rating: 评分[0-5]
        conversion_rate: 历史转化率[0-1](排序信号)
        margin: 毛利率[0-1](排序信号)
        in_stock: 是否有货
        on_shelf: 是否在架
        compliance_note: 特殊品类合规话术(医疗/食品等)
    """
    product_id: str
    name: str
    category: str
    price: float
    tags: list[str] = field(default_factory=list)
    features: dict[str, str] = field(default_factory=dict)
    rating: float = 4.5
    conversion_rate: float = 0.05
    margin: float = 0.2
    in_stock: bool = True
    on_shelf: bool = True
    compliance_note: str = ""


@dataclass
class Recommendation:
    """单条推荐结果(商品卡片 + 推荐理由)。"""
    product: Product
    score: float
    reason: str


@dataclass
class RecommendResponse:
    """导购推荐响应。"""
    answer: str
    recommendations: list[Recommendation] = field(default_factory=list)
    comparison: Optional[str] = None       # 比价/对比表(>=2 个商品时生成)
    warning: Optional[str] = None


class Recommender:
    """商品导购推荐器 — 召回→过滤→排序→理由生成。

    排序权重对应设计文档 §4.2.1: 转化率 × 利润 × 个性化(评分)。
    """

    # 多目标排序权重(可配置)
    W_CONVERSION = 0.45
    W_MARGIN = 0.25
    W_RATING = 0.30

    # 需求关键词 → 标签的映射(召回用), 把口语需求映射到商品标签
    NEED_TAG_MAP: dict[str, list[str]] = {
        "老人": ["老人", "大字", "简单", "大音量"],
        "老年": ["老人", "大字", "简单", "大音量"],
        "续航": ["长续航", "大电池"],
        "快充": ["快充"],
        "拍照": ["高像素", "拍照"],
        "游戏": ["高性能", "游戏"],
        "轻": ["轻便", "便携"],
        "便宜": ["高性价比"],
        "性价比": ["高性价比"],
        "降噪": ["降噪"],
        "运动": ["防水", "运动"],
    }

    def __init__(self, catalog: list[Product]):
        self.catalog = catalog

    def recommend(self, query: str, category: Optional[str] = None,
                  budget: Optional[float] = None, top_k: int = 3) -> RecommendResponse:
        """根据用户需求生成推荐。

        Args:
            query: 用户需求原文(如"给老人买个手机，预算两千以内")
            category: 限定类目(可选)
            budget: 预算上限(元，可选)
            top_k: 返回数量
        """
        # Step 1: 召回 — 标签 + 类目 + 关键词
        wanted_tags = self._extract_tags(query)
        recalled = []
        for p in self.catalog:
            if category and p.category != category:
                continue
            tag_hits = len(set(wanted_tags) & set(p.tags))
            name_hit = 1 if (category is None and p.category in query) else 0
            if tag_hits > 0 or name_hit > 0 or (category and not wanted_tags):
                recalled.append((p, tag_hits + name_hit))

        # Step 2: 过滤 — 库存/上架/预算
        filtered = [
            (p, hits) for (p, hits) in recalled
            if p.in_stock and p.on_shelf and (budget is None or p.price <= budget)
        ]

        if not filtered:
            return RecommendResponse(
                answer="抱歉，没有完全符合您要求的商品。您可以放宽预算或告诉我更多偏好，我再帮您找找。",
                warning="no_match",
            )

        # Step 3: 排序 — 召回匹配度优先，其次多目标加权分
        def rank_score(item):
            p, hits = item
            return hits * 1.0 + (
                p.conversion_rate * self.W_CONVERSION
                + p.margin * self.W_MARGIN
                + (p.rating / 5.0) * self.W_RATING
            )

        filtered.sort(key=rank_score, reverse=True)
        top = filtered[:top_k]

        # Step 4: 生成推荐理由(只基于真实属性，防夸大)
        recs = [
            Recommendation(product=p, score=round(rank_score((p, hits)), 3),
                           reason=self._build_reason(p, wanted_tags))
            for (p, hits) in top
        ]

        # Step 5: 输出 — 答案 + 比价
        answer = self._build_answer(recs, budget)
        comparison = self._build_comparison(recs) if len(recs) >= 2 else None

        return RecommendResponse(answer=answer, recommendations=recs, comparison=comparison)

    def _extract_tags(self, query: str) -> list[str]:
        """从需求文本中抽取标签(口语需求 → 商品标签)。"""
        tags: list[str] = []
        for kw, mapped in self.NEED_TAG_MAP.items():
            if kw in query:
                tags.extend(mapped)
        return list(dict.fromkeys(tags))  # 去重保序

    def _build_reason(self, p: Product, wanted_tags: list[str]) -> str:
        """生成推荐理由 — 仅引用商品真实属性(features)，避免编造。"""
        # 优先展示与用户需求匹配的属性
        matched = [t for t in wanted_tags if t in p.tags]
        parts = []
        if matched:
            parts.append("契合您关注的" + "、".join(matched[:3]))
        # 附上 1-2 条真实属性
        for k, v in list(p.features.items())[:2]:
            parts.append(f"{k}: {v}")
        reason = "；".join(parts) if parts else f"{p.category}热销款，评分{p.rating}"
        if p.compliance_note:
            reason += f"（{p.compliance_note}）"
        return reason

    def _build_answer(self, recs: list[Recommendation], budget: Optional[float]) -> str:
        budget_hint = f"（预算≤¥{budget:.0f}）" if budget else ""
        lines = [f"为您推荐以下{len(recs)}款商品{budget_hint}："]
        for i, r in enumerate(recs, 1):
            lines.append(f"{i}. {r.product.name} ¥{r.product.price:.0f} — {r.reason}")
        return "\n".join(lines)

    def _build_comparison(self, recs: list[Recommendation]) -> str:
        """生成简单比价/对比(价格 + 评分)。"""
        rows = ["商品 | 价格 | 评分"]
        for r in recs:
            rows.append(f"{r.product.name} | ¥{r.product.price:.0f} | {r.product.rating}")
        return "\n".join(rows)


def create_default_catalog() -> list[Product]:
    """创建默认商品目录(MVP 演示数据)。"""
    return [
        Product(
            product_id="P_PHONE_01", name="畅享老人手机 A1", category="手机",
            price=899.0, tags=["老人", "大字", "简单", "大音量", "长续航", "高性价比"],
            features={"屏幕": "6.5英寸大字模式", "电池": "5000mAh", "特色": "一键SOS"},
            rating=4.7, conversion_rate=0.09, margin=0.28,
        ),
        Product(
            product_id="P_PHONE_02", name="旗舰影像手机 Pro", category="手机",
            price=4999.0, tags=["高像素", "拍照", "高性能", "游戏", "快充"],
            features={"主摄": "5000万像素", "充电": "100W快充", "芯片": "旗舰处理器"},
            rating=4.8, conversion_rate=0.06, margin=0.18,
        ),
        Product(
            product_id="P_PHONE_03", name="千元长续航手机 C3", category="手机",
            price=1599.0, tags=["长续航", "大电池", "高性价比", "快充"],
            features={"电池": "6000mAh", "充电": "66W快充", "屏幕": "120Hz高刷"},
            rating=4.6, conversion_rate=0.08, margin=0.22,
        ),
        Product(
            product_id="P_EARBUD_01", name="主动降噪耳机 N2", category="耳机",
            price=699.0, tags=["降噪", "运动", "防水", "长续航"],
            features={"降噪": "42dB主动降噪", "续航": "30小时", "防护": "IPX5防水"},
            rating=4.6, conversion_rate=0.07, margin=0.30,
        ),
        Product(
            product_id="P_VACUUM_01", name="无线吸尘器 V15", category="吸尘器",
            price=3099.0, tags=["大吸力", "轻便", "便携", "长续航"],
            features={"吸力": "230AW", "续航": "60分钟", "重量": "1.6kg轻量"},
            rating=4.8, conversion_rate=0.05, margin=0.25,
        ),
    ]
