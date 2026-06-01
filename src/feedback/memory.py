"""
长期记忆系统 - 基于 SQLite 的持久化记忆管理

本模块实现架构文档中"记忆分层"的完整设计(docs/01-architecture-overview.md §4.2):
  · 会话内记忆(短期): Redis/内存,TTL=30min → 已在 pipeline.py 实现
  · 用户画像记忆(长期): SQLite持久化 → 本模块实现
  · 组织级经验(沉淀): SQLite持久化 → 本模块实现

═══════════════════════════════════════════════════════════════════════════════
为什么用 SQLite:
═══════════════════════════════════════════════════════════════════════════════

  1. 零部署: Python标准库自带,不需要安装额外服务
  2. 持久化: 数据写入磁盘文件,进程重启不丢失
  3. 事务安全: ACID完整支持,写入不会损坏
  4. 性能够用: 单机10万级记录查询<10ms
  5. 可迁移: 后续规模大了可平滑迁移到PostgreSQL

═══════════════════════════════════════════════════════════════════════════════
记忆分层设计:
═══════════════════════════════════════════════════════════════════════════════

  ┌────────────────────────────────────────────────────────┐
  │ Layer 1: 用户偏好记忆 (user_preferences)               │
  │   · 用户习惯的沟通方式(简洁/详细)                       │
  │   · 常用收货地址/联系方式偏好                           │
  │   · 历史偏好品类                                       │
  │   · 有效话术记录(对该用户什么话管用)                    │
  └────────────────────────────────────────────────────────┘
  ┌────────────────────────────────────────────────────────┐
  │ Layer 2: 用户事件记忆 (user_events)                     │
  │   · 历史投诉记录(时间/原因/结果)                        │
  │   · 历史退款记录                                       │
  │   · 满意度评分历史                                      │
  │   · 升级事件记录                                       │
  └────────────────────────────────────────────────────────┘
  ┌────────────────────────────────────────────────────────┐
  │ Layer 3: 会话摘要存档 (session_archives)                │
  │   · 每次会话结束后生成摘要存档                          │
  │   · 包含: 问题/意图/处理结果/满意度                     │
  │   · 下次同用户来时可快速回顾历史                        │
  └────────────────────────────────────────────────────────┘
  ┌────────────────────────────────────────────────────────┐
  │ Layer 4: 组织级经验库 (org_experience)                  │
  │   · 高分会话的处理话术沉淀                              │
  │   · 常见问题的最佳解决方案                              │
  │   · 坏案例及规避方式                                   │
  └────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════════
数据存储位置:
═══════════════════════════════════════════════════════════════════════════════

  默认: data/memory.db (项目根目录下的data文件夹)
  可配置: 通过 MemoryConfig.db_path 修改

═══════════════════════════════════════════════════════════════════════════════
"""
import sqlite3
import json
import time
import os
from dataclasses import dataclass, field
from typing import Any, Optional
from contextlib import contextmanager


# ═══════════════════════════════════════════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MemoryConfig:
    """长期记忆配置"""
    db_path: str = "data/memory.db"       # SQLite数据库文件路径
    max_events_per_user: int = 100        # 每用户最多保留的事件数
    max_archives_per_user: int = 50       # 每用户最多保留的会话存档数
    archive_session_min_turns: int = 2    # 至少2轮对话才存档(过短无价值)


# ═══════════════════════════════════════════════════════════════════════════════
# 数据模型
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class UserPreference:
    """用户偏好记录"""
    user_id: str
    key: str            # 偏好键(如 "communication_style", "language")
    value: str          # 偏好值(如 "concise", "zh-CN")
    confidence: float = 1.0   # 置信度(多次观察到的偏好更确定)
    updated_at: float = field(default_factory=time.time)


@dataclass
class UserEvent:
    """用户事件记录(历史行为)"""
    user_id: str
    event_type: str     # complaint / refund / escalation / rating / purchase
    summary: str        # 事件摘要
    result: str = ""    # 处理结果
    metadata: dict = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


@dataclass
class SessionArchive:
    """会话存档(历史会话摘要)"""
    session_id: str
    user_id: str
    intent: str         # 主意图
    summary: str        # 会话摘要(问题+处理+结果)
    turn_count: int     # 对话轮次
    satisfaction: Optional[float] = None   # 满意度评分(1-5)
    model_used: str = ""
    escalated: bool = False
    resolved: bool = True
    created_at: float = field(default_factory=time.time)


@dataclass
class OrgExperience:
    """组织级经验(沉淀的最佳实践)"""
    experience_id: str
    category: str       # 经验分类(如 "complaint_handling", "refund_flow")
    scenario: str       # 场景描述
    solution: str       # 解决方案/有效话术
    effectiveness: float = 0.0  # 有效性评分(来自质检/用户反馈)
    usage_count: int = 0        # 被引用次数
    created_at: float = field(default_factory=time.time)


# ═══════════════════════════════════════════════════════════════════════════════
# SQLite 长期记忆存储
# ═══════════════════════════════════════════════════════════════════════════════

class LongTermMemory:
    """
    长期记忆管理器 - 基于SQLite的持久化记忆系统

    提供:
      1. 用户偏好读写(学习用户习惯)
      2. 用户事件记录(投诉/退款/评分历史)
      3. 会话存档(每次会话结束后归档)
      4. 组织经验查询(最佳实践检索)
      5. 用户画像聚合(综合记忆生成用户画像)

    数据安全:
      · SQLite WAL模式保证写入原子性
      · 所有写操作在事务中执行
      · 用户数据按user_id隔离(查询时强制过滤)

    使用示例:
        >>> memory = LongTermMemory()
        >>> memory.record_event("user_001", "complaint", "物流延迟投诉", result="补偿5元券")
        >>> history = memory.get_user_events("user_001")
        >>> profile = memory.get_user_profile("user_001")
    """

    def __init__(self, config: Optional[MemoryConfig] = None):
        self.config = config or MemoryConfig()
        self._ensure_db_dir()
        self._init_db()

    def _ensure_db_dir(self):
        """确保数据库目录存在"""
        db_dir = os.path.dirname(self.config.db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)

    @contextmanager
    def _get_conn(self):
        """获取数据库连接(上下文管理器,自动提交/回滚)"""
        conn = sqlite3.connect(self.config.db_path)
        conn.row_factory = sqlite3.Row  # 返回dict-like行
        conn.execute("PRAGMA journal_mode=WAL")  # WAL模式,读写并发更好
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self):
        """初始化数据库表结构"""
        with self._get_conn() as conn:
            conn.executescript("""
                -- 用户偏好表
                CREATE TABLE IF NOT EXISTS user_preferences (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    confidence REAL DEFAULT 1.0,
                    updated_at REAL NOT NULL,
                    UNIQUE(user_id, key)
                );

                -- 用户事件表
                CREATE TABLE IF NOT EXISTS user_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    result TEXT DEFAULT '',
                    metadata TEXT DEFAULT '{}',
                    created_at REAL NOT NULL
                );

                -- 会话存档表
                CREATE TABLE IF NOT EXISTS session_archives (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT UNIQUE NOT NULL,
                    user_id TEXT NOT NULL,
                    intent TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    turn_count INTEGER DEFAULT 0,
                    satisfaction REAL,
                    model_used TEXT DEFAULT '',
                    escalated INTEGER DEFAULT 0,
                    resolved INTEGER DEFAULT 1,
                    created_at REAL NOT NULL
                );

                -- 客户长期情感画像表(对应架构 §4.2.2)
                CREATE TABLE IF NOT EXISTS emotion_profiles (
                    user_id TEXT PRIMARY KEY,
                    baseline_sentiment REAL DEFAULT 0.1,
                    recent_trend TEXT DEFAULT 'STABLE',
                    sensitive_topics TEXT DEFAULT '[]',
                    escalation_count INTEGER DEFAULT 0,
                    preferred_tone TEXT DEFAULT 'EFFICIENT',
                    churn_risk TEXT DEFAULT 'LOW',
                    sample_count INTEGER DEFAULT 0,
                    updated_at REAL NOT NULL
                );

                -- 组织经验表
                CREATE TABLE IF NOT EXISTS org_experiences (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experience_id TEXT UNIQUE NOT NULL,
                    category TEXT NOT NULL,
                    scenario TEXT NOT NULL,
                    solution TEXT NOT NULL,
                    effectiveness REAL DEFAULT 0.0,
                    usage_count INTEGER DEFAULT 0,
                    created_at REAL NOT NULL
                );

                -- 索引(加速查询)
                CREATE INDEX IF NOT EXISTS idx_prefs_user ON user_preferences(user_id);
                CREATE INDEX IF NOT EXISTS idx_events_user ON user_events(user_id, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_events_type ON user_events(event_type);
                CREATE INDEX IF NOT EXISTS idx_archives_user ON session_archives(user_id, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_exp_category ON org_experiences(category);
            """)

    # ═══════════════════════════════════════════════════════════════
    # 用户偏好
    # ═══════════════════════════════════════════════════════════════

    def set_preference(self, user_id: str, key: str, value: str, confidence: float = 1.0):
        """
        设置/更新用户偏好

        使用 UPSERT 语义: 存在则更新,不存在则插入。

        Args:
            user_id: 用户ID
            key: 偏好键(如 "style", "language", "contact_preference")
            value: 偏好值
            confidence: 置信度(重复观察到则提高)
        """
        with self._get_conn() as conn:
            conn.execute("""
                INSERT INTO user_preferences (user_id, key, value, confidence, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(user_id, key) DO UPDATE SET
                    value = excluded.value,
                    confidence = excluded.confidence,
                    updated_at = excluded.updated_at
            """, (user_id, key, value, confidence, time.time()))

    def get_preferences(self, user_id: str) -> dict[str, str]:
        """获取用户所有偏好(key→value字典)"""
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT key, value FROM user_preferences WHERE user_id = ?",
                (user_id,)
            ).fetchall()
            return {row["key"]: row["value"] for row in rows}

    # ═══════════════════════════════════════════════════════════════
    # 用户事件
    # ═══════════════════════════════════════════════════════════════

    def record_event(self, user_id: str, event_type: str, summary: str,
                     result: str = "", metadata: dict = None):
        """
        记录用户事件(append-only,不可修改已有记录)

        典型事件类型:
          · complaint: 投诉(记录原因和处理结果)
          · refund: 退款(记录金额和原因)
          · escalation: 升级转人工(记录触发原因)
          · rating: 满意度评分
          · purchase: 重要购买行为

        Args:
            user_id: 用户ID
            event_type: 事件类型
            summary: 事件摘要
            result: 处理结果
            metadata: 附加数据(JSON序列化存储)
        """
        with self._get_conn() as conn:
            conn.execute("""
                INSERT INTO user_events (user_id, event_type, summary, result, metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (user_id, event_type, summary, result,
                  json.dumps(metadata or {}, ensure_ascii=False), time.time()))

            # 超过上限时删除最早的记录
            count = conn.execute(
                "SELECT COUNT(*) as c FROM user_events WHERE user_id = ?", (user_id,)
            ).fetchone()["c"]
            if count > self.config.max_events_per_user:
                conn.execute("""
                    DELETE FROM user_events WHERE id IN (
                        SELECT id FROM user_events WHERE user_id = ?
                        ORDER BY created_at ASC LIMIT ?
                    )
                """, (user_id, count - self.config.max_events_per_user))

    def get_user_events(self, user_id: str, event_type: str = None,
                        limit: int = 20) -> list[dict]:
        """
        查询用户历史事件

        Args:
            user_id: 用户ID
            event_type: 可选过滤事件类型
            limit: 返回数量上限(默认最近20条)

        Returns:
            事件列表(按时间降序)
        """
        with self._get_conn() as conn:
            if event_type:
                rows = conn.execute("""
                    SELECT * FROM user_events
                    WHERE user_id = ? AND event_type = ?
                    ORDER BY created_at DESC LIMIT ?
                """, (user_id, event_type, limit)).fetchall()
            else:
                rows = conn.execute("""
                    SELECT * FROM user_events WHERE user_id = ?
                    ORDER BY created_at DESC LIMIT ?
                """, (user_id, limit)).fetchall()

            return [dict(row) for row in rows]

    # ═══════════════════════════════════════════════════════════════
    # 会话存档
    # ═══════════════════════════════════════════════════════════════

    def archive_session(self, session_id: str, user_id: str, intent: str,
                        summary: str, turn_count: int, satisfaction: float = None,
                        model_used: str = "", escalated: bool = False,
                        resolved: bool = True):
        """
        归档一次会话(会话结束时调用)

        只有 turn_count >= min_turns 的会话才归档(过短的对话无参考价值)。

        Args:
            session_id: 会话ID(唯一)
            user_id: 用户ID
            intent: 会话主意图
            summary: 会话摘要(问题+处理+结果)
            turn_count: 对话轮次
            satisfaction: 用户满意度评分(可选)
            model_used: 使用的模型
            escalated: 是否发生了转人工
            resolved: 是否解决了问题
        """
        if turn_count < self.config.archive_session_min_turns:
            return

        with self._get_conn() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO session_archives
                (session_id, user_id, intent, summary, turn_count, satisfaction,
                 model_used, escalated, resolved, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (session_id, user_id, intent, summary, turn_count,
                  satisfaction, model_used, int(escalated), int(resolved), time.time()))

    def get_user_archives(self, user_id: str, limit: int = 10) -> list[dict]:
        """获取用户历史会话存档"""
        with self._get_conn() as conn:
            rows = conn.execute("""
                SELECT * FROM session_archives WHERE user_id = ?
                ORDER BY created_at DESC LIMIT ?
            """, (user_id, limit)).fetchall()
            return [dict(row) for row in rows]

    # ═══════════════════════════════════════════════════════════════
    # 组织经验
    # ═══════════════════════════════════════════════════════════════

    def add_experience(self, experience_id: str, category: str,
                       scenario: str, solution: str, effectiveness: float = 0.0):
        """添加一条组织级经验"""
        with self._get_conn() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO org_experiences
                (experience_id, category, scenario, solution, effectiveness, usage_count, created_at)
                VALUES (?, ?, ?, ?, ?, 0, ?)
            """, (experience_id, category, scenario, solution, effectiveness, time.time()))

    def find_experiences(self, category: str, limit: int = 5) -> list[dict]:
        """按分类查找组织经验(按有效性降序)"""
        with self._get_conn() as conn:
            rows = conn.execute("""
                SELECT * FROM org_experiences WHERE category = ?
                ORDER BY effectiveness DESC, usage_count DESC LIMIT ?
            """, (category, limit)).fetchall()
            return [dict(row) for row in rows]

    def increment_experience_usage(self, experience_id: str):
        """记录经验被引用(用于排序优化)"""
        with self._get_conn() as conn:
            conn.execute("""
                UPDATE org_experiences SET usage_count = usage_count + 1
                WHERE experience_id = ?
            """, (experience_id,))

    # ═══════════════════════════════════════════════════════════════
    # 客户长期情感画像 (对应架构 §4.2.2)
    # ═══════════════════════════════════════════════════════════════

    def update_emotion_profile(self, user_id: str, emotion_score: float,
                               emotion_level: str, topic: str = "",
                               escalated: bool = False) -> dict:
        """增量更新用户长期情感画像。

        每次会话的实时情绪结果在此沉淀为长期画像:
          · baseline_sentiment: 滑动平均(识别"易怒型"客户)
          · recent_trend: 与历史基线对比得出 趋势
          · sensitive_topics: 触发负面情绪的话题累积
          · escalation_count / churn_risk: 升级与流失风险
          · preferred_tone: 据基线情绪推导偏好语气

        Args:
            user_id: 用户ID
            emotion_score: 本轮情绪强度[0-1]
            emotion_level: 本轮情绪级别(neutral/negative/angry)
            topic: 本轮话题/意图标签(负面时记入敏感话题)
            escalated: 本轮是否升级转人工

        Returns:
            更新后的情感画像 dict
        """
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM emotion_profiles WHERE user_id = ?", (user_id,)
            ).fetchone()

            if row is None:
                baseline = emotion_score
                trend = "STABLE"
                topics = []
                esc = 1 if escalated else 0
                samples = 1
            else:
                prev_baseline = row["baseline_sentiment"]
                samples = row["sample_count"] + 1
                # 滑动平均更新基线(新样本权重 1/samples)
                baseline = prev_baseline + (emotion_score - prev_baseline) / samples
                # 趋势: 本轮明显高于/低于历史基线
                if emotion_score > prev_baseline + 0.15:
                    trend = "DETERIORATING"
                elif emotion_score < prev_baseline - 0.15:
                    trend = "IMPROVING"
                else:
                    trend = "STABLE"
                topics = json.loads(row["sensitive_topics"])
                esc = row["escalation_count"] + (1 if escalated else 0)

            # 负面/愤怒时记录敏感话题
            if topic and emotion_level in ("negative", "angry") and topic not in topics:
                topics.append(topic)
                topics = topics[-10:]  # 最多保留10个

            # 流失风险评估
            if baseline > 0.6 or esc >= 3:
                churn = "HIGH"
            elif baseline > 0.4 or esc >= 1:
                churn = "MEDIUM"
            else:
                churn = "LOW"

            # 偏好语气: 易怒型→共情；正常→高效
            tone = "EMPATHETIC" if baseline > 0.45 else "EFFICIENT"

            conn.execute("""
                INSERT INTO emotion_profiles
                (user_id, baseline_sentiment, recent_trend, sensitive_topics,
                 escalation_count, preferred_tone, churn_risk, sample_count, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    baseline_sentiment = excluded.baseline_sentiment,
                    recent_trend = excluded.recent_trend,
                    sensitive_topics = excluded.sensitive_topics,
                    escalation_count = excluded.escalation_count,
                    preferred_tone = excluded.preferred_tone,
                    churn_risk = excluded.churn_risk,
                    sample_count = excluded.sample_count,
                    updated_at = excluded.updated_at
            """, (user_id, baseline, trend, json.dumps(topics, ensure_ascii=False),
                  esc, tone, churn, samples, time.time()))

        return {
            "user_id": user_id,
            "baseline_sentiment": round(baseline, 3),
            "recent_trend": trend,
            "sensitive_topics": topics,
            "escalation_count": esc,
            "preferred_tone": tone,
            "churn_risk": churn,
            "sample_count": samples,
        }

    def get_emotion_profile(self, user_id: str) -> Optional[dict]:
        """读取用户长期情感画像(无则返回 None)。"""
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM emotion_profiles WHERE user_id = ?", (user_id,)
            ).fetchone()
            if row is None:
                return None
            d = dict(row)
            d["sensitive_topics"] = json.loads(d["sensitive_topics"])
            return d

    # ═══════════════════════════════════════════════════════════════
    # 用户画像聚合(综合长期记忆生成画像)
    # ═══════════════════════════════════════════════════════════════

    def get_user_profile(self, user_id: str) -> dict:
        """
        聚合用户画像(供意图层/推理层使用)

        综合偏好、事件历史、会话存档,生成完整的用户认知:
          · 投诉倾向(历史投诉频率)
          · 满意度趋势
          · 偏好风格
          · 上次交互摘要
          · 风险标签

        Args:
            user_id: 用户ID

        Returns:
            聚合后的用户画像字典
        """
        preferences = self.get_preferences(user_id)
        events = self.get_user_events(user_id, limit=50)
        archives = self.get_user_archives(user_id, limit=10)

        # 统计投诉次数
        complaint_count = sum(1 for e in events if e.get("event_type") == "complaint")
        refund_count = sum(1 for e in events if e.get("event_type") == "refund")
        escalation_count = sum(1 for e in events if e.get("event_type") == "escalation")

        # 计算平均满意度
        ratings = [a["satisfaction"] for a in archives if a.get("satisfaction")]
        avg_satisfaction = sum(ratings) / len(ratings) if ratings else None

        # 最近一次交互
        last_session = archives[0] if archives else None

        # 风险评估
        risk_level = "low"
        if complaint_count >= 3 or escalation_count >= 2:
            risk_level = "high"
        elif complaint_count >= 1 or escalation_count >= 1:
            risk_level = "medium"

        return {
            "user_id": user_id,
            "preferences": preferences,
            "complaint_count": complaint_count,
            "refund_count": refund_count,
            "escalation_count": escalation_count,
            "avg_satisfaction": avg_satisfaction,
            "risk_level": risk_level,
            "total_sessions": len(archives),
            "last_session": {
                "intent": last_session["intent"],
                "summary": last_session["summary"],
                "resolved": bool(last_session["resolved"]),
                "time": last_session["created_at"],
            } if last_session else None,
            "labels": self._generate_labels(preferences, complaint_count, avg_satisfaction),
        }

    def _generate_labels(self, prefs: dict, complaints: int, satisfaction: float) -> list[str]:
        """根据记忆数据生成用户标签"""
        labels = []
        if complaints >= 3:
            labels.append("frequent_complainer")
        if satisfaction and satisfaction < 3.0:
            labels.append("low_satisfaction")
        if satisfaction and satisfaction >= 4.5:
            labels.append("high_satisfaction")
        if prefs.get("style") == "concise":
            labels.append("prefers_concise")
        if prefs.get("vip") == "true":
            labels.append("vip_user")
        return labels

    # ═══════════════════════════════════════════════════════════════
    # 统计
    # ═══════════════════════════════════════════════════════════════

    def stats(self) -> dict:
        """数据库统计信息"""
        with self._get_conn() as conn:
            prefs = conn.execute("SELECT COUNT(*) as c FROM user_preferences").fetchone()["c"]
            events = conn.execute("SELECT COUNT(*) as c FROM user_events").fetchone()["c"]
            archives = conn.execute("SELECT COUNT(*) as c FROM session_archives").fetchone()["c"]
            exps = conn.execute("SELECT COUNT(*) as c FROM org_experiences").fetchone()["c"]
            users = conn.execute("SELECT COUNT(DISTINCT user_id) as c FROM user_events").fetchone()["c"]

        db_size = os.path.getsize(self.config.db_path) if os.path.exists(self.config.db_path) else 0

        return {
            "db_path": self.config.db_path,
            "db_size_kb": round(db_size / 1024, 1),
            "total_users": users,
            "preferences": prefs,
            "events": events,
            "archives": archives,
            "experiences": exps,
        }
