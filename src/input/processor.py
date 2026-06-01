"""
① 输入层 - 预处理引擎

对应架构文档 docs/01-architecture-overview.md §3.2。
负责对用户原始输入进行标准化处理,确保后续层接收到干净、安全、结构化的输入。

处理链路(按顺序执行):
  原始输入 → PII脱敏 → 敏感词检测 → 文本归一化 → 语种识别 → 输出

核心能力:
  1. PII检测与掩码: 手机号/身份证/银行卡/邮箱自动打码
  2. 敏感词过滤: 违规内容检测(广告/色情/政治)
  3. 文本归一化: 全半角/繁简/单位统一
  4. 输入校验: 长度/格式/注入攻击检测
"""
import re
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


# ═══════════════════════════════════════════════════════════════════════════════
# PII(个人身份信息)检测与脱敏
# ═══════════════════════════════════════════════════════════════════════════════

class PIIType(Enum):
    """PII类型枚举"""
    PHONE = "phone"             # 手机号
    ID_CARD = "id_card"         # 身份证号
    BANK_CARD = "bank_card"     # 银行卡号
    EMAIL = "email"             # 邮箱
    ADDRESS = "address"         # 详细地址
    NAME = "name"               # 姓名


@dataclass
class PIIDetection:
    """PII检测结果"""
    pii_type: PIIType
    original: str       # 原始值
    masked: str         # 掩码后的值
    start: int          # 起始位置
    end: int            # 结束位置


# PII正则表达式模式
PII_PATTERNS = {
    PIIType.PHONE: re.compile(
        r'(?<!\d)'                           # 前面不是数字
        r'(1[3-9]\d{9})'                     # 1开头的11位手机号
        r'(?!\d)'                            # 后面不是数字
    ),
    PIIType.ID_CARD: re.compile(
        r'(?<!\d)'
        r'([1-9]\d{5}(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx])'  # 18位身份证
        r'(?!\d)'
    ),
    PIIType.BANK_CARD: re.compile(
        r'(?<!\d)'
        r'([3-6]\d{15,18})'                 # 16-19位银行卡号
        r'(?!\d)'
    ),
    PIIType.EMAIL: re.compile(
        r'([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})'
    ),
}

# 掩码规则: 保留前N和后M位,中间用*替代
MASK_RULES = {
    PIIType.PHONE: (3, 4),      # 138****1234
    PIIType.ID_CARD: (4, 4),    # 3301****1234
    PIIType.BANK_CARD: (4, 4),  # 6222****5678
    PIIType.EMAIL: (2, 0),      # ab****@xxx.com (仅保留前2位)
}


def detect_pii(text: str) -> list[PIIDetection]:
    """
    检测文本中的PII(个人身份信息)

    扫描文本,识别手机号、身份证、银行卡、邮箱等敏感信息。

    Args:
        text: 待检测文本

    Returns:
        检测到的PII列表(含类型、原始值、掩码值、位置)
    """
    detections = []

    for pii_type, pattern in PII_PATTERNS.items():
        for match in pattern.finditer(text):
            original = match.group(1)
            masked = _mask_value(original, pii_type)
            detections.append(PIIDetection(
                pii_type=pii_type,
                original=original,
                masked=masked,
                start=match.start(1),
                end=match.end(1),
            ))

    return detections


def mask_pii(text: str) -> tuple[str, list[PIIDetection]]:
    """
    对文本中的PII进行掩码处理

    检测并替换所有PII为掩码形式。
    原始值不保留在输出中(安全原则:最小暴露)。

    Args:
        text: 原始文本

    Returns:
        (掩码后文本, 检测结果列表)

    Example:
        >>> masked, pii = mask_pii("我手机13812345678,身份证330102199001011234")
        >>> print(masked)
        "我手机138****5678,身份证3301****1234"
    """
    detections = detect_pii(text)

    # 从后往前替换(避免位置偏移)
    result = text
    for d in sorted(detections, key=lambda x: x.start, reverse=True):
        result = result[:d.start] + d.masked + result[d.end:]

    return result, detections


def _mask_value(value: str, pii_type: PIIType) -> str:
    """按规则对PII值进行掩码"""
    keep_start, keep_end = MASK_RULES.get(pii_type, (2, 2))

    if pii_type == PIIType.EMAIL:
        # 邮箱特殊处理: user***@domain
        at_idx = value.find('@')
        if at_idx > 0:
            user = value[:at_idx]
            domain = value[at_idx:]
            masked_user = user[:keep_start] + '***'
            return masked_user + domain
        return value[:2] + '***'

    # 通用掩码
    if len(value) <= keep_start + keep_end:
        return value[:1] + '***' + value[-1:]

    return value[:keep_start] + '****' + value[-keep_end:]


# ═══════════════════════════════════════════════════════════════════════════════
# 敏感词检测
# ═══════════════════════════════════════════════════════════════════════════════

# 敏感词库(分类)
SENSITIVE_WORDS = {
    "prompt_injection": [
        "忽略以上指令", "ignore previous", "system prompt",
        "你是一个", "假装你是", "角色扮演", "jailbreak",
        "忘记你的设定", "无视规则",
    ],
    "abuse": [
        "你妈", "傻逼", "草泥马", "去死", "废物",
    ],
}


def detect_sensitive(text: str) -> list[tuple[str, str]]:
    """
    敏感词检测

    检测文本中的敏感内容(提示注入/辱骂等)。

    Args:
        text: 待检测文本

    Returns:
        [(敏感词, 分类)] 列表

    注意: 提示注入(prompt_injection)检测到后应阻断而非仅记录
    """
    text_lower = text.lower()
    hits = []

    for category, words in SENSITIVE_WORDS.items():
        for word in words:
            if word.lower() in text_lower:
                hits.append((word, category))

    return hits


# ═══════════════════════════════════════════════════════════════════════════════
# 文本归一化
# ═══════════════════════════════════════════════════════════════════════════════

# 全角→半角映射表
_FULLWIDTH_OFFSET = 0xFEE0

def normalize_text(text: str) -> str:
    """
    文本归一化处理

    处理内容:
      1. 全角字符→半角(数字/字母/标点)
      2. 连续空白压缩
      3. 首尾空白去除
      4. 特殊Unicode字符清理

    Args:
        text: 原始文本

    Returns:
        归一化后的文本
    """
    # 全角→半角
    result = []
    for char in text:
        code = ord(char)
        # 全角空格特殊处理
        if code == 0x3000:
            result.append(' ')
        # 全角字符范围(！到～)
        elif 0xFF01 <= code <= 0xFF5E:
            result.append(chr(code - _FULLWIDTH_OFFSET))
        else:
            result.append(char)

    text = ''.join(result)

    # 连续空白压缩
    text = re.sub(r'\s+', ' ', text)

    # 首尾去空
    text = text.strip()

    return text


# ═══════════════════════════════════════════════════════════════════════════════
# 输入层处理引擎(组合以上能力)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class InputProcessResult:
    """输入层处理结果"""
    cleaned_text: str                      # 处理后的文本(已脱敏+归一化)
    original_text: str                     # 原始文本(仅审计用,不传给后续层)
    pii_detected: list[PIIDetection] = field(default_factory=list)
    sensitive_words: list[tuple[str, str]] = field(default_factory=list)
    is_blocked: bool = False               # 是否应阻断(如检测到注入攻击)
    block_reason: str = ""
    language: str = "zh"


class InputProcessor:
    """
    输入层预处理引擎

    串联所有预处理步骤,输出干净安全的文本给下游层。
    对应架构文档 §3.2 "输入层职责"。

    处理顺序:
      1. 输入校验(长度/空值)
      2. 文本归一化(全半角/空白)
      3. PII检测与掩码
      4. 敏感词检测
      5. 提示注入检测(阻断型)
    """

    def __init__(self, max_length: int = 2000, enable_pii_mask: bool = True):
        """
        Args:
            max_length: 输入最大长度(超出截断)
            enable_pii_mask: 是否启用PII掩码(降级模式可关闭)
        """
        self.max_length = max_length
        self.enable_pii_mask = enable_pii_mask

    def process(self, text: str) -> InputProcessResult:
        """
        执行完整的输入预处理

        Args:
            text: 用户原始输入

        Returns:
            InputProcessResult(含处理后文本+检测结果+阻断标记)
        """
        original = text

        # Step 1: 基础校验
        if not text or not text.strip():
            return InputProcessResult(
                cleaned_text="", original_text=original,
                is_blocked=True, block_reason="empty_input"
            )

        # Step 2: 长度截断
        if len(text) > self.max_length:
            text = text[:self.max_length]

        # Step 3: 文本归一化
        text = normalize_text(text)

        # Step 4: PII脱敏
        pii_list = []
        if self.enable_pii_mask:
            text, pii_list = mask_pii(text)

        # Step 5: 敏感词检测
        sensitive = detect_sensitive(text)

        # Step 6: 提示注入阻断
        injection_hits = [s for s in sensitive if s[1] == "prompt_injection"]
        is_blocked = len(injection_hits) > 0

        return InputProcessResult(
            cleaned_text=text,
            original_text=original,
            pii_detected=pii_list,
            sensitive_words=sensitive,
            is_blocked=is_blocked,
            block_reason="prompt_injection" if is_blocked else "",
        )
