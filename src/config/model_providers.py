"""
NexusAI 智能客服系统 - 模型提供者配置

本模块定义了系统对接的大模型服务和Embedding服务配置。
实际环境说明:
  · 大模型(LLM): 通过 CodeBuddy OpenAI兼容代理(localhost:8765)调用
  · Embedding模型: 通过 Ollama(localhost:11434)调用本地千问模型
  · 本地推理: Ollama同时提供qwen2.5:7b等本地模型

可用模型资源:
  ┌──────────────────────────────────────────────────────────────────┐
  │ CodeBuddy Proxy (localhost:8765, OpenAI协议)                     │
  │  · gemini-3.0-flash (默认,快速)                                  │
  │  · claude-sonnet-4.6 (高质量对话)                                │
  │  · claude-opus-4.7 (最强推理)                                    │
  │  · gpt-5.5 (通用)                                               │
  │  · glm-5.1-ioa (中文优化)                                       │
  └──────────────────────────────────────────────────────────────────┘
  ┌──────────────────────────────────────────────────────────────────┐
  │ Ollama (localhost:11434)                                         │
  │  · qwen3-embedding:8b (首选Embedding, 4.7GB)                    │
  │  · text-embedding-v4 (备选Embedding, 4.7GB)                     │
  │  · bge-m3 (轻量Embedding, 1.2GB)                                │
  │  · qwen2.5:7b (本地小模型推理)                                   │
  │  · deepseek-r1:14b (本地中型推理)                                │
  └──────────────────────────────────────────────────────────────────┘
"""

# ═══════════════════════════════════════════════════════════════════════════════
# CodeBuddy 代理配置(OpenAI兼容API)
# ═══════════════════════════════════════════════════════════════════════════════

CODEBUDDY_CONFIG = {
    "base_url": "http://localhost:8765/v1",
    "api_key": "codebuddy-local",          # 本地代理无需真实key
    "default_model": "gemini-3.0-flash",   # 默认使用的模型
    "timeout": 45,

    # 按用途分配的模型
    "models": {
        # 快速响应(FAQ/简单对话) - 速度优先
        "fast": "gemini-3.0-flash",

        # 标准对话(多轮/工具调用) - 质量与速度平衡
        "standard": "claude-sonnet-4.6",

        # 复杂推理(多意图/长上下文) - 质量优先
        "reasoning": "claude-opus-4.7",

        # 中文优化(情绪安抚/共情) - 中文场景
        "chinese": "glm-5.1-ioa",

        # 意图分析(函数调用强) - 工具使用
        "tool_use": "claude-sonnet-4.6",
    },
}

# ═══════════════════════════════════════════════════════════════════════════════
# Ollama 配置(本地模型)
# ═══════════════════════════════════════════════════════════════════════════════

OLLAMA_CONFIG = {
    "base_url": "http://localhost:11434",

    # Embedding模型(RAG向量化用)
    "embedding": {
        # 首选: 千问3 Embedding(阿里出品,中文效果最佳)
        "primary": "qwen3-embedding:8b",
        # 备选: text-embedding-v4
        "fallback": "text-embedding-v4",
        # 轻量备选: bge-m3(资源紧张时用)
        "lightweight": "bge-m3",
        # 向量维度(qwen3-embedding输出)
        "dimension": 1024,
    },

    # 本地推理模型(隐私敏感/降级场景)
    "inference": {
        # 小模型(7B): FAQ生成/摘要/分类
        "small": "qwen2.5:7b",
        # 中型(14B): 复杂对话/降级使用
        "medium": "deepseek-r1:14b",
    },
}

# ═══════════════════════════════════════════════════════════════════════════════
# 模型路由策略映射
# ═══════════════════════════════════════════════════════════════════════════════

# 模型路由: 将推理层的ModelTier映射到实际模型
MODEL_ROUTING = {
    # MICRO: 本地分类器(不调用LLM)
    "micro": None,

    # SMALL: Ollama本地千问7B(隐私安全/低延迟)
    "small": {
        "provider": "ollama",
        "model": OLLAMA_CONFIG["inference"]["small"],
        "endpoint": f"{OLLAMA_CONFIG['base_url']}/v1/chat/completions",
    },

    # MEDIUM: CodeBuddy快速模型(标准对话)
    "medium": {
        "provider": "codebuddy",
        "model": CODEBUDDY_CONFIG["models"]["standard"],
        "endpoint": f"{CODEBUDDY_CONFIG['base_url']}/chat/completions",
    },

    # LARGE: CodeBuddy推理模型(复杂场景)
    "large": {
        "provider": "codebuddy",
        "model": CODEBUDDY_CONFIG["models"]["reasoning"],
        "endpoint": f"{CODEBUDDY_CONFIG['base_url']}/chat/completions",
    },

    # XLARGE: 同LARGE(当前环境最高级别)
    "xlarge": {
        "provider": "codebuddy",
        "model": CODEBUDDY_CONFIG["models"]["reasoning"],
        "endpoint": f"{CODEBUDDY_CONFIG['base_url']}/chat/completions",
    },
}

# ═══════════════════════════════════════════════════════════════════════════════
# Embedding 调用封装
# ═══════════════════════════════════════════════════════════════════════════════

import httpx
from typing import Optional


async def get_embedding(text: str, model: Optional[str] = None) -> list[float]:
    """
    调用Ollama获取文本向量

    优先使用千问Embedding模型,失败则降级到bge-m3。

    Args:
        text: 要向量化的文本
        model: 指定模型(默认用primary配置)

    Returns:
        向量列表(维度取决于模型,qwen3-embedding为1024维)

    Example:
        >>> vec = await get_embedding("7天无理由退货条件")
        >>> print(len(vec))  # 1024
    """
    model = model or OLLAMA_CONFIG["embedding"]["primary"]
    url = f"{OLLAMA_CONFIG['base_url']}/api/embed"

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(url, json={
                "model": model,
                "input": text,
            })
            resp.raise_for_status()
            data = resp.json()
            # Ollama embed API 返回格式: {"embeddings": [[...]], ...}
            return data["embeddings"][0]

    except Exception as e:
        # 降级到轻量模型
        if model != OLLAMA_CONFIG["embedding"]["lightweight"]:
            return await get_embedding(text, model=OLLAMA_CONFIG["embedding"]["lightweight"])
        raise RuntimeError(f"Embedding service unavailable: {e}")


async def get_embeddings_batch(texts: list[str], model: Optional[str] = None) -> list[list[float]]:
    """
    批量获取文本向量(用于知识库索引构建)

    Args:
        texts: 文本列表
        model: 指定模型

    Returns:
        向量列表的列表
    """
    model = model or OLLAMA_CONFIG["embedding"]["primary"]
    url = f"{OLLAMA_CONFIG['base_url']}/api/embed"

    try:
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(url, json={
                "model": model,
                "input": texts,
            })
            resp.raise_for_status()
            data = resp.json()
            return data["embeddings"]
    except Exception:
        # 降级: 逐条调用
        results = []
        for text in texts:
            vec = await get_embedding(text, model)
            results.append(vec)
        return results


def get_embedding_sync(text: str, model: Optional[str] = None) -> list[float]:
    """
    同步版本的Embedding调用(用于非异步上下文)

    Args:
        text: 要向量化的文本
        model: 指定模型

    Returns:
        向量列表
    """
    import requests

    model = model or OLLAMA_CONFIG["embedding"]["primary"]
    url = f"{OLLAMA_CONFIG['base_url']}/api/embed"

    try:
        resp = requests.post(url, json={"model": model, "input": text}, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return data["embeddings"][0]
    except Exception as e:
        if model != OLLAMA_CONFIG["embedding"]["lightweight"]:
            return get_embedding_sync(text, model=OLLAMA_CONFIG["embedding"]["lightweight"])
        raise RuntimeError(f"Embedding unavailable: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# LLM 调用封装
# ═══════════════════════════════════════════════════════════════════════════════


async def call_llm(
    messages: list[dict],
    model_tier: str = "medium",
    temperature: float = 0.3,
    max_tokens: int = 2048,
    tools: list[dict] = None,
) -> dict:
    """
    统一LLM调用入口

    根据model_tier路由到对应的模型提供者(CodeBuddy/Ollama)。
    所有调用走OpenAI兼容协议。

    Args:
        messages: OpenAI格式消息列表 [{"role":"user","content":"..."}]
        model_tier: 模型级别 "small"/"medium"/"large"/"xlarge"
        temperature: 采样温度
        max_tokens: 最大生成token数
        tools: 工具定义列表(函数调用)

    Returns:
        {
            "content": "回复文本",
            "model": "实际使用的模型ID",
            "usage": {"input_tokens": N, "output_tokens": N},
            "tool_calls": [...] (如有)
        }

    Example:
        >>> result = await call_llm(
        ...     messages=[{"role":"user","content":"怎么退货"}],
        ...     model_tier="medium"
        ... )
        >>> print(result["content"])
    """
    routing = MODEL_ROUTING.get(model_tier)
    if not routing:
        return {"content": "", "model": "none", "usage": {"input_tokens": 0, "output_tokens": 0}}

    endpoint = routing["endpoint"]
    model = routing["model"]

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if tools:
        payload["tools"] = tools

    headers = {"Content-Type": "application/json"}
    if routing["provider"] == "codebuddy":
        headers["Authorization"] = f"Bearer {CODEBUDDY_CONFIG['api_key']}"

    try:
        async with httpx.AsyncClient(timeout=CODEBUDDY_CONFIG["timeout"]) as client:
            resp = await client.post(endpoint, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()

            choice = data["choices"][0]
            message = choice["message"]

            return {
                "content": message.get("content", ""),
                "model": data.get("model", model),
                "usage": data.get("usage", {"input_tokens": 0, "output_tokens": 0}),
                "tool_calls": message.get("tool_calls", []),
            }

    except Exception as e:
        # Fallback: 降级到更低级别
        fallback_map = {"xlarge": "large", "large": "medium", "medium": "small"}
        fallback_tier = fallback_map.get(model_tier)
        if fallback_tier:
            return await call_llm(messages, model_tier=fallback_tier,
                                  temperature=temperature, max_tokens=max_tokens, tools=tools)
        return {"content": f"[模型调用失败: {e}]", "model": model, "usage": {"input_tokens": 0, "output_tokens": 0}}


def call_llm_sync(
    messages: list[dict],
    model_tier: str = "medium",
    temperature: float = 0.3,
    max_tokens: int = 2048,
) -> dict:
    """
    同步版本的LLM调用(用于非异步上下文)
    """
    import requests

    routing = MODEL_ROUTING.get(model_tier)
    if not routing:
        return {"content": "", "model": "none", "usage": {}}

    endpoint = routing["endpoint"]
    model = routing["model"]

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    headers = {"Content-Type": "application/json"}
    if routing["provider"] == "codebuddy":
        headers["Authorization"] = f"Bearer {CODEBUDDY_CONFIG['api_key']}"

    try:
        resp = requests.post(endpoint, json=payload, headers=headers, timeout=45)
        resp.raise_for_status()
        data = resp.json()
        choice = data["choices"][0]
        return {
            "content": choice["message"].get("content", ""),
            "model": data.get("model", model),
            "usage": data.get("usage", {}),
        }
    except Exception as e:
        return {"content": f"[调用失败: {e}]", "model": model, "usage": {}}
