"""LLM response helpers shared by Anthropic-compatible providers."""
from typing import Any, Dict, Iterable, List, Optional

# DeepSeek 的 Anthropic 兼容端点默认对 v4 系列模型开启扩展思考（extended thinking），
# 会先输出一个 `thinking` 内容块再输出 `text` 块。结构化/短输出场景（意图识别、
# LLM Judge、查询改写、重排等）的 max_tokens 很小，思考过程会把预算耗尽，导致
# 拿不到任何 text 块。这里统一禁用思考，换取稳定、可解析的输出。
NO_THINKING_KWARGS: Dict[str, Any] = {"extra_body": {"thinking": {"type": "disabled"}}}


def extract_text_content(content: Iterable[Any]) -> str:
    """Return text blocks from Anthropic-style response content."""
    texts: List[str] = []
    for block in content or []:
        if isinstance(block, str):
            texts.append(block)
            continue

        block_type = getattr(block, "type", None)
        text = getattr(block, "text", None)
        if isinstance(block, dict):
            block_type = block.get("type", block_type)
            text = block.get("text", text)

        if isinstance(text, str) and (block_type in (None, "text")):
            texts.append(text)

    return "\n".join(t for t in texts if t)


def safe_text(value: Any) -> str:
    """转成可安全编码的 UTF-8 字符串：去掉 Unicode 代理字符，避免发请求或写 Redis / ChromaDB 时报编码错误。"""
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)
    return value.encode("utf-8", errors="ignore").decode("utf-8")


def make_client(api_key: str, base_url: Optional[str], source: str) -> Any:
    """创建记录 token 用量的 Anthropic 异步客户端；source 是用量统计里的组件名（intent / agents / memory ...）。"""
    from anthropic import AsyncAnthropic

    from core.llm_usage import track

    kwargs: Dict[str, Any] = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return track(AsyncAnthropic(**kwargs), source)
