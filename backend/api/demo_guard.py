"""
公开演示的访客只读模式与每日花费上限。

访客（面试官等）通过专属链接进入 /studio，网关 Caddy 给他们的请求加上 ``X-Studio-Role: guest``；
外部请求自带的同名请求头会先被 Caddy 删掉，所以这个标记伪造不了。站长走 basic auth，不带标记。

访客规则：
- 只读：除了会被计入每日额度的对话、Skill 命中测试和检索，其余写操作一律 403。
  用白名单而不是黑名单，以后新增的写接口默认就只对站长开放。
- 隐私：线索和工具调用明细里的联系方式对访客打码。

每日花费上限（GOEUROOPS_DAILY_COST_LIMIT，单位元，0 或不设 = 不限）：读 core.llm_usage 按天累计的
费用估算，超过后对话等会调用模型的接口返回 429，第二天（北京时间）自动恢复。
"""
import os
import re
from typing import Any, Iterable, Optional, Set

GUEST_HEADER = "x-studio-role"

# 访客可以调用的非 GET 接口（路径不含网关的 /api 前缀）
GUEST_WRITE_ALLOWED = {
    ("POST", "/chat"),
    ("POST", "/chat/stream"),
    ("POST", "/skills/match"),
    ("POST", "/search"),
}

GUEST_READONLY_MESSAGE = "演示模式只读：这个操作只对站长开放。"
COST_LIMIT_MESSAGE = "今天的演示额度已经用完了，明天再来看看吧。"

_CONTACT_KEYS = {"contact", "phone", "email", "wechat", "mobile"}
_EMAIL_RE = re.compile(r"([A-Za-z0-9._%+-]{1,2})[A-Za-z0-9._%+-]*@([A-Za-z0-9.-]+\.[A-Za-z]{2,})")
_PHONE_RE = re.compile(r"(?<!\d)(\+?\d[\d -]{5,}\d)(?!\d)")
_WECHAT_RE = re.compile(r"((?:微信|vx|VX|wechat|WeChat|WX|wx)\s*(?:号)?\s*[:：是为]?\s*)([A-Za-z][-_A-Za-z0-9]{3,})")


def is_guest(headers: Any) -> bool:
    return (headers.get(GUEST_HEADER) or "").strip().lower() == "guest"


def guest_write_allowed(method: str, path: str) -> bool:
    method = method.upper()
    if method in ("GET", "HEAD", "OPTIONS"):
        return True
    if path.startswith("/api/"):
        path = path[4:]
    return (method, path.rstrip("/") or "/") in GUEST_WRITE_ALLOWED


def _mask_token(value: str) -> str:
    value = value.strip()
    if "@" in value:
        name, _, domain = value.partition("@")
        return f"{name[:2]}***@{domain}"
    if len(value) <= 4:
        return "*" * len(value)
    return value[:2] + "*" * (len(value) - 4) + value[-2:]


def _collect_contacts(obj: Any, found: Set[str]) -> None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key.lower() in _CONTACT_KEYS and isinstance(value, str) and value.strip():
                found.add(value.strip())
            else:
                _collect_contacts(value, found)
    elif isinstance(obj, list):
        for item in obj:
            _collect_contacts(item, found)


def _mask_text(text: str, known: Iterable[str]) -> str:
    for value in sorted(known, key=len, reverse=True):
        if len(value) >= 3:
            text = text.replace(value, _mask_token(value))
    text = _EMAIL_RE.sub(lambda m: f"{m.group(1)}***@{m.group(2)}", text)
    text = _PHONE_RE.sub(lambda m: _mask_token(m.group(1)) if sum(c.isdigit() for c in m.group(1)) >= 7 else m.group(1), text)
    return _WECHAT_RE.sub(lambda m: m.group(1) + _mask_token(m.group(2)), text)


def mask_pii(obj: Any, known: Optional[Set[str]] = None) -> Any:
    """返回打码后的副本：联系方式字段整体打码，其余文本里出现的同一联系方式、邮箱、手机号、微信号也打码。"""
    if known is None:
        known = set()
        _collect_contacts(obj, known)
    if isinstance(obj, dict):
        return {k: (_mask_token(v) if k.lower() in _CONTACT_KEYS and isinstance(v, str) and v.strip()
                    else mask_pii(v, known)) for k, v in obj.items()}
    if isinstance(obj, list):
        return [mask_pii(item, known) for item in obj]
    if isinstance(obj, str):
        return _mask_text(obj, known)
    return obj


def cost_limit() -> float:
    try:
        return max(0.0, float(os.getenv("GOEUROOPS_DAILY_COST_LIMIT", "0") or 0))
    except ValueError:
        return 0.0


async def cost_exceeded(redis_client: Any = None) -> bool:
    """今天（北京时间）的模型费用估算是否已达上限。统计不可用时放行：宁可多花一点也不误拦正常访问。"""
    limit = cost_limit()
    if not limit:
        return False
    from core import llm_usage

    r = redis_client if redis_client is not None else llm_usage._client()
    if r is None:
        return False
    try:
        spent = await r.hget(llm_usage.day_key(), "cost")
    except Exception:
        return False
    try:
        return float(spent or 0) >= limit
    except ValueError:
        return False
