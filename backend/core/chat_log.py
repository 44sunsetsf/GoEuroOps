"""
对话记录：每一轮问答写一条到 Redis 列表 ``goeuroops:chatlog``（最新在前，保留最近 2000 条），
供站长后台按访客查看「问了什么、答了什么、走了哪个 Agent」。

对话接口同时在响应头里返回 ``X-Conv-Id``，网关访问日志会记下它，后台据此把对话和访客会话对上。
Redis 不可用时静默跳过，绝不影响回答。
"""
import json
import logging
import time
from typing import Any, Dict

from core import llm_usage

log = logging.getLogger(__name__)

KEY = "goeuroops:chatlog"
KEEP = 2000


def entry(user_id: str, conv_id: str, message: str, resp: Any) -> Dict[str, Any]:
    return {
        "ts": round(time.time(), 3),
        "conv_id": conv_id,
        "request_id": getattr(resp, "request_id", ""),
        "user_id": user_id,
        "message": (message or "")[:1000],
        "answer": (getattr(resp, "response", "") or "")[:1500],
        "intent": getattr(resp, "intent", ""),
        "intent_group": getattr(resp, "intent_group", ""),
        "agents": list(getattr(resp, "agent_types", []) or []),
        "tools": list(getattr(resp, "tools_used", []) or []),
        "skills": list(getattr(resp, "skills_applied", []) or []),
        "escalated": bool(getattr(resp, "escalated", False)),
        "latency_ms": round(float(getattr(resp, "latency_ms", 0) or 0)),
    }


async def record(user_id: str, conv_id: str, message: str, resp: Any, redis_client: Any = None) -> None:
    r = redis_client if redis_client is not None else llm_usage._client()
    if r is None:
        return
    try:
        pipe = r.pipeline()
        pipe.lpush(KEY, json.dumps(entry(user_id, conv_id, message, resp), ensure_ascii=False))
        pipe.ltrim(KEY, 0, KEEP - 1)
        await pipe.execute()
    except Exception:
        log.debug("chat log not recorded", exc_info=True)
