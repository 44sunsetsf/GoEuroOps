"""
工具调用 trace 存储：每次请求一条，按 request_id 查。

原来只放在进程内存（deque，最多 200 条），重启就没了。现在：
- 内存里照旧留最近 N 条，写入和读取都不等 Redis；
- 同时异步写进 Redis：``goeuroops:trace:<request_id>``（带 TTL）+ 列表 ``goeuroops:traces``（最新在前，裁到 N 条），
  重启后 ``/trace/tool/{id}``、``/trace/tools`` 仍能查到；
- Redis 不可用或出错时只用内存，绝不影响回答。

可选：``GOEUROOPS_OTEL_EXPORT=1`` 且装了 opentelemetry 时，每条 trace 另外发一个 OpenTelemetry span
（GenAI 语义约定的几个字段 + 业务字段），可以接 Langfuse、Jaeger 等平台。默认关。
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Set

from core.config import env_int

logger = logging.getLogger(__name__)

KEY_PREFIX = "goeuroops:trace:"
INDEX_KEY = "goeuroops:traces"


def _default_redis() -> Any:
    from core import llm_usage
    return llm_usage._client()


class TraceStore:
    def __init__(self, max_items: Optional[int] = None, ttl_s: Optional[int] = None, redis_client: Any = ...):
        self.max_items = max_items or env_int("GOEUROOPS_TOOL_TRACE_MAX", 200)
        self.ttl_s = ttl_s or env_int("GOEUROOPS_TOOL_TRACE_TTL_S", 7 * 86400)
        self._redis_arg = redis_client          # ... = 用 REDIS_URL；None = 只用内存
        self._memory: Deque[Dict[str, Any]] = deque(maxlen=self.max_items)
        self._pending: Set[asyncio.Task] = set()
        self._warned = False

    def _redis(self) -> Any:
        return _default_redis() if self._redis_arg is ... else self._redis_arg

    def _redis_failed(self, what: str) -> None:
        if not self._warned:
            self._warned = True
            logger.warning("trace 写入/读取 Redis 失败（%s），先只用内存", what, exc_info=True)

    # ── 写 ──
    def add(self, trace: Dict[str, Any]) -> None:
        """同步调用：内存立即可查，Redis 和可选的 OTel 导出在后台做。"""
        self._memory.append(trace)
        export_otel(trace)
        if self._redis() is None:
            return
        try:
            task = asyncio.get_running_loop().create_task(self._persist(trace))
        except RuntimeError:                    # 没有事件循环（同步测试里），只留内存
            return
        self._pending.add(task)
        task.add_done_callback(self._pending.discard)

    async def _persist(self, trace: Dict[str, Any]) -> None:
        r = self._redis()
        rid = str(trace.get("request_id") or "")
        if r is None or not rid:
            return
        try:
            pipe = r.pipeline()
            pipe.set(KEY_PREFIX + rid, json.dumps(trace, ensure_ascii=False, default=str), ex=self.ttl_s)
            pipe.lpush(INDEX_KEY, rid)
            pipe.ltrim(INDEX_KEY, 0, self.max_items - 1)
            await pipe.execute()
        except Exception:                       # noqa: BLE001
            self._redis_failed("write")

    async def flush(self) -> None:
        """等后台写入做完（测试和优雅关闭用）。"""
        if self._pending:
            await asyncio.gather(*list(self._pending), return_exceptions=True)

    # ── 读 ──
    async def get(self, request_id: str) -> Optional[Dict[str, Any]]:
        for trace in reversed(self._memory):
            if trace.get("request_id") == request_id:
                return trace
        r = self._redis()
        if r is None:
            return None
        try:
            raw = await r.get(KEY_PREFIX + request_id)
        except Exception:                       # noqa: BLE001
            self._redis_failed("read")
            return None
        return json.loads(raw) if raw else None

    async def recent(self, limit: int = 20) -> List[Dict[str, Any]]:
        limit = max(1, min(int(limit or 20), self.max_items))
        r = self._redis()
        if r is not None:
            try:
                ids = await r.lrange(INDEX_KEY, 0, limit - 1)
                if ids:
                    raws = await r.mget([KEY_PREFIX + i for i in ids])
                    items = [json.loads(x) for x in raws if x]
                    if items:
                        return items
            except Exception:                   # noqa: BLE001
                self._redis_failed("read")
        return list(reversed(list(self._memory)[-limit:]))

    # ── 隐私删除 ──
    async def forget(self, user_id: str) -> int:
        """删掉这位用户的 trace（trace 里有原话片段）。返回删掉的条数。"""
        before = len(self._memory)
        kept = [t for t in self._memory if t.get("user_id") != user_id]
        self._memory.clear()
        self._memory.extend(kept)
        removed = before - len(kept)
        r = self._redis()
        if r is None:
            return removed
        try:
            ids = await r.lrange(INDEX_KEY, 0, self.max_items - 1)
            raws = await r.mget([KEY_PREFIX + i for i in ids]) if ids else []
            for rid, raw in zip(ids, raws):
                if raw and json.loads(raw).get("user_id") == user_id:
                    await r.delete(KEY_PREFIX + rid)
                    await r.lrem(INDEX_KEY, 0, rid)
                    removed += 1
        except Exception:                       # noqa: BLE001
            self._redis_failed("forget")
        return removed


# ── 可选：OpenTelemetry 导出 ─────────────────────────────────────────────────

_tracer: Any = None
_tracer_ready = False


def _get_tracer() -> Any:
    global _tracer, _tracer_ready
    if not _tracer_ready:
        _tracer_ready = True
        if os.getenv("GOEUROOPS_OTEL_EXPORT", "").strip().lower() in ("1", "true", "yes"):
            try:
                from opentelemetry import trace as otel_trace
                _tracer = otel_trace.get_tracer("goeuroops")
            except Exception:                   # noqa: BLE001 —— 没装 opentelemetry 就当没开
                logger.warning("GOEUROOPS_OTEL_EXPORT 已开，但没装 opentelemetry，跳过导出")
                _tracer = None
    return _tracer


def otel_attributes(trace: Dict[str, Any]) -> Dict[str, Any]:
    """trace → span 属性：GenAI 语义约定的字段 + 业务字段（门控、Skill、工具、升级）。"""
    calls = [c for c in trace.get("tool_calls") or [] if isinstance(c, dict)]
    return {
        "gen_ai.operation.name": "chat",
        "gen_ai.agent.name": trace.get("primary_agent") or "",
        "goeuroops.request_id": trace.get("request_id") or "",
        "goeuroops.intent": trace.get("intent") or "",
        "goeuroops.rag_gate.mode": (trace.get("rag_gate") or {}).get("mode", ""),
        "goeuroops.skills": [s.get("id", "") for s in trace.get("skills_applied") or [] if isinstance(s, dict)],
        "goeuroops.tools": list(trace.get("tools_used") or []),
        "goeuroops.tool_errors": sum(1 for c in calls if c.get("result_success") is False),
        "goeuroops.amount_guard": any(c.get("tool_name") == "amount_guard" for c in calls),
        "goeuroops.escalated": bool(trace.get("escalated")),
        "goeuroops.latency_ms": float(trace.get("latency_ms") or 0),
    }


def export_otel(trace: Dict[str, Any]) -> None:
    tracer = _get_tracer()
    if tracer is None:
        return
    try:
        with tracer.start_as_current_span("goeuroops.chat") as span:
            for key, value in otel_attributes(trace).items():
                span.set_attribute(key, value)
    except Exception:                           # noqa: BLE001
        logger.debug("OTel 导出失败", exc_info=True)
