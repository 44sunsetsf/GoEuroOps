"""
模型用量统计：每次调用模型后，把 token 数按天累加到 Redis，供站长后台展示。

用法：创建 AsyncAnthropic 客户端后包一层 ``track(client, "intent")``。普通调用和流式调用都会记录，
数据写在哈希 ``goeuroops:usage:<北京时间日期>`` 里，字段有 calls / in / out / cost，以及按来源拆开的
``calls:<source>`` 等，保留 120 天。
费用（元）按调用时刻估算：单价取 LLM_PRICE_INPUT / LLM_PRICE_CACHED_INPUT / LLM_PRICE_OUTPUT（元 / 百万 token，
默认是 DeepSeek V4 Pro 的闲时价 4.5 / 0.15 / 13.5），工作日北京时间 9–12 点和 14–18 点按忙时价乘 LLM_PEAK_MULTIPLIER（默认 2）。
Redis 不可用时静默跳过：统计失败绝不能影响回答。
"""
import asyncio
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

log = logging.getLogger(__name__)

KEY_PREFIX = "goeuroops:usage:"
KEEP = timedelta(days=120)
_TZ = timezone(timedelta(hours=8))
_redis: Any = None
_redis_ready = False


def _client() -> Any:
    global _redis, _redis_ready
    if not _redis_ready:
        _redis_ready = True
        url = os.getenv("REDIS_URL")
        if url:
            try:
                import redis.asyncio as redis
                _redis = redis.from_url(url, decode_responses=True)
            except Exception:
                _redis = None
    return _redis


def _price(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return default


def is_peak(now: datetime) -> bool:
    now = now.astimezone(_TZ)
    return now.weekday() < 5 and (9 <= now.hour < 12 or 14 <= now.hour < 18)


def cost_of(tokens_in: int, tokens_cached: int, tokens_out: int, now: Optional[datetime] = None) -> float:
    now = now or datetime.now(_TZ)
    cost = (tokens_in * _price("LLM_PRICE_INPUT", 4.5) + tokens_cached * _price("LLM_PRICE_CACHED_INPUT", 0.15)
            + tokens_out * _price("LLM_PRICE_OUTPUT", 13.5)) / 1_000_000
    return cost * (_price("LLM_PEAK_MULTIPLIER", 2.0) if is_peak(now) else 1.0)


def day_key(now: Optional[datetime] = None) -> str:
    return KEY_PREFIX + (now or datetime.now(_TZ)).astimezone(_TZ).strftime("%Y-%m-%d")


async def record(source: str, usage: Any, redis_client: Any = None) -> None:
    """Add one call's usage. ``usage`` is the Anthropic ``Usage`` object (or anything with the same fields)."""
    r = redis_client if redis_client is not None else _client()
    if r is None or usage is None:
        return
    fresh = int(getattr(usage, "input_tokens", 0) or 0) + int(getattr(usage, "cache_creation_input_tokens", 0) or 0)
    cached = int(getattr(usage, "cache_read_input_tokens", 0) or 0)
    tokens_in, tokens_out = fresh + cached, int(getattr(usage, "output_tokens", 0) or 0)
    cost = cost_of(fresh, cached, tokens_out)
    key = day_key()
    try:
        pipe = r.pipeline()
        for field, n in (("calls", 1), ("in", tokens_in), ("out", tokens_out),
                         (f"calls:{source}", 1), (f"in:{source}", tokens_in), (f"out:{source}", tokens_out)):
            pipe.hincrby(key, field, n)
        pipe.hincrbyfloat(key, "cost", cost)
        pipe.hincrbyfloat(key, f"cost:{source}", cost)
        pipe.expire(key, KEEP)
        await pipe.execute()
    except Exception:
        log.debug("llm usage not recorded", exc_info=True)


_pending: set = set()   # keep a reference so a pending record is not garbage-collected mid-flight


def _record_soon(source: str, usage: Any) -> None:
    try:
        task = asyncio.get_running_loop().create_task(record(source, usage))
    except RuntimeError:
        return
    _pending.add(task)
    task.add_done_callback(_pending.discard)


class _TrackedStream:
    """Wraps ``messages.stream(...)``: the usage of the final message is recorded when the stream closes."""

    def __init__(self, manager: Any, source: str):
        self._manager = manager
        self._source = source
        self._stream: Any = None

    async def __aenter__(self) -> Any:
        self._stream = await self._manager.__aenter__()
        return self._stream

    async def __aexit__(self, *exc: Any) -> Any:
        try:
            snapshot = getattr(self._stream, "current_message_snapshot", None)
            if snapshot is not None:
                _record_soon(self._source, getattr(snapshot, "usage", None))
        except Exception:
            pass
        return await self._manager.__aexit__(*exc)


def track(client: Any, source: str) -> Any:
    """Make ``client.messages.create`` and ``client.messages.stream`` record token usage. Returns the client."""
    messages = getattr(client, "messages", None)
    if messages is None or getattr(messages, "_usage_tracked", False):
        return client
    create, stream = messages.create, getattr(messages, "stream", None)

    async def tracked_create(*args: Any, **kwargs: Any) -> Any:
        resp = await create(*args, **kwargs)
        _record_soon(source, getattr(resp, "usage", None))
        return resp

    messages.create = tracked_create
    if stream is not None:
        messages.stream = lambda *args, **kwargs: _TrackedStream(stream(*args, **kwargs), source)
    messages._usage_tracked = True
    return client
