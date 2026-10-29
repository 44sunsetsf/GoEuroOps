"""trace 持久化（内存 + Redis，失败回退）和 bad case 复核队列。"""
import asyncio

import pytest

from core.review_queue import ReviewQueue, reasons_from_trace
from core.trace_store import TraceStore, otel_attributes


class FakeRedis:
    """只实现用到的几个命令。"""

    def __init__(self):
        self.kv, self.lists, self.hashes, self.zsets = {}, {}, {}, {}

    def pipeline(self):
        return FakePipe(self)

    async def set(self, key, value, ex=None):
        self.kv[key] = value

    async def get(self, key):
        return self.kv.get(key)

    async def mget(self, keys):
        return [self.kv.get(k) for k in keys]

    async def delete(self, key):
        self.kv.pop(key, None)

    async def lpush(self, key, value):
        self.lists.setdefault(key, []).insert(0, value)

    async def ltrim(self, key, a, b):
        self.lists[key] = self.lists.get(key, [])[a:b + 1]

    async def lrange(self, key, a, b):
        return self.lists.get(key, [])[a:b + 1]

    async def lrem(self, key, count, value):
        before = len(self.lists.get(key, []))
        self.lists[key] = [v for v in self.lists.get(key, []) if v != value]
        return before - len(self.lists[key])

    async def hset(self, key, field, value):
        self.hashes.setdefault(key, {})[field] = value

    async def hget(self, key, field):
        return self.hashes.get(key, {}).get(field)

    async def hmget(self, key, fields):
        return [self.hashes.get(key, {}).get(f) for f in fields]

    async def hdel(self, key, *fields):
        for f in fields:
            self.hashes.get(key, {}).pop(f, None)

    async def zadd(self, key, mapping):
        self.zsets.setdefault(key, {}).update(mapping)

    async def zcard(self, key):
        return len(self.zsets.get(key, {}))

    def _sorted(self, key):
        return [k for k, _ in sorted(self.zsets.get(key, {}).items(), key=lambda kv: kv[1])]

    async def zrange(self, key, a, b):
        return self._sorted(key)[a:b + 1]

    async def zrevrange(self, key, a, b):
        return list(reversed(self._sorted(key)))[a:b + 1]

    async def zrem(self, key, *members):
        for m in members:
            self.zsets.get(key, {}).pop(m, None)


class FakePipe:
    def __init__(self, r):
        self.r, self.ops = r, []

    def __getattr__(self, name):
        def queue(*args, **kwargs):
            self.ops.append((name, args, kwargs))
            return self
        return queue

    async def execute(self):
        for name, args, kwargs in self.ops:
            await getattr(self.r, name)(*args, **kwargs)


class BrokenRedis:
    def pipeline(self):
        raise ConnectionError("redis down")

    async def get(self, key):
        raise ConnectionError("redis down")

    async def lrange(self, *a):
        raise ConnectionError("redis down")

    async def hget(self, *a):
        raise ConnectionError("redis down")

    async def zrevrange(self, *a):
        raise ConnectionError("redis down")


def _trace(rid, user="u1", **extra):
    return {"request_id": rid, "user_id": user, "message": "想退定金", "response": "以顾问核对为准",
            "tool_calls": [], "tools_used": [], "skills_applied": [], **extra}


# ── trace ──

def test_trace_survives_restart_via_redis():
    r = FakeRedis()

    async def run():
        store = TraceStore(max_items=3, redis_client=r)
        for i in range(5):
            store.add(_trace(f"r{i}"))
        await store.flush()
        restarted = TraceStore(max_items=3, redis_client=r)      # 新进程：内存是空的
        return await restarted.get("r4"), await restarted.recent(10)

    got, recent = asyncio.run(run())
    assert got["request_id"] == "r4"
    assert [t["request_id"] for t in recent] == ["r4", "r3", "r2"]      # 最新在前，裁到 3 条
    assert r.lists["goeuroops:traces"] == ["r4", "r3", "r2"]


def test_trace_falls_back_to_memory_when_redis_fails():
    async def run():
        store = TraceStore(max_items=10, redis_client=BrokenRedis())
        store.add(_trace("r1"))
        await store.flush()
        return await store.get("r1"), await store.recent(5), await store.get("missing")

    got, recent, missing = asyncio.run(run())
    assert got["request_id"] == "r1"
    assert [t["request_id"] for t in recent] == ["r1"]
    assert missing is None


def test_trace_without_event_loop_keeps_memory_only():
    store = TraceStore(max_items=10, redis_client=FakeRedis())
    store.add(_trace("r1"))                     # 同步上下文：不建后台任务，也不报错
    assert asyncio.run(store.get("r1"))["request_id"] == "r1"


def test_trace_forget_removes_user_from_memory_and_redis():
    r = FakeRedis()

    async def run():
        store = TraceStore(max_items=10, redis_client=r)
        store.add(_trace("a", user="alice"))
        store.add(_trace("b", user="bob"))
        await store.flush()
        removed = await store.forget("alice")
        fresh = TraceStore(max_items=10, redis_client=r)
        return removed, await fresh.recent(10), await store.get("a")

    removed, recent, a = asyncio.run(run())
    assert removed == 2                         # 内存一条 + Redis 一条
    assert [t["request_id"] for t in recent] == ["b"]
    assert a is None


def test_otel_attributes_carry_business_fields():
    attrs = otel_attributes(_trace("r1", primary_agent="billing", intent="refund", escalated=True,
                                   rag_gate={"mode": "off"}, tools_used=["calculate_refund"],
                                   tool_calls=[{"tool_name": "amount_guard"}, {"tool_name": "x", "result_success": False}],
                                   skills_applied=[{"id": "billing_support"}]))
    assert attrs["gen_ai.agent.name"] == "billing"
    assert attrs["goeuroops.amount_guard"] is True
    assert attrs["goeuroops.tool_errors"] == 1
    assert attrs["goeuroops.skills"] == ["billing_support"]


# ── 复核队列 ──

def test_reasons_from_trace():
    assert reasons_from_trace(_trace("r", escalated=True)) == ["escalated"]
    assert reasons_from_trace(_trace("r", tool_calls=[{"tool_name": "amount_guard"}])) == ["amount_guard"]
    assert reasons_from_trace(_trace("r")) == []


@pytest.mark.parametrize("make_redis", [FakeRedis, lambda: None, BrokenRedis], ids=["redis", "memory", "broken"])
def test_review_flow_add_merge_confirm_export(make_redis):
    async def run():
        q = ReviewQueue(redis_client=make_redis())
        assert await q.add(_trace("r1"), []) is None                       # 没有原因不入队
        await q.add(_trace("r1", message="我身份证 110101199003077777 想退定金"), ["escalated"])
        await q.add(_trace("r1"), ["thumbs_down"], comment="没回答退多少")  # 同一轮合并
        pending = await q.list()
        with pytest.raises(ValueError):
            await q.review("r1", "confirmed", "")                         # 确认必须写期望
        await q.review("r1", "confirmed", "先确认服务进度，再调退款工具估算")
        return pending, await q.list(status="pending"), await q.export_cases()

    pending, after, cases = asyncio.run(run())
    assert len(pending) == 1 and pending[0]["reasons"] == ["escalated", "thumbs_down"]
    assert "110101199003077777" not in pending[0]["message"]              # 证件号入队前换成占位符
    assert pending[0]["comments"] == ["没回答退多少"]
    assert after == []                                                   # 确认后不在待处理里
    assert cases[0]["expected"].startswith("先确认") and cases[0]["source"] == "online_review"


def test_review_forget_and_trim():
    r = FakeRedis()

    async def run():
        q = ReviewQueue(redis_client=r)
        await q.add(_trace("a", user="alice"), ["escalated"])
        await q.add(_trace("b", user="bob"), ["escalated"])
        removed = await q.forget("alice")
        return removed, await q.list()

    removed, items = asyncio.run(run())
    assert removed == 1 and [x["request_id"] for x in items] == ["b"]
