"""工作记忆在对话主链路上：Redis 慢或挂了，对话要按"没有记忆"继续，而不是报错或卡住。"""
import asyncio

from memory.amem import MemoryManager, MsgRole
from tests.test_amem import FakeEmbedder, FakeNotes


class BrokenRedis:
    """所有命令都抛连接错误，模拟 Redis 挂了。"""

    def __getattr__(self, name):
        async def fail(*args, **kwargs):
            raise ConnectionError("redis down")
        return fail


class SlowRedis:
    """所有命令都卡住，模拟 Redis 变慢。"""

    def __getattr__(self, name):
        async def hang(*args, **kwargs):
            await asyncio.sleep(10)
        return hang


def _manager(redis_client, timeout=0.05):
    mgr = MemoryManager.__new__(MemoryManager)      # 不连真实的 Redis、ChromaDB 和模型
    mgr._redis = redis_client
    mgr._redis_timeout = timeout
    mgr._notes = FakeNotes()
    mgr._embedder = FakeEmbedder()
    return mgr


def test_get_context_returns_empty_memory_when_redis_is_down():
    ctx = asyncio.run(_manager(BrokenRedis()).get_context("u1", "c1", query="瑞典怎么申请"))
    assert ctx.recent_messages == [] and ctx.notes == [] and ctx.facts == []


def test_get_context_does_not_hang_when_redis_is_slow():
    async def run():
        return await asyncio.wait_for(_manager(SlowRedis()).get_context("u1", "c1", query="hi"), 2)

    ctx = asyncio.run(run())
    assert ctx.recent_messages == []


def test_add_message_swallows_redis_errors():
    asyncio.run(_manager(BrokenRedis()).add_message("u1", "c1", MsgRole.USER, "你好"))


def test_add_message_does_not_hang_when_redis_is_slow():
    async def run():
        await asyncio.wait_for(_manager(SlowRedis()).add_message("u1", "c1", MsgRole.USER, "你好"), 2)

    asyncio.run(run())


def test_after_turn_skips_when_redis_is_down():
    asyncio.run(_manager(BrokenRedis()).after_turn("u1", "c1"))
