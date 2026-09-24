import asyncio
from datetime import datetime, timedelta, timezone

from api.quota import DailyQuota

CN = timezone(timedelta(hours=8))


def test_unlimited_when_zero():
    quota = DailyQuota(limit=0)
    assert all(asyncio.run(quota.consume()) for _ in range(5))


def test_blocks_after_limit_and_resets_next_day():
    quota = DailyQuota(limit=2)
    day1 = datetime(2026, 9, 24, 23, 0, tzinfo=CN)
    day2 = datetime(2026, 9, 25, 0, 1, tzinfo=CN)
    assert asyncio.run(quota.consume(day1)) is True
    assert asyncio.run(quota.consume(day1)) is True
    assert asyncio.run(quota.consume(day1)) is False
    assert asyncio.run(quota.consume(day2)) is True


def test_falls_back_to_memory_when_redis_fails():
    class Broken:
        async def incr(self, key):
            raise ConnectionError("down")

    quota = DailyQuota(limit=1, redis_client=Broken())
    assert asyncio.run(quota.consume()) is True
    assert asyncio.run(quota.consume()) is False
