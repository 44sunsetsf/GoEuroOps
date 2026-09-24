"""
公开演示用的每日对话上限。

部署到公网后，任何拿到链接的人都能调用对话接口，消耗的是模型 API 额度。
GOEUROOPS_DAILY_CHAT_LIMIT 给全站每天的对话次数设一个上限（0 或不设 = 不限制），
计数按北京时间自然日重置；Redis 不可用时退回进程内计数，宁可多放行也不拦住正常访问。
"""
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

_KEY_PREFIX = "goeuroops:quota:chat:"
_TZ = timezone(timedelta(hours=8))


class DailyQuota:
    def __init__(self, limit: int, redis_url: Optional[str] = None, redis_client: Any = None):
        self.limit = max(0, limit)
        self._redis = redis_client
        if self._redis is None and redis_url:
            try:
                import redis.asyncio as redis
                self._redis = redis.from_url(redis_url, decode_responses=True)
            except Exception:
                self._redis = None
        self._memory: Dict[str, int] = {}

    @staticmethod
    def _today(now: Optional[datetime] = None) -> str:
        return (now or datetime.now(_TZ)).astimezone(_TZ).strftime("%Y%m%d")

    async def consume(self, now: Optional[datetime] = None) -> bool:
        """记一次对话；返回 False 表示今天的额度已经用完。"""
        if not self.limit:
            return True
        day = self._today(now)
        if self._redis is not None:
            try:
                key = _KEY_PREFIX + day
                count = await self._redis.incr(key)
                if count == 1:
                    await self._redis.expire(key, 2 * 86400)
                return count <= self.limit
            except Exception:
                pass
        self._memory = {day: self._memory.get(day, 0) + 1}
        return self._memory[day] <= self.limit
