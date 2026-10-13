"""
咨询记录：每个会话一份结构化摘要（这次想解决什么、聊到哪、还有什么没解决、约好的下一步）。

会话停下来（用户空闲 IDLE_SECONDS 秒或换了会话）时由整理步骤生成；同一会话后来又接着聊，就在原摘要上更新，
upto 记着摘要覆盖到哪条消息，下次只看之后的消息。每人只留最近 KEEP 份，用户要求删除时整份删掉。
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from core.llm_utils import safe_text

logger = logging.getLogger(__name__)

KEEP = 10


def render(case: Dict[str, Any]) -> str:
    """一份记录放进提示词或交接单的样子。"""
    out = f"{(case.get('updated_at') or '')[:10]}｜{case.get('summary', '')}"
    if case.get("open_items"):
        out += f"；还没解决：{'、'.join(case['open_items'])}"
    if case.get("next_step"):
        out += f"；约好的下一步：{case['next_step']}"
    return out


def clean(raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """模型给的摘要只留需要的字段并截断；没有摘要就不要。"""
    summary = safe_text(raw.get("summary") or "").strip()[:300]
    if not summary:
        return None
    items = [safe_text(str(x)).strip()[:80] for x in raw.get("open_items") or [] if str(x).strip()][:5]
    return {"summary": summary, "open_items": items, "next_step": safe_text(raw.get("next_step") or "").strip()[:120]}


class CaseStore:
    """Redis 列表，最新的在前。"""

    def __init__(self, redis_client: Any):
        self._redis = redis_client

    @staticmethod
    def key(user_id: str) -> str:
        return f"mem:cases:{user_id}"

    async def all(self, user_id: str) -> List[Dict[str, Any]]:
        out = []
        for raw in await self._redis.lrange(self.key(user_id), 0, KEEP - 1):
            try:
                out.append(json.loads(raw))
            except ValueError:
                continue
        return out

    async def get(self, user_id: str, conv_id: str) -> Optional[Dict[str, Any]]:
        return next((c for c in await self.all(user_id) if c.get("conv_id") == conv_id), None)

    async def recent(self, user_id: str, n: int, exclude_conv: str = "") -> List[Dict[str, Any]]:
        return [c for c in await self.all(user_id) if c.get("conv_id") != exclude_conv][:n]

    async def upsert(self, user_id: str, case: Dict[str, Any]) -> None:
        """同一会话的旧摘要换成新的，放到最前面，只留最近 KEEP 份。"""
        key = self.key(user_id)
        for raw in await self._redis.lrange(key, 0, -1):
            try:
                if json.loads(raw).get("conv_id") == case["conv_id"]:
                    await self._redis.lrem(key, 0, raw)
            except ValueError:
                continue
        await self._redis.lpush(key, json.dumps(case, ensure_ascii=False))
        await self._redis.ltrim(key, 0, KEEP - 1)

    async def delete(self, user_id: str) -> None:
        await self._redis.delete(self.key(user_id))
