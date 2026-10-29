"""
bad case 复核队列：线上值得人看一眼的对话自动进来，顾问确认后导出成回归用例。

进队列的三种情况（同一个 request_id 只有一条，原因合并）：
- ``thumbs_down``：用户点踩（POST /feedback）；
- ``escalated``：这一轮转了顾问；
- ``amount_guard``：金额护栏替换或警告过回复里的句子。

存储：Redis 哈希 ``goeuroops:review``（request_id → JSON）+ 有序集合 ``goeuroops:review:index``（按时间），
最多留 500 条；Redis 不可用时放内存。原话和回复入队前先把证件号换成占位符，访客查看时再打码（api 层）。

导出（GET /review/export）：一条一个 dict，``message`` 可直接喂给编排器，``expected`` 是顾问写的期望行为，
``bad_answer`` 留作对照；保存成 evaluation/datasets/ 下的 JSON 就能加进回归。
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional

from business.lead_store import redact_sensitive_ids

logger = logging.getLogger(__name__)

KEY = "goeuroops:review"
INDEX_KEY = "goeuroops:review:index"
KEEP = 500
REASONS = ("thumbs_down", "escalated", "amount_guard")
STATUSES = ("pending", "confirmed", "dismissed")


def _default_redis() -> Any:
    from core import llm_usage
    return llm_usage._client()


def reasons_from_trace(trace: Dict[str, Any]) -> List[str]:
    """从一条 trace 看它该不该自动进队列。"""
    reasons = []
    if trace.get("escalated"):
        reasons.append("escalated")
    if any(isinstance(c, dict) and c.get("tool_name") == "amount_guard" for c in trace.get("tool_calls") or []):
        reasons.append("amount_guard")
    return reasons


class ReviewQueue:
    def __init__(self, redis_client: Any = ...):
        self._redis_arg = redis_client          # ... = 用 REDIS_URL；None = 只用内存
        self._memory: Dict[str, Dict[str, Any]] = {}

    def _redis(self) -> Any:
        return _default_redis() if self._redis_arg is ... else self._redis_arg

    async def _load(self, request_id: str) -> Optional[Dict[str, Any]]:
        r = self._redis()
        if r is not None:
            try:
                raw = await r.hget(KEY, request_id)
                return json.loads(raw) if raw else None
            except Exception:                   # noqa: BLE001
                logger.warning("复核队列读取 Redis 失败，改用内存", exc_info=True)
        return self._memory.get(request_id)

    async def _save(self, item: Dict[str, Any]) -> None:
        r = self._redis()
        if r is not None:
            try:
                pipe = r.pipeline()
                pipe.hset(KEY, item["request_id"], json.dumps(item, ensure_ascii=False, default=str))
                pipe.zadd(INDEX_KEY, {item["request_id"]: item["created_ts"]})
                await pipe.execute()
                await self._trim(r)
                return
            except Exception:                   # noqa: BLE001
                logger.warning("复核队列写入 Redis 失败，改用内存", exc_info=True)
        self._memory[item["request_id"]] = item
        if len(self._memory) > KEEP:
            oldest = sorted(self._memory.values(), key=lambda x: x["created_ts"])[: len(self._memory) - KEEP]
            for old in oldest:
                self._memory.pop(old["request_id"], None)

    @staticmethod
    async def _trim(r: Any) -> None:
        extra = await r.zcard(INDEX_KEY) - KEEP
        if extra > 0:
            old = await r.zrange(INDEX_KEY, 0, extra - 1)
            if old:
                await r.hdel(KEY, *old)
                await r.zrem(INDEX_KEY, *old)

    async def add(self, trace: Dict[str, Any], reasons: List[str], comment: str = "") -> Optional[Dict[str, Any]]:
        """按 trace 入队；已在队列里就合并原因和点踩备注。reasons 为空时不入队。"""
        reasons = [x for x in reasons if x in REASONS]
        rid = str(trace.get("request_id") or "")
        if not reasons or not rid:
            return None
        try:
            item = await self._load(rid)
            if item is None:
                item = {
                    "request_id": rid,
                    "created_ts": round(time.time(), 3),
                    "user_id": trace.get("user_id", ""),
                    "conv_id": trace.get("conv_id", ""),
                    "message": redact_sensitive_ids(str(trace.get("message") or ""))[:500],
                    "answer": redact_sensitive_ids(str(trace.get("response") or ""))[:800],
                    "agent": trace.get("primary_agent"),
                    "intent": trace.get("intent"),
                    "tools_used": list(trace.get("tools_used") or []),
                    "skills": [s.get("id") for s in trace.get("skills_applied") or [] if isinstance(s, dict)],
                    "reasons": [],
                    "comments": [],
                    "status": "pending",
                    "expected": "",
                }
            for reason in reasons:
                if reason not in item["reasons"]:
                    item["reasons"].append(reason)
            if comment.strip():
                item["comments"].append(redact_sensitive_ids(comment.strip())[:300])
            await self._save(item)
            return item
        except Exception:                       # noqa: BLE001 —— 复核是旁路，失败不影响回答
            logger.warning("bad case 入队失败", exc_info=True)
            return None

    async def list(self, status: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        limit = max(1, min(int(limit or 50), KEEP))
        items: List[Dict[str, Any]] = []
        r = self._redis()
        if r is not None:
            try:
                ids = await r.zrevrange(INDEX_KEY, 0, KEEP - 1)
                raws = await r.hmget(KEY, ids) if ids else []
                items = [json.loads(x) for x in raws if x]
            except Exception:                   # noqa: BLE001
                logger.warning("复核队列读取 Redis 失败，改用内存", exc_info=True)
                items = []
        if not items:
            items = sorted(self._memory.values(), key=lambda x: x["created_ts"], reverse=True)
        if status:
            items = [x for x in items if x.get("status") == status]
        return items[:limit]

    async def review(self, request_id: str, status: str, expected: str = "") -> Optional[Dict[str, Any]]:
        """顾问处理：confirmed（确认是 bad case，写期望行为）或 dismissed（不是问题）。"""
        if status not in STATUSES:
            raise ValueError(f"status 只能是 {', '.join(STATUSES)}")
        if status == "confirmed" and not expected.strip():
            raise ValueError("确认为 bad case 时要写期望行为（expected），回归用例靠它判断")
        item = await self._load(request_id)
        if item is None:
            return None
        item["status"] = status
        item["expected"] = expected.strip()[:500]
        item["reviewed_ts"] = round(time.time(), 3)
        await self._save(item)
        return item

    async def export_cases(self) -> List[Dict[str, Any]]:
        """已确认的 bad case → 回归用例。"""
        cases = []
        for item in await self.list(status="confirmed", limit=KEEP):
            cases.append({
                "id": f"review_{item['request_id']}",
                "message": item["message"],
                "expected": item["expected"],
                "bad_answer": item["answer"],
                "reasons": item["reasons"],
                "agent": item.get("agent"),
                "intent": item.get("intent"),
                "tools_used": item.get("tools_used") or [],
                "source": "online_review",
            })
        return cases

    async def forget(self, user_id: str) -> int:
        removed = 0
        for item in await self.list(limit=KEEP):
            if item.get("user_id") != user_id:
                continue
            removed += 1
            self._memory.pop(item["request_id"], None)
            r = self._redis()
            if r is not None:
                try:
                    await r.hdel(KEY, item["request_id"])
                    await r.zrem(INDEX_KEY, item["request_id"])
                except Exception:               # noqa: BLE001
                    logger.warning("复核队列删除失败", exc_info=True)
        return removed
