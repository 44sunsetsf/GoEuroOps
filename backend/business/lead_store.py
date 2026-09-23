"""咨询线索与人工交接单存储。

- 线索（type=lead）：用户同意后由 ConsultingAgent 通过 create_consultation_lead 写入；
- 交接单（type=handoff）：EscalationAgent 转顾问时写入。

两者都给创始人在前端「线索」面板里跟进。数据存 Redis（不设 TTL），
Redis 不可用时自动降级为进程内存储，保证对话链路不会因为线索库故障而中断。
"""
from __future__ import annotations

import json
import logging
import re
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

LEAD_STATUSES = ("new", "contacted", "converted", "closed")
LEAD_TYPES = ("lead", "handoff")
CONTACT_CHANNELS = ("wechat", "email", "phone")
CHANNEL_LABELS = {"wechat": "微信", "email": "邮箱", "phone": "手机"}
STAGES = ("exploring", "preparing", "applying", "admitted")

_EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")
_PHONE_RE = re.compile(r"^\+?\d[\d -]{6,18}\d$")
_WECHAT_RE = re.compile(r"^[A-Za-z][-_A-Za-z0-9]{5,19}$")
# 身份证号 / 护照号之类的证件信息不应进入线索库
_ID_CARD_RE = re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)")
_PASSPORT_RE = re.compile(r"(?<![A-Za-z0-9])[EeGgPp]\d{8}(?![A-Za-z0-9])")

_KEY_PREFIX = "goeuroops:lead:"
_INDEX_KEY = "goeuroops:leads"


def mask_contact(value: str) -> str:
    """日志和默认列表展示用的脱敏：保留首尾各 2 个字符。"""
    value = value or ""
    if len(value) <= 4:
        return "*" * len(value)
    if "@" in value:
        name, _, domain = value.partition("@")
        return f"{name[:2]}***@{domain}"
    return f"{value[:2]}{'*' * (len(value) - 4)}{value[-2:]}"


def contains_sensitive_id(text: str) -> bool:
    return bool(_ID_CARD_RE.search(text or "") or _PASSPORT_RE.search(text or ""))


def validate_contact(channel: str, contact: str) -> Optional[str]:
    contact = (contact or "").strip()
    if channel not in CONTACT_CHANNELS:
        return f"contact_channel 必须是 {', '.join(CONTACT_CHANNELS)} 之一"
    if not contact:
        return "contact 不能为空"
    if channel == "email" and not _EMAIL_RE.match(contact):
        return "邮箱格式不正确"
    if channel == "phone" and not _PHONE_RE.match(contact):
        return "手机号格式不正确"
    if channel == "wechat" and not (_WECHAT_RE.match(contact) or _PHONE_RE.match(contact)):
        return "微信号格式不正确（6–20 位，字母开头，或绑定的手机号）"
    return None


def validate_lead_input(data: Dict[str, Any]) -> List[str]:
    """校验线索字段，返回错误列表（空列表表示通过）。"""
    errors: List[str] = []
    if data.get("consent") is not True:
        errors.append("需要用户明确同意由工作室顾问联系（consent 必须为 true）")
    if not str(data.get("name", "")).strip():
        errors.append("name（称呼）不能为空")
    contact_error = validate_contact(str(data.get("contact_channel", "")), str(data.get("contact", "")))
    if contact_error:
        errors.append(contact_error)
    stage = data.get("stage")
    if stage and stage not in STAGES:
        errors.append(f"stage 必须是 {', '.join(STAGES)} 之一")
    free_text = " ".join(str(data.get(k, "")) for k in ("background", "notes", "preferred_time", "name"))
    if contains_sensitive_id(free_text):
        errors.append("检测到疑似身份证/护照号码，线索中不能包含证件信息，请删除后重试")
    return errors


def _new_id(prefix: str) -> str:
    return f"{prefix}{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6]}"


class LeadStore:
    """线索仓库：Redis 优先，失败时退回内存。"""

    def __init__(self, redis_url: Optional[str] = None, redis_client: Any = None):
        self._redis = redis_client
        if self._redis is None and redis_url:
            try:
                import redis.asyncio as redis
                self._redis = redis.from_url(redis_url, decode_responses=True)
            except Exception as ex:  # pragma: no cover - 依赖缺失时的兜底
                logger.warning("线索库 Redis 初始化失败，使用内存存储: %s", ex)
                self._redis = None
        self._memory: Dict[str, Dict[str, Any]] = {}
        self._degraded = self._redis is None

    @property
    def backend(self) -> str:
        return "memory" if self._degraded else "redis"

    async def _save(self, lead: Dict[str, Any]) -> None:
        if not self._degraded:
            try:
                await self._redis.set(_KEY_PREFIX + lead["id"], json.dumps(lead, ensure_ascii=False))
                await self._redis.zadd(_INDEX_KEY, {lead["id"]: lead["created_ts"]})
                return
            except Exception as ex:
                logger.warning("线索写入 Redis 失败，降级为内存存储: %s", ex)
                self._degraded = True
        self._memory[lead["id"]] = lead

    async def _load(self, lead_id: str) -> Optional[Dict[str, Any]]:
        if not self._degraded:
            try:
                raw = await self._redis.get(_KEY_PREFIX + lead_id)
                return json.loads(raw) if raw else None
            except Exception as ex:
                logger.warning("读取线索失败，降级为内存存储: %s", ex)
                self._degraded = True
        return self._memory.get(lead_id)

    async def create(self, data: Dict[str, Any], *, lead_type: str = "lead") -> Dict[str, Any]:
        if lead_type not in LEAD_TYPES:
            raise ValueError(f"未知线索类型 {lead_type}")
        now = time.time()
        lead = {
            "id": _new_id("L" if lead_type == "lead" else "H"),
            "type": lead_type,
            "status": "new",
            "created_at": datetime.fromtimestamp(now, timezone.utc).isoformat(),
            "created_ts": now,
            "updated_at": datetime.fromtimestamp(now, timezone.utc).isoformat(),
            "notes": "",
            **{k: v for k, v in data.items() if k not in {"id", "type", "status", "created_at", "created_ts"}},
        }
        await self._save(lead)
        logger.info(
            "新%s已登记: id=%s channel=%s contact=%s",
            "线索" if lead_type == "lead" else "交接单",
            lead["id"],
            lead.get("contact_channel", "-"),
            mask_contact(str(lead.get("contact", ""))),
        )
        return lead

    async def list(
        self,
        *,
        status: Optional[str] = None,
        lead_type: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        leads: List[Dict[str, Any]] = []
        if not self._degraded:
            try:
                ids = await self._redis.zrevrange(_INDEX_KEY, 0, 999)
                if ids:
                    raws = await self._redis.mget([_KEY_PREFIX + i for i in ids])
                    leads = [json.loads(r) for r in raws if r]
            except Exception as ex:
                logger.warning("读取线索列表失败，降级为内存存储: %s", ex)
                self._degraded = True
        if self._degraded:
            leads = sorted(self._memory.values(), key=lambda l: l["created_ts"], reverse=True)
        if status:
            leads = [l for l in leads if l.get("status") == status]
        if lead_type:
            leads = [l for l in leads if l.get("type") == lead_type]
        return leads[: max(1, min(limit, 500))]

    async def update(self, lead_id: str, *, status: Optional[str] = None, notes: Optional[str] = None) -> Optional[Dict[str, Any]]:
        lead = await self._load(lead_id)
        if lead is None:
            return None
        if status is not None:
            if status not in LEAD_STATUSES:
                raise ValueError(f"status 必须是 {', '.join(LEAD_STATUSES)} 之一")
            lead["status"] = status
        if notes is not None:
            lead["notes"] = notes[:2000]
        lead["updated_at"] = datetime.now(timezone.utc).isoformat()
        await self._save(lead)
        return lead

    async def stats(self) -> Dict[str, Any]:
        leads = await self.list(limit=500)
        by_status = {s: 0 for s in LEAD_STATUSES}
        by_type = {t: 0 for t in LEAD_TYPES}
        for lead in leads:
            by_status[lead.get("status", "new")] = by_status.get(lead.get("status", "new"), 0) + 1
            by_type[lead.get("type", "lead")] = by_type.get(lead.get("type", "lead"), 0) + 1
        return {"total": len(leads), "by_status": by_status, "by_type": by_type, "backend": self.backend}
