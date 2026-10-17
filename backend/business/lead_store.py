"""咨询线索与人工交接单存储。

- 线索（type=lead）：用户同意后由 ConsultingAgent 通过 create_consultation_lead 写入；
- 交接单（type=handoff）：EscalationAgent 转顾问时写入。

两者都给创始人在前端「线索」面板里跟进。数据存 Redis（不设 TTL），
Redis 不可用时自动降级为进程内存储，保证对话链路不会因为线索库故障而中断。
"""
from __future__ import annotations

import hashlib
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
# 去重占位键：值是"还在跟进"的那张单的 id。用 SET NX 抢占，查重和建单是一步原子操作
_OPEN_KEY_PREFIX = "goeuroops:lead_open:"
_OPEN_KEY_TTL_S = 30 * 24 * 3600
# 这些状态下视为"顾问还在跟进"，同一会话（交接单）或同一联系方式（线索）不再新建
_OPEN_STATUSES = ("new", "contacted")
_MAX_FOLLOWUPS = 10
_URGENCY_RANK = {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}
# 线索重复登记时，用新值覆盖的字段
_LEAD_MERGE_FIELDS = ("name", "countries", "stage", "target_intake", "interested_services", "background", "preferred_time")


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


def redact_sensitive_ids(text: str) -> str:
    """把身份证号、护照号换成占位符（记忆模块存原话前用）。"""
    text = _ID_CARD_RE.sub("[证件号已隐去]", text or "")
    return _PASSPORT_RE.sub("[证件号已隐去]", text)


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


def handoff_open_key(user_id: str, conv_id: str) -> str:
    return f"handoff:{user_id}:{conv_id}"


def lead_open_key(channel: str, contact: str) -> str:
    """同一联系方式一张线索；键里只放哈希，不放联系方式原文。"""
    normalized = re.sub(r"[\s-]", "", (contact or "").strip().lower())
    return "lead:" + hashlib.sha256(f"{channel}:{normalized}".encode()).hexdigest()[:24]


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
        self._memory_open: Dict[str, str] = {}
        self._degraded = self._redis is None
        self.notifier: Any = None       # 可选：新建线索/交接单时推送通知（如飞书机器人），见 business/notifier.py

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

    async def create(self, data: Dict[str, Any], *, lead_type: str = "lead", lead_id: Optional[str] = None) -> Dict[str, Any]:
        if lead_type not in LEAD_TYPES:
            raise ValueError(f"未知线索类型 {lead_type}")
        now = time.time()
        lead = {
            "id": lead_id or _new_id("L" if lead_type == "lead" else "H"),
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

    async def _claim(self, key: str, lead_id: str) -> Optional[str]:
        """抢占去重键。抢到返回 None；已被占用返回占用者的 id。"""
        if not self._degraded:
            try:
                if await self._redis.set(_OPEN_KEY_PREFIX + key, lead_id, nx=True, ex=_OPEN_KEY_TTL_S):
                    return None
                holder = await self._redis.get(_OPEN_KEY_PREFIX + key)
                if holder:
                    return holder
                # 占用者恰好在这两步之间释放了：再抢一次
                ok = await self._redis.set(_OPEN_KEY_PREFIX + key, lead_id, nx=True, ex=_OPEN_KEY_TTL_S)
                return None if ok else await self._redis.get(_OPEN_KEY_PREFIX + key)
            except Exception as ex:
                logger.warning("去重键写入 Redis 失败，降级为内存存储: %s", ex)
                self._degraded = True
        # 内存模式：检查和写入之间没有 await，同一事件循环里天然原子
        holder = self._memory_open.get(key)
        if holder is None:
            self._memory_open[key] = lead_id
        return holder

    async def _release(self, key: str, lead_id: str) -> None:
        """释放去重键（只在它仍指向 lead_id 时）。"""
        if not self._degraded:
            try:
                if await self._redis.get(_OPEN_KEY_PREFIX + key) == lead_id:
                    await self._redis.delete(_OPEN_KEY_PREFIX + key)
                return
            except Exception as ex:
                logger.warning("去重键释放失败，降级为内存存储: %s", ex)
                self._degraded = True
        if self._memory_open.get(key) == lead_id:
            del self._memory_open[key]

    async def _create_once(self, key: str, data: Dict[str, Any], lead_type: str, merge) -> Dict[str, Any]:
        """按去重键建单：键空着就建新单；键指向一张还在跟进的单，就用 merge 把这次的内容并进去。
        merge 返回 True 时（如紧急度升到高），合并后再推送一次通知。

        占位键指向的单已关闭、或建单失败留下的空占位，释放后重抢一次。
        """
        for _ in range(2):
            new_id = _new_id("L" if lead_type == "lead" else "H")
            holder = await self._claim(key, new_id)
            if holder is None:
                try:
                    lead = await self.create({**data, "dedup_key": key}, lead_type=lead_type, lead_id=new_id)
                except Exception:
                    await self._release(key, new_id)
                    raise
                self._notify(lead)
                return {**lead, "deduplicated": False}
            existing = await self._load(holder)
            if existing is not None and existing.get("status") in _OPEN_STATUSES:
                renotify = merge(existing)
                existing["updated_at"] = datetime.now(timezone.utc).isoformat()
                await self._save(existing)
                logger.info("%s去重: 追加到 %s", "线索" if lead_type == "lead" else "交接单", existing["id"])
                if renotify:
                    self._notify(existing)
                return {**existing, "deduplicated": True}
            await self._release(key, holder)
        # 两次都被别的请求抢先又关掉，极少见：直接建单，不再去重
        lead = await self.create(data, lead_type=lead_type)
        self._notify(lead)
        return {**lead, "deduplicated": False}

    async def create_handoff(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """写入交接单，同一会话已有未关闭的交接单时不重复建单。

        重复的转人工请求（用户连发"转人工"、或隐私请求后又追问）只追加到原交接单的
        followups 里，顾问看到的仍是一张单子。返回的交接单带 deduplicated 标记。
        查重用 Redis SET NX 抢占去重键，两个请求同时到达也只会建一张单。
        """
        user_id, conv_id = data.get("user_id"), data.get("conv_id")
        if not (user_id and conv_id):
            lead = await self.create(data, lead_type="handoff")
            self._notify(lead)
            return {**lead, "deduplicated": False}

        def merge(lead: Dict[str, Any]) -> None:
            followups = list(lead.get("followups", []))
            followups.append({
                "at": datetime.now(timezone.utc).isoformat(),
                "reason": data.get("reason", ""),
                "message": data.get("last_message", ""),
            })
            lead["followups"] = followups[-_MAX_FOLLOWUPS:]
            # 紧急度只升不降：用户后来说"很急"，顾问那边要看得到；升到高或紧急时再推送一次
            new_rank = _URGENCY_RANK.get(str(data.get("urgency")), 0)
            if new_rank > _URGENCY_RANK.get(str(lead.get("urgency")), 0):
                lead["urgency"] = data["urgency"]
                return new_rank >= _URGENCY_RANK["HIGH"]
            return False

        return await self._create_once(handoff_open_key(user_id, conv_id), data, "handoff", merge)

    async def create_lead(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """登记咨询线索，同一联系方式已有未关闭的线索时更新那一张，不重复建。

        用户重试、或在不同会话里又留了一次同样的联系方式，顾问看到的仍是一张线索；
        这次新给的称呼、目标国家、背景等覆盖旧值。返回的线索带 deduplicated 标记。
        """
        key = lead_open_key(str(data.get("contact_channel", "")), str(data.get("contact", "")))

        def merge(lead: Dict[str, Any]) -> None:
            for name in _LEAD_MERGE_FIELDS:
                if data.get(name):
                    lead[name] = data[name]
            followups = list(lead.get("followups", []))
            followups.append({"at": datetime.now(timezone.utc).isoformat(), "reason": "用户再次登记",
                              "message": "", "conv_id": data.get("conv_id")})
            lead["followups"] = followups[-_MAX_FOLLOWUPS:]
            return False

        return await self._create_once(key, data, "lead", merge)

    def _notify(self, lead: Dict[str, Any]) -> None:
        if self.notifier is not None:
            self.notifier.notify(lead)

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
            leads = sorted(self._memory.values(), key=lambda lead: lead["created_ts"], reverse=True)
        if status:
            leads = [lead for lead in leads if lead.get("status") == status]
        if lead_type:
            leads = [lead for lead in leads if lead.get("type") == lead_type]
        return leads[: max(1, min(limit, 500))]

    async def update(self, lead_id: str, *, status: Optional[str] = None, notes: Optional[str] = None) -> Optional[Dict[str, Any]]:
        lead = await self._load(lead_id)
        if lead is None:
            return None
        if status is not None:
            if status not in LEAD_STATUSES:
                raise ValueError(f"status 必须是 {', '.join(LEAD_STATUSES)} 之一")
            lead["status"] = status
            if status not in _OPEN_STATUSES and lead.get("dedup_key"):
                # 关单后释放去重键，同一会话/联系方式再来会开新单
                await self._release(lead["dedup_key"], lead["id"])
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
