"""新线索 / 转顾问交接单的即时通知：推送到飞书群自定义机器人。

线索库建了新单后调用 notify()，在后台发送，不阻塞对话：
- 只推新单；同一会话重复转人工只并进原交接单，不再推送，免得刷屏；
- 只推摘要：意图、紧急度、原因、截短并去掉证件号的最后一句话、脱敏后的联系方式和线索面板链接；
  客户档案、完整聊天不出站，顾问到线索面板看详情；
- 失败只记日志和指标，交接单照样在线索库里；连续失败走熔断，不反复打一个挂掉的接口。

配置（都放服务器 .env，不进仓库）：
  GOEUROOPS_FEISHU_WEBHOOK        飞书群自定义机器人的 Webhook 地址；不设置就不推送
  GOEUROOPS_FEISHU_SECRET         机器人“签名校验”的密钥（建议开启）
  GOEUROOPS_LEADS_PANEL_URL       消息里“打开线索面板”的链接（可选）
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import logging
import os
import time
from typing import Any, Dict, Optional, Set

from business.lead_store import CHANNEL_LABELS, mask_contact, redact_sensitive_ids
from tooling.breaker import CircuitBreaker

logger = logging.getLogger(__name__)

_URGENCY_LABELS = {"LOW": "低", "MEDIUM": "中", "HIGH": "高", "CRITICAL": "紧急"}


def feishu_sign(secret: str, timestamp: int) -> str:
    """飞书自定义机器人签名：以 "timestamp\\nsecret" 为密钥对空串做 HmacSHA256，再 Base64。"""
    key = f"{timestamp}\n{secret}".encode("utf-8")
    return base64.b64encode(hmac.new(key, b"", digestmod=hashlib.sha256).digest()).decode("utf-8")


def build_message(lead: Dict[str, Any], panel_url: Optional[str] = None) -> Dict[str, Any]:
    """把线索 / 交接单转成飞书富文本消息（post）。只放摘要，不放客户档案。"""
    is_handoff = lead.get("type") == "handoff"
    urgency = str(lead.get("urgency") or "")
    if is_handoff:
        title = f"{'🔴 ' if urgency in ('HIGH', 'CRITICAL') else ''}转顾问交接单 {lead.get('id', '')}"
        rows = [
            f"原因：{lead.get('reason') or '-'}",
            f"意图：{lead.get('intent') or '-'}　紧急度：{_URGENCY_LABELS.get(urgency, urgency or '-')}",
            f"用户最后一句：{redact_sensitive_ids(str(lead.get('last_message') or ''))[:100] or '-'}",
        ]
    else:
        title = f"新咨询线索 {lead.get('id', '')}"
        channel = CHANNEL_LABELS.get(str(lead.get("contact_channel", "")), "")
        rows = [
            f"称呼：{str(lead.get('name') or '-')[:40]}",
            f"联系方式：{channel} {mask_contact(str(lead.get('contact', '')))}",
            f"目标国家：{'、'.join(lead.get('countries') or []) or '-'}　入学：{lead.get('target_intake') or '-'}",
        ]
    content = [[{"tag": "text", "text": row}] for row in rows]
    if panel_url:
        content.append([{"tag": "a", "text": "打开线索面板查看详情", "href": panel_url}])
    return {"msg_type": "post", "content": {"post": {"zh_cn": {"title": title, "content": content}}}}


class FeishuNotifier:
    """飞书群机器人推送。notify() 立即返回，真正的发送在后台任务里。"""

    def __init__(
        self,
        webhook_url: str,
        secret: Optional[str] = None,
        panel_url: Optional[str] = None,
        timeout_s: float = 5.0,
        client: Any = None,
    ):
        self._url = webhook_url
        self._secret = secret or None
        self._panel_url = panel_url or None
        self._timeout_s = timeout_s
        self._client = client
        self._breaker = CircuitBreaker()
        self._tasks: Set[asyncio.Task] = set()   # 持有引用，后台任务不会被垃圾回收掉

    @classmethod
    def from_env(cls) -> Optional["FeishuNotifier"]:
        url = os.getenv("GOEUROOPS_FEISHU_WEBHOOK", "").strip()
        if not url:
            return None
        return cls(
            url,
            secret=os.getenv("GOEUROOPS_FEISHU_SECRET", "").strip(),
            panel_url=os.getenv("GOEUROOPS_LEADS_PANEL_URL", "").strip(),
        )

    def notify(self, lead: Dict[str, Any]) -> None:
        try:
            task = asyncio.get_running_loop().create_task(self.send(lead))
        except RuntimeError:  # 没有事件循环（同步脚本里调用）：不推送
            return
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def send(self, lead: Dict[str, Any]) -> bool:
        """发送一条通知，返回是否成功。不抛异常。"""
        from core.metrics import LEAD_NOTIFY

        lead_type = str(lead.get("type", "lead"))
        ticket = self._breaker.acquire()
        if ticket is None:
            LEAD_NOTIFY.labels(type=lead_type, outcome="circuit_open").inc()
            logger.warning("飞书推送熔断中，跳过 %s", lead.get("id"))
            return False
        payload = build_message(lead, self._panel_url)
        if self._secret:
            ts = int(time.time())
            payload = {"timestamp": str(ts), "sign": feishu_sign(self._secret, ts), **payload}
        try:
            body = await self._post(payload)
            # 飞书出错时 HTTP 也是 200，要看返回体里的 code
            code = body.get("code", body.get("StatusCode", 0)) if isinstance(body, dict) else -1
            if code != 0:
                raise RuntimeError(f"飞书返回 code={code} msg={body.get('msg') if isinstance(body, dict) else body}")
        except Exception as ex:
            self._breaker.record_failure(ticket)
            LEAD_NOTIFY.labels(type=lead_type, outcome="error").inc()
            logger.warning("飞书推送失败 %s: %s", lead.get("id"), ex)
            return False
        self._breaker.record_success(ticket)
        LEAD_NOTIFY.labels(type=lead_type, outcome="ok").inc()
        logger.info("飞书推送成功 %s", lead.get("id"))
        return True

    async def _post(self, payload: Dict[str, Any]) -> Any:
        if self._client is not None:
            return await self._client.post_json(self._url, payload)
        import httpx

        async with httpx.AsyncClient(timeout=self._timeout_s) as client:
            resp = await client.post(self._url, json=payload)
            resp.raise_for_status()
            return resp.json()
