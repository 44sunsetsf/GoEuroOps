"""飞书推送：签名、消息内容（只放摘要）、失败不抛异常、熔断。"""
import asyncio
import base64
import hashlib
import hmac

from business.notifier import FeishuNotifier, build_message, feishu_sign


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.payloads = []

    async def post_json(self, url, payload):
        self.payloads.append(payload)
        r = self.responses.pop(0) if self.responses else {"code": 0}
        if isinstance(r, Exception):
            raise r
        return r


HANDOFF = {
    "id": "H20261017-abc123", "type": "handoff", "reason": "用户要求真人顾问", "intent": "human_handoff",
    "urgency": "HIGH", "last_message": "我的护照号是 E12345678，快找人联系我",
    "background": "客户档案（用户自述）：均分 85，雅思 7", "user_id": "u1",
}


def test_sign_matches_feishu_algorithm():
    expected = base64.b64encode(hmac.new(b"1700000000\nsecret", b"", hashlib.sha256).digest()).decode()
    assert feishu_sign("secret", 1700000000) == expected


def test_message_only_carries_summary():
    msg = build_message(HANDOFF, "https://example.com/leads")
    text = str(msg)
    assert msg["msg_type"] == "post"
    assert "🔴" in msg["content"]["post"]["zh_cn"]["title"]
    assert "E12345678" not in text            # 证件号去掉
    assert "雅思" not in text                  # 客户档案不出站
    assert "https://example.com/leads" in text

    lead = build_message({"id": "L1", "type": "lead", "name": "小王", "contact_channel": "wechat",
                          "contact": "wang_2027", "countries": ["瑞典"]})
    assert "wang_2027" not in str(lead) and "微信 wa*****27" in str(lead)


def test_send_signs_and_checks_body_code():
    async def run():
        client = FakeClient([{"code": 0}, {"code": 19021, "msg": "sign match fail"}])
        n = FeishuNotifier("https://hook", secret="s", client=client)
        assert await n.send(HANDOFF) is True
        assert {"timestamp", "sign", "msg_type"} <= set(client.payloads[0])
        assert await n.send(HANDOFF) is False      # HTTP 200 但 code≠0 也算失败

    asyncio.run(run())


def test_failures_never_raise_and_open_breaker():
    async def run():
        client = FakeClient([ConnectionError("down")] * 5)
        n = FeishuNotifier("https://hook", client=client)
        for _ in range(5):
            assert await n.send(HANDOFF) is False
        assert await n.send(HANDOFF) is False
        assert len(client.payloads) == 5           # 第 6 次熔断，不再打接口

    asyncio.run(run())


def test_notify_runs_in_background_from_lead_store():
    from business.lead_store import LeadStore

    async def run():
        client = FakeClient([{"code": 0}])
        store = LeadStore()
        store.notifier = FeishuNotifier("https://hook", client=client)
        lead = await store.create_handoff({"user_id": "u1", "conv_id": "c1", "reason": "r"})
        await asyncio.gather(*store.notifier._tasks)
        assert lead["deduplicated"] is False and len(client.payloads) == 1

    asyncio.run(run())


def test_from_env_disabled_without_webhook(monkeypatch):
    monkeypatch.delenv("GOEUROOPS_FEISHU_WEBHOOK", raising=False)
    assert FeishuNotifier.from_env() is None
    monkeypatch.setenv("GOEUROOPS_FEISHU_WEBHOOK", "https://hook")
    assert FeishuNotifier.from_env() is not None
