import asyncio

import pytest
from fastapi.testclient import TestClient

from api import demo_guard
from api.demo_guard import cost_exceeded, guest_write_allowed, mask_pii

GUEST = {"X-Studio-Role": "guest"}


def test_guest_may_only_read_and_use_counted_endpoints():
    assert guest_write_allowed("GET", "/leads")
    assert guest_write_allowed("POST", "/chat/stream")
    assert guest_write_allowed("POST", "/api/skills/match")
    assert guest_write_allowed("POST", "/search")
    for method, path in [("POST", "/skills/reload"), ("POST", "/knowledge/add"), ("POST", "/knowledge/upload"),
                         ("POST", "/eval/run"), ("PATCH", "/leads/abc"), ("DELETE", "/anything-new")]:
        assert not guest_write_allowed(method, path), (method, path)


def test_mask_contact_field_and_every_other_mention():
    lead = {
        "name": "小王",
        "contact_channel": "wechat",
        "contact": "wang_2027",
        "notes": "加微信 wang_2027 时备注留学",
        "summary": {"last_message": "我的邮箱 someone@example.com，电话 138 1234 5678，微信号：li_si_99"},
    }
    masked = mask_pii(lead)
    assert masked["contact"] == "wa*****27"
    assert "wang_2027" not in str(masked)
    assert "someone@example.com" not in str(masked) and "so***@example.com" in str(masked)
    assert "1234 5678" not in str(masked)
    assert "li_si_99" not in str(masked)
    assert masked["contact_channel"] == "wechat" and masked["name"] == "小王"
    assert lead["contact"] == "wang_2027"   # original untouched


def test_mask_leaves_ordinary_numbers_alone():
    assert mask_pii("均分 85，雅思 7.0，预算 30000 元") == "均分 85，雅思 7.0，预算 30000 元"


class FakeRedis:
    def __init__(self, spent):
        self.spent = spent

    async def hget(self, key, field):
        return self.spent


def test_cost_limit(monkeypatch):
    monkeypatch.setenv("GOEUROOPS_DAILY_COST_LIMIT", "3")
    assert asyncio.run(cost_exceeded(FakeRedis("2.99"))) is False
    assert asyncio.run(cost_exceeded(FakeRedis("3.0"))) is True
    assert asyncio.run(cost_exceeded(FakeRedis(None))) is False
    monkeypatch.setenv("GOEUROOPS_DAILY_COST_LIMIT", "0")
    assert asyncio.run(cost_exceeded(FakeRedis("999"))) is False


def test_cost_limit_fails_open_when_redis_breaks(monkeypatch):
    class Broken:
        async def hget(self, key, field):
            raise ConnectionError("down")

    monkeypatch.setenv("GOEUROOPS_DAILY_COST_LIMIT", "3")
    assert asyncio.run(cost_exceeded(Broken())) is False


@pytest.fixture
def client(monkeypatch):
    from api import main

    class FakeLeads:
        async def list(self, **kwargs):
            return [{"id": "1", "contact": "wang_2027", "notes": "微信 wang_2027"}]

        async def stats(self):
            return {"total": 1}

    monkeypatch.setattr(main, "_lead_store", FakeLeads())
    return TestClient(main.app)   # no `with`: lifespan (models, Redis) is not started


def test_session_role(client):
    assert client.get("/session", headers=GUEST).json()["role"] == "guest"
    assert client.get("/session").json()["role"] == "owner"


def test_guest_write_is_forbidden_but_owner_passes_the_guard(client):
    r = client.post("/skills/reload", headers=GUEST)
    assert r.status_code == 403 and "只读" in r.json()["detail"]
    assert client.post("/skills/reload").status_code == 503   # owner reaches the handler (not initialised in tests)


def test_leads_are_masked_for_guest_only(client):
    guest = client.get("/leads", headers=GUEST).json()["items"][0]
    owner = client.get("/leads").json()["items"][0]
    assert guest["contact"] == "wa*****27" and "wang_2027" not in guest["notes"]
    assert owner["contact"] == "wang_2027"


def test_chat_refused_once_daily_cost_is_spent(client, monkeypatch):
    from api import main

    async def spent(*_):
        return True

    monkeypatch.setattr(main, "_orchestrator", object())
    monkeypatch.setattr(main, "_memory", object())
    monkeypatch.setattr(main, "cost_exceeded", spent)
    r = client.post("/chat", json={"message": "hi", "user_id": "u"})
    assert r.status_code == 429 and r.json()["detail"] == demo_guard.COST_LIMIT_MESSAGE
