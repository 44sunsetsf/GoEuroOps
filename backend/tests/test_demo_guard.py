import asyncio

import pytest
from fastapi.testclient import TestClient

from api import demo_guard
from api.routes import chat as chat_routes
from api.routes import evals as eval_routes
from api.state import services
from api.demo_guard import cost_exceeded, guest_write_allowed, mask_pii

GUEST = {"X-Studio-Role": "guest"}


def test_guest_may_only_read_and_use_counted_endpoints():
    assert guest_write_allowed("GET", "/leads")
    assert guest_write_allowed("POST", "/chat/stream")
    assert guest_write_allowed("POST", "/api/skills/match")
    assert guest_write_allowed("POST", "/search")
    assert guest_write_allowed("POST", "/eval/run")
    for method, path in [("POST", "/skills/reload"), ("POST", "/knowledge/add"), ("POST", "/knowledge/upload"),
                         ("PATCH", "/leads/abc"), ("DELETE", "/anything-new")]:
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

    monkeypatch.setattr(services, "lead_store", FakeLeads())
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

    async def spent(*_):
        return True

    monkeypatch.setattr(services, "orchestrator", object())
    monkeypatch.setattr(services, "memory", object())
    monkeypatch.setattr(chat_routes, "cost_exceeded", spent)
    r = client.post("/chat", json={"message": "hi", "user_id": "u"})
    assert r.status_code == 429 and r.json()["detail"] == demo_guard.COST_LIMIT_MESSAGE


class FakeEvaluator:
    def __init__(self):
        self.calls = []

    async def run(self, **kwargs):
        from types import SimpleNamespace
        self.calls.append(kwargs)
        return SimpleNamespace(pass_rate=1.0, total=0, passed=0, avg_scores={}, regressions=[],
                               recommendations=[], results=[])


def test_guest_eval_uses_defaults_keeps_baseline_and_is_limited(client, monkeypatch):
    from api.quota import DailyQuota
    from evaluation.evaluator import DEFAULT_INTENT_CASES

    ev = FakeEvaluator()
    monkeypatch.setattr(services, "evaluator", ev)
    monkeypatch.setattr(services, "guest_eval_quota", DailyQuota(limit=2))

    async def not_spent(*_):
        return False

    monkeypatch.setattr(eval_routes, "cost_exceeded", not_spent)
    custom = {"intent_cases": [{"message": "x", "expected_intent": "greeting"}] * 500, "compare_rag_gate": True}
    assert client.post("/eval/run", json=custom, headers=GUEST).status_code == 200
    assert ev.calls[-1]["intent_cases"] is DEFAULT_INTENT_CASES           # custom cases ignored
    assert ev.calls[-1]["compare_rag_gate"] is False and ev.calls[-1]["save_baseline"] is False
    assert client.post("/eval/run", headers=GUEST).status_code == 200
    r = client.post("/eval/run", headers=GUEST)
    assert r.status_code == 429 and "次数" in r.json()["detail"]
    assert client.post("/eval/run").status_code == 200                    # owner is not limited
    assert ev.calls[-1]["save_baseline"] is True


def test_guest_eval_refused_while_another_runs(client, monkeypatch):
    import asyncio

    lock = asyncio.Lock()
    asyncio.run(lock.acquire())
    monkeypatch.setattr(services, "guest_eval_lock", lock)
    monkeypatch.setattr(services, "evaluator", FakeEvaluator())

    async def not_spent(*_):
        return False

    monkeypatch.setattr(eval_routes, "cost_exceeded", not_spent)
    r = client.post("/eval/run", headers=GUEST)
    assert r.status_code == 429 and "在运行" in r.json()["detail"]
