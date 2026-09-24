import asyncio
from datetime import date

from business.catalog import get_catalog, get_countries, match_service_mentions
from business.lead_store import LeadStore, mask_contact, validate_lead_input
from business.pricing import calculate_refund, quote_bundle

BEFORE_EARLY_BIRD = date(2026, 9, 1)
AFTER_EARLY_BIRD = date(2026, 11, 1)


# ── 目录 ─────────────────────────────────────────────────────────────────────

def test_catalog_loads_and_skus_unique():
    catalog = get_catalog()
    skus = [s.sku for s in catalog.services]
    assert len(skus) == len(set(skus))
    assert catalog.service("full_journey").price == 11800
    assert catalog.service("intro_call").price == 0


def test_country_lookup_by_alias():
    book = get_countries()
    assert book.find("Sweden").key == "瑞典"
    assert book.find("dk").key == "丹麦"
    assert book.find("美国") is None
    assert {c.key for c in book.countries} == {"瑞典", "德国", "荷兰", "芬兰", "丹麦"}


def test_service_mentions_prefers_specific_alias():
    assert match_service_mentions("我想了解全程陪跑Plus") == ["full_journey_plus"]
    assert match_service_mentions("选校全案加3个项目文书") == ["selection_full", "essay_pack_3"]
    # ASCII 别名按单词边界匹配，"cvs" 不应命中 cv
    assert match_service_mentions("cvs 药店") == []


# ── 报价 ─────────────────────────────────────────────────────────────────────

def test_quote_plain_sum_and_payment_plan():
    q = quote_bundle([{"sku": "selection_full"}, {"sku": "cv"}], today=AFTER_EARLY_BIRD)
    assert q["success"]
    assert q["subtotal"] == 2999 + 899
    assert q["total"] == 3898
    # 选校部分付 30% 定金，CV 全款
    assert q["payment_plan"]["due_at_signing"] == round(2999 * 0.3) + 899


def test_quote_early_bird_only_on_eligible_items():
    q = quote_bundle(
        [{"sku": "selection_full"}, {"sku": "ps_single"}],
        early_bird=True,
        today=BEFORE_EARLY_BIRD,
    )
    assert q["discounts"][0]["code"] == "early_bird"
    assert q["discounts"][0]["amount"] == -round(2999 * 0.1)
    assert q["total"] == 2999 + 2499 - 300


def test_quote_early_bird_expired_adds_note():
    q = quote_bundle([{"sku": "full_journey"}], early_bird=True, today=AFTER_EARLY_BIRD)
    assert q["total"] == 11800
    assert any("早鸟" in note for note in q["notes"])


def test_quote_picks_better_of_early_bird_and_group_then_stacks_referral():
    q = quote_bundle(
        [{"sku": "full_journey"}],
        early_bird=True,
        group_size=2,
        referral=True,
        today=BEFORE_EARLY_BIRD,
    )
    codes = [d["code"] for d in q["discounts"]]
    assert codes == ["early_bird", "referral"]
    assert q["total"] == 11800 - 1180 - 300


def test_quote_floor_protection():
    # 早鸟 9 折 + 老带新 300 对 ¥2,999 的选校全案：2999-300-300=2399 < 85%=2550
    q = quote_bundle([{"sku": "selection_full"}], early_bird=True, referral=True, today=BEFORE_EARLY_BIRD)
    assert q["total"] == 2550
    assert q["discounts"][-1]["code"] == "floor_adjust"


def test_quote_rejects_unknown_sku():
    q = quote_bundle([{"sku": "guaranteed_offer"}])
    assert not q["success"]
    assert "guaranteed_offer" in q["error"]


# ── 退款 ─────────────────────────────────────────────────────────────────────

def test_refund_not_started_is_full():
    r = calculate_refund("full_journey", 3540, "not_started")
    assert r["estimated_refund"] == 3540
    assert r["requires_human_review"]


def test_refund_delivered_is_zero():
    assert calculate_refund("selection_full", 2999, "delivered")["estimated_refund"] == 0


def test_refund_essay_by_remaining_rounds():
    r = calculate_refund("ps_single", 2499, "in_progress", completed_rounds=1)
    assert r["estimated_refund"] == round(2499 * 2 / 3, 2)


def test_refund_essay_without_rounds_asks_for_input():
    r = calculate_refund("ps_single", 2499, "in_progress")
    assert "completed_rounds" in r["needs_input"]


def test_refund_selection_in_progress_half():
    assert calculate_refund("selection_full", 2999, "in_progress")["estimated_refund"] == 1499.5


def test_refund_package_keeps_deposit():
    r = calculate_refund("full_journey", 11800, "in_progress", progress_percent=50)
    deposit = round(11800 * 0.3)
    assert r["estimated_refund"] == round((11800 - deposit) * 0.5, 2)


def test_refund_consult_in_progress_is_zero():
    assert calculate_refund("consult_single", 699, "in_progress")["estimated_refund"] == 0


# ── 线索 ─────────────────────────────────────────────────────────────────────

def _lead(**overrides):
    data = {
        "name": "小王",
        "contact_channel": "wechat",
        "contact": "wang_2027",
        "consent": True,
    }
    data.update(overrides)
    return data


def test_lead_validation_requires_consent_and_valid_contact():
    assert validate_lead_input(_lead()) == []
    assert any("consent" in e for e in validate_lead_input(_lead(consent=False)))
    assert any("邮箱" in e for e in validate_lead_input(_lead(contact_channel="email", contact="abc")))


def test_lead_validation_rejects_id_numbers():
    errors = validate_lead_input(_lead(background="身份证 110101199001011234"))
    assert any("证件" in e for e in errors)


def test_mask_contact():
    assert mask_contact("wang_2027") == "wa*****27"
    assert mask_contact("someone@example.com") == "so***@example.com"


class BrokenRedis:
    async def set(self, *a, **k):
        raise ConnectionError("down")

    async def get(self, *a, **k):
        raise ConnectionError("down")

    async def zrevrange(self, *a, **k):
        raise ConnectionError("down")


def test_lead_store_memory_fallback_roundtrip():
    async def run():
        store = LeadStore(redis_client=BrokenRedis())
        lead = await store.create(_lead(countries=["瑞典"]))
        assert store.backend == "memory"
        assert lead["id"].startswith("L")
        await store.create({"reason": "用户要求转人工"}, lead_type="handoff")
        assert len(await store.list()) == 2
        assert len(await store.list(lead_type="handoff")) == 1
        updated = await store.update(lead["id"], status="contacted", notes="已加微信")
        assert updated["status"] == "contacted"
        stats = await store.stats()
        assert stats["by_status"]["contacted"] == 1

    asyncio.run(run())


def test_handoff_is_idempotent_per_open_conversation():
    async def run():
        store = LeadStore()
        base = {"user_id": "u1", "conv_id": "c1", "reason": "用户要求真人顾问", "last_message": "转人工"}
        first = await store.create_handoff(base)
        second = await store.create_handoff({**base, "last_message": "怎么还没人联系我"})
        other_conv = await store.create_handoff({**base, "conv_id": "c2"})

        assert first["deduplicated"] is False
        assert second["deduplicated"] is True and second["id"] == first["id"]
        assert second["followups"][-1]["message"] == "怎么还没人联系我"
        assert other_conv["id"] != first["id"]
        assert len(await store.list(lead_type="handoff")) == 2

        # 顾问关单后，同一会话再要求真人会开新单
        await store.update(first["id"], status="closed")
        reopened = await store.create_handoff(base)
        assert reopened["deduplicated"] is False and reopened["id"] != first["id"]

    asyncio.run(run())
