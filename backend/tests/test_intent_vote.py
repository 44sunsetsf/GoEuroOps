"""意图投票：按大类汇总、同大类细化、确定性分歧检测。"""
from core.intent_recognizer import IntentCategory as I
from core.intent_recognizer import IntentRecognizer


def recognizer() -> IntentRecognizer:
    return IntentRecognizer(api_key="test")


def vote(llm, emb, pat, groups=None):
    r = recognizer()
    pat_result = {"intent": pat[0], "confidence": pat[1]}
    if groups is not None:
        pat_result["groups"] = set(groups)
    return r._vote(
        {"intent": llm[0], "confidence": llm[1]},
        {"intent": emb[0], "confidence": emb[1]},
        pat_result,
    )


def test_same_group_scores_are_merged_before_comparing_domains():
    # 费用 0.385 + 退款 0.12 = 0.505，单看任何一个都低于阈值，合并后才过线
    intent, conf, _ = vote(
        (I.BILLING, 0.55),          # 0.385
        (I.REFUND, 0.6),            # 0.12  → 费用大类合计 0.505
        (I.STUDY_CONSULT, 1.0),     # 0.10
    )
    assert intent == I.BILLING   # 旧逻辑只看费用 0.385，低于阈值会判成 OTHER
    assert conf >= 0.5


def test_pattern_refinement_stays_within_group():
    # 融合结果是留学咨询，关键词命中"退款"：不能被跨领域改写成退款
    intent, _, scores = vote(
        (I.STUDY_CONSULT, 0.9),
        (I.STUDY_CONSULT, 0.5),
        (I.REFUND, 0.5),
        groups=[I.BILLING, I.STUDY_CONSULT],
    )
    assert intent == I.STUDY_CONSULT
    assert "refined_by_pattern" not in scores


def test_pattern_refinement_inside_group_still_works():
    intent, conf, scores = vote((I.BILLING, 0.7), (I.BILLING, 0.3), (I.REFUND, 0.5), groups=[I.BILLING])
    assert intent == I.REFUND
    assert scores["refined_by_pattern"] == 0.5
    assert conf >= 0.5


def test_deterministic_signals_can_veto_confident_llm():
    # LLM 很自信地判成留学咨询，但向量和关键词都强烈指向退款，且消息里没有任何留学关键词
    intent, conf, scores = vote(
        (I.STUDY_CONSULT, 0.95),
        (I.REFUND, 0.8),
        (I.REFUND, 0.75),
        groups=[I.BILLING],
    )
    assert intent == I.OTHER
    assert conf < 0.5          # 低置信 OTHER → 编排器走澄清追问
    assert scores["conflict"] == 1.0


def test_composite_message_is_not_treated_as_conflict():
    # "瑞典怎么申请，另外定金能退吗"：消息里也有留学关键词，属于复合问题，交给协作层处理
    intent, _, scores = vote(
        (I.STUDY_CONSULT, 0.9),
        (I.REFUND, 0.8),
        (I.REFUND, 0.75),
        groups=[I.BILLING, I.STUDY_CONSULT],
    )
    assert intent == I.STUDY_CONSULT
    assert "conflict" not in scores


def test_weak_deterministic_signals_do_not_veto():
    intent, _, scores = vote((I.STUDY_CONSULT, 0.9), (I.REFUND, 0.4), (I.REFUND, 0.5), groups=[I.BILLING])
    assert intent == I.STUDY_CONSULT
    assert "conflict" not in scores


def test_pattern_recognizer_reports_all_keyword_groups():
    pat = recognizer()._pattern_recognize("瑞典CS硕士怎么申请？另外定金能退吗")
    assert pat["intent"] == I.REFUND
    assert {I.BILLING, I.STUDY_CONSULT} <= pat["groups"]


def test_end_to_end_conflict_uses_real_pattern_groups():
    r = recognizer()
    msg = "定金能退吗，我想申请退款"
    pat = r._pattern_recognize(msg)
    intent, _, scores = r._vote(
        {"intent": I.STUDY_CONSULT, "confidence": 0.95},
        {"intent": I.REFUND, "confidence": 0.8},
        pat,
    )
    assert intent == I.OTHER and scores["conflict"] == 1.0


def test_pragmatic_intents_cannot_veto_business_intent():
    # "帮我改成用邮件联系我"：LLM 判 account（正确），两路都判 request，不应否决
    intent, _, scores = vote((I.ACCOUNT, 0.9), (I.REQUEST, 0.85), (I.REQUEST, 0.5), groups=[I.REQUEST])
    assert intent == I.ACCOUNT
    assert "conflict" not in scores


def _counting_recognizer(llm_results):
    """大模型这一路按顺序返回给定结果，记录调用次数；向量路关掉。"""
    import asyncio  # noqa: F401

    r = recognizer()
    r._embedding_enabled = False
    calls = []

    async def fake_llm(message, history):
        calls.append(message)
        return llm_results[min(len(calls), len(llm_results)) - 1]

    r._llm_recognize = fake_llm
    return r, calls


def test_degraded_result_is_not_cached():
    import asyncio

    failed = {"intent": I.OTHER, "confidence": 0.0, "failed": True}
    ok = {"intent": I.REFUND, "confidence": 0.9}
    r, calls = _counting_recognizer([failed, ok])

    first = asyncio.run(r.recognize("定金能退吗"))
    second = asyncio.run(r.recognize("定金能退吗"))
    third = asyncio.run(r.recognize("定金能退吗"))
    assert first.intent == I.REFUND          # 降级：关键词兜底
    assert len(calls) == 2                   # 失败那次没进缓存，第二次重新调了大模型
    assert second.intent == I.REFUND and third is second  # 成功结果才缓存
    assert r.cache_stats["skipped_degraded"] == 1


def test_cache_key_uses_full_text_and_normalizes_writing():
    import asyncio

    r, calls = _counting_recognizer([{"intent": I.GREETING, "confidence": 0.9}])
    long_a = "我想问一下" * 50 + "退款"
    long_b = "我想问一下" * 50 + "发票"
    assert r._cache_key(long_a) != r._cache_key(long_b)   # 前 250 字相同，后面不同
    assert r._cache_key("你好") == r._cache_key(" 你好！") == r._cache_key("你好～")
    assert r._cache_key("ＡＰＳ要多久") == r._cache_key("aps要多久")
    asyncio.run(r.recognize("你好"))
    asyncio.run(r.recognize("你好！"))
    assert len(calls) == 1 and r.cache_hits == 1


def test_pattern_matches_full_width_keywords():
    assert recognizer()._pattern_recognize("ＡＰＳ审核要多久")["intent"] == recognizer()._pattern_recognize("APS审核要多久")["intent"]


def test_cache_metrics_split_first_turn_and_later():
    import asyncio
    from prometheus_client import REGISTRY

    def value(result, turn):
        return REGISTRY.get_sample_value("goeuroops_intent_cache_total", {"result": result, "turn": turn}) or 0.0

    before = (value("hit", "first"), value("miss", "later"))
    r, _ = _counting_recognizer([{"intent": I.GREETING, "confidence": 0.9}])
    asyncio.run(r.recognize("早上好呀"))
    asyncio.run(r.recognize("早上好呀"))
    asyncio.run(r.recognize("那德国呢", history=[{"role": "user", "content": "瑞典怎么申请"}]))
    assert value("hit", "first") - before[0] == 1
    assert value("miss", "later") - before[1] == 1
