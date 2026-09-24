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
