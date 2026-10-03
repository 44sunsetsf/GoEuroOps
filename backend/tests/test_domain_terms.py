"""业务词表的“漂移报警”。

意图识别的关键词（按意图分）和路由打分的关键词（按领域分）用途不同，不强求一致；
但新增业务词时最容易只改一边。这里把两边现有的差异固定下来：只改了一份时测试会失败，
提醒你决定另一份要不要跟着改，然后把这里的快照一起更新。
"""
from agents.agent_orchestrator import _BILLING_INTENTS, _CONSULTING_INTENTS, _GENERAL_INTENTS
from business.domain_terms import (
    INTENT_KEYWORDS_GENERIC,
    INTENT_KEYWORDS_SPECIFIC,
    ROUTING_KEYWORDS_BILLING,
    ROUTING_KEYWORDS_CONSULTING,
    ROUTING_KEYWORDS_CONSULTING_COLLAB,
    ROUTING_KEYWORDS_GENERAL,
)


def _intent_keywords(intents):
    return {kw for table in (INTENT_KEYWORDS_SPECIFIC, INTENT_KEYWORDS_GENERIC)
            for cat, kws in table.items() if cat in intents for kw in kws}


def test_routing_only_keywords_are_known():
    """只在路由里、意图识别不认识的词：这些词会给 Agent 加分，但不会帮助判意图。"""
    assert set(ROUTING_KEYWORDS_CONSULTING) - _intent_keywords(_CONSULTING_INTENTS) == {"留学", "申请", "选校", "文书"}
    assert set(ROUTING_KEYWORDS_BILLING) - _intent_keywords(_BILLING_INTENTS) == set()
    assert set(ROUTING_KEYWORDS_GENERAL) - _intent_keywords(_GENERAL_INTENTS) == {"你们是", "工作室", "帮助"}


def test_billing_intent_only_keywords_are_known():
    """费用领域最要紧：这些词能判出费用类意图，但路由打分时不给费用 Agent 加分。"""
    assert _intent_keywords(_BILLING_INTENTS) - set(ROUTING_KEYWORDS_BILLING) == {
        "能退吗", "抬头", "税号", "付款失败", "付不了", "重复付款", "扣了两次", "payment failed", "费用",
    }


def test_collaboration_keywords_are_a_subset_of_consulting_keywords():
    """复合问题检测用的咨询词，必须同时是咨询领域的打分词。"""
    assert set(ROUTING_KEYWORDS_CONSULTING_COLLAB) <= set(ROUTING_KEYWORDS_CONSULTING)
