import asyncio

from core.rag_gate import RAG_POLICY, RagGate, RagMode


def make_search(results_by_domain, calls=None):
    async def search(query, top_k, domain):
        if calls is not None:
            calls.append(domain)
        await asyncio.sleep(0)
        return list(results_by_domain.get(domain, []))

    return search


def resolve(gate, intent, confidence, domain="consulting", message="瑞典申请截止是什么时候"):
    async def run():
        speculative = gate.speculate(message)
        return await gate.resolve(speculative, intent=intent, confidence=confidence, domain=domain, message=message)

    return asyncio.run(run())


def test_policy_covers_knowledge_and_chitchat_intents():
    assert RAG_POLICY["application_process"] is RagMode.PREFETCH
    assert RAG_POLICY["refund"] is RagMode.PREFETCH
    assert RAG_POLICY["greeting"] is RagMode.OFF
    assert RAG_POLICY["data_privacy"] is RagMode.OFF
    assert RAG_POLICY["query"] is RagMode.ON_DEMAND


def test_prefetch_uses_routed_domain_results():
    gate = RagGate(make_search({
        "consulting": [{"title": "瑞典概览", "content": "1 月截止", "score": 0.5}],
        "billing": [{"title": "退款政策", "content": "...", "score": 0.9}],
    }), min_confidence=0.6, top_k=3, min_score=0.0)

    decision = resolve(gate, "application_process", 0.9)

    assert decision.mode is RagMode.PREFETCH
    assert [item["title"] for item in decision.items] == ["瑞典概览"]
    assert decision.to_dict()["prefetched"] == 1


def test_low_confidence_falls_back_to_on_demand():
    gate = RagGate(make_search({"consulting": [{"title": "x", "content": "y", "score": 0.9}]}),
                   min_confidence=0.6, top_k=3, min_score=0.0)

    decision = resolve(gate, "application_process", 0.4)

    assert decision.mode is RagMode.ON_DEMAND
    assert decision.items == []
    assert "置信度" in decision.reason


def test_low_confidence_does_not_switch_search_off():
    gate = RagGate(make_search({}), min_confidence=0.6, top_k=3, min_score=0.0)
    assert resolve(gate, "greeting", 0.3).mode is RagMode.ON_DEMAND
    assert resolve(gate, "greeting", 0.9).mode is RagMode.OFF


def test_low_score_items_are_dropped_and_empty_result_not_injected():
    gate = RagGate(make_search({"consulting": [{"title": "弱相关", "content": "...", "score": -0.2}]}),
                   min_confidence=0.6, top_k=3, min_score=0.1)

    decision = resolve(gate, "study_consult", 0.9)

    assert decision.mode is RagMode.PREFETCH
    assert decision.items == []
    assert decision.dropped_low_score == 1
    assert "不注入" in decision.reason


def test_speculative_tasks_are_cancelled_when_gate_is_off():
    started = []

    async def slow_search(query, top_k, domain):
        started.append(domain)
        await asyncio.sleep(10)
        return []

    gate = RagGate(slow_search, min_confidence=0.6, top_k=3, min_score=0.0)

    async def run():
        speculative = gate.speculate("你好")
        await asyncio.sleep(0)
        decision = await gate.resolve(speculative, intent="greeting", confidence=0.95, domain="general", message="你好")
        await asyncio.sleep(0)
        return decision, speculative

    decision, speculative = asyncio.run(run())
    assert decision.mode is RagMode.OFF
    assert all(task.cancelled() for task in speculative.values())


def test_gate_without_search_fn_never_prefetches():
    gate = RagGate(None, min_confidence=0.6)
    assert gate.speculate("瑞典") is None
    decision = resolve(gate, "study_consult", 0.9)
    assert decision.mode is RagMode.ON_DEMAND


def test_orchestrator_prefetch_runs_in_parallel_with_intent_recognition():
    """投机预取应该和意图识别同时开始，而不是等意图识别完成后才开始。"""
    from agents.agent_orchestrator import AgentOrchestrator, AgentType, Request
    from core.intent_recognizer import IntentCategory, IntentResult, UrgencyLevel

    timeline = []

    class SlowRecognizer:
        async def recognize(self, message, history=None):
            timeline.append("intent_start")
            await asyncio.sleep(0.05)
            timeline.append("intent_end")
            return IntentResult(
                intent=IntentCategory.APPLICATION_PROCESS, confidence=0.9, urgency=UrgencyLevel.LOW,
                intent_group="study_consult", entities={"country": ["瑞典"]}, reasoning="", latency_ms=50,
            )

    class ToolManager:
        async def search_fast(self, tool_name, query, top_k=3, domain=None):
            timeline.append(f"search_{domain}")
            return [{"title": f"{domain}-doc", "content": "内容", "score": 0.5}]

        async def search_with_rewrite(self, *a, **k):
            raise AssertionError("预取路径不应走改写+重排")

    captured = {}

    from agents.agent_orchestrator import AgentStats

    class FakeAgent:
        agent_type = AgentType.CONSULTING
        stats = AgentStats()

        async def handle(self, req, on_delta=None):
            from agents.agent_orchestrator import AgentResponse
            captured["knowledge"] = req.knowledge
            captured["rag_mode"] = req.rag_mode
            return AgentResponse(AgentType.CONSULTING, "回答", True)

    orch = AgentOrchestrator(api_key="test", rag_tool_manager=ToolManager())
    orch._intent_recognizer = SlowRecognizer()
    orch._pool[AgentType.CONSULTING] = [FakeAgent()]
    orch._rag_gate.min_score = 0.0

    result = asyncio.run(orch.run(Request(message="瑞典CS硕士申请截止是什么时候", user_id="u", conv_id="c")))

    assert timeline.index("search_consulting") < timeline.index("intent_end")
    assert captured["rag_mode"] == "prefetch"
    assert [item["title"] for item in captured["knowledge"]] == ["consulting-doc"]
    assert result.rag_gate["mode"] == "prefetch"
    assert result.entities == {"country": ["瑞典"]}
