import asyncio

from agents.agent_orchestrator import (
    AgentProfile,
    AgentResponse,
    AgentType,
    AgentOrchestrator,
    BillingAgent,
    ConsultingAgent,
    EscalationAgent,
    GeneralAgent,
    Request,
    ResponseComposer,
    RoutingDecision,
    build_shared_rag_tools,
)
from core.intent_recognizer import IntentCategory, UrgencyLevel


class FakeClient:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []

        class Messages:
            async def create(inner, **kwargs):
                self.calls.append(kwargs)
                if self.error:
                    raise self.error
                return self.response

        self.messages = Messages()


def make_request(**kwargs):
    values = {
        "message": "咨询瑞典的CS硕士申请，另外我交的定金能退款吗",
        "user_id": "u1",
        "conv_id": "c1",
        "intent": IntentCategory.APPLICATION_PROCESS,
        "intent_group": "study_consult",
        "urgency": UrgencyLevel.HIGH,
        "intent_confidence": 0.92,
        "entities": {"country": ["瑞典"], "amount": ["3540 元"]},
    }
    values.update(kwargs)
    return Request(**values)


def test_agent_profiles_have_distinct_contracts_and_generation_config():
    assert isinstance(GeneralAgent.profile, AgentProfile)
    assert GeneralAgent.profile.role != ConsultingAgent.profile.role
    assert ConsultingAgent.profile.workflow != BillingAgent.profile.workflow
    assert ConsultingAgent.profile.temperature != BillingAgent.profile.temperature
    assert "search_knowledge_base" in GeneralAgent.profile.tool_scope
    assert "lookup_country_admissions_overview" in ConsultingAgent.profile.tool_scope
    assert "calculate_refund" in BillingAgent.profile.tool_scope


def test_domain_agents_build_different_role_packets():
    req = make_request()
    general_packet = GeneralAgent(FakeClient(), "test-model")._build_role_packet(req)
    consulting_packet = ConsultingAgent(FakeClient(), "test-model")._build_role_packet(req)
    billing_packet = BillingAgent(FakeClient(), "test-model")._build_role_packet(req)

    assert "triage_targets" in general_packet
    assert "consulting_fields" in consulting_packet
    assert "verification_fields" in billing_packet
    assert general_packet != consulting_packet != billing_packet


def test_escalation_agent_is_a_real_non_llm_handoff_node():
    client = FakeClient()
    agent = EscalationAgent(client, "test-model")

    result = asyncio.run(agent.handle(make_request(
        intent=IntentCategory.HUMAN_HANDOFF,
        urgency=UrgencyLevel.CRITICAL,
    )))

    assert result.success is True
    assert result.escalate is True
    assert "转交给工作室顾问" in result.content
    assert client.calls == []


def test_escalation_agent_writes_handoff_ticket():
    from business.lead_store import LeadStore

    store = LeadStore()
    agent = EscalationAgent(FakeClient(), "test-model")
    agent.set_lead_store(store)

    result = asyncio.run(agent.handle(make_request(intent=IntentCategory.DATA_PRIVACY)))
    tickets = asyncio.run(store.list(lead_type="handoff"))

    assert len(tickets) == 1
    assert tickets[0]["id"] in result.content
    assert result.tools_used == ["create_handoff_summary"]
    assert result.tool_traces[0]["success"] is True


def test_composer_fallback_preserves_primary_and_supporting_results():
    composer = ResponseComposer(FakeClient(error=RuntimeError("provider down")), "test-model")
    req = make_request()
    responses = [
        AgentResponse(AgentType.CONSULTING, "瑞典CS硕士通常是2年制，具体以院校官网为准。", True),
        AgentResponse(AgentType.BILLING, "请提供合同号和两笔付款的时间和金额。", True),
    ]

    content = asyncio.run(composer.compose(req, responses))

    assert content.startswith("瑞典CS硕士通常是2年制，具体以院校官网为准。")
    assert "补充说明" in content
    assert "两笔付款" in content


def test_routing_decision_can_target_escalation_pool():
    # Keep this assertion close to the public data contract used by the API.
    decision = RoutingDecision(
        primary_agent=AgentType.ESCALATION,
        reason="critical request",
        confidence=1.0,
    )
    assert decision.agent_types == [AgentType.ESCALATION]
    assert not decision.multi_agent


def test_composite_request_routes_explicit_billing_signal_as_supporting_agent():
    orchestrator = AgentOrchestrator.__new__(AgentOrchestrator)
    orchestrator._pool = {
        AgentType.GENERAL: [object()],
        AgentType.CONSULTING: [object()],
        AgentType.BILLING: [object()],
    }

    decision = orchestrator._route_decision(make_request())

    assert decision.primary_agent is AgentType.CONSULTING
    assert decision.supporting_agents == [AgentType.BILLING]
    assert decision.multi_agent is True


def test_agent_tool_scopes_are_real_and_isolated():
    general_tools = set(GeneralAgent(FakeClient(), "test-model").get_tools())
    consulting_tools = set(ConsultingAgent(FakeClient(), "test-model").get_tools())
    billing_tools = set(BillingAgent(FakeClient(), "test-model").get_tools())
    escalation_tools = set(EscalationAgent(FakeClient(), "test-model").get_tools())

    assert general_tools == {"inspect_request_context", "suggest_required_fields", "get_studio_profile"}
    assert consulting_tools == {
        "lookup_country_admissions_overview",
        "lookup_service_offering",
        "quote_service_bundle",
        "create_consultation_lead",
    }
    assert billing_tools == {"check_payment_fields", "calculate_refund", "get_payment_policy", "compare_amounts"}
    assert escalation_tools == {"create_handoff_summary"}
    assert not general_tools & consulting_tools
    assert not consulting_tools & billing_tools


def test_shared_rag_tool_is_available_to_all_agents():
    class RagManager:
        async def search_with_rewrite(self, tool_name, query, top_k=5, context=None, domain=None):
            return type(
                "Result",
                (),
                {"success": True, "data": [{"title": "退款政策", "content": "7 天内可退款"}], "reranked": True},
            )()

    shared = build_shared_rag_tools(RagManager())

    general = GeneralAgent(FakeClient(), "test-model")
    consulting = ConsultingAgent(FakeClient(), "test-model")
    billing = BillingAgent(FakeClient(), "test-model")
    escalation = EscalationAgent(FakeClient(), "test-model")

    for agent in (general, consulting, billing, escalation):
        agent.set_shared_tools(shared)
        tools = agent.get_tools()
        assert "search_knowledge_base" in tools


def test_rag_tool_scopes_search_to_its_own_domain():
    """search_knowledge_base 必须把 Agent 自己的领域传给检索层，避免跨业务线串场。"""
    calls = []

    class RagManager:
        async def search_with_rewrite(self, tool_name, query, top_k=5, context=None, domain=None):
            calls.append(domain)
            return type("Result", (), {"success": True, "data": [], "reranked": True})()

    tools = build_shared_rag_tools(RagManager(), domain="consulting")
    req = make_request()
    asyncio.run(tools["search_knowledge_base"].handler(req, {"query": "瑞典硕士"}))

    assert calls == ["consulting"]


def test_tool_input_validation_rejects_unknown_fields():
    agent = ConsultingAgent(FakeClient(), "test-model")
    spec = agent.get_tools()["lookup_country_admissions_overview"]

    try:
        agent._validate_tool_input(spec, {"country": "瑞典", "secret": "nope"})
    except ValueError as exc:
        assert "不允许的工具参数" in str(exc)
    else:
        raise AssertionError("unknown tool fields should be rejected")


def test_tool_use_round_trip_executes_only_whitelisted_tool():
    class ToolUseBlock:
        type = "tool_use"
        id = "toolu_1"
        name = "lookup_country_admissions_overview"
        input = {"country": "瑞典"}

    class TextBlock:
        type = "text"
        text = "已根据瑞典的项目情况给出参考信息。"

    class ToolClient:
        def __init__(self):
            self.calls = []
            self.responses = [
                type("Response", (), {"content": [ToolUseBlock()]})(),
                type("Response", (), {"content": [TextBlock()]})(),
            ]

        class Messages:
            def __init__(self, owner):
                self.owner = owner

            async def create(self, **kwargs):
                self.owner.calls.append(kwargs)
                return self.owner.responses.pop(0)

        @property
        def messages(self):
            return self.Messages(self)

    client = ToolClient()
    agent = ConsultingAgent(client, "test-model")
    response = asyncio.run(agent.handle(make_request()))

    assert response.success is True
    assert response.tools_used == ["lookup_country_admissions_overview"]
    assert len(client.calls) == 2
    assert {tool["name"] for tool in client.calls[0]["tools"]} == {
        "lookup_country_admissions_overview",
        "lookup_service_offering",
        "quote_service_bundle",
        "create_consultation_lead",
    }
    assert "tool_result" in str(client.calls[1]["messages"])
    # 成功路径上也要保留工具轨迹（之前只有失败路径会写入）
    assert [t["tool_name"] for t in response.tool_traces] == ["lookup_country_admissions_overview"]
    assert response.tool_traces[0]["result_success"] is True


class TextOnlyClient:
    def __init__(self, text="好的"):
        self.calls = []
        owner = self

        class Messages:
            async def create(inner, **kwargs):
                owner.calls.append(kwargs)
                return type("R", (), {"content": [type("T", (), {"type": "text", "text": text})()]})()

        self.messages = Messages()


def _shared_rag():
    class RagManager:
        async def search_with_rewrite(self, *a, **k):
            return type("Result", (), {"success": True, "data": [], "reranked": False})()

    return build_shared_rag_tools(RagManager(), domain="general")


def test_rag_mode_off_hides_search_tool():
    client = TextOnlyClient()
    agent = GeneralAgent(client, "test-model")
    agent.set_shared_tools(_shared_rag())

    asyncio.run(agent.handle(make_request(intent=IntentCategory.GREETING, rag_mode="off")))
    names = {tool["name"] for tool in client.calls[0]["tools"]}
    assert "search_knowledge_base" not in names

    asyncio.run(agent.handle(make_request(rag_mode="on_demand")))
    names = {tool["name"] for tool in client.calls[1]["tools"]}
    assert "search_knowledge_base" in names


def test_prefetched_knowledge_is_injected_with_titles():
    client = TextOnlyClient()
    agent = ConsultingAgent(client, "test-model")
    req = make_request(
        rag_mode="prefetch",
        knowledge=[{"title": "瑞典CS硕士申请概览", "content": "1 月中旬截止", "score": 0.6}],
    )
    asyncio.run(agent.handle(req))
    messages = str(client.calls[0]["messages"])
    assert "[知识库上下文]" in messages
    assert "《瑞典CS硕士申请概览》" in messages


def test_skill_selection_is_reported_and_reference_tool_exposed(tmp_path):
    from core.skill_loader import SkillManager

    skill_dir = tmp_path / "study_consulting"
    (skill_dir / "references").mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: 留学咨询\ndescription: 测试\nagents: [consulting]\nintents: [application_process]\n---\n规则正文",
        encoding="utf-8",
    )
    (skill_dir / "references" / "faq.md").write_text("# 常见问题\n内容", encoding="utf-8")
    manager = SkillManager(str(tmp_path))
    manager.load()

    client = TextOnlyClient()
    agent = ConsultingAgent(client, "test-model", skill_manager=manager)
    response = asyncio.run(agent.handle(make_request()))

    assert [s["id"] for s in response.skills_applied] == ["study_consulting"]
    assert "规则正文" in client.calls[0]["system"]
    assert "read_skill_reference" in {tool["name"] for tool in client.calls[0]["tools"]}
    assert manager.summary()["skills"][0]["stats"]["hits"] == 1


def test_escalation_does_not_open_duplicate_tickets():
    from business.lead_store import LeadStore

    store = LeadStore()
    agent = EscalationAgent(FakeClient(), "test-model")
    agent.set_lead_store(store)

    first = asyncio.run(agent.handle(make_request(intent=IntentCategory.HUMAN_HANDOFF, message="转人工")))
    second = asyncio.run(agent.handle(make_request(intent=IntentCategory.HUMAN_HANDOFF, message="还没人联系我")))

    tickets = asyncio.run(store.list(lead_type="handoff"))
    assert len(tickets) == 1
    assert tickets[0]["id"] in first.content and tickets[0]["id"] in second.content
    assert "不会重复排队" in second.content
    assert second.tool_traces[0]["deduplicated"] is True


def test_deliverable_name_does_not_pull_in_consulting_agent():
    orch = AgentOrchestrator.__new__(AgentOrchestrator)
    orch._pool = {t: [object()] for t in AgentType}
    req = make_request(message="我想申请退款，选校报告还没交付", intent=IntentCategory.REFUND, entities={})
    targets = orch._collaboration_targets(req)
    assert AgentType.CONSULTING not in targets
    assert AgentType.BILLING in targets

    composite = make_request(message="想问问瑞典怎么选校，另外定金能退吗", intent=IntentCategory.REFUND, entities={})
    assert AgentType.CONSULTING in orch._collaboration_targets(composite)
