import asyncio
from types import SimpleNamespace

from agents.agent_orchestrator import (
    AgentProfile,
    AgentResponse,
    AgentStats,
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
        AgentType.GENERAL: [SimpleNamespace(stats=AgentStats())],
        AgentType.CONSULTING: [SimpleNamespace(stats=AgentStats())],
        AgentType.BILLING: [SimpleNamespace(stats=AgentStats())],
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
    """search_knowledge_base 必须把 Agent 自己的领域传给检索层，避免跨业务线串场；对话历史也要传，用来补全追问。"""
    calls = []

    class RagManager:
        async def search_with_rewrite(self, tool_name, query, top_k=5, context=None, domain=None, history=None):
            calls.append((domain, history))
            return type("Result", (), {"success": True, "data": [], "reranked": True})()

    tools = build_shared_rag_tools(RagManager(), domain="consulting")
    req = make_request()
    req.history = [{"role": "user", "content": "瑞典 CS 什么时候截止"}]
    asyncio.run(tools["search_knowledge_base"].handler(req, {"query": "瑞典硕士"}))

    assert calls == [("consulting", req.history)]


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


def test_rag_mode_off_keeps_tool_list_but_skips_search():
    """门控 off 不再拿掉检索工具（工具列表一变，前缀缓存就全部失效），而是让它直接返回“不需要检索”。"""
    client = TextOnlyClient()
    agent = GeneralAgent(client, "test-model")
    agent.set_shared_tools(_shared_rag())

    off_req = make_request(intent=IntentCategory.GREETING, rag_mode="off")
    asyncio.run(agent.handle(off_req))
    asyncio.run(agent.handle(make_request(rag_mode="on_demand")))
    assert client.calls[0]["tools"] == client.calls[1]["tools"]
    assert "search_knowledge_base" in {tool["name"] for tool in client.calls[0]["tools"]}
    assert "不需要检索知识库" in str(client.calls[0]["messages"])

    skip = agent._tools_for(off_req)["search_knowledge_base"].handler
    result = skip(off_req, {"query": "你好"})
    assert result["skipped"] is True


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
    # 按需 Skill 放进消息，system 保持不变（前缀缓存）
    assert "规则正文" not in client.calls[0]["system"]
    assert "规则正文" in str(client.calls[0]["messages"][0])
    assert "read_skill_reference" in {tool["name"] for tool in client.calls[0]["tools"]}
    assert manager.summary()["skills"][0]["stats"]["hits"] == 1


def test_system_and_tools_stay_identical_across_turns(tmp_path):
    """同一个 Agent：命中不同 Skill、门控不同的两轮，system 和工具列表逐字相同；常驻 Skill 留在 system。"""
    from core.skill_loader import SkillManager

    for sid, body in (("brand", "mode: always\n---\n品牌底线"),
                      ("sales", "intents: [service_inquiry]\n---\n报价规则")):
        d = tmp_path / sid
        (d / "references").mkdir(parents=True)
        (d / "SKILL.md").write_text(
            f"---\nname: {sid}\ndescription: 测试\nagents: [consulting]\n{body}", encoding="utf-8")
    (tmp_path / "sales" / "references" / "faq.md").write_text("# 价格问答\n内容", encoding="utf-8")
    manager = SkillManager(str(tmp_path))
    manager.load()

    client = TextOnlyClient()
    agent = ConsultingAgent(client, "test-model", skill_manager=manager)
    asyncio.run(agent.handle(make_request(intent=IntentCategory.SERVICE_INQUIRY, rag_mode="prefetch")))
    asyncio.run(agent.handle(make_request(intent=IntentCategory.GREETING, rag_mode="off")))

    first, second = client.calls
    assert first["system"] == second["system"]
    assert first["tools"] == second["tools"]
    assert "品牌底线" in first["system"] and "报价规则" not in first["system"]
    assert "报价规则" in str(first["messages"][0]) and "报价规则" not in str(second["messages"])


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
    orch._pool = {t: [SimpleNamespace(stats=AgentStats())] for t in AgentType}
    req = make_request(message="我想申请退款，选校报告还没交付", intent=IntentCategory.REFUND, entities={})
    targets = orch._collaboration_targets(req)
    assert AgentType.CONSULTING not in targets
    assert AgentType.BILLING in targets

    composite = make_request(message="想问问瑞典怎么选校，另外定金能退吗", intent=IntentCategory.REFUND, entities={})
    assert AgentType.CONSULTING in orch._collaboration_targets(composite)


def test_concurrent_requests_on_shared_agent_keep_their_own_records():
    """同一个 Agent 实例同时处理两个请求时，各自的 Skill 记录和工具轨迹不能串。

    Agent 在编排器里是单例，所有请求共用；按请求变化的记录必须跟着请求走。
    旧实现把命中的 Skill 在调模型之前就挂在 self 上，另一个请求一进来就把它覆盖了。
    """
    tool_for = {"问瑞典": ("lookup_country_admissions_overview", {"country": "瑞典"}),
                "问服务": ("lookup_service_offering", {})}

    class Selection:
        def __init__(self, message):
            self.message = message
            self.skill_ids = [f"skill-{message}"]
            self.prompt = ""
            self.has_references = False

        def applied(self):
            return [{"id": f"skill-{self.message}"}]

    class SkillManager:
        def select(self, message, agent_type, **kwargs):
            return Selection(message)

        def record(self, selection, agent_type):
            pass

    class InterleavingClient:
        def __init__(self):
            self.arrived = 0
            self.both_in = asyncio.Event()
            owner = self

            class Messages:
                async def create(inner, **kwargs):
                    msg = kwargs["messages"][-1]["content"]
                    if isinstance(msg, str):              # 第一轮：等两个请求都进来再返回，制造交错
                        owner.arrived += 1
                        if owner.arrived == 2:
                            owner.both_in.set()
                        await owner.both_in.wait()
                        name, args = tool_for[msg]
                        block = type("B", (), {"type": "tool_use", "id": f"id-{name}", "name": name, "input": args})()
                        return type("R", (), {"content": [block]})()
                    text = type("T", (), {"type": "text", "text": "好的"})()
                    return type("R", (), {"content": [text]})()

            self.messages = Messages()

    agent = ConsultingAgent(InterleavingClient(), "test-model", SkillManager())

    async def both():
        return await asyncio.gather(
            agent.handle(make_request(message="问瑞典")),
            agent.handle(make_request(message="问服务")),
        )

    a, b = asyncio.run(both())
    assert a.skills_applied == [{"id": "skill-问瑞典"}]
    assert b.skills_applied == [{"id": "skill-问服务"}]
    assert [t["tool_name"] for t in a.tool_traces] == ["lookup_country_admissions_overview"]
    assert [t["tool_name"] for t in b.tool_traces] == ["lookup_service_offering"]


class ScriptedClient:
    """按顺序返回预设的 content 列表；列表用完后重复最后一个。"""

    def __init__(self, *rounds):
        self.calls = []
        self.rounds = list(rounds)
        owner = self

        class Messages:
            async def create(inner, **kwargs):
                owner.calls.append(kwargs)
                content = owner.rounds.pop(0) if len(owner.rounds) > 1 else owner.rounds[0]
                return type("R", (), {"content": content})()

        self.messages = Messages()


def _tool_block(name, args, block_id=None):
    return type("B", (), {"type": "tool_use", "id": block_id or f"id-{name}", "name": name, "input": args})()


def _text_block(text):
    return type("T", (), {"type": "text", "text": text})()


def test_last_round_forbids_tools_and_answers_from_existing_results():
    lookup = [_tool_block("lookup_country_admissions_overview", {"country": "瑞典"})]
    client = ScriptedClient(lookup, lookup, [_text_block("瑞典的申请一般在一月中旬截止。")])
    agent = ConsultingAgent(client, "test-model")
    response = asyncio.run(agent.handle(make_request()))

    assert response.success is True
    assert response.content == "瑞典的申请一般在一月中旬截止。"
    assert len(client.calls) == 3
    assert "tool_choice" not in client.calls[0] and "tool_choice" not in client.calls[1]
    assert client.calls[2]["tool_choice"] == {"type": "none"}
    assert client.calls[2]["tools"]          # 历史里有 tool_use，工具定义必须保留


def test_tool_use_on_last_round_is_ignored_instead_of_failing():
    from agents.base import ROUND_LIMIT_FALLBACK

    # 模拟不认 tool_choice 的兼容服务：每一轮都要求调工具
    client = ScriptedClient([_tool_block("lookup_country_admissions_overview", {"country": "瑞典"})])
    agent = ConsultingAgent(client, "test-model")
    response = asyncio.run(agent.handle(make_request()))

    assert response.success is True
    assert response.content == ROUND_LIMIT_FALLBACK
    assert len(client.calls) == 3
    names = [t["tool_name"] for t in response.tool_traces]
    assert names == ["lookup_country_admissions_overview"] * 2 + ["tool_round_limit"]


def test_tool_use_on_last_round_keeps_text_if_model_wrote_some():
    client = ScriptedClient([_text_block("根据已查到的信息：瑞典一月截止。"),
                             _tool_block("lookup_country_admissions_overview", {"country": "瑞典"})])
    agent = ConsultingAgent(client, "test-model")
    response = asyncio.run(agent.handle(make_request()))

    assert response.success is True
    assert response.content == "根据已查到的信息：瑞典一月截止。"


def test_same_round_reads_run_concurrently_and_writes_run_in_order():
    from tooling.gateway import ToolOutcome

    class RecordingGateway:
        def __init__(self):
            self.active = 0
            self.max_reads_active = 0
            self.writes_active = 0
            self.max_writes_active = 0
            self.order = []

        async def call(self, spec, req, args):
            write = spec.policy.side_effect == "write"
            if write:
                self.writes_active += 1
                self.max_writes_active = max(self.max_writes_active, self.writes_active)
            else:
                self.active += 1
                self.max_reads_active = max(self.max_reads_active, self.active)
            await asyncio.sleep(0.01)
            if write:
                self.writes_active -= 1
            else:
                self.active -= 1
            self.order.append((spec.name, args.get("name") or args.get("country") or args.get("service_id")))
            return ToolOutcome(result={"success": True}, ok=True, kind="ok", result_success=True)

        def record_rejected(self, kind):
            pass

    batch = [
        _tool_block("create_consultation_lead", {"name": "A"}, "w1"),
        _tool_block("lookup_country_admissions_overview", {"country": "瑞典"}, "r1"),
        _tool_block("not_a_tool", {}, "x1"),
        _tool_block("lookup_country_admissions_overview", {"country": "德国"}, "r2"),
        _tool_block("create_consultation_lead", {"name": "B"}, "w2"),
    ]
    client = ScriptedClient(batch, [_text_block("好的")])
    agent = ConsultingAgent(client, "test-model")
    gateway = RecordingGateway()
    agent._gateway = gateway
    response = asyncio.run(agent.handle(make_request()))

    assert response.success is True
    assert gateway.max_reads_active == 2       # 两个只读查询同时在跑
    assert gateway.max_writes_active == 1      # 写入从不重叠
    writes = [x for x in gateway.order if x[0] == "create_consultation_lead"]
    assert writes == [("create_consultation_lead", "A"), ("create_consultation_lead", "B")]
    # trace 和 tool_result 仍按模型请求的顺序
    assert [t["tool_use_id"] for t in response.tool_traces] == ["w1", "r1", "x1", "r2", "w2"]
    results = client.calls[1]["messages"][-1]["content"]
    assert [r["tool_use_id"] for r in results] == ["w1", "r1", "x1", "r2", "w2"]
    assert response.tool_traces[2]["outcome"] == "not_allowed"
