import asyncio
import json
import time

import pytest

from agents.agent_orchestrator import BillingAgent, ConsultingAgent, Request
from agents.tools import billing_tools, consulting_tools, make_model_tool
from business.catalog import get_catalog
from business.pricing import calculate_refund, quote_bundle
from core.intent_recognizer import IntentCategory, UrgencyLevel
from tooling.amount_guard import NOTICE, AmountGrounding, AmountGuard, extract_amounts
from tooling.breaker import CircuitState
from tooling.gateway import ToolGateway, ToolPolicy, shrink_result, validate_args
from tooling.schemas import QuoteArgs, QuoteResult, RefundResult, schema_for


def _req(message="帮我报个价", **kwargs):
    values = dict(message=message, user_id="u1", conv_id="c1", intent=IntentCategory.APPLICATION_PROCESS,
                  intent_group="study_consult", urgency=UrgencyLevel.LOW, intent_confidence=0.9, entities={})
    values.update(kwargs)
    return Request(**values)


def _run(coro):
    return asyncio.run(coro)


# ── Pydantic 定义生成的 JSON Schema ─────────────────────────────────────────

def test_schema_is_flat_and_strict():
    schema = schema_for(QuoteArgs)
    text = json.dumps(schema)
    assert "$defs" not in text and "$ref" not in text and "anyOf" not in text
    assert schema["additionalProperties"] is False
    item = schema["properties"]["items"]["items"]
    assert item["additionalProperties"] is False
    assert item["properties"]["qty"]["maximum"] == 20


def test_every_agent_tool_has_a_pydantic_model_and_policy():
    tools = {**consulting_tools(None), **billing_tools()}
    for spec in tools.values():
        assert spec.input_model is not None, spec.name
        assert spec.policy.side_effect in ("read", "write", "external")
    assert tools["create_consultation_lead"].policy.side_effect == "write"
    assert tools["create_consultation_lead"].policy.breaker is True
    assert tools["quote_service_bundle"].output_model is QuoteResult
    assert tools["calculate_refund"].output_model is RefundResult


# ── 入参校验 ────────────────────────────────────────────────────────────────

def test_nested_arguments_are_validated():
    spec = consulting_tools(None)["quote_service_bundle"]
    with pytest.raises(ValueError, match="qty"):
        validate_args(spec, {"items": [{"sku": "selection_full", "qty": 99}]})
    with pytest.raises(ValueError, match="不允许的工具参数"):
        validate_args(spec, {"items": [{"sku": "selection_full", "price": 1}]})
    with pytest.raises(ValueError, match="缺少必需参数"):
        validate_args(spec, {})
    data = validate_args(spec, {"items": [{"sku": "selection_full"}]})
    assert data["items"] == [{"sku": "selection_full", "qty": 1}]


def test_enum_arguments_are_validated():
    spec = billing_tools()["calculate_refund"]
    with pytest.raises(ValueError, match="stage"):
        validate_args(spec, {"sku": "selection_full", "amount_paid": 100, "stage": "half_done"})
    with pytest.raises(ValueError, match="amount_paid"):
        validate_args(spec, {"sku": "selection_full", "amount_paid": -5, "stage": "not_started"})


# ── 网关：超时、熔断、异常隔离、返回值校验、截断 ─────────────────────────────

def _noargs_spec(handler, policy=ToolPolicy(), output_model=None, name="t"):
    from tooling.schemas import NoArgs
    return make_model_tool(name, "test", NoArgs, handler, output_model=output_model, policy=policy)


def test_gateway_times_out_sync_handlers():
    gw = ToolGateway()
    spec = _noargs_spec(lambda req, args: time.sleep(0.5), policy=ToolPolicy(timeout_s=0.05))
    outcome = _run(gw.call(spec, None, {}))
    assert outcome.kind == "timeout" and outcome.ok is False
    assert outcome.result["success"] is False


def test_gateway_isolates_handler_errors():
    def boom(req, args):
        raise RuntimeError("redis down")

    outcome = _run(ToolGateway().call(_noargs_spec(boom), None, {}))
    assert outcome.kind == "error" and "redis down" in outcome.result["error"]


def test_gateway_opens_circuit_for_external_tools():
    gw = ToolGateway()
    calls = []

    def flaky(req, args):
        calls.append(1)
        raise ConnectionError("down")

    spec = _noargs_spec(flaky, policy=ToolPolicy(side_effect="write", breaker=True), name="writer")
    for _ in range(5):
        _run(gw.call(spec, None, {}))
    outcome = _run(gw.call(spec, None, {}))
    assert outcome.kind == "circuit_open"
    assert len(calls) == 5                       # 熔断后不再真正调用
    assert gw._breakers["writer"].state == CircuitState.OPEN


def test_pure_tools_do_not_use_the_breaker():
    gw = ToolGateway()

    def boom(req, args):
        raise ValueError("bug")

    spec = _noargs_spec(boom, name="pure")
    for _ in range(8):
        assert _run(gw.call(spec, None, {})).kind == "error"
    assert "pure" not in gw._breakers


def test_gateway_rejects_results_that_break_money_invariants():
    bad = {"success": True, "subtotal": 1000, "total": 1200, "lines": [], "payment_plan": {"due_at_signing": 100}}
    outcome = _run(ToolGateway().call(_noargs_spec(lambda r, a: bad, output_model=QuoteResult), None, {}))
    assert outcome.kind == "invalid_output"
    assert outcome.result["success"] is False
    assert "1200" not in json.dumps(outcome.result)   # 错误的数字不交给模型


def test_long_results_are_truncated_and_flagged():
    big = {"success": True, "results": [{"text": "x" * 200} for _ in range(100)]}
    out, truncated = shrink_result(big, 3000)
    assert truncated and out["truncated"] is True
    assert out["truncated_lists"]["results"]["total"] == 100
    assert len(out["results"]) < 100
    assert len(json.dumps(out, ensure_ascii=False)) <= 3000


def test_gateway_stats_report_outcomes():
    gw = ToolGateway()
    spec = _noargs_spec(lambda r, a: {"success": True}, name="ok_tool")
    _run(gw.call(spec, None, {}))
    _run(gw.call(spec, None, {"extra": 1}))
    row = next(r for r in gw.stats() if r["tool"] == "ok_tool")
    assert row["calls"] == 2 and row["by_outcome"] == {"ok": 1, "invalid_args": 1}


# ── 金额工具的返回值不变量：真实计算结果都要通过 ──────────────────────────────

def test_real_quotes_and_refunds_satisfy_invariants():
    skus = [s.sku for s in get_catalog().services]
    for sku in skus:
        for kwargs in ({}, {"early_bird": True}, {"group_size": 3, "referral": True}):
            QuoteResult.model_validate(quote_bundle([{"sku": sku, "qty": 1}], **kwargs))
        for stage in ("not_started", "in_progress", "delivered"):
            RefundResult.model_validate(calculate_refund(sku, 5000, stage, completed_rounds=1, progress_percent=40))


# ── 金额护栏 ────────────────────────────────────────────────────────────────

def test_extract_amounts_handles_rmb_formats_only():
    text = "合计 ¥7,798，定金 2000 元，约 1.2万元；2027 年 9 折，申请费 SEK 900，3 轮修改。"
    assert sorted(extract_amounts(text)) == [2000.0, 7798.0, 12000.0]


def _guard(*grounded, mode="enforce"):
    g = AmountGrounding()
    g.add_many(grounded)
    return AmountGuard(g, mode=mode)


def test_guard_replaces_only_ungrounded_sentences():
    guard = _guard('{"total": 7798}')
    out = guard.sanitize("应付合计 ¥7,798。老学员再送你 500 元优惠！有问题随时问我。")
    assert "7,798" in out
    assert "500 元" not in out and NOTICE in out
    assert out.endswith("有问题随时问我。")
    assert guard.violations[0].amounts == [500.0]


def test_guard_accepts_amounts_the_user_gave():
    guard = _guard("我已经付了 3540 元")
    assert guard.sanitize("您已付的 3540 元会按进度核算。") == "您已付的 3540 元会按进度核算。"


def test_guard_warn_mode_keeps_text_but_records():
    guard = _guard(mode="warn")
    assert guard.sanitize("大概 9999 元。") == "大概 9999 元。"
    assert guard.violations


def test_guard_streams_sentence_by_sentence():
    guard = _guard('{"total": 7798}')
    sent = []
    for chunk in ["应付合", "计 ¥7,7", "98。再减", " 300 元", "吧。结束"]:
        sent.append(guard.feed(chunk))
    sent.append(guard.flush())
    streamed = "".join(sent)
    assert sent[0] == "" and sent[1] == ""            # 句子没结束前不放行
    assert "¥7,798。" in streamed
    assert "300 元" not in streamed and NOTICE in streamed
    assert streamed.endswith("结束")


# ── 接进 Agent 之后的端到端行为 ──────────────────────────────────────────────

class ScriptedClient:
    """按顺序返回预设的模型输出；stream=True 时走流式接口。"""

    def __init__(self, turns):
        self.turns = list(turns)
        self.calls = []
        owner = self

        class _Stream:
            def __init__(self, content):
                self.content = content

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            @property
            def text_stream(self):
                async def gen():
                    for block in self.content:
                        if block.type == "text":
                            for i in range(0, len(block.text), 3):
                                yield block.text[i:i + 3]
                return gen()

            async def get_final_message(self):
                return type("M", (), {"content": self.content})()

        class Messages:
            async def create(inner, **kwargs):
                owner.calls.append(kwargs)
                return type("R", (), {"content": owner.turns.pop(0)})()

            def stream(inner, **kwargs):
                owner.calls.append(kwargs)
                return _Stream(owner.turns.pop(0))

        self.messages = Messages()


def _text(t):
    return type("T", (), {"type": "text", "text": t})()


def _tool_use(name, args, tid="toolu_1"):
    return type("U", (), {"type": "tool_use", "id": tid, "name": name, "input": args})()


def test_agent_blocks_a_made_up_total_without_tool_call():
    client = ScriptedClient([[_text("选校全案加文书一共 9999 元，今天签还能再便宜。")]])
    response = _run(ConsultingAgent(client, "m").handle(_req("选校全案加三篇文书多少钱")))
    assert "9999" not in response.content and NOTICE in response.content
    guard_trace = [t for t in response.tool_traces if t["tool_name"] == "amount_guard"]
    assert guard_trace and 9999.0 in guard_trace[0]["input"]["violations"][0]["amounts"]


def test_agent_passes_amounts_that_came_from_the_quote_tool():
    quote = quote_bundle([{"sku": "selection_full", "qty": 1}], referral=True)
    total = quote["total"]
    client = ScriptedClient([
        [_tool_use("quote_service_bundle", {"items": [{"sku": "selection_full"}], "referral": True})],
        [_text(f"按公开规则，应付合计 {total:,.0f} 元。")],
    ])
    response = _run(ConsultingAgent(client, "m").handle(_req("选校全案老带新多少钱")))
    assert f"{total:,.0f} 元" in response.content
    assert not [t for t in response.tool_traces if t["tool_name"] == "amount_guard"]
    trace = response.tool_traces[0]
    assert trace["outcome"] == "ok" and trace["side_effect"] == "read"


def test_agent_reports_invalid_tool_args_back_to_the_model():
    client = ScriptedClient([
        [_tool_use("calculate_refund", {"sku": "selection_full", "amount_paid": 100, "stage": "halfway"})],
        [_text("请告诉我服务目前进行到哪一步。")],
    ])
    response = _run(BillingAgent(client, "m").handle(_req("退款")))
    assert response.tool_traces[0]["outcome"] == "invalid_args"
    tool_result = json.loads(client.calls[1]["messages"][-1]["content"][0]["content"])
    assert tool_result["success"] is False and "stage" in tool_result["error"]


def test_streaming_never_sends_an_ungrounded_amount():
    client = ScriptedClient([[_text("先说结论。总价 12345 元。其余请咨询顾问。")]])
    chunks = []

    async def on_delta(t):
        chunks.append(t)

    response = _run(ConsultingAgent(client, "m").handle(_req("多少钱"), on_delta=on_delta))
    streamed = "".join(chunks)
    assert "12345" not in streamed and "12345" not in response.content
    assert streamed == response.content
