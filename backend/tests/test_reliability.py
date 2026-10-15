"""熔断器、监控降级路由、专属 Agent 回退和评测评分失败的处理。"""
import asyncio
import json
import time
from types import SimpleNamespace

import pytest

from agents.agent_orchestrator import (
    AgentOrchestrator,
    AgentResponse,
    AgentStats,
    AgentType,
    Request,
)
from core.intent_recognizer import IntentCategory
from evaluation.evaluator import EndToEndEvaluator, EvalReport, LLMJudge
from monitor.performance_monitor import PerformanceMonitor
from tooling.breaker import CircuitBreaker, CircuitState


# ── 熔断器 ────────────────────────────────────────────────────────────────────

def _opened_breaker() -> CircuitBreaker:
    breaker = CircuitBreaker(failure_threshold=2, recovery_s=0.05)
    for _ in range(2):
        breaker.record_failure(breaker.acquire())
    assert breaker.state == CircuitState.OPEN
    return breaker


def test_half_open_lets_only_one_probe_through():
    breaker = _opened_breaker()
    assert breaker.acquire() is None              # 冷却中
    time.sleep(0.06)
    probe = breaker.acquire()
    assert probe is not None and breaker.state == CircuitState.HALF_OPEN
    assert breaker.acquire() is None              # 探测还没回来，其他请求不放行
    breaker.record_success(probe)
    assert breaker.state == CircuitState.CLOSED


def test_failed_probe_reopens():
    breaker = _opened_breaker()
    time.sleep(0.06)
    breaker.record_failure(breaker.acquire())
    assert breaker.state == CircuitState.OPEN
    assert breaker.acquire() is None


def test_stale_success_cannot_close_a_newer_open():
    breaker = CircuitBreaker(failure_threshold=2, recovery_s=60)
    slow = breaker.acquire()                      # 熔断前发出的慢请求
    for _ in range(2):
        breaker.record_failure(breaker.acquire())
    breaker.record_success(slow)                  # 熔断后才返回
    assert breaker.state == CircuitState.OPEN


def test_lost_probe_is_replaced_after_recovery_window():
    breaker = _opened_breaker()
    time.sleep(0.06)
    lost = breaker.acquire()                      # 探测请求被取消，永远不回报
    assert lost is not None
    time.sleep(0.06)
    probe = breaker.acquire()
    assert probe is not None and probe != lost
    breaker.record_success(lost)                  # 迟到的旧探测不算数
    assert breaker.state == CircuitState.HALF_OPEN
    breaker.record_success(probe)
    assert breaker.state == CircuitState.CLOSED


# ── 监控降级路由 ──────────────────────────────────────────────────────────────

def _orchestrator(**agents) -> AgentOrchestrator:
    orch = AgentOrchestrator.__new__(AgentOrchestrator)
    orch._pool = {
        AgentType.GENERAL: [agents.get("general") or SimpleNamespace(stats=AgentStats())],
        AgentType.CONSULTING: [agents.get("consulting") or SimpleNamespace(stats=AgentStats())],
        AgentType.BILLING: [agents.get("billing") or SimpleNamespace(stats=AgentStats())],
        AgentType.ESCALATION: [SimpleNamespace(stats=AgentStats())],
    }
    return orch


def _refund_request() -> Request:
    return Request(message="定金交了还能退款吗", user_id="u", conv_id="c",
                   intent=IntentCategory.REFUND, intent_group="billing", intent_confidence=0.9)


def test_degraded_specialist_is_skipped_and_reason_recorded():
    orch = _orchestrator()
    orch._pool[AgentType.BILLING][0].stats.monitor_penalty = 0.6
    decision = orch._route_decision(_refund_request())
    assert decision.primary_agent is AgentType.GENERAL
    assert "monitor_fallback=[billing]" in decision.reason


def test_penalty_below_threshold_keeps_routing():
    orch = _orchestrator()
    orch._pool[AgentType.BILLING][0].stats.monitor_penalty = 0.4
    assert orch._route_decision(_refund_request()).primary_agent is AgentType.BILLING


def test_monitor_penalty_uses_recent_window_and_recovers():
    stats = AgentStats()
    for _ in range(5):
        stats.total += 1
        stats.finish(False, 100.0)
    orch = _orchestrator(billing=SimpleNamespace(stats=stats))

    def collect():
        """和 PerformanceMonitor._collect 里算惩罚的规则一致。"""
        total, sr, first = stats.recent_window()
        penalty = PerformanceMonitor._routing_penalty(sr, first) if total >= PerformanceMonitor.MIN_SAMPLES else 0.0
        orch.update_routing_penalties({"billing_0": penalty})

    collect()
    assert orch._monitor_degraded() == [AgentType.BILLING]

    # 5 分钟没有新样本：窗口清空，惩罚归零，重新接流量
    stats.recent = type(stats.recent)(((t - 400, ok, f) for t, ok, f in stats.recent), maxlen=50)
    collect()
    assert orch._monitor_degraded() == []


def test_slow_first_token_alone_never_triggers_fallback():
    assert PerformanceMonitor._routing_penalty(1.0, 60_000) < 0.5
    assert PerformanceMonitor._routing_penalty(0.6, 0) >= 0.5


# ── 专属 Agent 失败回退 ───────────────────────────────────────────────────────

class _Agent:
    def __init__(self, agent_type, response_fn):
        self.agent_type = agent_type
        self.stats = AgentStats()
        self._fn = response_fn

    async def handle(self, req, on_delta=None):
        return await self._fn(on_delta)


def test_fallback_keeps_both_traces_and_resets_stream():
    async def broken(on_delta):
        await on_delta("瑞典的申请")
        return AgentResponse(agent_type=AgentType.CONSULTING, content="抱歉", success=False, streamed=True,
                             tool_traces=[{"tool_name": "lookup_country_admissions_overview"}])

    async def general(on_delta):
        await on_delta("我先简单说一下")
        return AgentResponse(agent_type=AgentType.GENERAL, content="我先简单说一下", success=True,
                             tool_traces=[{"tool_name": "search_knowledge_base"}])

    orch = _orchestrator(consulting=_Agent(AgentType.CONSULTING, broken), general=_Agent(AgentType.GENERAL, general))
    events = []

    async def on_delta(text):
        events.append(("token", text))

    async def on_reset():
        events.append(("reset", None))

    on_delta.reset = on_reset
    response = asyncio.run(orch._execute(_refund_request(), AgentType.CONSULTING, on_delta=on_delta))

    assert events == [("token", "瑞典的申请"), ("reset", None), ("token", "我先简单说一下")]
    assert [t["tool_name"] for t in response.tool_traces] == [
        "lookup_country_admissions_overview", "agent_fallback", "search_knowledge_base",
    ]
    assert response.agent_type is AgentType.GENERAL


# ── 评测：Judge 评分失败 ──────────────────────────────────────────────────────

@pytest.mark.parametrize("raw", [
    "",
    "评分如下",
    '{"relevance": 0.9}',
    '{"relevance": true, "accuracy": 1, "completeness": 1, "helpfulness": 1, "compliance": 1}',
    '{"relevance": 1.5, "accuracy": 1, "completeness": 1, "helpfulness": 1, "compliance": 1}',
    '{"relevance": NaN, "accuracy": 1, "completeness": 1, "helpfulness": 1, "compliance": 1}',
])
def test_judge_rejects_bad_output(raw):
    with pytest.raises(ValueError):
        LLMJudge.parse(raw)


def test_judge_accepts_fenced_json():
    raw = '```json\n{"relevance": 0.9, "accuracy": 1, "completeness": 0.8, "helpfulness": 0.9, "compliance": 1}\n```'
    assert LLMJudge.parse(raw)["completeness"] == 0.8


class _JudgeClient:
    def __init__(self, replies):
        self._replies = list(replies)
        self.calls = 0

        class Messages:
            async def create(inner, **kwargs):
                self.calls += 1
                reply = self._replies.pop(0)
                if isinstance(reply, Exception):
                    raise reply
                return SimpleNamespace(content=[SimpleNamespace(type="text", text=reply)], stop_reason="end_turn")

        self.messages = Messages()


def test_judge_retries_once_then_reports_failure():
    client = _JudgeClient(["不是 JSON", "还是不是"])
    scores = asyncio.run(LLMJudge(client, "m").judge("q", "a"))
    assert scores.judge_failed and client.calls == 2 and "JSON" in scores.error


def test_judge_does_not_retry_auth_errors():
    err = RuntimeError("401")
    err.status_code = 401
    client = _JudgeClient([err, "{}"])
    assert asyncio.run(LLMJudge(client, "m").judge("q", "a")).judge_failed
    assert client.calls == 1


GOOD = '{"relevance": 0.9, "accuracy": 0.9, "completeness": 0.9, "helpfulness": 0.9, "compliance": 1.0}'


class _Orch:
    async def run(self, req):
        return SimpleNamespace(response="回答", tools_used=[], agent_type=AgentType.GENERAL, intent=None,
                               skills_applied=[], rag_gate={})


def _evaluator(tmp_path, replies) -> EndToEndEvaluator:
    ev = EndToEndEvaluator.__new__(EndToEndEvaluator)
    ev._orchestrator = _Orch()
    ev._judge = LLMJudge(_JudgeClient(replies), "judge-model")
    ev._history = []
    ev._baseline_path = tmp_path / "baseline.json"
    ev._baseline = None
    ev._skill_manager = None
    ev.dim_threshold = 0.75
    return ev


CASES = [{"question": "q1"}, {"question": "q2"}]


def test_partial_report_excludes_failures_and_keeps_baseline(tmp_path):
    ev = _evaluator(tmp_path, [GOOD, "坏", "坏"])
    report = asyncio.run(ev.run(dialog_cases=CASES))
    assert report.status == "partial"
    assert (report.total, report.scored, report.judge_failed, report.passed) == (2, 1, 1, 1)
    assert report.pass_rate == 1.0
    assert report.avg_scores["relevance"] == 0.9            # 失败样本没有被当成 0.5 混进均分
    assert not ev._baseline_path.exists()                    # 不完整的报告不覆盖基线


def test_regression_only_compares_same_suite(tmp_path):
    ev = _evaluator(tmp_path, [GOOD, GOOD])
    first = asyncio.run(ev.run(dialog_cases=CASES))
    assert first.status == "complete" and json.loads(ev._baseline_path.read_text())["suite_id"] == first.suite_id

    worse = GOOD.replace("0.9", "0.5")
    ev._judge = LLMJudge(_JudgeClient([worse]), "judge-model")
    other_suite = asyncio.run(ev.run(dialog_cases=[{"question": "别的问题"}]))
    assert other_suite.suite_id != first.suite_id and other_suite.regressions == []

    ev._judge = LLMJudge(_JudgeClient([worse, worse]), "judge-model")
    same_suite = asyncio.run(ev.run(dialog_cases=CASES))
    assert any(r.startswith("relevance") for r in same_suite.regressions)


def test_old_baseline_without_suite_id_is_not_compared(tmp_path):
    ev = _evaluator(tmp_path, [GOOD])
    ev._baseline = EvalReport(timestamp="", total=1, passed=1, pass_rate=1.0, avg_scores={"relevance": 1.0},
                              regressions=[], recommendations=[], results=[])
    report = asyncio.run(ev.run(intent_cases=None, dialog_cases=[{"question": "q"}]))
    assert report.regressions == []


# ── 检索降级 ──────────────────────────────────────────────────────────────────

def test_knowledge_fallback_is_reported_as_failure_not_evidence():
    from retrieval.manager import RetrievalManager, Tool

    async def down(params, context):
        raise ConnectionError("chroma down")

    manager = RetrievalManager(api_key="test")
    manager.register(Tool(
        name="knowledge_search", description="test", handler=down,
        schema={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        fallback=lambda params, context, error: [{"title": "知识库降级结果", "content": "暂时不可用", "fallback": True}],
    ))

    async def no_rewrite(query, n=3):
        return [query]

    manager.rewrite_query = no_rewrite
    result = asyncio.run(manager.search_with_rewrite("knowledge_search", "瑞典截止时间"))
    assert not result.success and result.data == []
    assert "知识库暂时不可用" in result.error
