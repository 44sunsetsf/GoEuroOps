"""告警只在样本足够时触发、每个指标只挂一条、恢复后自动解决。"""
import asyncio

from monitor.performance_monitor import PerformanceMonitor


def make():
    return PerformanceMonitor(orchestrator=None, tool_manager=None)


def open_alerts(m):
    return [a for a in m._alerts if not a.resolved]


def test_no_alert_on_tiny_samples():
    m = make()
    m._check_threshold("tool_success_rate", 0.0, "knowledge_search", samples=1)
    assert open_alerts(m) == []


def test_one_open_alert_per_metric_and_auto_resolve():
    m = make()
    for _ in range(3):  # three collection cycles while failing
        m._check_threshold("tool_success_rate", 0.5, "knowledge_search", samples=10)
    assert len(open_alerts(m)) == 1
    m._check_threshold("tool_success_rate", 1.0, "knowledge_search", samples=20)
    assert open_alerts(m) == []
    assert m._alerts[-1].resolved


def test_other_metrics_are_independent():
    m = make()
    m._check_threshold("tool_success_rate", 0.5, "a", samples=10)
    m._check_threshold("tool_success_rate", 0.5, "b", samples=10)
    m._check_threshold("tool_success_rate", 1.0, "a", samples=10)
    assert [a.metric for a in open_alerts(m)] == ["tool_success_rate:b"]


def test_latency_alert_uses_time_to_first_output():
    # 知识类回答整段生成要 4.7 秒，但第一个字 1.2 秒就出来了：不该告警
    m = make()
    m._check_threshold("agent_avg_ms", 4751, "consulting_0", samples=23)
    m._check_threshold("agent_first_ms", 1200, "consulting_0", samples=23)
    assert open_alerts(m) == []
    # 用户等了 3.5 秒还一个字都没看到，才告警
    m._check_threshold("agent_first_ms", 3500, "consulting_0", samples=23)
    assert [a.metric for a in open_alerts(m)] == ["agent_first_ms:consulting_0"]


class _Orch:
    def __init__(self, success_rate):
        self.success_rate = success_rate

    def get_stats(self):
        return {"consulting_0": {"total": 20, "success_rate": self.success_rate, "avg_ms": 4000.0,
                                 "streamed": 20, "avg_first_ms": 1000.0, "routing_score": 0.5}}


class _Tools:
    def get_stats(self):
        return {}


def test_suggestions_clear_after_recovery():
    orch = _Orch(0.5)
    m = PerformanceMonitor(orchestrator=orch, tool_manager=_Tools())
    asyncio.run(m._collect())
    assert [s["title"] for s in m.summary()["suggestions"]] == ["Agent consulting_0 成功率偏低"]
    orch.success_rate = 1.0
    asyncio.run(m._collect())
    assert m.summary()["suggestions"] == []


def test_agent_records_time_to_first_output():
    from agents.agent_orchestrator import AgentStats, ConsultingAgent

    agent = ConsultingAgent.__new__(ConsultingAgent)   # 不连模型，只验证 handle 对 on_delta 的计时包装
    agent.stats = AgentStats()
    agent._needs_escalation = lambda content: False

    async def fake_llm(req, on_delta=None, run=None):
        for t in ("你好", "，这里是回答"):
            await on_delta(t)
        return "你好，这里是回答"

    agent._call_llm = fake_llm
    got = []

    async def on_delta(t):
        got.append(t)

    asyncio.run(agent.handle(object(), on_delta=on_delta))
    assert got == ["你好", "，这里是回答"]              # 文本原样转发
    assert agent.stats.first_count == 1 and agent.stats.total == 1
