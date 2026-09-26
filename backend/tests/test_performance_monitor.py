"""告警只在样本足够时触发、每个指标只挂一条、恢复后自动解决。"""
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
