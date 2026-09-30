"""业务链路的 Prometheus 指标（注册在默认 registry，经 /metrics 暴露）。"""
from prometheus_client import Counter, Histogram

SKILL_HITS = Counter(
    "goeuroops_skill_hits_total",
    "Skill 注入次数",
    ["skill", "agent"],
)

RAG_GATE_DECISIONS = Counter(
    "goeuroops_rag_gate_total",
    "意图门控 RAG 决策次数",
    ["mode"],
)

LEADS_CREATED = Counter(
    "goeuroops_leads_created_total",
    "登记的线索 / 交接单数量",
    ["type"],
)

TOOL_CALLS = Counter(
    "goeuroops_tool_calls_total",
    "Agent 工具调用次数",
    ["tool", "outcome"],   # ok / invalid_args / invalid_output / timeout / error / circuit_open
)

TOOL_LATENCY = Histogram(
    "goeuroops_tool_latency_seconds",
    "Agent 工具调用耗时",
    ["tool"],
    buckets=(0.005, 0.02, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 20),
)

AMOUNT_GUARD = Counter(
    "goeuroops_amount_guard_total",
    "金额护栏命中次数：模型回复里出现了没有依据的金额",
    ["action"],            # replaced（已替换）/ warned（只告警）
)
