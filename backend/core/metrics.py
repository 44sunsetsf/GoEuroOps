"""业务链路的 Prometheus 指标（注册在默认 registry，经 /metrics 暴露）。"""
from prometheus_client import Counter

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
