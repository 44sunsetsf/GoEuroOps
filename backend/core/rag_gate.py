"""意图门控 RAG。

按识别出的意图决定本轮知识库检索的策略：

  prefetch   知识密集型意图：编排器预先检索，把结果作为「知识库上下文」注入，
             省掉模型"先决定调用检索工具、再生成回答"的一轮 LLM 往返；
             检索工具仍保留，模型觉得不够时可以补查。
  on_demand  泛问题：不预取，由模型自己决定是否调用检索工具（原有行为）。
  off        问候、投诉、转人工、隐私：不检索，并从本轮工具列表里移除检索工具，
             避免无意义的检索调用。

意图置信度低于阈值时一律退回 on_demand，避免意图识别出错导致该查的没查。

延迟：意图识别本身要一次 LLM 调用，预检索与它并行启动（投机执行）——
对各业务 domain 各做一次纯向量召回（不做 LLM 改写/重排），意图出来后
只采用路由到的那个 domain 的结果，其余丢弃。
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

SearchFn = Callable[[str, int, Optional[str]], Awaitable[List[Dict[str, Any]]]]

SPECULATIVE_DOMAINS = ("consulting", "billing", "general")


class RagMode(str, Enum):
    PREFETCH = "prefetch"
    ON_DEMAND = "on_demand"
    OFF = "off"


# 意图 → 策略。未列出的意图默认 on_demand。
RAG_POLICY: Dict[str, RagMode] = {
    "study_consult": RagMode.PREFETCH,
    "application_process": RagMode.PREFETCH,
    "service_inquiry": RagMode.PREFETCH,
    "booking": RagMode.PREFETCH,
    "refund": RagMode.PREFETCH,
    "invoice": RagMode.PREFETCH,
    "payment_issue": RagMode.PREFETCH,
    "billing": RagMode.PREFETCH,
    "query": RagMode.ON_DEMAND,
    "request": RagMode.ON_DEMAND,
    "service_progress": RagMode.ON_DEMAND,
    "account": RagMode.ON_DEMAND,
    "other": RagMode.ON_DEMAND,
    "greeting": RagMode.OFF,
    "feedback": RagMode.OFF,
    "complaint": RagMode.OFF,
    "escalation": RagMode.OFF,
    "human_handoff": RagMode.OFF,
    "data_privacy": RagMode.OFF,
}


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


@dataclass
class RagGateDecision:
    mode: RagMode
    reason: str
    intent: Optional[str] = None
    confidence: float = 0.0
    domain: Optional[str] = None
    items: List[Dict[str, Any]] = field(default_factory=list)
    dropped_low_score: int = 0
    wait_ms: float = 0.0

    @property
    def prefetched(self) -> int:
        return len(self.items)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode.value,
            "reason": self.reason,
            "intent": self.intent,
            "confidence": round(self.confidence, 3),
            "domain": self.domain,
            "prefetched": self.prefetched,
            "hits": [
                {"title": item.get("title", ""), "score": item.get("score")}
                for item in self.items
            ],
            "dropped_low_score": self.dropped_low_score,
            "wait_ms": round(self.wait_ms, 1),
        }


class RagGate:
    def __init__(
        self,
        search_fn: Optional[SearchFn] = None,
        *,
        min_confidence: Optional[float] = None,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
        enabled: Optional[bool] = None,
    ):
        self._search_fn = search_fn
        self.min_confidence = min_confidence if min_confidence is not None else _env_float("GOEUROOPS_RAG_GATE_MIN_CONF", 0.6)
        self.top_k = top_k if top_k is not None else _env_int("GOEUROOPS_RAG_PREFETCH_TOP_K", 3)
        # 混合检索分数下限（向量 0.75 + 字面覆盖 0.25），低于它的片段不注入
        self.min_score = min_score if min_score is not None else _env_float("GOEUROOPS_RAG_PREFETCH_MIN_SCORE", 0.4)
        if enabled is None:
            enabled = os.getenv("GOEUROOPS_RAG_GATE_ENABLED", "true").lower() not in {"0", "false", "no", "off"}
        self.enabled = enabled

    def set_search_fn(self, search_fn: Optional[SearchFn]) -> None:
        self._search_fn = search_fn

    @property
    def can_prefetch(self) -> bool:
        return self.enabled and self._search_fn is not None

    def decide(self, intent: Optional[str], confidence: float) -> tuple[RagMode, str]:
        if not self.enabled:
            return RagMode.ON_DEMAND, "门控已关闭，由模型自行决定是否检索"
        policy = RAG_POLICY.get(intent or "other", RagMode.ON_DEMAND)
        if policy != RagMode.ON_DEMAND and confidence < self.min_confidence:
            return RagMode.ON_DEMAND, (
                f"意图 {intent} 置信度 {confidence:.2f} 低于 {self.min_confidence}，"
                f"不采用 {policy.value} 策略，退回按需检索"
            )
        if policy == RagMode.PREFETCH:
            return policy, f"意图 {intent} 属于知识密集型，预取知识库"
        if policy == RagMode.OFF:
            return policy, f"意图 {intent} 不需要知识库，本轮不检索"
        return policy, f"意图 {intent or 'other'} 由模型按需检索"

    def speculate(self, message: str) -> Optional[Dict[str, "asyncio.Task[List[Dict[str, Any]]]"]]:
        """在意图识别的同时，为各业务 domain 启动纯向量召回。"""
        if not self.can_prefetch or not (message or "").strip():
            return None
        return {
            domain: asyncio.create_task(self._safe_search(message, domain))
            for domain in SPECULATIVE_DOMAINS
        }

    async def _safe_search(self, message: str, domain: str) -> List[Dict[str, Any]]:
        try:
            return await self._search_fn(message, self.top_k, domain)  # type: ignore[misc]
        except Exception as ex:
            logger.warning("RAG 预取失败 domain=%s: %s", domain, ex)
            return []

    async def resolve(
        self,
        speculative: Optional[Dict[str, "asyncio.Task[List[Dict[str, Any]]]"]],
        *,
        intent: Optional[str],
        confidence: float,
        domain: str,
        message: str,
    ) -> RagGateDecision:
        """意图和路由确定后，决定采用哪份投机召回结果（或丢弃）。"""
        mode, reason = self.decide(intent, confidence)
        decision = RagGateDecision(mode=mode, reason=reason, intent=intent, confidence=confidence, domain=domain)

        if mode != RagMode.PREFETCH or not self.can_prefetch:
            cancel_speculative(speculative)
            if mode == RagMode.PREFETCH:
                decision.mode = RagMode.ON_DEMAND
                decision.reason = "检索工具未初始化，退回按需检索"
            self._count(decision)
            return decision

        t0 = time.monotonic()
        task = (speculative or {}).get(domain)
        # 路由到的 domain 不在投机范围内（如 escalation）时，现场检索一次
        items = await task if task is not None else await self._safe_search(message, domain)
        cancel_speculative({k: v for k, v in (speculative or {}).items() if k != domain})
        decision.wait_ms = (time.monotonic() - t0) * 1000

        kept = [item for item in items if float(item.get("score", 0) or 0) >= self.min_score]
        decision.dropped_low_score = len(items) - len(kept)
        decision.items = kept[: self.top_k]
        if not decision.items:
            decision.reason += "；但没有足够相关的片段，不注入，模型仍可自行检索"
        self._count(decision)
        return decision

    @staticmethod
    def _count(decision: RagGateDecision) -> None:
        from core.metrics import RAG_GATE_DECISIONS
        RAG_GATE_DECISIONS.labels(mode=decision.mode.value).inc()

    @staticmethod
    def format_context(items: List[Dict[str, Any]]) -> str:
        """把预取结果格式化成带编号和来源标题的上下文块。"""
        lines = [
            "以下是根据用户问题预先检索到的知识库片段（按相关度排序）。"
            "回答时优先依据这些内容，引用时说明来源标题；片段未覆盖的内容不要编造，"
            "可以调用 search_knowledge_base 补查，仍然没有就如实说明需要顾问确认。",
        ]
        for idx, item in enumerate(items, start=1):
            lines.append(f"[{idx}] 《{item.get('title', '未命名')}》\n{item.get('content', '')}")
        return "\n\n".join(lines)


def cancel_speculative(speculative: Optional[Dict[str, "asyncio.Task[Any]"]]) -> None:
    for task in (speculative or {}).values():
        if not task.done():
            task.cancel()
