"""路由：意图识别之后，决定主 Agent 和协作 Agent。

全部是规则，不调大模型：紧急和转人工直接升级；其余按意图映射给基础分，
领域关键词（business/domain_terms.py）和实体再加分，最高分是主 Agent，够高的第二名做协作。
"""
import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List


from business.domain_terms import (
    ROUTING_KEYWORDS_BILLING,
    ROUTING_KEYWORDS_CONSULTING,
    ROUTING_KEYWORDS_CONSULTING_COLLAB,
    ROUTING_KEYWORDS_GENERAL,
)
from core.intent_recognizer import IntentCategory, UrgencyLevel


from agents.base import AgentType, Request

logger = logging.getLogger(__name__)


_GENERAL_INTENTS = {
    IntentCategory.QUERY,
    IntentCategory.SERVICE_PROGRESS,
    IntentCategory.REQUEST,
    IntentCategory.COMPLAINT,
    IntentCategory.GREETING,
    IntentCategory.FEEDBACK,
    IntentCategory.ACCOUNT,
    IntentCategory.OTHER,
}
_CONSULTING_INTENTS = {
    IntentCategory.STUDY_CONSULT,
    IntentCategory.APPLICATION_PROCESS,
    IntentCategory.SERVICE_INQUIRY,
    IntentCategory.BOOKING,
}
_BILLING_INTENTS = {
    IntentCategory.BILLING,
    IntentCategory.REFUND,
    IntentCategory.INVOICE,
    IntentCategory.PAYMENT_ISSUE,
}


def _service_terms() -> List[str]:
    """业务目录里的服务名、交付物名和"申请退款"这类费用短语，按长度降序，保证先去掉长词。"""
    from business.catalog import get_catalog

    terms = {"选校报告", "申请退款", "申请发票", "申请开票", "申请退还"}
    for service in get_catalog().services:
        terms.add(service.name.lower())
        terms.update(d.lower() for d in getattr(service, "deliverables", None) or [])
    return sorted(terms, key=len, reverse=True)


def _strip_service_terms(msg: str) -> str:
    for term in _service_terms():
        msg = msg.replace(term, " ")
    return msg



@dataclass
class RoutingDecision:
    """一次请求的结构化路由决策。"""
    primary_agent: AgentType
    supporting_agents: List[AgentType] = field(default_factory=list)
    reason: str = ""
    confidence: float = 0.0

    @property
    def agent_types(self) -> List[AgentType]:
        return [self.primary_agent] + self.supporting_agents

    @property
    def multi_agent(self) -> bool:
        return bool(self.supporting_agents)



class Router:
    """路由规则。available(agent_type) 告诉它当前哪些类型的 Agent 有可用实例。"""

    def __init__(self, available: Callable[[AgentType], bool]):
        self._available = available

    def decide(self, req: Request) -> RoutingDecision:
        """
        结构化路由决策。

        先处理紧急/转人工，再用领域分数决定主 Agent 和辅助 Agent。
        这样可以表达“主处理 + 辅助诊断”，避免关键词命中后无主次地拼接。
        """
        if req.urgency == UrgencyLevel.CRITICAL:
            return RoutingDecision(
                primary_agent=AgentType.ESCALATION,
                reason="紧急度为 CRITICAL，触发升级路由",
                confidence=1.0,
            )

        if req.intent in (IntentCategory.ESCALATION, IntentCategory.HUMAN_HANDOFF, IntentCategory.DATA_PRIVACY):
            return RoutingDecision(
                primary_agent=AgentType.ESCALATION,
                reason=f"意图为 {req.intent.value if req.intent else 'unknown'}，触发升级路由",
                confidence=max(req.intent_confidence, 0.8),
            )

        scores = self.domain_scores(req)
        available_scores = {
            agent_type: score
            for agent_type, score in scores.items()
            if agent_type == AgentType.GENERAL or self._available(agent_type)
        }
        if not available_scores:
            return RoutingDecision(
                primary_agent=AgentType.GENERAL,
                reason="无可用专属 Agent，降级到 GeneralAgent",
                confidence=0.1,
            )

        ordered = sorted(available_scores.items(), key=lambda item: item[1], reverse=True)
        primary_agent, primary_score = ordered[0]

        collaboration_targets = self.collaboration_targets(req)
        supporting_agents = [
            agent_type
            for agent_type in collaboration_targets
            if agent_type != primary_agent and agent_type in available_scores
        ]

        if not supporting_agents:
            supporting_agents = [
                agent_type
                for agent_type, score in ordered[1:]
                if agent_type != AgentType.GENERAL
                and score >= 0.45
                and score >= primary_score * 0.55
            ]

        reason = self._reason(req, available_scores, primary_agent, supporting_agents)
        return RoutingDecision(
            primary_agent=primary_agent,
            supporting_agents=supporting_agents,
            reason=reason,
            confidence=round(min(primary_score, 1.0), 3),
        )

    def domain_scores(self, req: Request) -> Dict[AgentType, float]:
        """按意图、关键词和实体为各领域 Agent 打分。"""
        msg = req.message.lower()
        scores = {
            AgentType.GENERAL: 0.1,
            AgentType.CONSULTING: 0.0,
            AgentType.BILLING: 0.0,
        }

        if req.intent in _GENERAL_INTENTS:
            scores[AgentType.GENERAL] += 0.55
        if req.intent in _CONSULTING_INTENTS:
            scores[AgentType.CONSULTING] += 0.75
        if req.intent in _BILLING_INTENTS:
            scores[AgentType.BILLING] += 0.75

        consulting_hits = sum(1 for kw in ROUTING_KEYWORDS_CONSULTING if kw in msg)
        billing_hits = sum(1 for kw in ROUTING_KEYWORDS_BILLING if kw in msg)
        general_hits = sum(1 for kw in ROUTING_KEYWORDS_GENERAL if kw in msg)

        scores[AgentType.CONSULTING] += min(0.45, consulting_hits * 0.18)
        scores[AgentType.BILLING] += min(0.45, billing_hits * 0.18)
        scores[AgentType.GENERAL] += min(0.35, general_hits * 0.12)

        entities = req.entities or {}
        if entities.get("country") or entities.get("service"):
            scores[AgentType.CONSULTING] += 0.2
        if entities.get("amount") or entities.get("contract_id"):
            scores[AgentType.BILLING] += 0.15

        return {agent_type: round(score, 3) for agent_type, score in scores.items()}

    @staticmethod
    def _reason(
        req: Request,
        scores: Dict[AgentType, float],
        primary_agent: AgentType,
        supporting_agents: List[AgentType],
    ) -> str:
        score_text = ", ".join(
            f"{agent_type.value}={score:.2f}"
            for agent_type, score in sorted(scores.items(), key=lambda item: item[1], reverse=True)
        )
        support_text = ", ".join(agent.value for agent in supporting_agents) or "none"
        intent = req.intent.value if req.intent else "unknown"
        return (
            f"intent={intent}, group={req.intent_group or 'unknown'}, "
            f"primary={primary_agent.value}, supporting={support_text}, scores=[{score_text}]"
        )

    def collaboration_targets(self, req: Request) -> List[AgentType]:
        """
        判断是否需要多个 Agent 并行协作。

        意图识别通常只返回一个主意图；这里用领域关键词补充检测复合问题，
        例如"咨询瑞典项目的同时又被重复扣款"需要留学咨询和账单 Agent 同时处理。
        """
        msg = req.message.lower()
        targets: List[AgentType] = []

        # 服务名、交付物（"选校报告"）和费用短语（"申请退款"）不是在问留学本身，扫描前去掉
        consult_msg = _strip_service_terms(msg)
        if req.intent in _CONSULTING_INTENTS or any(kw in consult_msg for kw in ROUTING_KEYWORDS_CONSULTING_COLLAB):
            targets.append(AgentType.CONSULTING)
        if req.intent in _BILLING_INTENTS or any(kw in msg for kw in ROUTING_KEYWORDS_BILLING):
            targets.append(AgentType.BILLING)

        # 保持顺序去重，并只返回当前有实例的 Agent 类型。
        deduped = list(dict.fromkeys(targets))
        return [agent_type for agent_type in deduped if self._available(agent_type)]

    @staticmethod
    def needs_clarification(req: Request) -> bool:
        """低置信度且无明确意图时，先追问，避免误路由。"""
        if req.intent != IntentCategory.OTHER:
            return False
        text = (req.message or "").strip()
        if len(text) <= 2:
            return False
        return req.intent_confidence < 0.5

