"""
亮点：多 Agent 路由与编排

核心问题：多 Agent 情况下如何做 Routing？

路由策略（三层决策）：
  1. 意图路由 —— 根据 IntentCategory 直接映射到专属 Agent
  2. 性能路由 —— 同类 Agent 有多个时，选成功率最高、延迟最低的
  3. 降级路由 —— 专属 Agent 不可用时，自动降级到 GeneralAgent

并行协作：
  - 复合问题（如"问瑞典项目 + 问退款"）可同时派发给多个 Agent
  - 结果由 ResponseComposer 合并后返回

每次 Agent 调用前还有两道确定性的前置处理：
  - 意图门控 RAG（core/rag_gate.py）：按意图决定预取 / 按需 / 不检索
  - Skills 路由（core/skill_loader.py）：按意图、关键词、语义样例挑选业务规范注入

升级机制：
  - 紧急度 CRITICAL 或转人工意图 → EscalationAgent 生成交接单写入线索面板
"""
import asyncio
import logging
import os
import time
from collections import deque
from datetime import datetime
from dataclasses import replace
from typing import Any, Dict, List, Optional

from anthropic import AsyncAnthropic

from agents.tools import (
    build_shared_rag_tools,
)
from business.lead_store import LeadStore
from core.intent_recognizer import IntentCategory, IntentRecognizer, UrgencyLevel
from core.llm_utils import make_client
from core.rag_gate import RagGate, RagGateDecision, RagMode, cancel_speculative
from core.config import DEFAULT_MODEL, env_int


from agents.base import (  # noqa: F401 —— 部分名字供旧的导入路径使用
    AgentProfile,
    AgentResponse,
    AgentStats,
    AgentType,
    BaseAgent,
    OnDelta,
    OrchestratorResult,
    Request,
)
from agents.roles import (  # noqa: F401 —— 部分名字供旧的导入路径使用
    BillingAgent,
    ConsultingAgent,
    EscalationAgent,
    GeneralAgent,
)
from agents.composer import (  # noqa: F401 —— 部分名字供旧的导入路径使用
    ResponseComposer,
)
from agents.routing import (  # noqa: F401 —— 部分名字供旧的导入路径使用
    Router,
    RoutingDecision,
    _BILLING_INTENTS,
    _CONSULTING_INTENTS,
    _GENERAL_INTENTS,
)

logger = logging.getLogger(__name__)


class AgentOrchestrator:
    """
    多 Agent 编排器。

    路由逻辑（三层）：
      1. 意图 → Agent 类型映射
      2. 同类多实例时按 routing_score() 选最优
      3. 专属 Agent 失败时降级到 GeneralAgent
    """

    def __init__(
        self,
        api_key:  str,
        base_url: Optional[str] = None,
        model:    str = DEFAULT_MODEL,
        skill_manager: Optional[Any] = None,
        rag_tool_manager: Optional[Any] = None,
        lead_store: Optional[LeadStore] = None,
        rag_gate: Optional[RagGate] = None,
    ):
        client = make_client(api_key, base_url, "agents")

        self._intent_recognizer = IntentRecognizer(api_key=api_key, base_url=base_url, model=model)
        self._skill_manager = skill_manager
        self._composer = ResponseComposer(client, model, skill_manager)
        self._recent_tool_traces = deque(maxlen=env_int("GOEUROOPS_TOOL_TRACE_MAX", 200))
        self._rag_gate = rag_gate or RagGate()

        # Agent 池：每种类型可有多个实例（水平扩展）
        self._pool: Dict[AgentType, List[BaseAgent]] = {
            AgentType.GENERAL: [self._make_agent(GeneralAgent, client, model, skill_manager)],
            AgentType.CONSULTING: [self._make_agent(ConsultingAgent, client, model, skill_manager)],
            AgentType.BILLING: [self._make_agent(BillingAgent, client, model, skill_manager)],
            AgentType.ESCALATION: [self._make_agent(EscalationAgent, client, model, skill_manager)],
        }
        self.set_shared_tools(rag_tool_manager)
        self.set_lead_store(lead_store or LeadStore())

    @property
    def rag_gate(self) -> RagGate:
        return self._rag_gate

    def set_lead_store(self, lead_store: Optional[LeadStore]) -> None:
        """线索库注入给需要写线索/交接单的 Agent（咨询、转顾问）。"""
        self._lead_store = lead_store
        for agents in self._pool.values():
            for agent in agents:
                agent.set_lead_store(lead_store)

    @staticmethod
    def _make_agent(
        agent_cls: type[BaseAgent],
        client: AsyncAnthropic,
        default_model: str,
        skill_manager: Optional[Any],
    ) -> BaseAgent:
        """按角色创建 Agent，并允许用环境变量覆盖该角色的模型。

        可使用更强模型，通用接待可使用更快模型，升级节点本身不需要调用 LLM。
        """
        profile = agent_cls.profile
        env_name = f"GOEUROOPS_{agent_cls.agent_type.value.upper()}_MODEL"
        model = os.getenv(env_name, "").strip() or profile.model
        configured_profile = replace(profile, model=model) if model else profile
        return agent_cls(client, default_model, skill_manager, profile=configured_profile)

    def set_skill_manager(self, skill_manager: Optional[Any]) -> None:
        """更新 SkillManager 引用，供运行时重载或测试替换使用。"""
        self._skill_manager = skill_manager
        self._composer._skill_manager = skill_manager
        for agents in self._pool.values():
            for agent in agents:
                agent._skill_manager = skill_manager

    def set_shared_tools(self, rag_tool_manager: Optional[Any]) -> None:
        """
        为每个 Agent 绑定各自领域范围内的 RAG 工具。

        不同 Agent 类型拿到的是各自 domain 限定的检索范围（例如 ConsultingAgent
        只能检索 domain="consulting" 和 "shared" 的知识），而不是所有 Agent
        共享同一份不做区分的知识库入口——避免跨业务线内容互相串场。
        """
        for agent_type, agents in self._pool.items():
            # 转顾问节点不做知识问答，不暴露检索工具
            if agent_type == AgentType.ESCALATION:
                continue
            domain_tools = build_shared_rag_tools(rag_tool_manager, domain=agent_type.value)
            for agent in agents:
                agent.set_shared_tools(domain_tools)

        if rag_tool_manager is not None and hasattr(rag_tool_manager, "search_fast"):
            async def search_fn(query: str, top_k: int, domain: Optional[str]) -> List[Dict[str, Any]]:
                return await rag_tool_manager.search_fast("knowledge_search", query, top_k=top_k, domain=domain)
            self._rag_gate.set_search_fn(search_fn)
        else:
            self._rag_gate.set_search_fn(None)

    async def recognize_intent(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]] = None,
    ):
        """对外暴露意图识别，供 API 层先判断是否需要 RAG 等前置能力。"""
        return await self._intent_recognizer.recognize(message, history=history)

    def _record_tool_trace(self, result: OrchestratorResult) -> None:
        trace = {
            "request_id": result.request_id,
            "timestamp": datetime.now().isoformat(),
            "intent": result.intent.value if result.intent else None,
            "primary_agent": result.primary_agent.value if result.primary_agent else None,
            "supporting_agents": [agent.value for agent in result.supporting_agents],
            "tools_used": list(result.tools_used),
            "tool_calls": list(result.tool_traces),
            "skills_applied": list(result.skills_applied),
            "rag_gate": dict(result.rag_gate),
            "escalated": result.escalated,
            "latency_ms": round(result.latency_ms, 1),
        }
        self._recent_tool_traces.append(trace)

    def get_tool_trace(self, request_id: str) -> Optional[Dict[str, Any]]:
        for trace in reversed(self._recent_tool_traces):
            if trace.get("request_id") == request_id:
                return trace
        return None

    def get_recent_tool_traces(self, limit: int = 20) -> List[Dict[str, Any]]:
        if not self._recent_tool_traces:
            return []
        limit = max(1, min(int(limit or 20), len(self._recent_tool_traces)))
        return list(reversed(list(self._recent_tool_traces)[-limit:]))

    # ── 主入口 ────────────────────────────────────────────────────────────────

    async def run(self, req: Request, on_delta: Optional[OnDelta] = None) -> OrchestratorResult:
        """
        处理一次请求的完整流程：
          意图识别 → 路由选 Agent → 执行 → 检查升级 → 返回结果

        on_delta（可选）：逐 token 转发最终回复文本，用于流式输出。
        只有单 Agent 路径是真正逐 token 转发；澄清追问和多 Agent 并行汇总
        这两个边缘路径在拿到完整文本后一次性转发，对调用方而言接口一致。
        """
        t0 = time.monotonic()

        # 0. 投机预取：与意图识别并行启动各 domain 的纯向量召回（门控决定是否采用）
        speculative = self._rag_gate.speculate(req.message) if req.rag_mode is None else None

        # 1. 意图识别（如果调用方已识别则跳过）
        try:
            if req.intent is None:
                intent_result = await self._intent_recognizer.recognize(req.message, history=req.history)
                req.intent  = intent_result.intent
                req.intent_group = intent_result.intent_group
                req.urgency = intent_result.urgency
                req.intent_confidence = intent_result.confidence
                req.intent_source_scores = dict(intent_result.source_scores)
                if not req.entities:
                    req.entities = intent_result.entities
        except BaseException:
            cancel_speculative(speculative)
            raise

        if self._needs_clarification(req):
            cancel_speculative(speculative)
            clarification = (
                "我还不太确定你想了解哪方面～可以告诉我是以下哪类吗？\n\n"
                "- 瑞典/德国/荷兰/芬兰/丹麦 CS 硕士的申请信息\n"
                "- 我们的选校、文书服务和价格\n"
                "- 已购服务的进度、付款、退款或发票\n"
                "- 直接联系真人顾问"
            )
            if on_delta is not None:
                await on_delta(clarification)
            result = OrchestratorResult(
                request_id=req.request_id,
                response=clarification,
                agent_type=AgentType.GENERAL,
                intent=req.intent,
                escalated=False,
                latency_ms=(time.monotonic() - t0) * 1000,
                agent_types=[AgentType.GENERAL],
                primary_agent=AgentType.GENERAL,
                routing_reason="低置信度 OTHER 意图，先澄清用户需求",
                routing_confidence=req.intent_confidence,
                **self._intent_fields(req),
                rag_gate={"mode": "skipped", "reason": "澄清追问，不检索"},
            )
            self._record_tool_trace(result)
            return result

        decision = self._route_decision(req)

        # 2. 意图门控 RAG：按意图决定预取 / 按需 / 不检索
        gate = await self._apply_rag_gate(req, decision, speculative)

        # 复合问题自动并行协作，例如同一句同时问项目信息和退款。
        if decision.multi_agent:
            return await self.run_parallel(req, decision, on_delta=on_delta, t0=t0, gate=gate)

        # 3. 执行主 Agent（含降级）
        response = await self._execute(req, decision.primary_agent, on_delta=on_delta)

        # 4. 升级检查
        escalated = False
        if response.escalate or req.urgency == UrgencyLevel.CRITICAL or req.intent in (
            IntentCategory.ESCALATION,
            IntentCategory.HUMAN_HANDOFF,
            IntentCategory.DATA_PRIVACY,
        ):
            escalated = True
            logger.warning(f"请求 {req.request_id} 触发升级: intent={req.intent} urgency={req.urgency}")

        result = OrchestratorResult(
            request_id=req.request_id,
            response=response.content,
            agent_type=response.agent_type,
            intent=req.intent,
            escalated=escalated,
            latency_ms=(time.monotonic() - t0) * 1000,
            agent_types=[response.agent_type],
            primary_agent=decision.primary_agent,
            supporting_agents=[],
            tools_used=list(response.tools_used),
            tool_traces=list(response.tool_traces),
            routing_reason=decision.reason,
            routing_confidence=decision.confidence,
            **self._intent_fields(req),
            skills_applied=list(response.skills_applied),
            rag_gate=gate.to_dict(),
        )
        self._record_tool_trace(result)
        return result

    @staticmethod
    def _intent_fields(req: Request) -> Dict[str, Any]:
        return {
            "intent_group": req.intent_group,
            "intent_confidence": req.intent_confidence,
            "intent_source_scores": dict(req.intent_source_scores),
            "entities": dict(req.entities or {}),
        }

    async def _apply_rag_gate(
        self,
        req: Request,
        decision: RoutingDecision,
        speculative: Optional[Dict[str, Any]],
    ) -> RagGateDecision:
        if req.rag_mode is not None:
            # 调用方已经指定了策略（如评测对照组），不再门控
            cancel_speculative(speculative)
            return RagGateDecision(mode=RagMode(req.rag_mode), reason="调用方指定策略", intent=req.intent.value if req.intent else None)
        gate = await self._rag_gate.resolve(
            speculative,
            intent=req.intent.value if req.intent else None,
            confidence=req.intent_confidence,
            domain=decision.primary_agent.value,
            message=req.message,
        )
        req.rag_mode = gate.mode.value
        req.knowledge = list(gate.items)
        logger.info(
            "RAG 门控: request=%s mode=%s prefetched=%d reason=%s",
            req.request_id, gate.mode.value, gate.prefetched, gate.reason,
        )
        return gate

    async def run_parallel(
        self,
        req: Request,
        decision: RoutingDecision,
        on_delta: Optional[OnDelta] = None,
        t0: Optional[float] = None,
        gate: Optional[RagGateDecision] = None,
    ) -> OrchestratorResult:
        """
        并行派发给多个 Agent，合并结果。
        适用于复合问题（如同时问项目信息和退款）。

        每个子 Agent 仍各自内部完成（不逐 token 转发，多路并行文本交错
        转发给用户没有意义），合并后的最终文本一次性通过 on_delta 转发。
        """
        t0 = t0 if t0 is not None else time.monotonic()
        agent_types = decision.agent_types
        tasks = [self._execute(req, at) for at in agent_types]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        valid_responses = [r for r in responses if isinstance(r, AgentResponse)]
        combined = await self._composer.compose(req, valid_responses)
        if on_delta is not None:
            await on_delta(combined)
        escalated = any(isinstance(r, AgentResponse) and r.escalate for r in responses)
        tools_used = list(dict.fromkeys(
            tool_name
            for response in valid_responses
            for tool_name in response.tools_used
        ))
        tool_traces = [
            trace
            for response in valid_responses
            for trace in response.tool_traces
        ]
        result = OrchestratorResult(
            request_id=req.request_id,
            response=combined,
            agent_type=decision.primary_agent,
            intent=req.intent,
            escalated=escalated,
            latency_ms=(time.monotonic() - t0) * 1000,
            agent_types=[
                r.agent_type for r in responses
                if isinstance(r, AgentResponse) and r.success
            ] or agent_types,
            primary_agent=decision.primary_agent,
            supporting_agents=decision.supporting_agents,
            tools_used=tools_used,
            tool_traces=tool_traces,
            routing_reason=decision.reason,
            routing_confidence=decision.confidence,
            **self._intent_fields(req),
            skills_applied=[
                {**skill, "agent": response.agent_type.value}
                for response in valid_responses
                for skill in response.skills_applied
            ],
            rag_gate=gate.to_dict() if gate else {},
        )
        self._record_tool_trace(result)
        return result

    # ── 路由逻辑 ──────────────────────────────────────────────────────────────

    # ── 路由（规则在 agents/routing.py 的 Router 里，这里保留原来的方法名）──────────

    @property
    def _router(self) -> Router:
        # 每次按当前的 Agent 池判断可用性；Router 本身无状态，创建很便宜
        return Router(lambda agent_type: bool(self._pool.get(agent_type)))

    def _route_decision(self, req: Request) -> RoutingDecision:
        return self._router.decide(req)

    def _collaboration_targets(self, req: Request) -> List[AgentType]:
        return self._router.collaboration_targets(req)

    @staticmethod
    def _needs_clarification(req: Request) -> bool:
        return Router.needs_clarification(req)

    def _best_agent(self, agent_type: AgentType) -> Optional[BaseAgent]:
        """
        性能路由：从同类 Agent 中选 routing_score() 最高的。
        这是"基于在线表现动态调整路由"的核心。
        """
        agents = self._pool.get(agent_type, [])
        if not agents:
            return None
        return max(agents, key=lambda a: a.stats.routing_score())

    async def _execute(
        self,
        req: Request,
        agent_type: AgentType,
        on_delta: Optional[OnDelta] = None,
    ) -> AgentResponse:
        """执行 Agent，失败时降级到 GeneralAgent。"""
        agent = self._best_agent(agent_type)
        if agent is None:
            agent = self._best_agent(AgentType.GENERAL)
        if agent is None:
            return AgentResponse(
                agent_type=AgentType.GENERAL,
                content="服务暂时不可用，请稍后重试。",
                success=False,
            )

        response = await agent.handle(req, on_delta=on_delta)

        # 专属 Agent 失败时降级到 GeneralAgent
        if not response.success and agent_type not in (AgentType.GENERAL, AgentType.ESCALATION):
            logger.warning(f"{agent_type.value} 失败，降级到 GeneralAgent")
            fallback = self._best_agent(AgentType.GENERAL)
            if fallback:
                response = await fallback.handle(req, on_delta=on_delta)

        return response

    # ── 统计（供 Monitor 读取）────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        result = {}
        for agent_type, agents in self._pool.items():
            for i, agent in enumerate(agents):
                key = f"{agent_type.value}_{i}"
                result[key] = {
                    "total":        agent.stats.total,
                    "success_rate": round(agent.stats.success_rate, 3),
                    "avg_ms":       round(agent.stats.avg_ms, 1),
                    "streamed":     agent.stats.first_count,
                    "avg_first_ms": round(agent.stats.avg_first_ms, 1),
                    "monitor_penalty": round(agent.stats.monitor_penalty, 3),
                    "routing_score": round(agent.stats.routing_score(), 3),
                    "role": agent.profile.role,
                    "workflow": list(agent.profile.workflow),
                    "tool_scope": list(agent.profile.tool_scope),
                    "available_tools": list(agent.get_tools()),
                    "model": agent._model,
                }
        return result

    def update_routing_penalties(self, penalties: Dict[str, float]) -> None:
        """
        接收 Monitor 的在线表现反馈，动态调整路由惩罚项。

        penalties 的 key 使用 get_stats() 中的 agent key，例如 consulting_0。
        """
        for agent_type, agents in self._pool.items():
            for i, agent in enumerate(agents):
                key = f"{agent_type.value}_{i}"
                penalty = penalties.get(key, 0.0)
                agent.stats.monitor_penalty = min(max(penalty, 0.0), 0.9)
