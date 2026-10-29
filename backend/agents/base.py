"""Agent 的公共部分：请求和回复的数据结构，以及所有角色共用的工具循环（BaseAgent）。"""
import asyncio
import json
import logging
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Deque, Dict, List, Optional, Tuple

from anthropic import AsyncAnthropic

from agents.tools import (
    AgentToolSpec,
    make_model_tool,
)
from business.catalog import get_catalog
from business.lead_store import LeadStore
from core.intent_recognizer import IntentCategory, UrgencyLevel
from tooling.amount_guard import AmountGrounding, AmountGuard
from tooling.gateway import get_gateway, validate_args
from tooling.schemas import ReadSkillReferenceArgs
from core.llm_utils import NO_THINKING_KWARGS, extract_text_content, safe_text
from core.rag_gate import RagGate, RagMode


OnDelta = Callable[[str], Awaitable[None]]

MAX_TOOL_ROUNDS = 3
ROUND_LIMIT_FALLBACK = "抱歉，这个问题我暂时没能查全。您可以换个说法再问一次，或者预约顾问为您解答。"


async def reset_stream(on_delta: Optional[OnDelta]) -> None:
    """让前端清掉已经流出去的半段回复。调用方在 on_delta 上挂一个 reset 协程函数才生效。"""
    reset = getattr(on_delta, "reset", None)
    if reset is not None:
        await reset()

logger = logging.getLogger(__name__)


# ── 数据结构 ──────────────────────────────────────────────────────────────────

class AgentType(Enum):
    GENERAL   = "general"    # 前台接待与分诊
    CONSULTING = "consulting"  # 留学咨询与服务介绍/报价/线索
    BILLING   = "billing"    # 服务费用、退款与发票
    ESCALATION = "escalation" # 转顾问交接


@dataclass(frozen=True)
class AgentProfile:

    role: str
    mission: str
    workflow: Tuple[str, ...]
    input_contract: Tuple[str, ...]
    output_contract: Tuple[str, ...]
    handoff_conditions: Tuple[str, ...] = ()
    tool_scope: Tuple[str, ...] = ()
    model: Optional[str] = None
    temperature: float = 0.2
    max_tokens: int = 1024


@dataclass
class AgentStats:
    """Agent 运行时统计，供 Monitor 和路由决策使用。"""
    total:     int   = 0
    success:   int   = 0
    total_ms:  float = 0.0
    # 流式输出时用户看到第一个字的耗时。回答写多长由问题决定，整段生成时间会随之变长，
    # 用户实际等待的是第一个字，所以延迟告警看这个值。
    first_count:    int   = 0
    first_total_ms: float = 0.0
    monitor_penalty: float = 0.0
    # 最近完成的调用：(完成时刻, 是否成功, 首字耗时 ms 或 None)。Monitor 按它算路由惩罚，
    # 累计值会让很久以前的故障一直拖着分数，被排除出路由的 Agent 也就永远恢复不了。
    recent: Deque[Tuple[float, bool, Optional[float]]] = field(
        default_factory=lambda: deque(maxlen=50), repr=False,
    )

    @property
    def success_rate(self) -> float:
        return self.success / self.total if self.total else 1.0

    @property
    def avg_ms(self) -> float:
        return self.total_ms / self.total if self.total else 0.0

    @property
    def avg_first_ms(self) -> float:
        return self.first_total_ms / self.first_count if self.first_count else 0.0

    def first_output(self, t0: float) -> float:
        ms = (time.monotonic() - t0) * 1000
        self.first_count += 1
        self.first_total_ms += ms
        return ms

    def finish(self, success: bool, ms: float, first_ms: Optional[float] = None) -> None:
        """一次调用结束（total 在开始时已经加过）。"""
        if success:
            self.success += 1
        self.total_ms += ms
        self.recent.append((time.monotonic(), success, first_ms))

    def recent_window(self, window_s: float = 300.0) -> Tuple[int, float, float]:
        """最近 window_s 秒内（最多 50 次）完成的调用：样本数、成功率、平均首字耗时（没有流式样本时为 0）。"""
        cutoff = time.monotonic() - window_s
        rows = [r for r in self.recent if r[0] >= cutoff]
        if not rows:
            return 0, 1.0, 0.0
        firsts = [r[2] for r in rows if r[2] is not None]
        return (
            len(rows),
            sum(1 for r in rows if r[1]) / len(rows),
            sum(firsts) / len(firsts) if firsts else 0.0,
        )

    def routing_score(self) -> float:
        """路由评分：成功率高、延迟低的 Agent 得分高。"""
        latency_score = 1.0 / (1.0 + self.avg_ms / 1000)
        base_score = self.success_rate * 0.7 + latency_score * 0.3
        return base_score * max(0.0, 1.0 - self.monitor_penalty)


@dataclass
class AgentResponse:
    agent_type:  AgentType
    content:     str
    success:     bool
    confidence:  float = 1.0
    latency_ms:  float = 0.0
    escalate:    bool  = False   # 是否需要升级
    tools_used:  List[str] = field(default_factory=list)
    tool_traces: List[Dict[str, Any]] = field(default_factory=list)
    skills_applied: List[Dict[str, Any]] = field(default_factory=list)
    streamed:    bool = False        # 失败前是否已经有文字流给了用户


@dataclass
class Request:
    message:     str
    user_id:     str
    conv_id:     str
    context:     str = ""        # 来自 MemoryManager 的格式化上下文
    memory_facts: List[str] = field(default_factory=list)   # 客户档案（交接单用）
    memory_cases: List[str] = field(default_factory=list)   # 以前的咨询记录（交接单用）
    history:     Optional[List[Dict[str, str]]] = None  # 对话历史，传给意图识别
    entities:    Dict[str, List[str]] = field(default_factory=dict)
    intent:      Optional[IntentCategory] = None
    intent_group: Optional[str] = None
    urgency:     Optional[UrgencyLevel]   = None
    intent_confidence: float = 1.0
    intent_source_scores: Dict[str, float] = field(default_factory=dict)
    request_id:  str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    # 意图门控 RAG 的结果：prefetch 时 knowledge 为预取片段；off 时本轮不暴露检索工具
    rag_mode:    Optional[str] = None
    knowledge:   List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class OrchestratorResult:
    request_id:  str
    response:    str
    agent_type:  AgentType
    intent:      Optional[IntentCategory]
    escalated:   bool  = False
    latency_ms:  float = 0.0
    agent_types: List[AgentType] = field(default_factory=list)
    primary_agent: Optional[AgentType] = None
    supporting_agents: List[AgentType] = field(default_factory=list)
    tools_used: List[str] = field(default_factory=list)
    tool_traces: List[Dict[str, Any]] = field(default_factory=list)
    routing_reason: str = ""
    routing_confidence: float = 0.0
    intent_group: Optional[str] = None
    intent_confidence: float = 0.0
    intent_source_scores: Dict[str, float] = field(default_factory=dict)
    entities: Dict[str, List[str]] = field(default_factory=dict)
    skills_applied: List[Dict[str, Any]] = field(default_factory=list)
    rag_gate: Dict[str, Any] = field(default_factory=dict)


# ── 基础 Agent ────────────────────────────────────────────────────────────────

@dataclass
class _RunRecord:
    """一次 handle() 调用的执行记录。

    Agent 实例在所有并发请求之间共享，所以这些按请求变化的数据不能挂在 self 上，
    否则两个请求交错 await 时会互相覆盖 trace。
    """

    tools_used: List[str] = field(default_factory=list)
    tool_traces: List[Dict[str, Any]] = field(default_factory=list)
    skills: List[Dict[str, Any]] = field(default_factory=list)


class BaseAgent:
    """所有 Agent 的基类，封装 LLM 调用、角色契约和统计。"""

    agent_type: AgentType
    system_prompt: str
    profile: AgentProfile

    def __init__(
        self,
        client: AsyncAnthropic,
        model: str,
        skill_manager: Optional[Any] = None,
        profile: Optional[AgentProfile] = None,
    ):
        self._client = client
        self.profile = profile or self.profile
        self._model  = self.profile.model or model
        self._skill_manager = skill_manager
        self.stats   = AgentStats()
        self._shared_tools: Dict[str, AgentToolSpec] = {}
        self._lead_store: Optional[LeadStore] = None
        self._gateway = get_gateway()

    def set_lead_store(self, lead_store: Optional[LeadStore]) -> None:
        self._lead_store = lead_store

    def get_tools(self) -> Dict[str, AgentToolSpec]:
        """返回该角色真实可调用的工具白名单。"""
        return dict(self._shared_tools)

    def _select_skills(self, req: Request) -> Any:
        """按意图、关键词、语义样例和会话历史挑选本轮注入的 Skills。"""
        if self._skill_manager is None:
            return None
        if not hasattr(self._skill_manager, "select"):
            return None
        selection = self._skill_manager.select(
            req.message,
            self.agent_type.value,
            intent=req.intent.value if req.intent else None,
            intent_group=req.intent_group,
            history=req.history,
        )
        if hasattr(self._skill_manager, "record"):
            self._skill_manager.record(selection, self.agent_type.value)
        return selection

    def _tools_for(self, req: Request, selection: Any = None) -> Dict[str, AgentToolSpec]:
        """本轮真正暴露给模型的工具：角色白名单 + 门控调整 + Skill 参考资料工具。"""
        tools = self.get_tools()
        if req.rag_mode == RagMode.OFF.value:
            tools.pop("search_knowledge_base", None)
        if selection is not None and getattr(selection, "has_references", False):
            skill_manager = self._skill_manager
            allowed = list(selection.skill_ids)

            def read_skill_reference(_req: Request, args: Dict[str, Any]) -> Dict[str, Any]:
                return skill_manager.read_reference(
                    str(args.get("skill", "")), str(args.get("file", "")), allowed_skill_ids=allowed
                )

            tools["read_skill_reference"] = make_model_tool(
                "read_skill_reference",
                "读取本轮已命中 Skill 的参考资料（话术库、FAQ、案例等），skill 和 file 取自 Skill 说明中列出的资料目录。",
                ReadSkillReferenceArgs,
                read_skill_reference,
            )
        return tools

    def set_shared_tools(self, tools: Optional[Dict[str, AgentToolSpec]]) -> None:
        self._shared_tools = dict(tools or {})

    async def handle(self, req: Request, on_delta: Optional[OnDelta] = None) -> AgentResponse:
        t0 = time.monotonic()
        self.stats.total += 1
        run = _RunRecord()
        first_ms: Optional[float] = None
        if on_delta is not None:
            forward = on_delta

            async def on_delta(text: str) -> None:
                nonlocal first_ms
                if first_ms is None:
                    first_ms = self.stats.first_output(t0)
                await forward(text)
        try:
            content = await self._call_llm(req, on_delta=on_delta, run=run)
            ms = (time.monotonic() - t0) * 1000
            self.stats.finish(True, ms, first_ms)
            escalate = self._needs_escalation(content)
            return AgentResponse(
                agent_type=self.agent_type,
                content=content,
                success=True,
                latency_ms=ms,
                escalate=escalate,
                tools_used=list(run.tools_used),
                tool_traces=list(run.tool_traces),
                skills_applied=list(run.skills),
            )
        except Exception as ex:
            ms = (time.monotonic() - t0) * 1000
            self.stats.finish(False, ms, first_ms)
            logger.error(f"{self.agent_type.value} 处理失败: {ex}")
            return AgentResponse(
                agent_type=self.agent_type,
                content="抱歉，处理您的请求时出现问题，请稍后重试。",
                success=False,
                latency_ms=ms,
                streamed=first_ms is not None,
                tool_traces=list(run.tool_traces),
                skills_applied=list(run.skills),
            )

    async def _call_llm(
        self, req: Request, on_delta: Optional[OnDelta] = None, run: Optional[_RunRecord] = None,
    ) -> str:
        run = run if run is not None else _RunRecord()
        messages = self._build_messages(req)

        selection = self._select_skills(req)
        run.skills = selection.applied() if selection is not None else []
        system_prompt = self._build_system_prompt(req, selection)
        tools = self._tools_for(req, selection)
        tool_traces = run.tool_traces        # 边执行边写进 run，出异常时 handle() 也能拿到已有的 trace

        grounding = self._amount_grounding(req, messages, system_prompt)
        guard = AmountGuard(grounding)

        async def guarded_delta(text: str) -> None:
            safe = guard.feed(text)
            if safe and on_delta is not None:
                await on_delta(safe)

        for round_no in range(MAX_TOOL_ROUNDS):
            last_round = round_no == MAX_TOOL_ROUNDS - 1
            request_kwargs = self._request_kwargs(system_prompt, messages, tools, last_round)
            content_blocks = await self._complete(request_kwargs, guarded_delta if on_delta is not None else None)
            if on_delta is not None:
                tail = guard.flush()
                if tail:
                    await on_delta(tail)
            tool_uses = [block for block in content_blocks if self._block_type(block) == "tool_use"]
            if tool_uses and last_round:
                # 兼容不支持 tool_choice 的 Anthropic 兼容服务：不再执行，只用这一轮的文字
                ignored = [self._block_value(b, "name") for b in tool_uses]
                logger.warning(f"{self.agent_type.value} 最后一轮仍请求工具 {ignored}，已忽略")
                tool_traces.append(self._guard_trace(
                    "tool_round_limit", {"ignored": ignored},
                    f"工具调用达到 {MAX_TOOL_ROUNDS} 轮上限，最后一轮的工具请求未执行",
                ))
                tool_uses = []
                if not extract_text_content(content_blocks).strip():
                    if on_delta is not None:
                        await on_delta(ROUND_LIMIT_FALLBACK)
                    return ROUND_LIMIT_FALLBACK
            if not tool_uses:
                text = guard.sanitize(extract_text_content(content_blocks))
                if guard.violations:
                    amounts = sorted({a for v in guard.violations for a in v.amounts})
                    tool_traces.append(self._guard_trace(
                        "amount_guard", {"violations": [v.to_dict() for v in guard.violations]},
                        f"回复里的金额没有依据：{amounts}（模式 {guard.mode}）",
                    ))
                return text

            messages.append({"role": "assistant", "content": content_blocks})
            outcomes = await self._run_tool_batch(tool_uses, tools, req)
            tool_results = self._record_tool_outcomes(tool_uses, outcomes, tools, run, grounding)
            messages.append({"role": "user", "content": tool_results})

        raise RuntimeError(f"{self.agent_type.value} 工具调用超过最大轮数")   # 最后一轮必定返回，走不到这里

    def _build_messages(self, req: Request) -> List[Dict[str, Any]]:
        """背景、实体、角色契约、知识库各用一对 user/assistant 消息注入，最后是用户原话。"""
        messages: List[Dict[str, Any]] = []

        def inject(label: str, body: str, ack: str) -> None:
            messages.append({"role": "user", "content": f"[{label}]\n{safe_text(body)}"})
            messages.append({"role": "assistant", "content": ack})

        if req.context:
            inject("背景信息", req.context, "好的，我已了解背景信息。")
        if req.entities:
            inject("结构化实体", json.dumps(req.entities, ensure_ascii=False), "好的，我会结合这些结构化实体处理。")
        role_packet = self._build_role_packet(req)
        if role_packet:
            inject("角色输入契约", role_packet, "好的，我会按照该角色的输入和输出契约处理。")
        if req.knowledge:
            inject("知识库上下文", RagGate.format_context(req.knowledge), "好的，我会优先依据这些知识库片段回答，并注明来源。")
        messages.append({"role": "user", "content": safe_text(req.message)})
        return messages

    @staticmethod
    def _amount_grounding(req: Request, messages: List[Dict[str, Any]], system_prompt: str) -> AmountGrounding:
        """金额护栏的依据：回复里的人民币金额必须能在这些地方找到（工具结果在执行后追加）。"""
        grounding = AmountGrounding()
        grounding.add_many(m["content"] for m in messages if isinstance(m["content"], str))
        grounding.add_text(system_prompt)
        grounding.add_json(req.history)
        grounding.add_json(get_catalog().model_dump())
        return grounding

    def _request_kwargs(
        self, system_prompt: str, messages: List[Dict[str, Any]], tools: Dict[str, AgentToolSpec], last_round: bool,
    ) -> Dict[str, Any]:
        request_kwargs: Dict[str, Any] = {
            "model": self._model,
            "max_tokens": self.profile.max_tokens,
            "temperature": self.profile.temperature,
            "system": system_prompt,
            "messages": messages,
            **NO_THINKING_KWARGS,
        }
        if tools:
            request_kwargs["tools"] = [
                {"name": spec.name, "description": spec.description, "input_schema": spec.input_schema}
                for spec in tools.values()
            ]
            # 最后一轮禁止再调工具，逼模型用已有结果作答。tools 不能去掉：
            # 消息里已有 tool_use / tool_result 时，API 要求请求里仍然带着工具定义
            if last_round:
                request_kwargs["tool_choice"] = {"type": "none"}
        return request_kwargs

    def _guard_trace(self, name: str, payload: Dict[str, Any], error: str) -> Dict[str, Any]:
        """框架自己拦下的事件（轮数上限、金额护栏）也记成一条工具轨迹，方便在 Trace 里看到。"""
        return {
            "agent_type": self.agent_type.value,
            "tool_name": name,
            "tool_use_id": "",
            "input": payload,
            "success": True,
            "result_success": False,
            "latency_ms": 0.0,
            "cached": False,
            "reranked": False,
            "error": error,
        }

    def _record_tool_outcomes(
        self, tool_uses: List[Any], outcomes: List[Any], tools: Dict[str, AgentToolSpec],
        run: _RunRecord, grounding: AmountGrounding,
    ) -> List[Dict[str, Any]]:
        """把一批工具结果写进轨迹和金额依据，返回发回给模型的 tool_result 列表。"""
        tool_results = []
        for block, outcome in zip(tool_uses, outcomes):
            name = self._block_value(block, "name")
            tool_use_id = self._block_value(block, "id")
            args = self._block_value(block, "input") or {}
            spec = tools.get(name)
            if outcome is None:
                error_text = f"工具不在 {self.agent_type.value} Agent 白名单中"
                result: Any = {"success": False, "error": error_text}
                call_success, result_success, kind, tool_latency_ms, truncated = False, None, "not_allowed", 0.0, False
            else:
                result, call_success, kind = outcome.result, outcome.ok, outcome.kind
                error_text, result_success = outcome.error, outcome.result_success
                tool_latency_ms, truncated = outcome.latency_ms, outcome.truncated
                if call_success:
                    run.tools_used.append(name)
            if not error_text and isinstance(result, dict):
                error_text = str(result.get("error", "") or "")
            grounding.add_json(result)
            run.tool_traces.append(
                {
                    "agent_type": self.agent_type.value,
                    "tool_name": name,
                    "tool_use_id": tool_use_id,
                    "input": dict(args) if isinstance(args, dict) else {"raw": str(args)[:200]},
                    "success": call_success,
                    "result_success": result_success,
                    "outcome": kind,
                    "side_effect": spec.policy.side_effect if spec is not None else None,
                    "truncated": truncated,
                    "latency_ms": round(tool_latency_ms, 1),
                    "cached": bool(result.get("cached")) if isinstance(result, dict) else False,
                    "reranked": bool(result.get("reranked")) if isinstance(result, dict) else False,
                    "error": error_text,
                }
            )
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": json.dumps(result, ensure_ascii=False),
            })
        return tool_results

    async def _run_tool_batch(
        self, tool_uses: List[Any], tools: Dict[str, AgentToolSpec], req: Request,
    ) -> List[Optional[Any]]:
        """执行同一轮里模型请求的所有工具，结果按请求顺序返回；不在白名单里的位置为 None。

        同一轮的请求是模型一次性发出的，彼此不依赖对方的结果，所以只读和外部检索类
        并行执行；写入类（如登记线索）之间按顺序执行，避免对同一份数据并发写。
        """
        outcomes: List[Optional[Any]] = [None] * len(tool_uses)
        concurrent: List[Tuple[int, AgentToolSpec, Any]] = []
        writes: List[Tuple[int, AgentToolSpec, Any]] = []
        for i, block in enumerate(tool_uses):
            spec = tools.get(self._block_value(block, "name"))
            if spec is None:
                self._gateway.record_rejected("not_allowed")
                continue
            args = self._block_value(block, "input") or {}
            (writes if spec.policy.side_effect == "write" else concurrent).append((i, spec, args))

        async def run_one(i: int, spec: AgentToolSpec, args: Any) -> None:
            outcomes[i] = await self._gateway.call(spec, req, args)

        async def run_writes() -> None:
            for item in writes:
                await run_one(*item)

        await asyncio.gather(*(run_one(*item) for item in concurrent), run_writes())
        return outcomes

    async def _complete(
        self,
        request_kwargs: Dict[str, Any],
        on_delta: Optional[OnDelta],
    ) -> List[Any]:
        """执行一次 LLM 调用；有 on_delta 时用流式接口，逐 token 转发文本增量。

        工具调用轮次也走流式接口：模型在决定调用工具前偶尔会先吐出一小段
        文本，直接转发即可；真正的最终回答轮（不含 tool_use）会完整地
        逐 token 流式返回。
        """
        if on_delta is None:
            resp = await self._client.messages.create(**request_kwargs)
            return list(resp.content or [])

        async with self._client.messages.stream(**request_kwargs) as stream:
            async for text in stream.text_stream:
                if text:
                    await on_delta(text)
            final_message = await stream.get_final_message()
        return list(final_message.content or [])

    @staticmethod
    def _block_type(block: Any) -> Optional[str]:
        if isinstance(block, dict):
            return block.get("type")
        return getattr(block, "type", None)

    @staticmethod
    def _block_value(block: Any, key: str) -> Any:
        if isinstance(block, dict):
            return block.get(key)
        return getattr(block, key, None)

    @staticmethod
    def _validate_tool_input(spec: AgentToolSpec, args: Any) -> None:
        validate_args(spec, args)

    def _build_system_prompt(self, req: Request, selection: Any = None) -> str:
        """把角色契约和动态 Skills 拼入 system prompt。"""
        profile_prompt = (
            f"\n\n[角色契约]\n"
            f"角色：{self.profile.role}\n"
            f"职责：{self.profile.mission}\n"
            f"处理流程：{' -> '.join(self.profile.workflow)}\n"
            f"可用输入：{'；'.join(self.profile.input_contract)}\n"
            f"输出要求：{'；'.join(self.profile.output_contract)}\n"
            f"升级条件：{'；'.join(self.profile.handoff_conditions) or '无，按通用接待规则处理'}\n"
            f"允许的数据/工具范围：{'、'.join(self.profile.tool_scope) or '仅使用当前请求上下文'}\n"
            "不要声称执行了未提供的查询、修改或退款操作；缺少证据时明确说明需要核验。"
        )
        prompt = f"{self.system_prompt}{profile_prompt}"
        if self._skill_manager is not None:
            if selection is not None:
                skill_prompt = selection.prompt
            else:
                skill_prompt = self._skill_manager.prompt_for(req.message, self.agent_type.value)
            if skill_prompt:
                prompt = f"{prompt}\n\n[动态 Skills]\n{skill_prompt}"
        # 放在最后：参考资料和 Skills 都是中文，语言要求写在前面会被淹没
        return (
            f"{prompt}\n\n[回答语言]\n"
            "看用户最近一条消息：用英文提问就必须整段用英文回答（含标题和要点），即使参考资料是中文，"
            "并把资料内容翻译成英文；用中文提问就用中文。院校名、服务名可保留原文。"
        )

    def _build_role_packet(self, req: Request) -> str:
        """给子 Agent 的确定性输入包；子类可补充领域字段。"""
        packet = {
            "agent_type": self.agent_type.value,
            "intent": req.intent.value if req.intent else None,
            "intent_group": req.intent_group,
            "urgency": req.urgency.name if req.urgency else None,
            "intent_confidence": round(req.intent_confidence, 4),
            "available_entities": req.entities or {},
            "rag_mode": req.rag_mode,
        }
        return json.dumps(packet, ensure_ascii=False)

    def _needs_escalation(self, content: str) -> bool:
        """检测 Agent 是否已经明确把用户交给人工（关键词检测）。"""
        keywords = ["转人工", "已为你转交顾问", "已转交顾问", "escalate", "无法处理"]
        return any(kw in content for kw in keywords)


