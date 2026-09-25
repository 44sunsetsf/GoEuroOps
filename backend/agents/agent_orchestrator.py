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
import inspect
import json
import logging
import os
import time
import uuid
from collections import deque
from datetime import datetime
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

OnDelta = Callable[[str], Awaitable[None]]

from anthropic import AsyncAnthropic
from core.llm_usage import track

from agents.tools import (
    AgentToolSpec,
    build_handoff_summary,
    build_shared_rag_tools,
    billing_tools,
    consulting_tools,
    escalation_tools,
    general_tools,
    make_tool,
)
from business.catalog import get_catalog
from business.lead_store import LeadStore
from core.intent_recognizer import IntentCategory, IntentRecognizer, UrgencyLevel
from core.llm_utils import NO_THINKING_KWARGS, extract_text_content
from core.rag_gate import RagGate, RagGateDecision, RagMode, cancel_speculative

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


def _env_float(name: str, default: float) -> float:
    """读取可选浮点配置；错误配置不应阻塞服务启动。"""
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        logger.warning("忽略非法浮点配置 %s=%r", name, os.getenv(name))
        return default


def _env_int(name: str, default: int) -> int:
    """读取可选整数配置；错误配置不应阻塞服务启动。"""
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        logger.warning("忽略非法整数配置 %s=%r", name, os.getenv(name))
        return default


@dataclass
class AgentStats:
    """Agent 运行时统计，供 Monitor 和路由决策使用。"""
    total:     int   = 0
    success:   int   = 0
    total_ms:  float = 0.0
    monitor_penalty: float = 0.0

    @property
    def success_rate(self) -> float:
        return self.success / self.total if self.total else 1.0

    @property
    def avg_ms(self) -> float:
        return self.total_ms / self.total if self.total else 0.0

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


@dataclass
class Request:
    message:     str
    user_id:     str
    conv_id:     str
    context:     str = ""        # 来自 MemoryManager 的格式化上下文
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


# ── 基础 Agent ────────────────────────────────────────────────────────────────

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
        self._last_tools_used: List[str] = []
        self._last_tool_traces: List[Dict[str, Any]] = []
        self._last_skills: List[Dict[str, Any]] = []
        self._shared_tools: Dict[str, AgentToolSpec] = {}
        self._lead_store: Optional[LeadStore] = None

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

            tools["read_skill_reference"] = make_tool(
                "read_skill_reference",
                "读取本轮已命中 Skill 的参考资料（话术库、FAQ、案例等），skill 和 file 取自 Skill 说明中列出的资料目录。",
                {
                    "skill": {"type": "string", "description": "Skill id"},
                    "file": {"type": "string", "description": "参考资料文件名，如 objection_handling.md"},
                },
                read_skill_reference,
                required=["skill", "file"],
            )
        return tools

    def set_shared_tools(self, tools: Optional[Dict[str, AgentToolSpec]]) -> None:
        self._shared_tools = dict(tools or {})

    async def handle(self, req: Request, on_delta: Optional[OnDelta] = None) -> AgentResponse:
        t0 = time.monotonic()
        self.stats.total += 1
        self._last_tools_used = []
        self._last_tool_traces = []
        self._last_skills = []
        try:
            content = await self._call_llm(req, on_delta=on_delta)
            ms = (time.monotonic() - t0) * 1000
            self.stats.success += 1
            self.stats.total_ms += ms
            escalate = self._needs_escalation(content)
            return AgentResponse(
                agent_type=self.agent_type,
                content=content,
                success=True,
                latency_ms=ms,
                escalate=escalate,
                tools_used=list(self._last_tools_used),
                tool_traces=list(self._last_tool_traces),
                skills_applied=list(self._last_skills),
            )
        except Exception as ex:
            ms = (time.monotonic() - t0) * 1000
            self.stats.total_ms += ms
            logger.error(f"{self.agent_type.value} 处理失败: {ex}")
            return AgentResponse(
                agent_type=self.agent_type,
                content="抱歉，处理您的请求时出现问题，请稍后重试。",
                success=False,
                latency_ms=ms,
                tool_traces=list(self._last_tool_traces),
                skills_applied=list(self._last_skills),
            )

    async def _call_llm(self, req: Request, on_delta: Optional[OnDelta] = None) -> str:
        def _clean(s: str) -> str:
            return s.encode("utf-8", errors="ignore").decode("utf-8")

        messages = []
        if req.context:
            messages.append({"role": "user", "content": f"[背景信息]\n{_clean(req.context)}"})
            messages.append({"role": "assistant", "content": "好的，我已了解背景信息。"})
        if req.entities:
            entities_text = json.dumps(req.entities, ensure_ascii=False)
            messages.append({"role": "user", "content": f"[结构化实体]\n{_clean(entities_text)}"})
            messages.append({"role": "assistant", "content": "好的，我会结合这些结构化实体处理。"})
        role_packet = self._build_role_packet(req)
        if role_packet:
            messages.append({"role": "user", "content": f"[角色输入契约]\n{_clean(role_packet)}"})
            messages.append({"role": "assistant", "content": "好的，我会按照该角色的输入和输出契约处理。"})
        if req.knowledge:
            messages.append({"role": "user", "content": f"[知识库上下文]\n{_clean(RagGate.format_context(req.knowledge))}"})
            messages.append({"role": "assistant", "content": "好的，我会优先依据这些知识库片段回答，并注明来源。"})
        messages.append({"role": "user", "content": _clean(req.message)})

        selection = self._select_skills(req)
        self._last_skills = selection.applied() if selection is not None else []
        system_prompt = self._build_system_prompt(req, selection)
        tools = self._tools_for(req, selection)
        tools_used: List[str] = []
        tool_traces: List[Dict[str, Any]] = []
        for _ in range(3):
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
                    {
                        "name": spec.name,
                        "description": spec.description,
                        "input_schema": spec.input_schema,
                    }
                    for spec in tools.values()
                ]
            content_blocks = await self._complete(request_kwargs, on_delta)
            tool_uses = [block for block in content_blocks if self._block_type(block) == "tool_use"]
            if not tool_uses:
                self._last_tools_used = tools_used
                self._last_tool_traces = tool_traces
                return extract_text_content(content_blocks)

            messages.append({"role": "assistant", "content": content_blocks})
            tool_results = []
            for block in tool_uses:
                name = self._block_value(block, "name")
                tool_use_id = self._block_value(block, "id")
                args = self._block_value(block, "input") or {}
                spec = tools.get(name)
                tool_t0 = time.monotonic()
                call_success = True
                result_success: Optional[bool] = None
                error_text = ""
                if spec is None:
                    call_success = False
                    result: Any = {"success": False, "error": f"工具不在 {self.agent_type.value} Agent 白名单中"}
                    error_text = result["error"]
                else:
                    try:
                        self._validate_tool_input(spec, args)
                        result = spec.handler(req, args)
                        if inspect.isawaitable(result):
                            result = await result
                        tools_used.append(name)
                        if isinstance(result, dict) and "success" in result:
                            result_success = bool(result.get("success"))
                    except Exception as ex:
                        call_success = False
                        logger.warning("Agent 工具 %s 执行失败: %s", name, ex)
                        error_text = str(ex)
                        result = {"success": False, "error": error_text}
                tool_latency_ms = (time.monotonic() - tool_t0) * 1000
                if not error_text and isinstance(result, dict):
                    error_text = str(result.get("error", "") or "")
                tool_traces.append(
                    {
                        "agent_type": self.agent_type.value,
                        "tool_name": name,
                        "tool_use_id": tool_use_id,
                        "input": dict(args),
                        "success": call_success,
                        "result_success": result_success,
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
            messages.append({"role": "user", "content": tool_results})

        self._last_tools_used = tools_used
        self._last_tool_traces = tool_traces
        raise RuntimeError(f"{self.agent_type.value} 工具调用超过最大轮数")

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
        if not isinstance(args, dict):
            raise ValueError("工具参数必须是 JSON 对象")
        schema = spec.input_schema
        for field_name in schema.get("required", []):
            if field_name not in args:
                raise ValueError(f"缺少必需参数: {field_name}")
        properties = schema.get("properties", {})
        unknown = set(args) - set(properties)
        if unknown and schema.get("additionalProperties") is False:
            raise ValueError(f"不允许的工具参数: {', '.join(sorted(unknown))}")
        type_map = {"string": str, "number": (int, float), "integer": int, "boolean": bool}
        for key, value in args.items():
            expected = properties.get(key, {}).get("type")
            if expected in type_map and not isinstance(value, type_map[expected]):
                raise ValueError(f"参数 {key} 类型错误，期望 {expected}")

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
        base_prompt = f"{self.system_prompt}{profile_prompt}"
        if self._skill_manager is None:
            return base_prompt
        if selection is not None:
            skill_prompt = selection.prompt
        else:
            skill_prompt = self._skill_manager.prompt_for(req.message, self.agent_type.value)
        if not skill_prompt:
            return base_prompt
        return f"{base_prompt}\n\n[动态 Skills]\n{skill_prompt}"

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


class GeneralAgent(BaseAgent):
    agent_type    = AgentType.GENERAL
    profile = AgentProfile(
        role="前台接待与分诊",
        mission="接待来访学生，介绍工作室是谁、能做什么，澄清不完整的需求，处理售后进度和投诉的首轮沟通，并把专业问题引导到对应环节。",
        workflow=("回应问候或复述诉求", "判断属于咨询/费用/售后/投诉哪一类", "直接回答或只追问必要字段", "给出下一步"),
        input_contract=("对话历史", "用户画像", "意图与紧急度", "知识库上下文"),
        output_contract=("先回应核心问题", "信息不足时只追问必要字段", "明确下一步和时效", "不编造进度或承诺"),
        handoff_conditions=(
            "查询已购服务的具体进度（需要顾问核对后回复）",
            "投诉或强烈不满",
            "涉及退款、合同或隐私删除",
            "用户明确要求真人顾问",
        ),
        tool_scope=("search_knowledge_base", "inspect_request_context", "suggest_required_fields", "get_studio_profile"),
        temperature=0.3,
        max_tokens=900,
    )
    system_prompt = (
        "你是留学咨询工作室「指北」的前台接待助手。工作室由几位在瑞典的 CS 留学生创办，"
        "专注瑞典、德国、荷兰、芬兰、丹麦的英语授课计算机硕士申请，提供选校咨询和文书辅导。"
        "你负责友好接待、介绍工作室、分诊和售后首轮沟通。"
        "你看不到任何合同、付款或文书进度数据，涉及这些时收集必要信息并说明会由顾问核对后回复，不要编造进度。"
    )

    def _build_role_packet(self, req: Request) -> str:
        packet = json.loads(super()._build_role_packet(req))
        packet["triage_targets"] = {
            "consulting": "选校/申请/服务价格/预约",
            "billing": "付款/退款/发票",
            "escalation": "真人顾问/投诉/隐私",
        }
        packet["response_mode"] = "answer_or_clarify"
        return json.dumps(packet, ensure_ascii=False)

    def get_tools(self) -> Dict[str, AgentToolSpec]:
        tools = super().get_tools()
        tools.update(general_tools())
        return tools


class ConsultingAgent(BaseAgent):
    agent_type    = AgentType.CONSULTING
    profile = AgentProfile(
        role="留学咨询答疑与服务顾问助理",
        mission="解答五国英语授课 CS 硕士的公开知识问题，介绍工作室服务和公开价格，用报价工具给出准确报价，在用户同意后登记咨询线索，并在需要个性化判断时引导预约真人顾问。",
        workflow=(
            "判断问题类型：公开知识 / 服务与价格 / 个性化请求",
            "公开知识结合知识库和国家资料工具回答并标注参考性质",
            "价格问题调用服务查询或报价工具，不自行计算",
            "触及个性化边界时说明原因并引导预约",
            "用户愿意留联系方式时确认同意后登记线索",
        ),
        input_contract=("对话历史", "识别到的国家/服务/入学季实体", "意图与紧急度", "知识库上下文"),
        output_contract=(
            "先回应核心问题",
            "具体数字注明参考来源和核实建议",
            "价格与优惠只来自工具结果",
            "需要个性化判断时明确建议预约顾问并说明下一步",
        ),
        handoff_conditions=(
            "用户要求具体选校/定校建议或个性化项目排序",
            "用户要求撰写、修改或评价文书内容",
            "用户要求个性化申请策略或时间规划",
            "用户要求公开优惠以外的折扣或修改合同条款",
            "用户明确要求真人顾问或表达明确购买意向",
        ),
        tool_scope=(
            "search_knowledge_base",
            "lookup_country_admissions_overview",
            "lookup_service_offering",
            "quote_service_bundle",
            "create_consultation_lead",
        ),
        temperature=0.3,
        max_tokens=1200,
    )
    system_prompt = (
        "你是留学咨询工作室「指北」的咨询助理，服务对象是想申请瑞典、德国、荷兰、芬兰、丹麦英语授课计算机硕士的学生。"
        "你可以：回答这五国 CS 硕士的公开知识问题；介绍工作室的服务、交付内容和公开价格；用报价工具计算报价；"
        "在用户明确同意后登记咨询线索，由顾问联系。"
        "你不能：针对某个人的背景给出选校结论或录取概率判断；撰写、修改或评价文书内容；"
        "承诺录取结果；给出公开规则以外的任何折扣；把参考资料说成最新官方政策。"
        "遇到这些请求时，说明这是顾问 1 对 1 服务的范围，并介绍对应服务和预约方式。"
        "任何金额都必须来自工具结果，不要心算或估算。"
    )

    def _build_role_packet(self, req: Request) -> str:
        packet = json.loads(super()._build_role_packet(req))
        packet["consulting_fields"] = {
            "countries_mentioned": req.entities.get("country", []),
            "services_mentioned": req.entities.get("service", []),
            "intake": req.entities.get("intake", []),
            "test_score": req.entities.get("test_score", []),
            "boundary": "不给个性化选校结论、不写改文书、不承诺录取、不给额外折扣；触及即引导预约顾问",
        }
        return json.dumps(packet, ensure_ascii=False)

    def get_tools(self) -> Dict[str, AgentToolSpec]:
        tools = super().get_tools()
        tools.update(consulting_tools(self._lead_store))
        return tools


class BillingAgent(BaseAgent):
    agent_type    = AgentType.BILLING
    profile = AgentProfile(
        role="服务费用与售后",
        mission="解答付款方式、定金与尾款、退款政策和发票问题；用退款计算工具给出估算，明确所有实际退款都需要顾问核验协议后处理。",
        workflow=("确认费用场景", "收集必要核验字段", "按政策解释或用工具估算", "说明处理路径与时效", "需要实际操作时转顾问"),
        input_contract=("合同号", "金额", "付款时间", "付款渠道", "服务进度", "知识库上下文"),
        output_contract=("需要核验的信息", "按政策可以判断的内容", "估算结果及依据", "下一步处理路径与时效"),
        handoff_conditions=("实际发起退款", "多付或付款成功但服务未确认", "发票作废重开", "对退款金额有异议"),
        tool_scope=("search_knowledge_base", "check_payment_fields", "calculate_refund", "get_payment_policy", "compare_amounts"),
        temperature=0.0,
        max_tokens=1100,
    )
    system_prompt = (
        "你是留学咨询工作室「指北」的费用与售后助理，负责付款、定金尾款、退款和发票问题。"
        "你看不到真实的付款和合同记录；退款金额只能用 calculate_refund 工具按政策估算，并说明最终以顾问核验协议为准。"
        "不要承诺退款一定成功或到账时间早于政策说明。"
    )

    def _build_role_packet(self, req: Request) -> str:
        packet = json.loads(super()._build_role_packet(req))
        packet["verification_fields"] = {
            "contract_id": req.entities.get("contract_id", []),
            "amount": req.entities.get("amount", []),
            "date": req.entities.get("date", []),
            "services_mentioned": req.entities.get("service", []),
            "missing_fields": [
                label for label, values in (
                    ("合同号", req.entities.get("contract_id", [])),
                    ("付款金额", req.entities.get("amount", [])),
                ) if not values
            ],
            "risk_boundary": "不得承诺退款成功、立即到账或直接修改合同",
        }
        return json.dumps(packet, ensure_ascii=False)

    def get_tools(self) -> Dict[str, AgentToolSpec]:
        tools = super().get_tools()
        tools.update(billing_tools())
        return tools


class EscalationAgent(BaseAgent):
    """转顾问节点。

    升级不是一个普通问答 Prompt：它生成标准化的交接单写入线索面板，
    并给用户一个确定的答复（谁、多久、通过什么方式联系），而不是让 LLM
    继续编造处理结果。
    """

    agent_type = AgentType.ESCALATION
    profile = AgentProfile(
        role="转顾问交接",
        mission="确认转交原因，整理已知上下文写入交接单，告知用户顾问的响应时效和时差，不执行未经授权的操作。",
        workflow=("确认转交原因", "整理已知信息", "写入交接单", "告知响应时效"),
        input_contract=("用户消息", "意图", "紧急度", "结构化实体", "对话背景"),
        output_contract=("已转交说明", "交接单编号", "响应时效与时差", "隐私提醒"),
        handoff_conditions=("用户明确要求真人顾问", "紧急、投诉或隐私删除请求"),
        tool_scope=("create_handoff_summary",),
        temperature=0.0,
        max_tokens=500,
    )
    system_prompt = "你负责把用户转交给工作室顾问，不要继续模拟已完成的后台操作。"

    def get_tools(self) -> Dict[str, AgentToolSpec]:
        tools = super().get_tools()
        tools.update(escalation_tools(self._lead_store))
        return tools

    async def handle(self, req: Request, on_delta: Optional[OnDelta] = None) -> AgentResponse:
        t0 = time.monotonic()
        self.stats.total += 1
        intent = req.intent.value if req.intent else "unknown"
        reason = {
            "data_privacy": "用户提出个人资料/隐私相关请求",
            "human_handoff": "用户要求真人顾问",
            "escalation": "用户要求升级处理或提出投诉",
        }.get(intent, f"紧急度 {req.urgency.name if req.urgency else 'UNKNOWN'} 需要顾问尽快跟进")

        ticket_id = None
        deduplicated = False
        tool_traces: List[Dict[str, Any]] = []
        if self._lead_store is not None:
            tool_t0 = time.monotonic()
            try:
                ticket = await self._lead_store.create_handoff(build_handoff_summary(req, reason))
                ticket_id = ticket["id"]
                deduplicated = bool(ticket.get("deduplicated"))
                if not deduplicated:
                    from core.metrics import LEADS_CREATED
                    LEADS_CREATED.labels(type="handoff").inc()
                success, error = True, ""
            except Exception as ex:
                logger.warning("交接单写入失败: %s", ex)
                success, error = False, str(ex)
            tool_traces.append({
                "agent_type": self.agent_type.value,
                "tool_name": "create_handoff_summary",
                "tool_use_id": None,
                "input": {"reason": reason},
                "deduplicated": deduplicated,
                "success": success,
                "result_success": success,
                "latency_ms": round((time.monotonic() - tool_t0) * 1000, 1),
                "cached": False,
                "reranked": False,
                "error": error,
            })

        studio = get_catalog().studio
        if deduplicated:
            lines = ["这个会话已经有一张交接单在顾问那里跟进，我把你刚才的消息补充进去了，不会重复排队。", ""]
        else:
            lines = ["我已经把你的问题转交给工作室顾问。", ""]
        if ticket_id:
            lines.append(f"- 交接单编号：{ticket_id}")
        lines.append(f"- 转交原因：{reason}")
        lines.append(f"- 响应时效：{studio.response_sla}")
        lines.append(f"- 时差提醒：{studio.timezone_note}")
        if intent == "data_privacy":
            lines.append("- 资料删除或停止联系的请求会由顾问确认身份后处理，完成后会告知你。")
        lines += [
            "",
            "如果还没留过联系方式，可以直接回复称呼和微信/邮箱/手机中的一种，方便顾问联系你。"
            "请不要发送身份证号、护照号、银行卡号或任何密码。",
        ]
        content = "\n".join(lines)
        if on_delta is not None:
            await on_delta(content)
        ms = (time.monotonic() - t0) * 1000
        self.stats.success += 1
        self.stats.total_ms += ms
        return AgentResponse(
            agent_type=self.agent_type,
            content=content,
            success=True,
            latency_ms=ms,
            escalate=True,
            tools_used=["create_handoff_summary"] if ticket_id else [],
            tool_traces=tool_traces,
        )


class ResponseComposer:
    """多 Agent 汇总节点，统一主次、去重和输出边界。"""

    def __init__(self, client: AsyncAnthropic, model: str, skill_manager: Optional[Any] = None):
        self._client = client
        self._model = model
        self._skill_manager = skill_manager

    async def compose(self, req: Request, responses: List[AgentResponse]) -> str:
        successful = [response for response in responses if response.success and response.content.strip()]
        if not successful:
            return "抱歉，所有 Agent 均处理失败。"
        if len(successful) == 1:
            return successful[0].content

        evidence = "\n\n".join(
            f"[{response.agent_type.value} Agent 输出]\n{response.content}"
            for response in successful
        )
        prompt = (
            "你是留学咨询工作室「指北」的回复整合助手，负责把多个专业 Agent 的结果合并成一条最终回复。\n"
            "要求：以主 Agent 的结论为主，按用户问题优先级组织内容；去掉重复和冲突表述；"
            "不能补造价格、折扣、退款金额或进度；金额只保留工具算出的数字；如果结论冲突，明确说明需要顾问核实；"
            "保留必要的核验字段和转顾问边界。只输出给用户看的中文回复，不要提及 Agent。\n\n"
            f"主 Agent：{successful[0].agent_type.value}\n"
            f"用户问题：{req.message}\n"
            f"候选结果：\n{evidence}"
        )
        if self._skill_manager is not None and hasattr(self._skill_manager, "select"):
            # 只取常驻的品牌语气类 Skill，保证合并后的口吻和边界统一
            selection = self._skill_manager.select(req.message, "general")
            voice = [m.skill for m in selection.matches if m.skill.mode == "always"]
            if voice:
                prompt += "\n\n[品牌语气与输出边界]\n" + "\n\n".join(skill.to_prompt_block() for skill in voice)
        try:
            response = await self._client.messages.create(
                model=self._model,
                max_tokens=_env_int("GOEUROOPS_COMPOSER_MAX_TOKENS", 1000),
                temperature=_env_float("GOEUROOPS_COMPOSER_TEMPERATURE", 0.1),
                messages=[{"role": "user", "content": prompt}],
                **NO_THINKING_KWARGS,
            )
            content = extract_text_content(response.content).strip()
            if content:
                return content
        except Exception as ex:
            logger.warning("Response Composer 失败，使用确定性合并: %s", ex)

        # 汇总节点不可用时保留主次标签，避免丢失某个专业 Agent 的结论。
        return "\n\n".join(
            f"{response.content}" if index == 0 else f"补充说明：\n{response.content}"
            for index, response in enumerate(successful)
        )


# ── 编排器 ────────────────────────────────────────────────────────────────────

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
# 领域关键词：只用于主/辅 Agent 打分和复合问题检测，不直接决定路由
_CONSULTING_KWS = ["留学", "申请", "硕士", "研究生", "选校", "文书", "雅思", "托福", "aps", "报价", "多少钱",
                   "套餐", "陪跑", "瑞典", "德国", "荷兰", "芬兰", "丹麦", "北欧"]
# 复合问题检测只看"问留学本身"的词：问退款时顺口提到"全程陪跑"不算咨询问题
_CONSULTING_COLLAB_KWS = ["留学", "申请", "硕士", "研究生", "选校", "雅思", "托福", "aps",
                          "瑞典", "德国", "荷兰", "芬兰", "丹麦", "北欧"]
_BILLING_KWS = ["退款", "退钱", "定金", "尾款", "发票", "付款", "多付", "refund", "invoice"]
_GENERAL_KWS = ["进度", "第几轮", "你们是", "工作室", "联系方式", "帮助"]


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

class AgentOrchestrator:
    """
    多 Agent 编排器。

    路由逻辑（三层）：
      1. 意图 → Agent 类型映射
      2. 同类多实例时按 routing_score() 选最优
      3. 专属 Agent 失败时降级到 GeneralAgent
    """

    # 意图 → Agent 类型的静态映射（路由表）
    _INTENT_ROUTING: Dict[IntentCategory, AgentType] = {
        IntentCategory.STUDY_CONSULT:  AgentType.CONSULTING,
        IntentCategory.APPLICATION_PROCESS: AgentType.CONSULTING,
        IntentCategory.SERVICE_INQUIRY: AgentType.CONSULTING,
        IntentCategory.BOOKING:    AgentType.CONSULTING,
        IntentCategory.BILLING:    AgentType.BILLING,
        IntentCategory.REFUND:     AgentType.BILLING,
        IntentCategory.INVOICE:    AgentType.BILLING,
        IntentCategory.PAYMENT_ISSUE: AgentType.BILLING,
        IntentCategory.ESCALATION: AgentType.ESCALATION,
        IntentCategory.HUMAN_HANDOFF: AgentType.ESCALATION,
        IntentCategory.DATA_PRIVACY: AgentType.ESCALATION,
        # 其余意图 → GENERAL（默认）
    }

    def __init__(
        self,
        api_key:  str,
        base_url: Optional[str] = None,
        model:    str = "claude-3-5-sonnet-20241022",
        skill_manager: Optional[Any] = None,
        rag_tool_manager: Optional[Any] = None,
        lead_store: Optional[LeadStore] = None,
        rag_gate: Optional[RagGate] = None,
    ):
        kwargs: Dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        client = track(AsyncAnthropic(**kwargs), "agents")

        self._intent_recognizer = IntentRecognizer(api_key=api_key, base_url=base_url, model=model)
        self._skill_manager = skill_manager
        self._composer = ResponseComposer(client, model, skill_manager)
        self._recent_tool_traces = deque(maxlen=_env_int("GOEUROOPS_TOOL_TRACE_MAX", 200))
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

    def _route(self, intent: Optional[IntentCategory], urgency: Optional[UrgencyLevel]) -> AgentType:
        """
        三层路由决策：
          1. 意图映射
          2. 紧急度覆盖（CRITICAL 直接升级）
          3. 默认 GENERAL
        """
        if urgency == UrgencyLevel.CRITICAL:
            return AgentType.ESCALATION

        if intent and intent in self._INTENT_ROUTING:
            target = self._INTENT_ROUTING[intent]
            # 如果目标类型有可用实例则使用，否则降级
            if target in self._pool and self._pool[target]:
                return target

        return AgentType.GENERAL

    def _route_decision(self, req: Request) -> RoutingDecision:
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

        scores = self._domain_scores(req)
        available_scores = {
            agent_type: score
            for agent_type, score in scores.items()
            if agent_type == AgentType.GENERAL or self._pool.get(agent_type)
        }
        if not available_scores:
            return RoutingDecision(
                primary_agent=AgentType.GENERAL,
                reason="无可用专属 Agent，降级到 GeneralAgent",
                confidence=0.1,
            )

        ordered = sorted(available_scores.items(), key=lambda item: item[1], reverse=True)
        primary_agent, primary_score = ordered[0]

        collaboration_targets = self._collaboration_targets(req)
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

        reason = self._routing_reason(req, available_scores, primary_agent, supporting_agents)
        return RoutingDecision(
            primary_agent=primary_agent,
            supporting_agents=supporting_agents,
            reason=reason,
            confidence=round(min(primary_score, 1.0), 3),
        )

    def _domain_scores(self, req: Request) -> Dict[AgentType, float]:
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

        consulting_hits = sum(1 for kw in _CONSULTING_KWS if kw in msg)
        billing_hits = sum(1 for kw in _BILLING_KWS if kw in msg)
        general_hits = sum(1 for kw in _GENERAL_KWS if kw in msg)

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
    def _routing_reason(
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

    def _collaboration_targets(self, req: Request) -> List[AgentType]:
        """
        判断是否需要多个 Agent 并行协作。

        意图识别通常只返回一个主意图；这里用领域关键词补充检测复合问题，
        例如"咨询瑞典项目的同时又被重复扣款"需要留学咨询和账单 Agent 同时处理。
        """
        msg = req.message.lower()
        targets: List[AgentType] = []

        # 服务名、交付物（"选校报告"）和费用短语（"申请退款"）不是在问留学本身，扫描前去掉
        consult_msg = _strip_service_terms(msg)
        if req.intent in _CONSULTING_INTENTS or any(kw in consult_msg for kw in _CONSULTING_COLLAB_KWS):
            targets.append(AgentType.CONSULTING)
        if req.intent in _BILLING_INTENTS or any(kw in msg for kw in _BILLING_KWS):
            targets.append(AgentType.BILLING)

        # 保持顺序去重，并只返回当前有实例的 Agent 类型。
        deduped = list(dict.fromkeys(targets))
        return [agent_type for agent_type in deduped if self._pool.get(agent_type)]

    @staticmethod
    def _needs_clarification(req: Request) -> bool:
        """低置信度且无明确意图时，先追问，避免误路由。"""
        if req.intent != IntentCategory.OTHER:
            return False
        text = (req.message or "").strip()
        if len(text) <= 2:
            return False
        return req.intent_confidence < 0.5

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
