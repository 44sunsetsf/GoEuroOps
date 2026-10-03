"""接口的请求和响应模型。"""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message:     str
    user_id:     str = "anonymous"
    conv_id:     Optional[str] = None


class ChatResponse(BaseModel):
    conv_id:     str
    request_id:  str = ""
    response:    str
    intent:      str
    intent_group: str = "other"
    agent_type:  str
    agent_types: List[str] = Field(default_factory=list)
    primary_agent: str = ""
    supporting_agents: List[str] = Field(default_factory=list)
    tools_used: List[str] = Field(default_factory=list)
    routing_reason: str = ""
    routing_confidence: float = 0.0
    escalated:   bool
    latency_ms:  float
    knowledge_used: bool = False
    entities: Dict[str, List[str]] = Field(default_factory=dict)
    intent_confidence: float = 0.0
    intent_source_scores: Dict[str, float] = Field(default_factory=dict)
    skills_applied: List[Dict[str, Any]] = Field(default_factory=list)
    rag_gate: Dict[str, Any] = Field(default_factory=dict)


class ToolTraceResponse(BaseModel):
    request_id: str
    found: bool
    trace: Dict[str, Any] = Field(default_factory=dict)


class RecentToolTracesResponse(BaseModel):
    items: List[Dict[str, Any]] = Field(default_factory=list)


class SkillMatchInput(BaseModel):
    message: str
    agent_type: Optional[str] = None
    intent: Optional[str] = None
    history: Optional[List[Dict[str, str]]] = None


class LeadUpdateInput(BaseModel):
    status: Optional[str] = None
    notes: Optional[str] = None


class DocInput(BaseModel):
    """单篇文档输入。"""
    title:   str
    content: str
    domain:  Optional[str] = None  # 归属 Agent 领域（consulting/billing/general 等），不填则所有 Agent 可见


class BatchDocInput(BaseModel):
    """批量文档导入请求体。"""
    documents: List[DocInput]


class EvalIntentInput(BaseModel):
    """意图识别评测用例。"""
    message: str
    expected_intent: str
    context: Optional[Dict[str, Any]] = None


class EvalDialogInput(BaseModel):
    """对话质量评测用例。question 单轮，turns 多轮。"""
    question: Optional[str] = None
    turns: Optional[List[str]] = None
    user_id: Optional[str] = None
    conv_id: Optional[str] = None
    expected_behavior: Optional[str] = None   # 该场景的期望行为，供 LLM Judge 按场景打分
    expected_tools: Optional[List[str]] = None  # 期望调用的工具，确定性检查
    knowledge: bool = False                     # 是否为知识类问题（参与 RAG 门控对照实验）


class EvalRunInput(BaseModel):
    """评测请求。为空时使用内置默认用例。"""
    intent_cases: Optional[List[EvalIntentInput]] = None
    dialog_cases: Optional[List[EvalDialogInput]] = None
    include_skill_evals: bool = True
    compare_rag_gate: bool = False   # 额外跑一组"模型自行检索"对照，会多调用 LLM
