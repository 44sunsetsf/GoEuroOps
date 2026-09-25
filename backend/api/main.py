"""
GoEuroOps 留学业务智能运营中枢 — FastAPI 入口

启动时打印小熊饼干图案。
所有核心组件在 lifespan 中初始化，通过环境变量配置。
"""
import asyncio
import json
import logging
import os
import pathlib
import secrets
import sys
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional


_ROOT = str(pathlib.Path(__file__).parent.parent.resolve())
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import uvicorn
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Header, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

load_dotenv()

logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

BANNER = r"""
    ʕ•ᴥ•ʔ  ʕ•ᴥ•ʔ  ʕ•ᴥ•ʔ
   ╔══════════════════════╗
   ║   GoEuroOps  v3.0     ║
   ║ 留学业务智能运营中枢 ║
   ╚══════════════════════╝
    ʕ•ᴥ•ʔ  ʕ•ᴥ•ʔ  ʕ•ᴥ•ʔ
"""

# ── 全局组件（lifespan 中初始化）─────────────────────────────────────────────
_orchestrator = None
_memory       = None
_tool_manager = None
_monitor      = None
_evaluator    = None
_skill_manager = None
_lead_store   = None
_quota        = None

def _anthropic_cfg() -> Dict[str, Any]:
    key = os.getenv("ANTHROPIC_API_KEY", "")
    if not key:
        raise RuntimeError("未设置 ANTHROPIC_API_KEY")
    cfg: Dict[str, Any] = {
        "api_key":  key,
        "model":    os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022").strip(),
    }
    base_url = os.getenv("ANTHROPIC_BASE_URL", "").strip()
    if base_url:
        cfg["base_url"] = base_url
    return cfg


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _orchestrator, _memory, _tool_manager, _monitor, _evaluator, _skill_manager, _lead_store, _quota

    print(BANNER, flush=True)

    from agents.agent_orchestrator import AgentOrchestrator, Request
    from api.quota import DailyQuota
    from core.intent_recognizer import IntentRecognizer
    from evaluation.evaluator import EndToEndEvaluator
    from mcp.knowledge_base import KnowledgeBase
    from mcp.tool_manager import MCPToolManager, Tool
    from memory.conversation_memory import MemoryManager
    from monitor.performance_monitor import PerformanceMonitor
    from core.skill_loader import SkillManager
    from business.catalog import get_catalog, get_countries
    from business.lead_store import LeadStore

    cfg = _anthropic_cfg()
    # 业务目录在启动时校验一次，YAML 写错直接启动失败，而不是等用户问价格时才暴露
    catalog = get_catalog()
    logger.info(
        "业务目录已加载: %s，%d 项服务，%d 个国家",
        catalog.catalog_version, len(catalog.services), len(get_countries().countries),
    )
    logger.info(f"模型: {cfg['model']}  base_url: {cfg.get('base_url', '(官方)')}")

    # 意图识别器（Orchestrator 内部也会创建，这里单独暴露给 Evaluator）
    recognizer = IntentRecognizer(
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url"),
        model=cfg["model"],
    )

    # Skills：启动时从目录加载业务能力说明，并在 Agent 调用 LLM 时动态注入。
    skills_dir = os.getenv("GOEUROOPS_SKILLS_DIR", str(pathlib.Path(_ROOT) / "skills"))
    _skill_manager = SkillManager(
        root_dir=skills_dir,
        max_prompt_chars=int(os.getenv("GOEUROOPS_SKILLS_MAX_PROMPT_CHARS", "6000")),
    )
    _skill_manager.load()

    # 公开演示的每日对话上限（GOEUROOPS_DAILY_CHAT_LIMIT，0 = 不限）
    _quota = DailyQuota(
        limit=int(os.getenv("GOEUROOPS_DAILY_CHAT_LIMIT", "0") or 0),
        redis_url=os.getenv("REDIS_URL", "redis://redis:6379/0"),
    )

    # 线索库：咨询线索和转顾问交接单，前端「线索」面板读取
    _lead_store = LeadStore(redis_url=os.getenv("REDIS_URL", "redis://redis:6379/0"))

    # Agent 编排器
    _orchestrator = AgentOrchestrator(
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url"),
        model=cfg["model"],
        skill_manager=_skill_manager,
        lead_store=_lead_store,
    )

    # 记忆管理器（Redis 工作记忆 + ChromaDB 情景记忆/用户画像）
    _memory = MemoryManager(
        redis_url=os.getenv("REDIS_URL", "redis://redis:6379/0"),
        chroma_host=os.getenv("CHROMA_HOST", "chromadb"),
        chroma_port=int(os.getenv("CHROMA_PORT", "8000")),
        chroma_path=os.getenv("CHROMA_PERSIST_DIRECTORY", "/app/data/chroma"),
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url"),
        model=cfg["model"],
    )

    # MCP 工具管理器 + RAG 知识库（基于 ChromaDB 的真实检索）
    _tool_manager = MCPToolManager(
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url"),
        model=cfg["model"],
    )
    kb = KnowledgeBase(
        chroma_host=os.getenv("CHROMA_HOST", "chromadb"),
        chroma_port=int(os.getenv("CHROMA_PORT", "8000")),
        chroma_path=os.getenv("CHROMA_PERSIST_DIRECTORY", "/app/data/chroma"),
    )
    logger.info(f"知识库已加载: {await kb.doc_count_async()} 个文档片段")

    def knowledge_fallback(params: Dict[str, Any], context: Optional[Dict[str, Any]], error: str):
        query = params.get("query", "")
        return [{
            "title": "知识库降级结果",
            "content": f"知识库暂时不可用，未能完成对“{query}”的语义检索。请稍后重试，或请顾问确认。",
            "score": 0.0,
            "fallback": True,
            "error": error,
        }]

    _tool_manager.register(Tool(
        name="knowledge_search",
        description="搜索知识库（基于 ChromaDB 向量检索）",
        handler=kb.search_handler,
        schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "top_k": {"type": "integer"},
            },
            "required": ["query"],
        },
        cache_ttl=300.0,
        supports_rerank=True,
        fallback=knowledge_fallback,
    ))
    if _orchestrator is not None:
        _orchestrator.set_shared_tools(_tool_manager)

    # 性能监控（可选启动 Prometheus）
    prom_port = int(os.getenv("PROMETHEUS_PORT", "0")) or None
    _monitor = PerformanceMonitor(
        orchestrator=_orchestrator,
        tool_manager=_tool_manager,
        interval_s=float(os.getenv("MONITOR_INTERVAL", "10")),
        webhook_url=os.getenv("ALERT_WEBHOOK_URL") or None,
        prometheus_port=prom_port,
    )
    await _monitor.start()

    # 评测器
    _evaluator = EndToEndEvaluator(
        orchestrator=_orchestrator,
        recognizer=recognizer,
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url"),
        model=cfg["model"],
        baseline_path=os.getenv("EVAL_BASELINE_PATH", "/app/data/eval/baseline.json"),
        skill_manager=_skill_manager,
    )

    logger.info("GoEuroOps 已就绪")
    yield

    await _monitor.stop()
    if _memory is not None:
        await _memory.close()
    logger.info("GoEuroOps 已关闭")


# ── FastAPI ───────────────────────────────────────────────────────────────────
app = FastAPI(
    title="GoEuroOps 留学业务智能运营中枢",
    version="3.0.0",
    lifespan=lifespan,
    docs_url="/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── 请求/响应模型 ─────────────────────────────────────────────────────────────
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


# ── 路由 ──────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    if _orchestrator is None:
        raise HTTPException(503, "服务未就绪")
    return {"status": "ok", "agents": _orchestrator.get_stats()}


@app.get("/skills", tags=["Skills"])
async def skills_summary():
    """查看当前已加载的 Skills，便于确认热加载结果和排查解析错误。"""
    if _skill_manager is None:
        raise HTTPException(503, "Skills 未初始化")
    return _skill_manager.summary()


@app.post("/skills/reload", tags=["Skills"])
async def reload_skills():
    """运行时重新扫描 Skill 目录，不需要重启服务。"""
    if _skill_manager is None:
        raise HTTPException(503, "Skills 未初始化")
    _skill_manager.reload()
    if _orchestrator is not None:
        _orchestrator.set_skill_manager(_skill_manager)
    return _skill_manager.summary()


class SkillMatchInput(BaseModel):
    message: str
    agent_type: Optional[str] = None
    intent: Optional[str] = None
    history: Optional[List[Dict[str, str]]] = None


@app.post("/skills/match", tags=["Skills"])
async def match_skills(body: SkillMatchInput):
    """
    命中测试（干跑）：给一句话，返回每个 Skill 的得分明细和最终会注入哪些。

    不传 intent 时会现场做一次意图识别，与真实对话链路保持一致。
    """
    if _skill_manager is None:
        raise HTTPException(503, "Skills 未初始化")
    intent, intent_group, intent_confidence = body.intent, None, None
    if intent is None and _orchestrator is not None:
        result = await _orchestrator.recognize_intent(body.message, history=body.history)
        intent, intent_group, intent_confidence = result.intent.value, result.intent_group, result.confidence
    report = _skill_manager.match_report(
        body.message,
        body.agent_type,
        intent=intent,
        intent_group=intent_group,
        history=body.history,
    )
    report["intent_confidence"] = intent_confidence
    return report


@app.get("/skills/evals", tags=["Skills"])
async def run_skill_evals():
    """跑所有 Skill 自带的命中回归用例（skills/*/evals/cases.json）。"""
    if _skill_manager is None:
        raise HTTPException(503, "Skills 未初始化")
    return _skill_manager.run_evals()


@app.get("/skills/{skill_id}", tags=["Skills"])
async def skill_detail(skill_id: str):
    """查看单个 Skill 的完整正文、参考资料目录和命中统计。"""
    if _skill_manager is None:
        raise HTTPException(503, "Skills 未初始化")
    detail = _skill_manager.detail(skill_id)
    if detail is None:
        raise HTTPException(404, f"Skill 不存在: {skill_id}")
    return detail


# ── 业务目录 / 线索 ───────────────────────────────────────────────────────────

@app.get("/catalog", tags=["业务"])
async def catalog_view():
    """服务价目、优惠、付款退款政策和五国资料（前端价目面板使用）。"""
    from business.catalog import get_catalog, get_countries

    return {"catalog": get_catalog().model_dump(), "countries": get_countries().model_dump()}


def _require_admin(x_admin_token: Optional[str] = Header(default=None)) -> None:
    """设置了 GOEUROOPS_ADMIN_TOKEN 时，线索接口必须带 X-Admin-Token。"""
    expected = os.getenv("GOEUROOPS_ADMIN_TOKEN", "").strip()
    if expected and not secrets.compare_digest(x_admin_token or "", expected):
        raise HTTPException(401, "需要有效的 X-Admin-Token")


class LeadUpdateInput(BaseModel):
    status: Optional[str] = None
    notes: Optional[str] = None


@app.get("/leads", tags=["线索"], dependencies=[Depends(_require_admin)])
async def list_leads(status: Optional[str] = None, type: Optional[str] = None, limit: int = 100):
    """咨询线索与转顾问交接单列表（按时间倒序）。"""
    if _lead_store is None:
        raise HTTPException(503, "线索库未初始化")
    items = await _lead_store.list(status=status, lead_type=type, limit=limit)
    return {"items": items, "stats": await _lead_store.stats()}


@app.patch("/leads/{lead_id}", tags=["线索"], dependencies=[Depends(_require_admin)])
async def update_lead(lead_id: str, body: LeadUpdateInput):
    """更新线索状态（new / contacted / converted / closed）和跟进备注。"""
    if _lead_store is None:
        raise HTTPException(503, "线索库未初始化")
    try:
        lead = await _lead_store.update(lead_id, status=body.status, notes=body.notes)
    except ValueError as ex:
        raise HTTPException(400, str(ex))
    if lead is None:
        raise HTTPException(404, f"线索不存在: {lead_id}")
    return lead


async def _run_chat(req: ChatRequest, conv_id: str, on_delta=None) -> ChatResponse:
    """
    一次完整对话：记忆读取 → 编排（意图识别 ∥ 投机预取 → 路由 → RAG 门控 →
    Skills 注入 → Agent 执行）→ 记忆写入。/chat 和 /chat/stream 共用。

    意图识别交给编排器内部完成，这样知识库预取才能和意图识别并行。
    """
    from agents.agent_orchestrator import Request as OrcReq
    from core import chat_log
    from memory.conversation_memory import MsgRole

    mem_ctx = await _memory.get_context(req.user_id, conv_id, query=req.message)
    history = [
        {"role": m.role.value, "content": m.content}
        for m in mem_ctx.recent_messages[-5:]
    ] if mem_ctx.recent_messages else None

    orch_req = OrcReq(
        message=req.message,
        user_id=req.user_id,
        conv_id=conv_id,
        context=mem_ctx.to_prompt_text(),
        history=history,
    )
    result = await _orchestrator.run(orch_req, on_delta=on_delta)

    await _memory.add_message(req.user_id, conv_id, MsgRole.USER, req.message)
    await _memory.add_message(req.user_id, conv_id, MsgRole.ASSISTANT, result.response)
    # 异步更新用户画像（不阻塞响应）
    asyncio.create_task(_memory.update_profile(req.user_id, conv_id))

    response = ChatResponse(
        conv_id=conv_id,
        request_id=result.request_id,
        response=result.response,
        intent=result.intent.value if result.intent else "other",
        intent_group=result.intent_group or "other",
        agent_type=result.agent_type.value,
        agent_types=[agent_type.value for agent_type in result.agent_types],
        primary_agent=result.primary_agent.value if result.primary_agent else result.agent_type.value,
        supporting_agents=[agent_type.value for agent_type in result.supporting_agents],
        tools_used=result.tools_used,
        routing_reason=result.routing_reason,
        routing_confidence=result.routing_confidence,
        escalated=result.escalated,
        latency_ms=round(result.latency_ms, 1),
        knowledge_used=(
            "search_knowledge_base" in result.tools_used
            or bool(result.rag_gate.get("prefetched"))
        ),
        entities=result.entities,
        intent_confidence=round(result.intent_confidence, 4),
        intent_source_scores=result.intent_source_scores,
        skills_applied=result.skills_applied,
        rag_gate=result.rag_gate,
    )
    # 站长后台按访客查看对话：记下这一轮问答（失败不影响回答）
    await chat_log.record(req.user_id, conv_id, req.message, response)
    return response


async def _check_quota() -> None:
    if _quota is not None and not await _quota.consume():
        raise HTTPException(429, "今天的体验名额已经用完了，明天再来看看吧。")


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, response: Response):
    """主对话接口（一次性返回）。"""
    if _orchestrator is None or _memory is None:
        raise HTTPException(503, "服务未就绪")
    await _check_quota()
    conv_id = req.conv_id or str(uuid.uuid4())
    response.headers["X-Conv-Id"] = conv_id    # 网关日志据此把对话和访客会话对上
    return await _run_chat(req, conv_id)


def _sse(event: str, data: Any) -> str:
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """
    流式对话接口。与 /chat 流程一致，区别是 Agent 生成最终回复时逐 token
    通过 Server-Sent Events 推给前端，而不是等全部生成完再一次性返回。

    事件类型：
      token — {"text": "..."}          增量文本片段
      done  — 完整 ChatResponse 字段     流结束时发一次，供前端更新侧栏统计
      error — {"message": "..."}        出错时发一次
    """
    if _orchestrator is None or _memory is None:
        raise HTTPException(503, "服务未就绪")
    await _check_quota()

    conv_id = req.conv_id or str(uuid.uuid4())

    async def event_gen():
        queue: "asyncio.Queue[Any]" = asyncio.Queue()
        DONE = object()

        async def on_delta(text: str) -> None:
            await queue.put(("token", text))

        async def produce() -> None:
            try:
                response = await _run_chat(req, conv_id, on_delta=on_delta)
                await queue.put(("done", response.model_dump()))
            except Exception as ex:
                logger.exception("流式对话处理失败")
                await queue.put(("error", {"message": str(ex)}))
            finally:
                await queue.put(("__end__", DONE))

        producer_task = asyncio.create_task(produce())
        try:
            while True:
                event, data = await queue.get()
                if event == "__end__":
                    break
                if event == "token":
                    yield _sse("token", {"text": data})
                else:
                    yield _sse(event, data)
        finally:
            if not producer_task.done():
                producer_task.cancel()

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
            "X-Conv-Id": conv_id,
        },
    )


@app.get("/monitor")
async def monitor_summary():
    """实时监控摘要：Agent 成功率、工具统计、告警、优化建议。"""
    if _monitor is None:
        raise HTTPException(503, "服务未就绪")
    return _monitor.summary()


@app.get("/trace/tool/{request_id}", response_model=ToolTraceResponse)
async def get_tool_trace(request_id: str):
    """查看某次请求的工具调用明细。"""
    if _orchestrator is None:
        raise HTTPException(503, "服务未就绪")
    trace = _orchestrator.get_tool_trace(request_id)
    return ToolTraceResponse(
        request_id=request_id,
        found=trace is not None,
        trace=trace or {},
    )


@app.get("/trace/tools", response_model=RecentToolTracesResponse)
async def list_recent_tool_traces(limit: int = 20):
    """查看最近 N 次请求的工具调用明细。"""
    if _orchestrator is None:
        raise HTTPException(503, "服务未就绪")
    return RecentToolTracesResponse(items=_orchestrator.get_recent_tool_traces(limit=limit))


@app.get("/metrics")
async def prometheus_metrics():
    """Prometheus 指标入口。"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/search")
async def search(query: str, top_k: int = 5, domain: Optional[str] = None):
    """
    演示检索优化链路：查询改写 → 并行召回 → 去重 → 相关性重排/过滤 → Top-K。
    展示 MCP 工具调用的核心亮点。domain 可选，用于验证按 Agent 领域隔离检索的效果。
    """
    if _tool_manager is None:
        raise HTTPException(503, "服务未就绪")
    result = await _tool_manager.search_with_rewrite("knowledge_search", query, top_k=top_k, domain=domain)
    return {"query": query, "results": result.data, "reranked": result.reranked, "success": result.success, "error": result.error}


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


@app.post("/knowledge/add", tags=["知识库"])
async def add_knowledge(body: BatchDocInput):
    """
    批量导入文档到知识库。

    文档会自动切片（每片 500 字）并存入 ChromaDB，ChromaDB 内置 Embedding 模型自动向量化。

    示例请求体：
    ```json
    {
      "documents": [
        {"title": "退款政策", "content": "用户在购买后 7 天内可以申请无理由退款..."},
        {"title": "配送说明", "content": "标准配送 3-5 个工作日..."}
      ]
    }
    ```
    """
    tool = _tool_manager._tools.get("knowledge_search") if _tool_manager else None
    if tool is None:
        raise HTTPException(503, "知识库未初始化")
    kb = tool.handler.__self__
    count = await kb.add_documents_async([
        {"title": d.title, "content": d.content, "domain": d.domain} for d in body.documents
    ])
    # 知识库变了，旧的检索缓存可能不含新文档，立即失效而不是等 TTL
    _tool_manager.invalidate_cache("knowledge_search")
    total = await kb.doc_count_async()
    return {"message": f"成功导入 {count} 个文档片段", "added_chunks": count, "total_chunks": total}


@app.post("/knowledge/upload", tags=["知识库"])
async def upload_knowledge(file: UploadFile = File(...), domain: Optional[str] = None):
    """
    上传文件导入知识库。

    支持格式：
    - `.txt` / `.md`：整个文件作为一篇文档，文件名作为标题
    - `.json`：JSON 数组格式 `[{"title": "...", "content": "...", "domain": "..."}, ...]`

    domain 参数（可选）：归属 Agent 领域（consulting/billing/general 等），
    没有在文档自己的字段里指定 domain 时会用这个值兜底；都不填则所有 Agent 可见。

    文件大小限制：10MB
    """
    tool = _tool_manager._tools.get("knowledge_search") if _tool_manager else None
    if tool is None:
        raise HTTPException(503, "知识库未初始化")
    kb = tool.handler.__self__

    content = await file.read()
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(413, "文件大小超过 10MB 限制")

    text = content.decode("utf-8", errors="ignore")
    filename = file.filename or "unknown"

    if filename.endswith(".json"):
        import json as _json
        try:
            docs = _json.loads(text)
            if not isinstance(docs, list):
                raise HTTPException(400, "JSON 文件应为数组格式: [{title, content}, ...]")
        except _json.JSONDecodeError as e:
            raise HTTPException(400, f"JSON 解析失败: {e}")
    else:
        # txt / md：整个文件作为一篇文档
        title = filename.rsplit(".", 1)[0] if "." in filename else filename
        docs = [{"title": title, "content": text}]

    if domain:
        for doc in docs:
            if isinstance(doc, dict):
                doc.setdefault("domain", domain)

    count = await kb.add_documents_async(docs)
    _tool_manager.invalidate_cache("knowledge_search")
    total = await kb.doc_count_async()
    return {
        "message": f"文件 {filename} 导入成功",
        "added_chunks": count,
        "total_chunks": total,
    }


@app.get("/knowledge/stats", tags=["知识库"])
async def knowledge_stats():
    """查看知识库统计信息（文档片段总数）。"""
    tool = _tool_manager._tools.get("knowledge_search") if _tool_manager else None
    if tool is None:
        raise HTTPException(503, "知识库未初始化")
    kb = tool.handler.__self__
    return await kb.stats_async()


@app.post("/eval/run")
async def run_eval(body: Optional[EvalRunInput] = None):
    """运行内置评测用例，返回评测报告。"""
    if _evaluator is None:
        raise HTTPException(503, "服务未就绪")
    from evaluation.evaluator import DEFAULT_DIALOG_CASES, DEFAULT_INTENT_CASES, IntentTestCase

    if body and body.intent_cases is not None:
        intent_cases = [
            IntentTestCase(
                message=c.message,
                expected_intent=c.expected_intent,
                context=c.context,
            )
            for c in body.intent_cases
        ]
    else:
        intent_cases = DEFAULT_INTENT_CASES

    if body and body.dialog_cases is not None:
        dialog_cases = [
            c.model_dump(exclude_none=True)
            for c in body.dialog_cases
        ]
    else:
        dialog_cases = DEFAULT_DIALOG_CASES

    report = await _evaluator.run(
        intent_cases=intent_cases,
        dialog_cases=dialog_cases,
        include_skill_evals=body.include_skill_evals if body else True,
        compare_rag_gate=body.compare_rag_gate if body else False,
    )
    return {
        "pass_rate":       report.pass_rate,
        "total":           report.total,
        "passed":          report.passed,
        "avg_scores":      report.avg_scores,
        "regressions":     report.regressions,
        "recommendations": report.recommendations,
        "results": [
            {
                "test_id": r.test_id,
                "passed": r.passed,
                "scores": r.scores,
                "detail": r.detail,
                "metadata": r.metadata,
            }
            for r in report.results
        ],
    }


# ── 交互式 CLI ────────────────────────────────────────────────────────────────
async def _cli():
    print(BANNER)
    print("GoEuroOps CLI — 输入 quit 退出\n")

    from agents.agent_orchestrator import AgentOrchestrator, Request
    from memory.conversation_memory import MemoryManager, MsgRole
    from core.skill_loader import SkillManager

    cfg = _anthropic_cfg()
    skill_manager = SkillManager(
        root_dir=os.getenv("GOEUROOPS_SKILLS_DIR", str(pathlib.Path(_ROOT) / "skills")),
        max_prompt_chars=int(os.getenv("GOEUROOPS_SKILLS_MAX_PROMPT_CHARS", "6000")),
    )
    skill_manager.load()
    orch = AgentOrchestrator(
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url"),
        model=cfg["model"],
        skill_manager=skill_manager,
    )
    mem  = MemoryManager(
        redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
        chroma_host=os.getenv("CHROMA_HOST", "localhost"),
        chroma_port=int(os.getenv("CHROMA_PORT", "8000")),
        chroma_path=os.getenv("CHROMA_PERSIST_DIRECTORY", "/tmp/chroma"),
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url"),
        model=cfg["model"],
    )

    user_id, conv_id = "cli_user", str(uuid.uuid4())

    while True:
        try:
            msg = input("你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见 ʕ•ᴥ•ʔ")
            break
        if not msg or msg.lower() in ("quit", "exit", "退出"):
            print("再见 ʕ•ᴥ•ʔ")
            break

        ctx = await mem.get_context(user_id, conv_id, query=msg)
        history = [
            {"role": m.role.value, "content": m.content}
            for m in ctx.recent_messages[-5:]
        ] if ctx.recent_messages else None
        req = Request(message=msg, user_id=user_id, conv_id=conv_id, context=ctx.to_prompt_text(), history=history)
        result = await orch.run(req)

        await mem.add_message(user_id, conv_id, MsgRole.USER, msg)
        await mem.add_message(user_id, conv_id, MsgRole.ASSISTANT, result.response)

        print(f"\nGoEuroOps [{result.agent_type.value}]: {result.response}\n")

    await mem.close()


if __name__ == "__main__":
    if "--cli" in sys.argv:
        asyncio.run(_cli())
    else:
        uvicorn.run(
            "api.main:app",
            host=os.getenv("API_HOST", "0.0.0.0"),
            port=int(os.getenv("API_PORT", "8000")),
            reload=os.getenv("APP_ENV") == "development",
        )
