"""
GoEuroOps 留学业务智能运营中枢 — FastAPI 入口

启动时打印小熊饼干图案。
所有核心组件在 lifespan 中初始化，通过环境变量配置。
"""
import asyncio
import logging
import os
import pathlib
import sys
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional


_ROOT = str(pathlib.Path(__file__).parent.parent.resolve())
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.demo_guard import GUEST_READONLY_MESSAGE, guest_write_allowed, is_guest
from api.state import services
from core.config import DEFAULT_MODEL

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


def _anthropic_cfg() -> Dict[str, Any]:
    key = os.getenv("ANTHROPIC_API_KEY", "")
    if not key:
        raise RuntimeError("未设置 ANTHROPIC_API_KEY")
    cfg: Dict[str, Any] = {
        "api_key":  key,
        "model":    os.getenv("ANTHROPIC_MODEL", DEFAULT_MODEL).strip(),
    }
    base_url = os.getenv("ANTHROPIC_BASE_URL", "").strip()
    if base_url:
        cfg["base_url"] = base_url
    return cfg


@asynccontextmanager
async def lifespan(app: FastAPI):

    print(BANNER, flush=True)

    from agents.agent_orchestrator import AgentOrchestrator
    from api.quota import DailyQuota
    from core.intent_recognizer import IntentRecognizer
    from evaluation.evaluator import EndToEndEvaluator
    from retrieval.knowledge_base import KnowledgeBase
    from retrieval.manager import RetrievalManager, Tool
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
    services.skill_manager = SkillManager(
        root_dir=skills_dir,
        max_prompt_chars=int(os.getenv("GOEUROOPS_SKILLS_MAX_PROMPT_CHARS", "6000")),
    )
    services.skill_manager.load()

    # 公开演示的每日对话上限（GOEUROOPS_DAILY_CHAT_LIMIT，0 = 不限）
    services.quota = DailyQuota(
        limit=int(os.getenv("GOEUROOPS_DAILY_CHAT_LIMIT", "0") or 0),
        redis_url=os.getenv("REDIS_URL", "redis://redis:6379/0"),
    )

    # 演示访客每天可运行内置评测的次数（GOEUROOPS_GUEST_EVAL_LIMIT，默认 5）
    services.guest_eval_quota = DailyQuota(
        limit=int(os.getenv("GOEUROOPS_GUEST_EVAL_LIMIT", "5") or 0),
        redis_url=os.getenv("REDIS_URL", "redis://redis:6379/0"),
        key_prefix="goeuroops:quota:guest_eval:",
    )

    # 线索库：咨询线索和转顾问交接单，前端「线索」面板读取
    services.lead_store = LeadStore(redis_url=os.getenv("REDIS_URL", "redis://redis:6379/0"))

    # Agent 编排器
    services.orchestrator = AgentOrchestrator(
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url"),
        model=cfg["model"],
        skill_manager=services.skill_manager,
        lead_store=services.lead_store,
    )

    # 记忆管理器（Redis 工作记忆 + ChromaDB 情景记忆/用户画像）
    services.memory = MemoryManager(
        redis_url=os.getenv("REDIS_URL", "redis://redis:6379/0"),
        chroma_host=os.getenv("CHROMA_HOST", "chromadb"),
        chroma_port=int(os.getenv("CHROMA_PORT", "8000")),
        chroma_path=os.getenv("CHROMA_PERSIST_DIRECTORY", "/app/data/chroma"),
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url"),
        model=cfg["model"],
    )

    # 检索工具管理器 + RAG 知识库（基于 ChromaDB 的真实检索）
    services.tool_manager = RetrievalManager(
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

    services.tool_manager.register(Tool(
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
    if services.orchestrator is not None:
        services.orchestrator.set_shared_tools(services.tool_manager)

    # 性能监控（可选启动 Prometheus）
    prom_port = int(os.getenv("PROMETHEUS_PORT", "0")) or None
    services.monitor = PerformanceMonitor(
        orchestrator=services.orchestrator,
        tool_manager=services.tool_manager,
        interval_s=float(os.getenv("MONITOR_INTERVAL", "10")),
        webhook_url=os.getenv("ALERT_WEBHOOK_URL") or None,
        prometheus_port=prom_port,
    )
    await services.monitor.start()

    # 评测器
    services.evaluator = EndToEndEvaluator(
        orchestrator=services.orchestrator,
        recognizer=recognizer,
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url"),
        model=cfg["model"],
        baseline_path=os.getenv("EVAL_BASELINE_PATH", "/app/data/eval/baseline.json"),
        skill_manager=services.skill_manager,
    )

    logger.info("GoEuroOps 已就绪")
    yield

    await services.monitor.stop()
    if services.memory is not None:
        await services.memory.close()
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


@app.middleware("http")
async def _guest_guard(request: Request, call_next):
    """演示访客只读：白名单以外的写操作直接拒绝（规则见 api/demo_guard.py）。"""
    if is_guest(request.headers) and not guest_write_allowed(request.method, request.url.path):
        return JSONResponse({"detail": GUEST_READONLY_MESSAGE}, status_code=403)
    return await call_next(request)


# ── 路由（各模块在 api/routes/ 下）──────────────────────────────────────────────
from api.routes import chat, evals, knowledge, leads, skills, system  # noqa: E402

for _module in (system, skills, leads, chat, knowledge, evals):
    app.include_router(_module.router)


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
