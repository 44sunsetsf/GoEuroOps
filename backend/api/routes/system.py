"""会话身份、健康检查、价目、监控、工具轨迹和指标接口。"""
import logging


from fastapi import HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from api.demo_guard import (
    cost_limit, is_guest, mask_pii,
)

from fastapi import APIRouter
from api.schemas import RecentToolTracesResponse, ToolTraceResponse
from api.state import services

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/session")
async def session_info(request: Request):
    """当前访问者的身份：guest（演示只读）或 owner（站长）。前端据此显示提示、禁用写操作。"""
    guest = is_guest(request.headers)
    return {"role": "guest" if guest else "owner", "readonly": guest, "daily_cost_limit": cost_limit() or None}


@router.get("/health")
async def health():
    if services.orchestrator is None:
        raise HTTPException(503, "服务未就绪")
    return {"status": "ok", "agents": services.orchestrator.get_stats()}


@router.get("/catalog", tags=["业务"])
async def catalog_view():
    """服务价目、优惠、付款退款政策和五国资料（前端价目面板使用）。"""
    from business.catalog import get_catalog, get_countries

    return {"catalog": get_catalog().model_dump(), "countries": get_countries().model_dump()}


@router.get("/monitor")
async def monitor_summary():
    """实时监控摘要：Agent 成功率、工具统计、告警、优化建议。"""
    if services.monitor is None:
        raise HTTPException(503, "服务未就绪")
    return services.monitor.summary()


@router.get("/trace/tool/{request_id}", response_model=ToolTraceResponse)
async def get_tool_trace(request_id: str, request: Request):
    """查看某次请求的工具调用明细。"""
    if services.orchestrator is None:
        raise HTTPException(503, "服务未就绪")
    trace = services.orchestrator.get_tool_trace(request_id)
    if trace is not None and is_guest(request.headers):
        trace = mask_pii(trace)
    return ToolTraceResponse(
        request_id=request_id,
        found=trace is not None,
        trace=trace or {},
    )


@router.get("/trace/tools", response_model=RecentToolTracesResponse)
async def list_recent_tool_traces(request: Request, limit: int = 20):
    """查看最近 N 次请求的工具调用明细。"""
    if services.orchestrator is None:
        raise HTTPException(503, "服务未就绪")
    items = services.orchestrator.get_recent_tool_traces(limit=limit)
    if is_guest(request.headers):
        items = mask_pii(items)
    return RecentToolTracesResponse(items=items)


@router.get("/tools/stats")
async def tool_stats():
    """每个 Agent 工具的调用次数、各类结果（ok / invalid_args / timeout / error / circuit_open）、平均和最大耗时、熔断状态。"""
    from tooling.gateway import get_gateway
    return {"tools": get_gateway().stats()}


@router.get("/metrics")
async def prometheus_metrics():
    """Prometheus 指标入口。"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
