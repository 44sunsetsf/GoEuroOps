"""bad case 闭环：用户点踩、转顾问、金额护栏触发的对话进复核队列，顾问确认后导出成回归用例。"""
import logging

from fastapi import APIRouter, HTTPException, Request

from api.demo_guard import is_guest, mask_pii
from api.schemas import FeedbackInput, ReviewInput
from api.state import services

logger = logging.getLogger(__name__)
router = APIRouter()


def _queue():
    if services.review_queue is None or services.orchestrator is None:
        raise HTTPException(503, "服务未就绪")
    return services.review_queue


@router.post("/feedback", tags=["复核"])
async def feedback(body: FeedbackInput):
    """用户对一轮回答点赞或点踩；点踩（可附一句备注）进复核队列。"""
    queue = _queue()
    trace = await services.orchestrator.get_tool_trace(body.request_id)
    if trace is None:
        raise HTTPException(404, "找不到这一轮对话，可能已过期")
    if body.rating == "up":
        return {"ok": True, "queued": False}
    item = await queue.add(trace, ["thumbs_down"], comment=body.comment)
    return {"ok": True, "queued": item is not None}


@router.get("/review", tags=["复核"])
async def list_review(request: Request, status: str = "pending", limit: int = 50):
    """复核列表（默认只看待处理的）；访客看到的联系方式打码。"""
    items = await _queue().list(status=status or None, limit=limit)
    return {"items": mask_pii(items) if is_guest(request.headers) else items}


@router.post("/review/{request_id}", tags=["复核"])
async def review_item(request_id: str, body: ReviewInput):
    """顾问确认或驳回一条复核。"""
    try:
        item = await _queue().review(request_id, body.status, body.expected)
    except ValueError as ex:
        raise HTTPException(400, str(ex))
    if item is None:
        raise HTTPException(404, "复核队列里没有这一条")
    return item


@router.get("/review/export", tags=["复核"])
async def export_review(request: Request):
    """已确认的 bad case 导出为回归用例 JSON（保存到 evaluation/datasets/ 即可加进回归）。"""
    cases = await _queue().export_cases()
    return {"cases": mask_pii(cases) if is_guest(request.headers) else cases}
