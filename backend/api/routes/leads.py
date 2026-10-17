"""线索与转顾问交接单接口。"""
import logging
from typing import Optional

from fastapi import HTTPException, Request

from api.demo_guard import (
    is_guest, mask_pii,
)

from fastapi import APIRouter
from api.schemas import LeadUpdateInput
from api.state import services

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/leads", tags=["线索"])
async def list_leads(request: Request, status: Optional[str] = None, type: Optional[str] = None, limit: int = 100):
    """咨询线索与转顾问交接单列表（按时间倒序）。演示访客看到的联系方式是打码的。"""
    if services.lead_store is None:
        raise HTTPException(503, "线索库未初始化")
    items = await services.lead_store.list(status=status, lead_type=type, limit=limit)
    if is_guest(request.headers):
        items = [mask_pii(item) for item in items]
    return {"items": items, "stats": await services.lead_store.stats()}


@router.patch("/leads/{lead_id}", tags=["线索"])
async def update_lead(lead_id: str, body: LeadUpdateInput):
    """更新线索状态（new / contacted / converted / closed）和跟进备注。"""
    if services.lead_store is None:
        raise HTTPException(503, "线索库未初始化")
    try:
        lead = await services.lead_store.update(lead_id, status=body.status, notes=body.notes)
    except ValueError as ex:
        raise HTTPException(400, str(ex))
    if lead is None:
        raise HTTPException(404, f"线索不存在: {lead_id}")
    return lead
