"""记忆的顾问接口：看客户档案、核实后改字段、按用户要求删除全部记忆。"""
import logging

from fastapi import APIRouter, HTTPException, Request

from api.demo_guard import is_guest
from api.schemas import ProfileFieldInput
from api.state import services

logger = logging.getLogger(__name__)
router = APIRouter()


def _memory(request: Request):
    if is_guest(request.headers):                   # 档案是访客的个人情况，演示访客不能看
        raise HTTPException(403, "演示访客不能查看客户档案")
    if services.memory is None:
        raise HTTPException(503, "记忆模块未初始化")
    return services.memory


@router.get("/memory/{user_id}/profile", tags=["记忆"])
async def get_profile(user_id: str, request: Request):
    """客户档案原样返回：每个字段的取值、出处原话、说话时间、来源（user / advisor）、以前的取值、待核实的说法。"""
    return {"user_id": user_id, "profile": await _memory(request).get_profile(user_id)}


@router.put("/memory/{user_id}/profile/{slot}", tags=["记忆"])
async def set_profile_field(user_id: str, slot: str, body: ProfileFieldInput, request: Request):
    """顾问核实后改一个字段，以后模型不会覆盖它。"""
    try:
        profile = await _memory(request).set_profile_field(user_id, slot, body.value)
    except ValueError as ex:
        raise HTTPException(400, str(ex))
    return {"user_id": user_id, "profile": profile}


@router.delete("/memory/{user_id}", tags=["记忆"])
async def forget_user(user_id: str, request: Request):
    """隐私删除：清掉这位用户的会话、档案、咨询记录、笔记和后台对话记录。线索和交接单是业务记录，由顾问在线索面板处理。"""
    deleted = await _memory(request).forget_user(user_id)
    # trace 和复核队列里也有原话片段，一起删
    if services.orchestrator is not None:
        deleted["traces"] = await services.orchestrator.trace_store.forget(user_id)
    if services.review_queue is not None:
        deleted["review_items"] = await services.review_queue.forget(user_id)
    return {"user_id": user_id, "deleted": deleted}
