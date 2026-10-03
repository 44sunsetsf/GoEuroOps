"""Skills 的查看、热加载、命中测试和回归接口。"""
import logging


from fastapi import HTTPException, Request

from api.demo_guard import (
    is_guest,
)

from fastapi import APIRouter
from api.schemas import SkillMatchInput
from api.routes.chat import _check_quota
from api.state import services

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/skills", tags=["Skills"])
async def skills_summary():
    """查看当前已加载的 Skills，便于确认热加载结果和排查解析错误。"""
    if services.skill_manager is None:
        raise HTTPException(503, "Skills 未初始化")
    return services.skill_manager.summary()


@router.post("/skills/reload", tags=["Skills"])
async def reload_skills():
    """运行时重新扫描 Skill 目录，不需要重启服务。"""
    if services.skill_manager is None:
        raise HTTPException(503, "Skills 未初始化")
    services.skill_manager.reload()
    if services.orchestrator is not None:
        services.orchestrator.set_skill_manager(services.skill_manager)
    return services.skill_manager.summary()


@router.post("/skills/match", tags=["Skills"])
async def match_skills(body: SkillMatchInput, request: Request):
    """
    命中测试（干跑）：给一句话，返回每个 Skill 的得分明细和最终会注入哪些。

    不传 intent 时会现场做一次意图识别，与真实对话链路保持一致。
    """
    if services.skill_manager is None:
        raise HTTPException(503, "Skills 未初始化")
    intent, intent_group, intent_confidence = body.intent, None, None
    if intent is None and services.orchestrator is not None:
        if is_guest(request.headers):
            await _check_quota()   # 现场意图识别要调用模型，访客的调用计入每日额度
        result = await services.orchestrator.recognize_intent(body.message, history=body.history)
        intent, intent_group, intent_confidence = result.intent.value, result.intent_group, result.confidence
    report = services.skill_manager.match_report(
        body.message,
        body.agent_type,
        intent=intent,
        intent_group=intent_group,
        history=body.history,
    )
    report["intent_confidence"] = intent_confidence
    return report


@router.get("/skills/evals", tags=["Skills"])
async def run_skill_evals():
    """跑所有 Skill 自带的命中回归用例（skills/*/evals/cases.json）。"""
    if services.skill_manager is None:
        raise HTTPException(503, "Skills 未初始化")
    return services.skill_manager.run_evals()


@router.get("/skills/{skill_id}", tags=["Skills"])
async def skill_detail(skill_id: str):
    """查看单个 Skill 的完整正文、参考资料目录和命中统计。"""
    if services.skill_manager is None:
        raise HTTPException(503, "Skills 未初始化")
    detail = services.skill_manager.detail(skill_id)
    if detail is None:
        raise HTTPException(404, f"Skill 不存在: {skill_id}")
    return detail
