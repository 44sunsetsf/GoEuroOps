"""端到端评测接口。"""
import logging
from typing import Any, Dict, Optional


from fastapi import HTTPException, Request

from api.demo_guard import (
    COST_LIMIT_MESSAGE, cost_exceeded, is_guest,
)

from fastapi import APIRouter
from api.schemas import EvalRunInput
from api.state import services

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/eval/run")
async def run_eval(request: Request, body: Optional[EvalRunInput] = None):
    """运行内置评测用例，返回评测报告。

    演示访客也能运行，但只跑内置用例、不做 RAG 对照、不覆盖站长的基线，每天有次数上限，同一时间只跑一个。
    """
    if services.evaluator is None:
        raise HTTPException(503, "服务未就绪")
    if is_guest(request.headers):
        if await cost_exceeded():
            raise HTTPException(429, COST_LIMIT_MESSAGE)
        if services.guest_eval_lock.locked():
            raise HTTPException(429, "已经有一个评测在运行，请一两分钟后再试。")
        if services.guest_eval_quota is not None and not await services.guest_eval_quota.consume():
            raise HTTPException(429, "今天演示模式的评测次数已经用完了，明天再来看看吧。")
        async with services.guest_eval_lock:
            return await _run_eval(None, save_baseline=False)
    return await _run_eval(body)


async def _run_eval(body: Optional[EvalRunInput], save_baseline: bool = True) -> Dict[str, Any]:
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

    report = await services.evaluator.run(
        intent_cases=intent_cases,
        dialog_cases=dialog_cases,
        include_skill_evals=body.include_skill_evals if body else True,
        compare_rag_gate=body.compare_rag_gate if body else False,
        save_baseline=save_baseline,
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
