"""对话接口：/chat 一次性返回，/chat/stream 用 SSE 逐段推送。"""
import asyncio
import json
import logging
import uuid
from typing import Any


from fastapi import HTTPException, Response
from fastapi.responses import StreamingResponse

from api.demo_guard import (
    COST_LIMIT_MESSAGE, cost_exceeded,
)

from fastapi import APIRouter
from api.schemas import ChatRequest, ChatResponse
from api.state import services

logger = logging.getLogger(__name__)
router = APIRouter()


async def _run_chat(req: ChatRequest, conv_id: str, on_delta=None) -> ChatResponse:
    """
    一次完整对话：记忆读取 → 编排（意图识别 ∥ 投机预取 → 路由 → RAG 门控 →
    Skills 注入 → Agent 执行）→ 记忆写入。/chat 和 /chat/stream 共用。

    意图识别交给编排器内部完成，这样知识库预取才能和意图识别并行。
    """
    from agents.agent_orchestrator import Request as OrcReq
    from core import chat_log
    from memory.conversation_memory import MsgRole

    mem_ctx = await services.memory.get_context(req.user_id, conv_id, query=req.message)
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
    result = await services.orchestrator.run(orch_req, on_delta=on_delta)

    await services.memory.add_message(req.user_id, conv_id, MsgRole.USER, req.message)
    await services.memory.add_message(req.user_id, conv_id, MsgRole.ASSISTANT, result.response)
    # 异步更新用户画像（不阻塞响应）
    asyncio.create_task(services.memory.update_profile(req.user_id, conv_id))

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
    if await cost_exceeded():
        raise HTTPException(429, COST_LIMIT_MESSAGE)
    if services.quota is not None and not await services.quota.consume():
        raise HTTPException(429, "今天的体验名额已经用完了，明天再来看看吧。")


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, response: Response):
    """主对话接口（一次性返回）。"""
    if services.orchestrator is None or services.memory is None:
        raise HTTPException(503, "服务未就绪")
    await _check_quota()
    conv_id = req.conv_id or str(uuid.uuid4())
    response.headers["X-Conv-Id"] = conv_id    # 网关日志据此把对话和访客会话对上
    return await _run_chat(req, conv_id)


def _sse(event: str, data: Any) -> str:
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


@router.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """
    流式对话接口。与 /chat 流程一致，区别是 Agent 生成最终回复时逐 token
    通过 Server-Sent Events 推给前端，而不是等全部生成完再一次性返回。

    事件类型：
      token — {"text": "..."}          增量文本片段
      done  — 完整 ChatResponse 字段     流结束时发一次，供前端更新侧栏统计
      error — {"message": "..."}        出错时发一次
    """
    if services.orchestrator is None or services.memory is None:
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
