"""多个 Agent 并行协作时，把各自的回答合并成一条。"""
import logging
from typing import Any, List, Optional

from anthropic import AsyncAnthropic

from core.llm_utils import NO_THINKING_KWARGS, extract_text_content
from core.config import env_float, env_int


from agents.base import AgentResponse, Request

logger = logging.getLogger(__name__)


class ResponseComposer:
    """多 Agent 汇总节点，统一主次、去重和输出边界。"""

    def __init__(self, client: AsyncAnthropic, model: str, skill_manager: Optional[Any] = None):
        self._client = client
        self._model = model
        self._skill_manager = skill_manager

    async def compose(self, req: Request, responses: List[AgentResponse]) -> str:
        successful = [response for response in responses if response.success and response.content.strip()]
        if not successful:
            return "抱歉，所有 Agent 均处理失败。"
        if len(successful) == 1:
            return successful[0].content

        evidence = "\n\n".join(
            f"[{response.agent_type.value} Agent 输出]\n{response.content}"
            for response in successful
        )
        prompt = (
            "你是留学咨询工作室「指北」的回复整合助手，负责把多个专业 Agent 的结果合并成一条最终回复。\n"
            "要求：以主 Agent 的结论为主，按用户问题优先级组织内容；去掉重复和冲突表述；"
            "不能补造价格、折扣、退款金额或进度；金额只保留工具算出的数字；如果结论冲突，明确说明需要顾问核实；"
            "保留必要的核验字段和转顾问边界。只输出给用户看的中文回复，不要提及 Agent。\n\n"
            f"主 Agent：{successful[0].agent_type.value}\n"
            f"用户问题：{req.message}\n"
            f"候选结果：\n{evidence}"
        )
        if self._skill_manager is not None and hasattr(self._skill_manager, "select"):
            # 只取常驻的品牌语气类 Skill，保证合并后的口吻和边界统一
            selection = self._skill_manager.select(req.message, "general")
            voice = [m.skill for m in selection.matches if m.skill.mode == "always"]
            if voice:
                prompt += "\n\n[品牌语气与输出边界]\n" + "\n\n".join(skill.to_prompt_block() for skill in voice)
        try:
            response = await self._client.messages.create(
                model=self._model,
                max_tokens=env_int("GOEUROOPS_COMPOSER_MAX_TOKENS", 1000),
                temperature=env_float("GOEUROOPS_COMPOSER_TEMPERATURE", 0.1),
                messages=[{"role": "user", "content": prompt}],
                **NO_THINKING_KWARGS,
            )
            content = extract_text_content(response.content).strip()
            if content:
                return content
        except Exception as ex:
            logger.warning("Response Composer 失败，使用确定性合并: %s", ex)

        # 汇总节点不可用时保留主次标签，避免丢失某个专业 Agent 的结论。
        return "\n\n".join(
            f"{response.content}" if index == 0 else f"补充说明：\n{response.content}"
            for index, response in enumerate(successful)
        )


