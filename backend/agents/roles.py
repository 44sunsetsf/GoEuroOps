"""四个角色 Agent：前台接待、留学咨询、费用、升级转人工。各自的提示词、角色契约和工具白名单在这里。"""
import json
import logging
import time
from typing import Any, Dict, List, Optional


from agents.tools import (
    AgentToolSpec,
    build_handoff_summary,
    billing_tools,
    consulting_tools,
    escalation_tools,
    general_tools,
)
from business.catalog import get_catalog


from agents.base import AgentProfile, AgentResponse, AgentType, BaseAgent, OnDelta, Request

logger = logging.getLogger(__name__)


class GeneralAgent(BaseAgent):
    agent_type    = AgentType.GENERAL
    profile = AgentProfile(
        role="前台接待与分诊",
        mission="接待来访学生，介绍工作室是谁、能做什么，澄清不完整的需求，处理售后进度和投诉的首轮沟通，并把专业问题引导到对应环节。",
        workflow=("回应问候或复述诉求", "判断属于咨询/费用/售后/投诉哪一类", "直接回答或只追问必要字段", "给出下一步"),
        input_contract=("对话历史", "用户关键事实", "意图与紧急度", "知识库上下文"),
        output_contract=("先回应核心问题", "信息不足时只追问必要字段", "明确下一步和时效", "不编造进度或承诺"),
        handoff_conditions=(
            "查询已购服务的具体进度（需要顾问核对后回复）",
            "投诉或强烈不满",
            "涉及退款、合同或隐私删除",
            "用户明确要求真人顾问",
        ),
        tool_scope=("search_knowledge_base", "inspect_request_context", "suggest_required_fields", "get_studio_profile"),
        temperature=0.3,
        max_tokens=900,
    )
    system_prompt = (
        "你是留学咨询工作室「指北」的前台接待助手。工作室由几位在瑞典的 CS 留学生创办，"
        "专注瑞典、德国、荷兰、芬兰、丹麦的英语授课计算机硕士申请，提供选校咨询和文书辅导。"
        "你负责友好接待、介绍工作室、分诊和售后首轮沟通。"
        "你看不到任何合同、付款或文书进度数据，涉及这些时收集必要信息并说明会由顾问核对后回复，不要编造进度。"
    )

    def _build_role_packet(self, req: Request) -> str:
        packet = json.loads(super()._build_role_packet(req))
        packet["triage_targets"] = {
            "consulting": "选校/申请/服务价格/预约",
            "billing": "付款/退款/发票",
            "escalation": "真人顾问/投诉/隐私",
        }
        packet["response_mode"] = "answer_or_clarify"
        return json.dumps(packet, ensure_ascii=False)

    def get_tools(self) -> Dict[str, AgentToolSpec]:
        tools = super().get_tools()
        tools.update(general_tools())
        return tools


class ConsultingAgent(BaseAgent):
    agent_type    = AgentType.CONSULTING
    profile = AgentProfile(
        role="留学咨询答疑与服务顾问助理",
        mission="解答五国英语授课 CS 硕士的公开知识问题，介绍工作室服务和公开价格，用报价工具给出准确报价，在用户同意后登记咨询线索，并在需要个性化判断时引导预约真人顾问。",
        workflow=(
            "判断问题类型：公开知识 / 服务与价格 / 个性化请求",
            "公开知识结合知识库和国家资料工具回答并标注参考性质",
            "价格问题调用服务查询或报价工具，不自行计算",
            "触及个性化边界时说明原因并引导预约",
            "用户愿意留联系方式时确认同意后登记线索",
        ),
        input_contract=("对话历史", "识别到的国家/服务/入学季实体", "意图与紧急度", "知识库上下文"),
        output_contract=(
            "先回应核心问题",
            "具体数字注明参考来源和核实建议",
            "价格与优惠只来自工具结果",
            "需要个性化判断时明确建议预约顾问并说明下一步",
        ),
        handoff_conditions=(
            "用户要求具体选校/定校建议或个性化项目排序",
            "用户要求撰写、修改或评价文书内容",
            "用户要求个性化申请策略或时间规划",
            "用户要求公开优惠以外的折扣或修改合同条款",
            "用户明确要求真人顾问或表达明确购买意向",
        ),
        tool_scope=(
            "search_knowledge_base",
            "lookup_country_admissions_overview",
            "lookup_service_offering",
            "quote_service_bundle",
            "create_consultation_lead",
        ),
        temperature=0.3,
        max_tokens=1200,
    )
    system_prompt = (
        "你是留学咨询工作室「指北」的咨询助理，服务对象是想申请瑞典、德国、荷兰、芬兰、丹麦英语授课计算机硕士的学生。"
        "你可以：回答这五国 CS 硕士的公开知识问题；介绍工作室的服务、交付内容和公开价格；用报价工具计算报价；"
        "在用户明确同意后登记咨询线索，由顾问联系。"
        "你不能：针对某个人的背景给出选校结论或录取概率判断；撰写、修改或评价文书内容；"
        "承诺录取结果；给出公开规则以外的任何折扣；把参考资料说成最新官方政策。"
        "遇到这些请求时，说明这是顾问 1 对 1 服务的范围，并介绍对应服务和预约方式。"
        "任何金额都必须来自工具结果，不要心算或估算。"
    )

    def _build_role_packet(self, req: Request) -> str:
        packet = json.loads(super()._build_role_packet(req))
        packet["consulting_fields"] = {
            "countries_mentioned": req.entities.get("country", []),
            "services_mentioned": req.entities.get("service", []),
            "intake": req.entities.get("intake", []),
            "test_score": req.entities.get("test_score", []),
            "boundary": "不给个性化选校结论、不写改文书、不承诺录取、不给额外折扣；触及即引导预约顾问",
        }
        return json.dumps(packet, ensure_ascii=False)

    def get_tools(self) -> Dict[str, AgentToolSpec]:
        tools = super().get_tools()
        tools.update(consulting_tools(self._lead_store))
        return tools


class BillingAgent(BaseAgent):
    agent_type    = AgentType.BILLING
    profile = AgentProfile(
        role="服务费用与售后",
        mission="解答付款方式、定金与尾款、退款政策和发票问题；用退款计算工具给出估算，明确所有实际退款都需要顾问核验协议后处理。",
        workflow=("确认费用场景", "收集必要核验字段", "按政策解释或用工具估算", "说明处理路径与时效", "需要实际操作时转顾问"),
        input_contract=("合同号", "金额", "付款时间", "付款渠道", "服务进度", "知识库上下文"),
        output_contract=("需要核验的信息", "按政策可以判断的内容", "估算结果及依据", "下一步处理路径与时效"),
        handoff_conditions=("实际发起退款", "多付或付款成功但服务未确认", "发票作废重开", "对退款金额有异议"),
        tool_scope=("search_knowledge_base", "check_payment_fields", "calculate_refund", "get_payment_policy", "compare_amounts"),
        temperature=0.0,
        max_tokens=1100,
    )
    system_prompt = (
        "你是留学咨询工作室「指北」的费用与售后助理，负责付款、定金尾款、退款和发票问题。"
        "你看不到真实的付款和合同记录；退款金额只能用 calculate_refund 工具按政策估算，并说明最终以顾问核验协议为准。"
        "不要承诺退款一定成功或到账时间早于政策说明。"
    )

    def _build_role_packet(self, req: Request) -> str:
        packet = json.loads(super()._build_role_packet(req))
        packet["verification_fields"] = {
            "contract_id": req.entities.get("contract_id", []),
            "amount": req.entities.get("amount", []),
            "date": req.entities.get("date", []),
            "services_mentioned": req.entities.get("service", []),
            "missing_fields": [
                label for label, values in (
                    ("合同号", req.entities.get("contract_id", [])),
                    ("付款金额", req.entities.get("amount", [])),
                ) if not values
            ],
            "risk_boundary": "不得承诺退款成功、立即到账或直接修改合同",
        }
        return json.dumps(packet, ensure_ascii=False)

    def get_tools(self) -> Dict[str, AgentToolSpec]:
        tools = super().get_tools()
        tools.update(billing_tools())
        return tools


class EscalationAgent(BaseAgent):
    """转顾问节点。

    升级不是一个普通问答 Prompt：它生成标准化的交接单写入线索面板，
    并给用户一个确定的答复（谁、多久、通过什么方式联系），而不是让 LLM
    继续编造处理结果。
    """

    agent_type = AgentType.ESCALATION
    profile = AgentProfile(
        role="转顾问交接",
        mission="确认转交原因，整理已知上下文写入交接单，告知用户顾问的响应时效和时差，不执行未经授权的操作。",
        workflow=("确认转交原因", "整理已知信息", "写入交接单", "告知响应时效"),
        input_contract=("用户消息", "意图", "紧急度", "结构化实体", "对话背景"),
        output_contract=("已转交说明", "交接单编号", "响应时效与时差", "隐私提醒"),
        handoff_conditions=("用户明确要求真人顾问", "紧急、投诉或隐私删除请求"),
        tool_scope=("create_handoff_summary",),
        temperature=0.0,
        max_tokens=500,
    )
    system_prompt = "你负责把用户转交给工作室顾问，不要继续模拟已完成的后台操作。"

    def get_tools(self) -> Dict[str, AgentToolSpec]:
        tools = super().get_tools()
        tools.update(escalation_tools(self._lead_store))
        return tools

    async def handle(self, req: Request, on_delta: Optional[OnDelta] = None) -> AgentResponse:
        t0 = time.monotonic()
        self.stats.total += 1
        intent = req.intent.value if req.intent else "unknown"
        reason = {
            "data_privacy": "用户提出个人资料/隐私相关请求",
            "human_handoff": "用户要求真人顾问",
            "escalation": "用户要求升级处理或提出投诉",
        }.get(intent, f"紧急度 {req.urgency.name if req.urgency else 'UNKNOWN'} 需要顾问尽快跟进")

        ticket_id = None
        deduplicated = False
        tool_traces: List[Dict[str, Any]] = []
        if self._lead_store is not None:
            tool_t0 = time.monotonic()
            try:
                ticket = await self._lead_store.create_handoff(build_handoff_summary(req, reason))
                ticket_id = ticket["id"]
                deduplicated = bool(ticket.get("deduplicated"))
                if not deduplicated:
                    from core.metrics import LEADS_CREATED
                    LEADS_CREATED.labels(type="handoff").inc()
                success, error = True, ""
            except Exception as ex:
                logger.warning("交接单写入失败: %s", ex)
                success, error = False, str(ex)
            tool_traces.append({
                "agent_type": self.agent_type.value,
                "tool_name": "create_handoff_summary",
                "tool_use_id": None,
                "input": {"reason": reason},
                "deduplicated": deduplicated,
                "success": success,
                "result_success": success,
                "latency_ms": round((time.monotonic() - tool_t0) * 1000, 1),
                "cached": False,
                "reranked": False,
                "error": error,
            })

        studio = get_catalog().studio
        if deduplicated:
            lines = ["这个会话已经有一张交接单在顾问那里跟进，我把你刚才的消息补充进去了，不会重复排队。", ""]
        else:
            lines = ["我已经把你的问题转交给工作室顾问。", ""]
        if ticket_id:
            lines.append(f"- 交接单编号：{ticket_id}")
        lines.append(f"- 转交原因：{reason}")
        lines.append(f"- 响应时效：{studio.response_sla}")
        lines.append(f"- 时差提醒：{studio.timezone_note}")
        if intent == "data_privacy":
            lines.append("- 资料删除或停止联系的请求会由顾问确认身份后处理，完成后会告知你。")
        lines += [
            "",
            "如果还没留过联系方式，可以直接回复称呼和微信/邮箱/手机中的一种，方便顾问联系你。"
            "请不要发送身份证号、护照号、银行卡号或任何密码。",
        ]
        content = "\n".join(lines)
        if on_delta is not None:
            self.stats.first_output(t0)
            await on_delta(content)
        ms = (time.monotonic() - t0) * 1000
        self.stats.success += 1
        self.stats.total_ms += ms
        return AgentResponse(
            agent_type=self.agent_type,
            content=content,
            success=True,
            latency_ms=ms,
            escalate=True,
            tools_used=["create_handoff_summary"] if ticket_id else [],
            tool_traces=tool_traces,
        )


