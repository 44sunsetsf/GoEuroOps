"""Agent 工具定义与实现。

所有 Agent 工具集中在这里，编排器只负责：
  1. 根据 Agent 类型暴露工具白名单
  2. 执行 LLM 返回的 tool_use
  3. 将工具结果回传给 LLM

工具本身保持确定性、可测试，并明确区分：
  - 当前请求分析
  - 留学咨询参考信息
  - 账单字段核验
  - 人工升级摘要
  - 共享知识库 RAG

订单查询、退款执行、账单修改等需要真实业务系统授权的动作不在这里伪造。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional, TYPE_CHECKING, Union

if TYPE_CHECKING:
    from agents.agent_orchestrator import Request


AgentToolHandler = Callable[["Request", Dict[str, Any]], Union[Any, Awaitable[Any]]]


@dataclass(frozen=True)
class AgentToolSpec:
    """Agent 可见工具的定义和执行函数。"""

    name: str
    description: str
    input_schema: Dict[str, Any]
    handler: AgentToolHandler


def make_tool(
    name: str,
    description: str,
    properties: Dict[str, Any],
    handler: AgentToolHandler,
    required: Optional[List[str]] = None,
) -> AgentToolSpec:
    """创建带 JSON Schema 的 Agent 工具。"""
    return AgentToolSpec(
        name=name,
        description=description,
        input_schema={
            "type": "object",
            "properties": properties,
            "required": required or [],
            "additionalProperties": False,
        },
        handler=handler,
    )


def inspect_request_context(req: Request, args: Dict[str, Any]) -> Dict[str, Any]:
    """通用客服工具：返回脱敏后的当前请求快照。"""
    return {
        "intent": req.intent.value if req.intent else None,
        "intent_group": req.intent_group,
        "urgency": req.urgency.name if req.urgency else None,
        "intent_confidence": round(req.intent_confidence, 4),
        "entities": req.entities or {},
        "context_available": bool(req.context),
        "requested_focus": str(args.get("focus", "general"))[:40],
    }


def suggest_required_fields(req: Request, args: Dict[str, Any]) -> Dict[str, Any]:
    """通用客服工具：按业务类型计算下一轮只需询问的字段。"""
    intent = req.intent.value if req.intent else "other"
    fields: List[str] = []
    if intent in {"order_status", "logistics"}:
        fields = ["订单号或下单时间"]
    elif intent in {"account", "account_security"}:
        fields = ["登录方式或账号标识", "问题发生时间"]
    elif intent in {"complaint", "request"}:
        fields = ["事件时间", "期望处理方式"]
    elif intent == "other":
        fields = ["希望解决的具体问题"]
    return {
        "intent": intent,
        "required_fields": fields,
        "known_entities": req.entities or {},
    }


COUNTRY_ADMISSIONS_OVERVIEW: Dict[str, Dict[str, str]] = {
    "瑞典": {
        "name_en": "Sweden",
        "typical_duration": "多数CS硕士为2年（120学分）",
        "english_test_note": "通常要求雅思6.5左右或同等托福成绩，具体以院校官网为准",
        "application_rounds": "主申请季集中在每年1月中旬截止（次年秋季入学）",
        "tuition_note": "欧盟/欧洲经济区学生通常免学费，非欧盟学生一般需缴纳学费",
    },
    "德国": {
        "name_en": "Germany",
        "typical_duration": "多数CS硕士为2年（含部分1.5年项目）",
        "english_test_note": "英语授课项目通常要求雅思6.5或同等托福成绩，部分项目需德语基础",
        "application_rounds": "冬季入学申请多在夏季前截止，具体因校而异",
        "tuition_note": "多数联邦州公立大学免学费或仅收少量注册费，具体以院校官网为准",
    },
    "荷兰": {
        "name_en": "Netherlands",
        "typical_duration": "多数CS硕士为1-2年，1年制项目较常见",
        "english_test_note": "通常要求雅思6.5左右或同等托福成绩，具体以院校官网为准",
        "application_rounds": "申请截止时间因校而异，部分热门项目截止较早",
        "tuition_note": "欧盟学生学费较低，非欧盟学生学费较高，具体以院校官网为准",
    },
    "芬兰": {
        "name_en": "Finland",
        "typical_duration": "多数CS硕士为2年",
        "english_test_note": "通常要求雅思6.5左右或同等托福成绩，具体以院校官网为准",
        "application_rounds": "主申请季通常在每年1月截止（次年秋季入学）",
        "tuition_note": "欧盟/欧洲经济区学生通常免学费，非欧盟学生一般需缴纳学费（部分设奖学金）",
    },
    "丹麦": {
        "name_en": "Denmark",
        "typical_duration": "多数CS硕士为2年",
        "english_test_note": "通常要求雅思6.5左右或同等托福成绩，具体以院校官网为准",
        "application_rounds": "非欧盟申请者截止较早（约每年1-3月），欧盟申请者相对较晚",
        "tuition_note": "欧盟/欧洲经济区学生通常免学费，非欧盟学生一般需缴纳学费",
    },
}


def lookup_country_admissions_overview(req: Request, args: Dict[str, Any]) -> Dict[str, Any]:
    """留学咨询工具：返回五国CS硕士申请的静态参考信息，不代表实时官方数据。"""
    country = str(args.get("country", "")).strip()
    info = COUNTRY_ADMISSIONS_OVERVIEW.get(country)
    return {
        "country": country or "未识别",
        "found": info is not None,
        "overview": info or {},
        "disclaimer": "以上为通用参考信息，具体录取要求、学费和截止日期请以院校官网及工作室顾问最新核实结果为准，不构成录取承诺。",
        "source": "static_reference_not_live_data",
    }


SERVICE_OFFERINGS: Dict[str, Dict[str, Any]] = {
    "school_selection": {
        "name": "选校与项目定位咨询",
        "includes": ["背景评估与目标国家/项目初步梳理", "候选院校与项目清单及匹配理由", "1对1顾问沟通答疑"],
        "excludes": ["不代为提交申请", "不保证录取结果"],
        "how_to_book": "请留下联系方式和基本背景（专业、GPA、语言成绩），工作室顾问会联系安排咨询时间。",
    },
    "essay_writing": {
        "name": "文书/个人陈述写作辅导",
        "includes": ["文书方向梳理与头脑风暴", "结构与内容打磨建议", "人工顾问多轮修改"],
        "excludes": ["机器人本身不代写、不代改文书内容", "不承诺特定院校录取结果"],
        "how_to_book": "请先完成选校沟通或说明目标项目，工作室顾问会安排文书辅导对接。",
    },
}


def lookup_service_offering(req: Request, args: Dict[str, Any]) -> Dict[str, Any]:
    """留学咨询工具：返回工作室服务介绍和预约方式，不代表实时价格或档期。"""
    service = str(args.get("service", "")).strip()
    info = SERVICE_OFFERINGS.get(service)
    return {
        "service": service or "未识别",
        "found": info is not None,
        "offering": info or {},
        "all_services": list(SERVICE_OFFERINGS.keys()),
        "disclaimer": "以上为服务介绍性参考信息，具体价格、档期和服务细节请以工作室顾问当面/沟通确认的信息为准。",
    }


def check_billing_fields(req: Request, args: Dict[str, Any]) -> Dict[str, Any]:
    """账单工具：检查必要核验字段是否齐全。"""
    fields = {
        "order_id": bool(req.entities.get("order_id")),
        "amount": bool(req.entities.get("amount")),
        "date": bool(req.entities.get("date")),
        "payment_channel": bool(args.get("payment_channel")),
    }
    return {
        "fields": fields,
        "missing_fields": [name for name, present in fields.items() if not present],
        "can_confirm_refund": False,
        "reason": "当前工具只做字段检查，不连接订单或支付系统",
    }


def compare_amounts(req: Request, args: Dict[str, Any]) -> Dict[str, Any]:
    """账单工具：只做用户明确提供金额之间的算术。"""
    try:
        first = float(args["amount_a"])
        second = float(args["amount_b"])
    except (KeyError, TypeError, ValueError):
        return {"success": False, "error": "amount_a 和 amount_b 必须是数字"}
    return {
        "success": True,
        "amount_a": first,
        "amount_b": second,
        "difference": round(first - second, 2),
        "interpretation": "仅表示金额差值，不代表重复扣款或退款结论",
    }


def create_handoff_summary(req: Request, args: Dict[str, Any]) -> Dict[str, Any]:
    """升级工具：生成可交给人工客服的结构化摘要。"""
    return {
        "request_id": req.request_id,
        "reason": str(args.get("reason", "需要人工客服继续核验"))[:120],
        "intent": req.intent.value if req.intent else "unknown",
        "urgency": req.urgency.name if req.urgency else "UNKNOWN",
        "entities": req.entities or {},
        "sensitive_data_required": False,
    }


def build_shared_rag_tools(tool_manager: Any, domain: Optional[str] = None) -> Dict[str, AgentToolSpec]:
    """构建 RAG 工具。

    domain 非空时把检索范围限定在该 Agent 自己的知识领域（加上标记为
    "shared" 的通用内容），避免不同业务线的知识库内容互相串场。
    """

    async def search_knowledge_base(req: Request, args: Dict[str, Any]) -> Dict[str, Any]:
        query = str(args.get("query") or req.message or "").strip()
        top_k = int(args.get("top_k", 5) or 5)
        if not query:
            return {"success": False, "error": "query 不能为空", "results": []}
        if tool_manager is None:
            return {"success": False, "error": "RAG 工具未初始化", "results": []}

        result = await tool_manager.search_with_rewrite(
            "knowledge_search",
            query,
            top_k=top_k,
            domain=domain,
        )
        if not getattr(result, "success", False):
            return {
                "success": False,
                "query": query,
                "error": getattr(result, "error", "知识库检索失败"),
                "results": [],
                "reranked": False,
            }

        return {
            "success": True,
            "query": query,
            "top_k": top_k,
            "results": result.data,
            "reranked": bool(getattr(result, "reranked", False)),
        }

    return {
        "search_knowledge_base": make_tool(
            "search_knowledge_base",
            "检索知识库并返回最相关的文档片段；可用于通用、留学咨询、账单和升级场景。",
            {
                "query": {"type": "string", "description": "用户问题或检索关键词"},
                "top_k": {"type": "integer", "description": "返回结果条数"},
            },
            search_knowledge_base,
            required=["query"],
        )
    }


def general_tools() -> Dict[str, AgentToolSpec]:
    return {
        "inspect_request_context": make_tool(
            "inspect_request_context",
            "查看当前请求的意图、紧急度、实体和上下文可用性；不查询外部业务系统。",
            {"focus": {"type": "string", "description": "希望关注的业务方向"}},
            inspect_request_context,
        ),
        "suggest_required_fields": make_tool(
            "suggest_required_fields",
            "根据当前意图建议下一轮只需向用户补充的字段。",
            {},
            suggest_required_fields,
        ),
    }


def consulting_tools() -> Dict[str, AgentToolSpec]:
    return {
        "lookup_country_admissions_overview": make_tool(
            "lookup_country_admissions_overview",
            "查询瑞典/德国/荷兰/芬兰/丹麦五国CS硕士的通用参考信息（学制、语言要求、申请季、学费性质）；为静态参考数据，不代表实时官方信息。",
            {"country": {"type": "string", "description": "国家中文名，如 瑞典、德国、荷兰、芬兰、丹麦"}},
            lookup_country_admissions_overview,
            required=["country"],
        ),
        "lookup_service_offering": make_tool(
            "lookup_service_offering",
            "查询工作室的选校咨询或文书辅导服务介绍与预约方式；为静态介绍信息，不代表当前价格或档期。",
            {"service": {"type": "string", "description": "school_selection 或 essay_writing"}},
            lookup_service_offering,
            required=["service"],
        ),
    }


def billing_tools() -> Dict[str, AgentToolSpec]:
    return {
        "check_billing_fields": make_tool(
            "check_billing_fields",
            "检查账单核验字段是否齐全；不连接订单、支付或退款系统。",
            {"payment_channel": {"type": "string", "description": "支付渠道，例如微信、支付宝、银行卡"}},
            check_billing_fields,
        ),
        "compare_amounts": make_tool(
            "compare_amounts",
            "计算用户明确提供的两笔金额差值；不判断是否重复扣款，也不执行退款。",
            {
                "amount_a": {"type": "number", "description": "第一笔金额"},
                "amount_b": {"type": "number", "description": "第二笔金额"},
            },
            compare_amounts,
            required=["amount_a", "amount_b"],
        ),
    }


def escalation_tools() -> Dict[str, AgentToolSpec]:
    return {
        "create_handoff_summary": make_tool(
            "create_handoff_summary",
            "生成交给人工客服的结构化交接摘要，不会创建真实工单。",
            {"reason": {"type": "string", "description": "需要升级的原因"}},
            create_handoff_summary,
        ),
    }
