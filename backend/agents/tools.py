"""Agent 工具定义与实现。

所有 Agent 工具集中在这里，编排器只负责：
  1. 根据 Agent 类型暴露工具白名单
  2. 执行 LLM 返回的 tool_use
  3. 将工具结果回传给 LLM

工具保持确定性、可测试：价格、优惠、退款金额全部由 business/ 下的代码按
catalog.yaml 计算，LLM 只负责解释结果，不自己做算术，也不自己编折扣。
真正的收款、退款执行、合同修改需要人工操作，这里不伪造。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional, TYPE_CHECKING, Union

from business.catalog import CATEGORY_LABELS, get_catalog, get_countries
from business.lead_store import CHANNEL_LABELS, STAGES, LeadStore, mask_contact, validate_lead_input
from business.pricing import REFUND_STAGES, calculate_refund, quote_bundle

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


# ── 通用接待 ──────────────────────────────────────────────────────────────────

def inspect_request_context(req: Request, args: Dict[str, Any]) -> Dict[str, Any]:
    """通用工具：返回脱敏后的当前请求快照。"""
    return {
        "intent": req.intent.value if req.intent else None,
        "intent_group": req.intent_group,
        "urgency": req.urgency.name if req.urgency else None,
        "intent_confidence": round(req.intent_confidence, 4),
        "entities": req.entities or {},
        "context_available": bool(req.context),
        "requested_focus": str(args.get("focus", "general"))[:40],
    }


_REQUIRED_FIELDS: Dict[str, List[str]] = {
    "service_progress": ["合同号或签约时使用的称呼/联系方式", "想了解进度的服务（如 PS 第几轮、选校报告）"],
    "booking": ["想预约的服务", "方便的时间段（注明北京时间）", "一种联系方式"],
    "refund": ["合同号", "已付金额", "服务目前进度"],
    "payment_issue": ["合同号", "付款金额与时间", "付款渠道"],
    "invoice": ["合同号", "发票抬头（个人/企业，企业需税号）", "接收邮箱"],
    "account": ["需要更新的联系方式"],
    "complaint": ["涉及的服务和时间", "希望的处理方式"],
    "service_inquiry": ["感兴趣的服务或目标国家", "计划入学时间"],
    "other": ["希望解决的具体问题"],
}


def suggest_required_fields(req: Request, args: Dict[str, Any]) -> Dict[str, Any]:
    """通用工具：按业务类型给出下一轮只需询问的字段。"""
    intent = req.intent.value if req.intent else "other"
    return {
        "intent": intent,
        "required_fields": _REQUIRED_FIELDS.get(intent, []),
        "known_entities": req.entities or {},
        "privacy_note": "不要收集身份证号、护照号、银行卡号、密码或验证码。",
    }


def get_studio_profile(req: Request, args: Dict[str, Any]) -> Dict[str, Any]:
    """通用工具：工作室基本信息（品牌、所在地、时区、响应时效、覆盖范围）。"""
    catalog = get_catalog()
    studio = catalog.studio
    return {
        "name": studio.name,
        "tagline": studio.tagline,
        "founders_note": studio.founders_note,
        "base": studio.base,
        "timezone_note": studio.timezone_note,
        "response_sla": studio.response_sla,
        "capacity_note": studio.capacity_note,
        "coverage": studio.coverage,
        "not_covered": studio.not_covered,
        "booking_steps": catalog.booking.steps,
    }


# ── 留学咨询 / 服务销售 ───────────────────────────────────────────────────────

def lookup_country_admissions_overview(req: Request, args: Dict[str, Any]) -> Dict[str, Any]:
    """留学咨询工具：返回五国 CS 硕士申请的参考资料（非实时官方数据）。"""
    book = get_countries()
    query = str(args.get("country", "")).strip()
    country = book.find(query)
    if country is None:
        return {
            "success": False,
            "country": query or "未识别",
            "error": "不在工作室覆盖范围内或未识别的国家",
            "supported": [c.key for c in book.countries],
        }
    return {
        "success": True,
        "country": country.key,
        "overview": country.model_dump(exclude={"aliases", "key"}),
        "target_intake": book.target_intake,
        "last_verified": book.last_verified,
        "disclaimer": "以上为工作室整理的参考资料，录取要求、学费和截止日期每年可能调整，请以 source_urls 中的官网为准，不构成录取承诺。",
    }


def _service_payload(service) -> Dict[str, Any]:
    catalog = get_catalog()
    data = service.model_dump(exclude_none=True)
    data["price_display"] = "免费" if service.price == 0 else f"{catalog.money(service.price)} / {service.unit}"
    data["category_label"] = service.category_label
    return data


def lookup_service_offering(req: Request, args: Dict[str, Any]) -> Dict[str, Any]:
    """服务工具：按 SKU 或类别返回服务详情与价格；不传参数返回完整价目概览。"""
    catalog = get_catalog()
    sku = str(args.get("sku", "") or "").strip()
    category = str(args.get("category", "") or "").strip()
    if sku:
        service = catalog.service(sku)
        if service is None:
            return {"success": False, "error": f"未知 SKU: {sku}", "valid_skus": [s.sku for s in catalog.services]}
        services = [service]
    elif category:
        services = [s for s in catalog.services if s.category == category]
        if not services:
            return {"success": False, "error": f"未知类别: {category}", "categories": CATEGORY_LABELS}
    else:
        return {
            "success": True,
            "overview": [
                {"sku": s.sku, "name": s.name, "category": s.category_label,
                 "price_display": "免费" if s.price == 0 else f"{catalog.money(s.price)} / {s.unit}",
                 "summary": s.summary}
                for s in catalog.services
            ],
            "discount_rules": [catalog.discounts.early_bird.rule, catalog.discounts.group.rule,
                               catalog.discounts.referral.rule, catalog.discounts.stacking_rule],
            "payment_rule": catalog.payment.rule,
            "hint": "需要某项服务的完整内容时，再用 sku 参数查询。",
        }
    return {
        "success": True,
        "services": [_service_payload(s) for s in services],
        "payment_rule": catalog.payment.rule,
        "no_negotiation": catalog.discounts.no_negotiation,
        "disclaimer": "价格为工作室公开标价，最终以电子服务协议为准；档期以顾问确认为准。",
    }


def quote_service_bundle(req: Request, args: Dict[str, Any]) -> Dict[str, Any]:
    """报价工具：按价目表和公开优惠规则计算报价明细。"""
    items = args.get("items") or []
    if not isinstance(items, list):
        return {"success": False, "error": "items 必须是数组"}
    return quote_bundle(
        items,
        early_bird=bool(args.get("early_bird", False)),
        group_size=int(args.get("group_size", 1) or 1),
        referral=bool(args.get("referral", False)),
    )


def build_lead_tool(lead_store: Optional[LeadStore]) -> AgentToolSpec:
    async def create_consultation_lead(req: Request, args: Dict[str, Any]) -> Dict[str, Any]:
        errors = validate_lead_input(args)
        if errors:
            return {"success": False, "errors": errors,
                    "hint": "请向用户补充或更正以上信息后再登记；未获得明确同意时不要登记。"}
        if lead_store is None:
            return {"success": False, "error": "线索库未初始化，请引导用户稍后再试或直接联系顾问"}
        data = {k: v for k, v in args.items() if k != "consent"}
        data.update({"consent": True, "source": "chat", "user_id": req.user_id, "conv_id": req.conv_id,
                     "request_id": req.request_id})
        lead = await lead_store.create(data, lead_type="lead")
        _count_lead("lead")
        catalog = get_catalog()
        return {
            "success": True,
            "lead_id": lead["id"],
            "contact_masked": f"{CHANNEL_LABELS.get(args['contact_channel'], '')} {mask_contact(args['contact'])}",
            "next_step": f"顾问会在{catalog.studio.response_sla.split('；')[0]}通过该联系方式联系用户。",
            "timezone_note": catalog.studio.timezone_note,
        }

    return make_tool(
        "create_consultation_lead",
        "在用户明确同意后登记咨询线索，交给工作室顾问跟进。只收集必要信息，禁止收集证件号、成绩单原件、密码等。",
        {
            "name": {"type": "string", "description": "用户希望的称呼"},
            "contact_channel": {"type": "string", "enum": list(CHANNEL_LABELS), "description": "联系渠道：wechat / email / phone"},
            "contact": {"type": "string", "description": "对应渠道的联系方式"},
            "countries": {"type": "array", "items": {"type": "string"}, "description": "目标国家"},
            "stage": {"type": "string", "enum": list(STAGES), "description": "所处阶段：exploring 了解中 / preparing 准备中 / applying 申请中 / admitted 已录取"},
            "target_intake": {"type": "string", "description": "计划入学时间，如 2027 秋"},
            "interested_services": {"type": "array", "items": {"type": "string"}, "description": "意向服务 SKU"},
            "background": {"type": "string", "description": "用户主动提供的背景摘要（专业、均分区间、语言成绩等），不要包含证件号"},
            "preferred_time": {"type": "string", "description": "方便沟通的时间（注明时区）"},
            "consent": {"type": "boolean", "description": "用户是否明确同意由顾问联系，必须为 true"},
        },
        create_consultation_lead,
        required=["name", "contact_channel", "contact", "consent"],
    )


def _count_lead(lead_type: str) -> None:
    from core.metrics import LEADS_CREATED
    LEADS_CREATED.labels(type=lead_type).inc()


# ── 费用与售后 ────────────────────────────────────────────────────────────────

def check_payment_fields(req: Request, args: Dict[str, Any]) -> Dict[str, Any]:
    """费用工具：检查付款/退款核验需要的字段是否齐全。"""
    fields = {
        "contract_id": bool(req.entities.get("contract_id") or args.get("contract_id")),
        "amount": bool(req.entities.get("amount") or args.get("amount")),
        "date": bool(req.entities.get("date") or args.get("date")),
        "payment_channel": bool(args.get("payment_channel")),
    }
    labels = {"contract_id": "合同号", "amount": "付款金额", "date": "付款时间", "payment_channel": "付款渠道"}
    return {
        "fields": fields,
        "missing_fields": [labels[name] for name, present in fields.items() if not present],
        "can_confirm_refund": False,
        "reason": "工具只做字段检查，不连接收款系统；实际退款需顾问核验协议后处理。",
    }


def calculate_refund_tool(req: Request, args: Dict[str, Any]) -> Dict[str, Any]:
    """费用工具：按退款政策估算可退金额，结果需人工核验。"""
    return calculate_refund(
        str(args.get("sku", "")),
        args.get("amount_paid"),
        str(args.get("stage", "")),
        completed_rounds=args.get("completed_rounds"),
        progress_percent=args.get("progress_percent"),
    )


def get_payment_policy(req: Request, args: Dict[str, Any]) -> Dict[str, Any]:
    """费用工具：付款、退款、发票政策原文。"""
    catalog = get_catalog()
    return {
        "payment": catalog.payment.model_dump(),
        "refund_policy": catalog.refund_policy.model_dump(),
        "invoice": catalog.invoice.model_dump(),
    }


def compare_amounts(req: Request, args: Dict[str, Any]) -> Dict[str, Any]:
    """费用工具：只做用户明确提供金额之间的算术。"""
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
        "interpretation": "仅表示金额差值，不代表多付或退款结论",
    }


# ── 转顾问 ────────────────────────────────────────────────────────────────────

def build_handoff_summary(req: Request, reason: str) -> Dict[str, Any]:
    return {
        "request_id": req.request_id,
        "user_id": req.user_id,
        "conv_id": req.conv_id,
        "reason": reason[:200],
        "intent": req.intent.value if req.intent else "unknown",
        "urgency": req.urgency.name if req.urgency else "UNKNOWN",
        "entities": req.entities or {},
        "last_message": (req.message or "")[:500],
    }


def build_handoff_tool(lead_store: Optional[LeadStore]) -> AgentToolSpec:
    async def create_handoff_summary(req: Request, args: Dict[str, Any]) -> Dict[str, Any]:
        summary = build_handoff_summary(req, str(args.get("reason", "需要顾问继续跟进")))
        ticket_id = None
        if lead_store is not None:
            ticket = await lead_store.create(summary, lead_type="handoff")
            ticket_id = ticket["id"]
            _count_lead("handoff")
        return {**summary, "ticket_id": ticket_id, "sensitive_data_required": False}

    return make_tool(
        "create_handoff_summary",
        "生成交给工作室顾问的结构化交接单，并写入线索面板供顾问跟进。",
        {"reason": {"type": "string", "description": "需要转顾问的原因"}},
        create_handoff_summary,
    )


# ── 共享：知识库 RAG ─────────────────────────────────────────────────────────

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
            "检索工作室知识库（服务、政策、五国申请资料、FAQ），返回最相关的文档片段。",
            {
                "query": {"type": "string", "description": "用户问题或检索关键词"},
                "top_k": {"type": "integer", "description": "返回结果条数"},
            },
            search_knowledge_base,
            required=["query"],
        )
    }


# ── 按角色组装 ────────────────────────────────────────────────────────────────

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
        "get_studio_profile": make_tool(
            "get_studio_profile",
            "获取工作室基本信息：品牌、所在地与时差、响应时效、服务覆盖范围和预约流程。",
            {},
            get_studio_profile,
        ),
    }


def consulting_tools(lead_store: Optional[LeadStore] = None) -> Dict[str, AgentToolSpec]:
    return {
        "lookup_country_admissions_overview": make_tool(
            "lookup_country_admissions_overview",
            "查询瑞典/德国/荷兰/芬兰/丹麦 CS 硕士的参考资料：申请平台、时间窗口、申请费、学费、语言要求、代表项目、居留许可、奖学金和官网链接。为整理的参考数据，不代表实时官方信息。",
            {"country": {"type": "string", "description": "国家名，中英文均可，如 瑞典 / Germany"}},
            lookup_country_admissions_overview,
            required=["country"],
        ),
        "lookup_service_offering": make_tool(
            "lookup_service_offering",
            "查询工作室服务与公开价格。不传参数返回全部服务价目概览；传 sku 返回该服务的完整内容、交付物、轮次和时效；传 category 返回某类服务。",
            {
                "sku": {"type": "string", "description": "服务 SKU，如 selection_full / essay_pack_3 / full_journey"},
                "category": {"type": "string", "enum": list(CATEGORY_LABELS), "description": "服务类别"},
            },
            lookup_service_offering,
        ),
        "quote_service_bundle": make_tool(
            "quote_service_bundle",
            "按价目表和公开优惠规则计算报价明细（小计、优惠、应付、定金）。任何涉及具体金额或折扣的回答都必须调用此工具，禁止自行计算或承诺额外优惠。",
            {
                "items": {
                    "type": "array",
                    "description": "服务清单",
                    "items": {
                        "type": "object",
                        "properties": {
                            "sku": {"type": "string"},
                            "qty": {"type": "integer"},
                        },
                        "required": ["sku"],
                    },
                },
                "early_bird": {"type": "boolean", "description": "用户是否计划在早鸟截止前签约"},
                "group_size": {"type": "integer", "description": "一起报名的人数（含本人）"},
                "referral": {"type": "boolean", "description": "是否由老学员推荐"},
            },
            quote_service_bundle,
            required=["items"],
        ),
        "create_consultation_lead": build_lead_tool(lead_store),
    }


def billing_tools() -> Dict[str, AgentToolSpec]:
    return {
        "check_payment_fields": make_tool(
            "check_payment_fields",
            "检查付款/退款核验字段（合同号、金额、时间、渠道）是否齐全；不连接收款系统。",
            {"payment_channel": {"type": "string", "description": "付款渠道，例如微信支付、支付宝、银行转账"}},
            check_payment_fields,
        ),
        "calculate_refund": make_tool(
            "calculate_refund",
            "按退款政策估算可退金额。stage：not_started 未启动 / in_progress 进行中 / delivered 已交付。结果仅为估算，需顾问核验协议后确认。",
            {
                "sku": {"type": "string", "description": "服务 SKU"},
                "amount_paid": {"type": "number", "description": "已付金额（元）"},
                "stage": {"type": "string", "enum": list(REFUND_STAGES)},
                "completed_rounds": {"type": "integer", "description": "文书类已完成的修改轮次"},
                "progress_percent": {"type": "number", "description": "套餐类服务的进度百分比（由顾问确认）"},
            },
            calculate_refund_tool,
            required=["sku", "amount_paid", "stage"],
        ),
        "get_payment_policy": make_tool(
            "get_payment_policy",
            "获取付款方式、定金规则、退款政策和发票政策原文。",
            {},
            get_payment_policy,
        ),
        "compare_amounts": make_tool(
            "compare_amounts",
            "计算用户明确提供的两笔金额差值；不判断是否多付，也不执行退款。",
            {
                "amount_a": {"type": "number", "description": "第一笔金额"},
                "amount_b": {"type": "number", "description": "第二笔金额"},
            },
            compare_amounts,
            required=["amount_a", "amount_b"],
        ),
    }


def escalation_tools(lead_store: Optional[LeadStore] = None) -> Dict[str, AgentToolSpec]:
    return {"create_handoff_summary": build_handoff_tool(lead_store)}
