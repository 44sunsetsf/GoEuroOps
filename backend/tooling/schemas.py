"""工具的入参与返回值：用 Pydantic 定义一次，校验和发给模型的 JSON Schema 都从这里来。

入参模型一律 extra="forbid"：模型多传的参数直接拒绝，而不是悄悄忽略。
返回值模型只给金额类工具加：除了字段齐全，还检查业务不变量（实付不会超过标价、
退款不会超过已付），让"工具自己算错"也能被发现，而不是原样交给模型转述。
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Type

from pydantic import BaseModel, ConfigDict, Field, model_validator

from business.catalog import CATEGORY_LABELS
from business.lead_store import CHANNEL_LABELS, STAGES
from business.pricing import REFUND_STAGES


class _Args(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


# ── 入参 ──────────────────────────────────────────────────────────────────────

class NoArgs(_Args):
    pass


class InspectContextArgs(_Args):
    focus: str = Field("general", max_length=40, description="希望关注的业务方向")


class CountryArgs(_Args):
    country: str = Field(..., min_length=1, description="国家名，中英文均可，如 瑞典 / Germany")


class ServiceOfferingArgs(_Args):
    sku: Optional[str] = Field(None, description="服务 SKU，如 selection_full / essay_pack_3 / full_journey")
    category: Optional[Literal[tuple(CATEGORY_LABELS)]] = Field(None, description="服务类别")  # type: ignore[valid-type]


class QuoteItem(_Args):
    sku: str = Field(..., min_length=1, description="服务 SKU")
    qty: int = Field(1, ge=1, le=20, description="数量（1–20）")


class QuoteArgs(_Args):
    items: List[QuoteItem] = Field(..., min_length=1, description="服务清单")
    early_bird: bool = Field(False, description="用户是否计划在早鸟截止前签约")
    group_size: int = Field(1, ge=1, le=50, description="一起报名的人数（含本人）")
    referral: bool = Field(False, description="是否由老学员推荐")


class RefundArgs(_Args):
    sku: str = Field(..., min_length=1, description="服务 SKU")
    amount_paid: float = Field(..., ge=0, description="已付金额（元）")
    stage: Literal[tuple(REFUND_STAGES)]  # type: ignore[valid-type]
    completed_rounds: Optional[int] = Field(None, ge=0, description="文书类已完成的修改轮次")
    progress_percent: Optional[float] = Field(None, ge=0, le=100, description="套餐类服务的进度百分比（由顾问确认）")


class PaymentFieldsArgs(_Args):
    payment_channel: Optional[str] = Field(None, description="付款渠道，例如微信支付、支付宝、银行转账")


class CompareAmountsArgs(_Args):
    amount_a: float = Field(..., description="第一笔金额")
    amount_b: float = Field(..., description="第二笔金额")


class LeadArgs(_Args):
    name: str = Field(..., min_length=1, max_length=40, description="用户希望的称呼")
    contact_channel: Literal[tuple(CHANNEL_LABELS)] = Field(..., description="联系渠道：wechat / email / phone")  # type: ignore[valid-type]
    contact: str = Field(..., min_length=1, max_length=120, description="对应渠道的联系方式")
    countries: Optional[List[str]] = Field(None, description="目标国家")
    stage: Optional[Literal[tuple(STAGES)]] = Field(  # type: ignore[valid-type]
        None, description="所处阶段：exploring 了解中 / preparing 准备中 / applying 申请中 / admitted 已录取")
    target_intake: Optional[str] = Field(None, description="计划入学时间，如 2027 秋")
    interested_services: Optional[List[str]] = Field(None, description="意向服务 SKU")
    background: Optional[str] = Field(None, max_length=500, description="用户主动提供的背景摘要（专业、均分区间、语言成绩等），不要包含证件号")
    preferred_time: Optional[str] = Field(None, description="方便沟通的时间（注明时区）")
    consent: bool = Field(..., description="用户是否明确同意由顾问联系，必须为 true")


class HandoffArgs(_Args):
    reason: str = Field("需要顾问继续跟进", max_length=200, description="需要转顾问的原因")


class SearchKnowledgeArgs(_Args):
    query: str = Field(..., description="用户问题或检索关键词")
    top_k: int = Field(5, ge=1, le=20, description="返回结果条数")


class ReadSkillReferenceArgs(_Args):
    skill: str = Field(..., description="Skill id")
    file: str = Field(..., description="参考资料文件名，如 objection_handling.md")


# ── 返回值（金额类工具）──────────────────────────────────────────────────────

class _Result(BaseModel):
    model_config = ConfigDict(extra="allow")
    success: bool
    error: Optional[str] = None


class QuoteResult(_Result):
    subtotal: Optional[float] = None
    total: Optional[float] = None
    payment_plan: Optional[Dict[str, Any]] = None
    lines: Optional[List[Dict[str, Any]]] = None

    @model_validator(mode="after")
    def _invariants(self) -> "QuoteResult":
        if not self.success:
            return self
        missing = [n for n in ("subtotal", "total", "payment_plan", "lines") if getattr(self, n) is None]
        if missing:
            raise ValueError(f"报价结果缺少字段: {', '.join(missing)}")
        if self.total < 0 or self.total > self.subtotal:
            raise ValueError(f"报价不变量被破坏：应付 {self.total} 必须在 0 与标价 {self.subtotal} 之间")
        due = self.payment_plan.get("due_at_signing")
        if due is None or due < 0 or due > self.total:
            raise ValueError(f"报价不变量被破坏：签约时应付 {due} 必须在 0 与应付合计 {self.total} 之间")
        return self


class RefundResult(_Result):
    estimated_refund: Optional[float] = None
    amount_paid: Optional[float] = None
    requires_human_review: Optional[bool] = None

    @model_validator(mode="after")
    def _invariants(self) -> "RefundResult":
        if not self.success:
            return self
        if self.estimated_refund is None or self.amount_paid is None:
            raise ValueError("退款结果缺少 estimated_refund 或 amount_paid")
        if not (0 <= self.estimated_refund <= self.amount_paid):
            raise ValueError(f"退款不变量被破坏：可退 {self.estimated_refund} 必须在 0 与已付 {self.amount_paid} 之间")
        if self.requires_human_review is not True:
            raise ValueError("退款估算必须标记 requires_human_review")
        return self


# ── 发给模型的 JSON Schema ────────────────────────────────────────────────────

def schema_for(model: Type[BaseModel]) -> Dict[str, Any]:
    """由 Pydantic 模型生成精简的 JSON Schema：内联 $ref、去掉 title、把 Optional 折叠成单一类型。

    部分兼容 Anthropic 接口的服务对 anyOf / $defs 支持不完整，所以这里统一展开。
    """
    raw = model.model_json_schema()
    defs = raw.pop("$defs", {})

    def walk(node: Any) -> Any:
        if isinstance(node, list):
            return [walk(item) for item in node]
        if not isinstance(node, dict):
            return node
        if "$ref" in node:
            return walk(defs[node["$ref"].split("/")[-1]])
        out: Dict[str, Any] = {}
        for key, value in node.items():
            if key == "title":
                continue
            if key == "properties":          # 属性名本身可能叫 title，不能当关键字删
                out[key] = {name: walk(sub) for name, sub in value.items()}
            else:
                out[key] = walk(value)
        any_of = out.get("anyOf")
        if isinstance(any_of, list):
            non_null = [item for item in any_of if item.get("type") != "null"]
            if len(non_null) == 1 and len(non_null) < len(any_of):
                merged = {k: v for k, v in out.items() if k != "anyOf"}
                merged.update(non_null[0])
                out = merged
        if out.get("default", 0) is None:
            out.pop("default")
        return out

    return walk(raw)
