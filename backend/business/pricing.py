"""报价与退款的确定性计算。

价格和优惠只能由这里按 catalog.yaml 的公开规则算出来，LLM 只负责解释结果。
这样可以杜绝模型"自由发挥"给出不存在的折扣，也方便单元测试逐分核对。
"""
from __future__ import annotations

import math
from datetime import date
from typing import Any, Dict, List, Optional

from business.catalog import Catalog, get_catalog

REFUND_STAGES = ("not_started", "in_progress", "delivered")


def _early_bird_open(catalog: Catalog, today: date) -> bool:
    month, day = (int(part) for part in catalog.discounts.early_bird.deadline_month_day.split("-"))
    return today <= date(today.year, month, day)


def quote_bundle(
    items: List[Dict[str, Any]],
    *,
    early_bird: bool = False,
    group_size: int = 1,
    referral: bool = False,
    today: Optional[date] = None,
    catalog: Optional[Catalog] = None,
) -> Dict[str, Any]:
    """按公开价目和优惠规则计算报价。

    优惠规则（见 catalog.discounts）：
      - 早鸟和团报二选一，取折扣力度更大的一项，只作用于 discount_eligible 的服务；
      - 老带新立减固定金额，可叠加；
      - 叠加后不低于标价 floor_ratio。
    """
    catalog = catalog or get_catalog()
    today = today or date.today()
    rules = catalog.discounts

    if not items:
        return {"success": False, "error": "items 不能为空，请至少选择一项服务"}

    lines: List[Dict[str, Any]] = []
    unknown: List[str] = []
    for raw in items:
        sku = str((raw or {}).get("sku", "")).strip()
        qty = int((raw or {}).get("qty", 1) or 1)
        if qty < 1 or qty > 20:
            return {"success": False, "error": f"{sku} 的数量 {qty} 不合理（1–20）"}
        service = catalog.service(sku)
        if service is None:
            unknown.append(sku or "<空>")
            continue
        lines.append({
            "sku": sku,
            "name": service.name,
            "category": service.category,
            "unit_price": service.price,
            "unit": service.unit,
            "qty": qty,
            "amount": service.price * qty,
            "discount_eligible": service.discount_eligible,
        })
    if unknown:
        return {
            "success": False,
            "error": f"未知的服务 SKU: {', '.join(unknown)}",
            "valid_skus": [s.sku for s in catalog.services],
        }

    subtotal = sum(line["amount"] for line in lines)
    eligible_subtotal = sum(line["amount"] for line in lines if line["discount_eligible"])
    notes: List[str] = []
    discounts: List[Dict[str, Any]] = []

    # 早鸟 / 团报：二选一取最优
    candidates: List[tuple[float, str, str]] = []
    if early_bird:
        if _early_bird_open(catalog, today):
            candidates.append((rules.early_bird.rate, "early_bird", rules.early_bird.name))
        else:
            notes.append(f"今天（{today.isoformat()}）已过本年度早鸟截止日（{rules.early_bird.deadline_month_day}），不适用早鸟优惠。")
    if group_size >= rules.group.min_people:
        candidates.append((rules.group.rate, "group", rules.group.name))
    elif group_size > 1:
        notes.append(f"团报需至少 {rules.group.min_people} 人。")

    if candidates and eligible_subtotal > 0:
        rate, code, name = min(candidates, key=lambda c: c[0])
        off = round(eligible_subtotal * (1 - rate))
        discounts.append({"code": code, "name": name, "rate": rate, "amount": -off,
                          "applies_to": "选校全案、文书套餐和全程陪跑类服务"})
        if len(candidates) > 1:
            notes.append("早鸟与团报不可叠加，已自动选择更优惠的一项。")
    elif candidates:
        notes.append("所选服务均不参与早鸟/团报折扣（仅选校全案、套餐类服务参与）。")

    if referral and subtotal > 0:
        discounts.append({"code": "referral", "name": rules.referral.name,
                          "amount": -min(rules.referral.amount, subtotal)})

    total = subtotal + sum(d["amount"] for d in discounts)
    floor = math.ceil(subtotal * rules.floor_ratio)
    if discounts and total < floor:
        discounts.append({"code": "floor_adjust", "name": "最低实付保护", "amount": floor - total})
        notes.append(rules.floor_rule)
        total = floor

    # 付款安排：套餐/选校部分收定金，其余全款
    deposit_share = sum(
        line["amount"] for line in lines if line["category"] in catalog.payment.deposit_applies_to
    )
    if deposit_share and subtotal:
        deposit_part_net = round(total * deposit_share / subtotal)
        other_net = total - deposit_part_net
        due_now = round(deposit_part_net * catalog.payment.deposit_ratio) + other_net
    else:
        deposit_part_net = 0
        due_now = total

    return {
        "success": True,
        "currency": catalog.currency,
        "lines": lines,
        "subtotal": subtotal,
        "discounts": discounts,
        "total": total,
        "payment_plan": {
            "due_at_signing": due_now,
            "due_before_start": total - due_now,
            "rule": catalog.payment.rule,
        },
        "valid_days": catalog.payment.quote_validity_days,
        "quote_date": today.isoformat(),
        "notes": notes,
        "no_negotiation": rules.no_negotiation,
        "disclaimer": "报价按公开价目表和公开优惠规则自动计算，最终以电子服务协议为准。",
        "display": _format_quote(catalog, lines, subtotal, discounts, total, due_now),
    }


def _format_quote(catalog: Catalog, lines, subtotal, discounts, total, due_now) -> str:
    rows = [f"- {l['name']} × {l['qty']}：{catalog.money(l['amount'])}" for l in lines]
    rows.append(f"- 小计：{catalog.money(subtotal)}")
    rows += [f"- {d['name']}：{'-' if d['amount'] < 0 else '+'}{catalog.money(abs(d['amount']))}" for d in discounts]
    rows.append(f"- 应付合计：{catalog.money(total)}（签约时付 {catalog.money(due_now)}）")
    return "\n".join(rows)


def calculate_refund(
    sku: str,
    amount_paid: float,
    stage: str,
    *,
    completed_rounds: Optional[int] = None,
    progress_percent: Optional[float] = None,
    catalog: Optional[Catalog] = None,
) -> Dict[str, Any]:
    """按退款政策估算可退金额。结果始终需要人工核验后才生效。"""
    catalog = catalog or get_catalog()
    service = catalog.service(sku)
    if service is None:
        return {"success": False, "error": f"未知的服务 SKU: {sku}",
                "valid_skus": [s.sku for s in catalog.services]}
    if stage not in REFUND_STAGES:
        return {"success": False, "error": f"stage 必须是 {', '.join(REFUND_STAGES)} 之一"}
    try:
        paid = float(amount_paid)
    except (TypeError, ValueError):
        return {"success": False, "error": "amount_paid 必须是数字"}
    if paid < 0:
        return {"success": False, "error": "amount_paid 不能为负数"}

    base = {
        "success": True,
        "sku": sku,
        "service": service.name,
        "stage": stage,
        "amount_paid": paid,
        "requires_human_review": True,
        "policy_summary": catalog.refund_policy.summary,
        "notes": list(catalog.refund_policy.notes),
    }

    def result(refund: float, basis: str, needs_input: Optional[str] = None) -> Dict[str, Any]:
        out = dict(base, estimated_refund=round(max(0.0, min(refund, paid)), 2), basis=basis)
        if needs_input:
            out["needs_input"] = needs_input
        return out

    if stage == "not_started":
        return result(paid, "服务尚未启动，全额退还已付款项。")
    if stage == "delivered":
        return result(0, "服务已交付完成，按政策不予退款。")

    # in_progress
    category = service.category
    if category in {"consult", "addon"} and service.revision_rounds is None and sku != "extra_round":
        return result(0, "该服务已开始提供即视为已使用，按政策不退款（单次咨询可在 24 小时前改期）。")

    if category == "essay" or sku == "extra_round":
        total_rounds = service.revision_rounds or 1
        if completed_rounds is None:
            return result(0, "文书类按剩余轮次比例退还。", needs_input="completed_rounds（已完成的修改轮次）")
        done = max(0, min(int(completed_rounds), total_rounds))
        return result(paid * (total_rounds - done) / total_rounds,
                      f"文书类按剩余轮次比例退还：共 {total_rounds} 轮，已完成 {done} 轮。")

    if category == "selection":
        return result(paid * 0.5, "选校全案在报告交付前退还已付款的 50%。")

    # package：定金不退，已付尾款按剩余进度比例退
    deposit = round(service.price * catalog.payment.deposit_ratio)
    refundable_base = max(0.0, paid - deposit)
    if service.revision_rounds and completed_rounds is not None:
        total_rounds = service.revision_rounds
        done = max(0, min(int(completed_rounds), total_rounds))
        ratio = (total_rounds - done) / total_rounds
        basis = f"套餐启动后定金 {catalog.money(deposit)} 不退，已付尾款按剩余轮次（{total_rounds - done}/{total_rounds}）比例退还。"
    elif progress_percent is not None:
        pct = max(0.0, min(float(progress_percent), 100.0))
        ratio = (100 - pct) / 100
        basis = f"套餐启动后定金 {catalog.money(deposit)} 不退，已付尾款按剩余进度（{100 - pct:.0f}%）比例退还。"
    else:
        need = "completed_rounds（已完成轮次）" if service.revision_rounds else "progress_percent（顾问确认的服务进度百分比）"
        out = result(0, f"套餐启动后定金 {catalog.money(deposit)} 不退，已付尾款按剩余进度比例退还。", needs_input=need)
        out["max_possible_refund"] = round(refundable_base, 2)
        return out
    return result(refundable_base * ratio, basis)
