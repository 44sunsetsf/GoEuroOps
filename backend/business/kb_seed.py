"""知识库种子文档：由 catalog.yaml / countries.yaml / articles/*.md 生成。

种子文档带 seed_version（内容哈希），KnowledgeBase 启动时发现版本变化会
删除旧种子并重新写入——改完 YAML 重启即可生效，不用手动清 ChromaDB 卷；
用户自己上传的文档不受影响。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Tuple

import yaml

from business.catalog import Catalog, CountryBook, get_catalog, get_countries

_ARTICLES_DIR = Path(__file__).resolve().parent / "articles"


def _join(lines: List[str]) -> str:
    return "\n".join(line for line in lines if line)


def _studio_doc(c: Catalog) -> Dict[str, str]:
    s = c.studio
    return {
        "title": f"关于{s.brand}：工作室介绍",
        "domain": "shared",
        "content": _join([
            f"{s.name}。{s.tagline}。",
            s.founders_note,
            f"所在地：{s.base}。{s.timezone_note}",
            f"服务范围：{'、'.join(s.coverage['countries'])}的{s.coverage['programs']}。",
            "不提供的服务：" + "；".join(s.not_covered) + "。",
            f"响应时效：{s.response_sla}",
            s.capacity_note,
        ]),
    }


def _service_docs(c: Catalog) -> List[Dict[str, str]]:
    docs = []
    for sv in c.services:
        price = "免费" if sv.price == 0 else f"{c.money(sv.price)} / {sv.unit}"
        docs.append({
            "title": f"服务介绍：{sv.name}（{price}）",
            "domain": "consulting",
            "content": _join([
                f"{sv.name}，价格 {price}，类别：{sv.category_label}。{sv.summary}",
                f"时长：{sv.duration}。" if sv.duration else "",
                "包含：" + "；".join(sv.includes) + "。" if sv.includes else "",
                "不包含：" + "；".join(sv.excludes) + "。" if sv.excludes else "",
                "交付物：" + "、".join(sv.deliverables) + "。" if sv.deliverables else "",
                f"修改轮次：{sv.revision_rounds} 轮。" if sv.revision_rounds else "",
                f"交付时效：{sv.turnaround}。" if sv.turnaround else "",
                "".join(f"{e.name} {c.money(e.price)} / {e.unit}。" for e in sv.extras),
                sv.upgrade_note or "",
                "参与早鸟/团报优惠。" if sv.discount_eligible else "不参与早鸟/团报折扣（老带新立减仍可用）。",
            ]),
        })
    return docs


def _package_compare_doc(c: Catalog) -> Dict[str, str]:
    rows = [f"{s.name}：{c.money(s.price)}。{s.summary}" for s in c.services if s.category in {"selection", "package"}]
    return {
        "title": "套餐对比与怎么选",
        "domain": "consulting",
        "content": _join([
            "工作室的全案和套餐类服务对比如下。",
            *rows,
            "怎么选：只想确定去哪些国家、申哪些项目，选「选校定位全案」；已经定好项目、只需要文书，选文书套餐（按项目数选 3 个或 5 个）；"
            "希望从选校到录取都有人跟进，选「全程陪跑」；还需要面试、签证材料和落地准备，选「全程陪跑 Plus」。"
            "不确定时可以先预约免费的 15 分钟初步沟通。",
        ]),
    }


def _discount_doc(c: Catalog) -> Dict[str, str]:
    d = c.discounts
    return {
        "title": "优惠规则（早鸟、团报、老带新）",
        "domain": "shared",
        "content": _join([d.early_bird.rule, d.group.rule, d.referral.rule, d.stacking_rule, d.floor_rule, d.no_negotiation]),
    }


def _payment_docs(c: Catalog) -> List[Dict[str, str]]:
    rp = c.refund_policy
    return [
        {
            "title": "付款方式与定金规则",
            "domain": "shared",
            "content": _join([
                c.payment.rule,
                "支持的付款方式：" + "、".join(c.payment.channels) + "。",
                c.payment.contract_note,
                f"报价有效期 {c.payment.quote_validity_days} 天。",
            ]),
        },
        {
            "title": "退款政策",
            "domain": "billing",
            "content": _join([
                rp.summary,
                *[f"{r.label}：{r.rule}" for r in rp.rules],
                *rp.notes,
            ]),
        },
        {
            "title": "发票说明",
            "domain": "billing",
            "content": _join([
                f"可开具{c.invoice.type}，{c.invoice.timing}。",
                "抬头类型：" + "、".join(c.invoice.title_types) + "。",
                c.invoice.note,
            ]),
        },
        {
            "title": "预约咨询流程",
            "domain": "consulting",
            "content": _join([
                "预约流程：" + "；".join(f"{i}. {step}" for i, step in enumerate(c.booking.steps, 1)) + "。",
                c.booking.reschedule,
                c.studio.timezone_note,
            ]),
        },
    ]


def _country_docs(book: CountryBook) -> List[Dict[str, str]]:
    docs = []
    for ct in book.countries:
        docs.append({
            "title": f"{ct.key}（{ct.name_en}）CS 硕士申请概览",
            "domain": "consulting",
            "content": _join([
                f"{ct.key}英语授课计算机硕士申请参考资料（{book.last_verified}，面向{book.target_intake}，请以官网为准）。",
                f"申请平台：{ct.portal}。",
                f"申请时间：{ct.application_window}",
                f"申请费：{ct.application_fee}。",
                f"学费：{ct.tuition_non_eu}。",
                f"学制：{ct.duration}。",
                f"语言要求：{ct.language}。",
                "学术要求要点：" + "；".join(ct.academic_notes) + "。",
                "代表院校与项目：" + "；".join(ct.representative_programs) + "。",
                f"居留许可：{ct.residence_permit}",
                "奖学金：" + "；".join(ct.scholarships) + "。",
                f"特点：{ct.highlights}",
                "官方信息来源：" + "、".join(ct.source_urls),
            ]),
        })
    return docs


def _article_docs() -> List[Dict[str, str]]:
    docs = []
    if not _ARTICLES_DIR.is_dir():
        return docs
    for path in sorted(_ARTICLES_DIR.glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        meta: Dict[str, str] = {}
        body = raw
        if raw.startswith("---"):
            _, front, body = raw.split("---", 2)
            meta = yaml.safe_load(front) or {}
        docs.append({
            "title": str(meta.get("title") or path.stem),
            "domain": str(meta.get("domain") or "shared"),
            "content": body.strip(),
        })
    return docs


def build_seed_documents() -> Tuple[str, List[Dict[str, str]]]:
    """返回 (seed_version, 文档列表)。seed_version 随内容变化而变化。"""
    catalog, book = get_catalog(), get_countries()
    docs = [
        _studio_doc(catalog),
        *_service_docs(catalog),
        _package_compare_doc(catalog),
        _discount_doc(catalog),
        *_payment_docs(catalog),
        *_country_docs(book),
        *_article_docs(),
    ]
    digest = hashlib.sha256(json.dumps(docs, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:10]
    return f"{catalog.catalog_version}-{digest}", docs
