"""业务目录加载器。

`catalog.yaml`（工作室、服务、价格、政策）和 `countries.yaml`（五国资料）是
整个系统的单一数据源：Agent 工具、知识库种子、Skill 参考资料和前端价目面板
都从这里读取，改价格只需要改 YAML。

加载时用 pydantic 做结构校验，配置写错会在启动时直接报错，而不是等到用户
问价格时才暴露出来。
"""
from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field, field_validator

_DIR = Path(__file__).resolve().parent

SERVICE_CATEGORIES = {"consult", "selection", "essay", "package", "addon"}
CATEGORY_LABELS = {
    "consult": "咨询",
    "selection": "选校",
    "essay": "文书单项",
    "package": "套餐",
    "addon": "增值服务",
}


class ServiceExtra(BaseModel):
    name: str
    price: int
    unit: str


class Service(BaseModel):
    sku: str
    name: str
    category: str
    price: int = Field(ge=0)
    unit: str
    summary: str
    includes: List[str] = Field(default_factory=list)
    excludes: List[str] = Field(default_factory=list)
    deliverables: List[str] = Field(default_factory=list)
    duration: Optional[str] = None
    turnaround: Optional[str] = None
    revision_rounds: Optional[int] = Field(default=None, ge=1)
    extras: List[ServiceExtra] = Field(default_factory=list)
    discount_eligible: bool = False
    upgrade_note: Optional[str] = None

    @field_validator("category")
    @classmethod
    def _check_category(cls, value: str) -> str:
        if value not in SERVICE_CATEGORIES:
            raise ValueError(f"未知服务类别 {value}，可选 {sorted(SERVICE_CATEGORIES)}")
        return value

    @property
    def category_label(self) -> str:
        return CATEGORY_LABELS[self.category]


class EarlyBird(BaseModel):
    name: str
    rate: float = Field(gt=0, le=1)
    deadline_month_day: str
    rule: str

    @field_validator("deadline_month_day")
    @classmethod
    def _check_md(cls, value: str) -> str:
        if not re.fullmatch(r"\d{2}-\d{2}", value):
            raise ValueError("deadline_month_day 格式应为 MM-DD")
        return value


class GroupDiscount(BaseModel):
    name: str
    rate: float = Field(gt=0, le=1)
    min_people: int = Field(ge=2)
    rule: str


class Referral(BaseModel):
    name: str
    amount: int = Field(ge=0)
    rule: str


class Discounts(BaseModel):
    early_bird: EarlyBird
    group: GroupDiscount
    referral: Referral
    stacking_rule: str
    floor_ratio: float = Field(gt=0, le=1)
    floor_rule: str
    no_negotiation: str


class Payment(BaseModel):
    deposit_ratio: float = Field(gt=0, lt=1)
    deposit_applies_to: List[str]
    rule: str
    channels: List[str]
    contract_note: str
    quote_validity_days: int = Field(ge=1)


class RefundRule(BaseModel):
    stage: str
    label: str
    rule: str


class RefundPolicy(BaseModel):
    summary: str
    rules: List[RefundRule]
    notes: List[str] = Field(default_factory=list)


class Invoice(BaseModel):
    type: str
    timing: str
    title_types: List[str]
    note: str


class Booking(BaseModel):
    steps: List[str]
    reschedule: str


class Studio(BaseModel):
    name: str
    brand: str
    tagline: str
    founders_note: str
    base: str
    timezone: str
    timezone_note: str
    response_sla: str
    capacity_note: str
    coverage: Dict[str, Any]
    not_covered: List[str]
    contact: Dict[str, Any]


class Catalog(BaseModel):
    catalog_version: str
    currency: str
    currency_symbol: str
    studio: Studio
    services: List[Service]
    discounts: Discounts
    payment: Payment
    refund_policy: RefundPolicy
    invoice: Invoice
    booking: Booking

    @field_validator("services")
    @classmethod
    def _unique_sku(cls, services: List[Service]) -> List[Service]:
        seen = set()
        for service in services:
            if service.sku in seen:
                raise ValueError(f"重复的 SKU: {service.sku}")
            seen.add(service.sku)
        return services

    def service(self, sku: str) -> Optional[Service]:
        return next((s for s in self.services if s.sku == sku), None)

    def money(self, amount: float) -> str:
        return f"{self.currency_symbol}{amount:,.0f}"


class Country(BaseModel):
    key: str
    name_en: str
    aliases: List[str]
    portal: str
    application_window: str
    application_fee: str
    tuition_non_eu: str
    duration: str
    language: str
    academic_notes: List[str] = Field(default_factory=list)
    representative_programs: List[str] = Field(default_factory=list)
    residence_permit: str
    scholarships: List[str] = Field(default_factory=list)
    highlights: str
    source_urls: List[str] = Field(default_factory=list)


class CountryBook(BaseModel):
    last_verified: str
    target_intake: str
    countries: List[Country]

    def find(self, name: str) -> Optional[Country]:
        needle = (name or "").strip().lower()
        if not needle:
            return None
        for country in self.countries:
            if needle == country.key or needle in (alias.lower() for alias in country.aliases):
                return country
        return None


def _read_yaml(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} 顶层必须是对象")
    return data


def _business_dir() -> Path:
    return Path(os.getenv("GOEUROOPS_BUSINESS_DIR", str(_DIR))).expanduser().resolve()


@lru_cache(maxsize=1)
def get_catalog() -> Catalog:
    return Catalog.model_validate(_read_yaml(_business_dir() / "catalog.yaml"))


@lru_cache(maxsize=1)
def get_countries() -> CountryBook:
    return CountryBook.model_validate(_read_yaml(_business_dir() / "countries.yaml"))


def reload_business_data() -> None:
    """清掉缓存，下次访问时重新读取 YAML。"""
    get_catalog.cache_clear()
    get_countries.cache_clear()


# 用户口语里的服务说法 → SKU，用于实体抽取和报价工具的容错。
# 顺序有意义：更具体的说法放在前面。
SERVICE_ALIASES: List[tuple[str, str]] = [
    ("全程陪跑plus", "full_journey_plus"),
    ("陪跑plus", "full_journey_plus"),
    ("全程陪跑", "full_journey"),
    ("陪跑", "full_journey"),
    ("全程", "full_journey"),
    ("选校全案", "selection_full"),
    ("选校定位", "selection_full"),
    ("单次咨询", "consult_single"),
    ("单次选校", "consult_single"),
    ("免费沟通", "intro_call"),
    ("初步沟通", "intro_call"),
    ("5个项目文书", "essay_pack_5"),
    ("五个项目文书", "essay_pack_5"),
    ("3个项目文书", "essay_pack_3"),
    ("三个项目文书", "essay_pack_3"),
    ("文书套餐", "essay_pack_3"),
    ("动机信", "ps_single"),
    ("个人陈述", "ps_single"),
    ("ps", "ps_single"),
    ("简历", "cv"),
    ("cv", "cv"),
    ("推荐信", "rl_guide"),
    ("网申审核", "app_review"),
    ("模拟面试", "mock_interview"),
    ("签证", "visa_guide"),
    ("居留", "visa_guide"),
]


def match_service_mentions(text: str) -> List[str]:
    """从一句话里找出提到的服务 SKU（去重、保持出现顺序）。"""
    lowered = re.sub(r"\s+", "", (text or "").lower())
    found: List[str] = []
    consumed = lowered
    for alias, sku in SERVICE_ALIASES:
        key = alias.lower()
        if key.isascii():
            hit = re.search(rf"(?<![a-z]){re.escape(key)}(?![a-z])", consumed)
        else:
            hit = key in consumed
        if hit:
            if sku not in found:
                found.append(sku)
            # 避免"全程陪跑plus"同时命中"全程陪跑"
            consumed = consumed.replace(key, " ")
    return found
