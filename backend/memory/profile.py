"""
客户档案：用户本人的关键事实，一个字段一个当前值，带出处原话、说话时间、来源和以前的取值。

模型只提候选（字段、取值、出处），怎么写进档案由这里的代码决定：
  - 后说的覆盖先说的，旧值进 history；说话时间更早的候选（补全顺序乱了）只进 history；
  - 顾问改过的字段（source=advisor）模型不能覆盖：用户后来的不同说法记在 pending 里，等顾问核实；
  - experience / other 可以有多条，全部保留（最多 LIST_KEEP 条）。
档案整份存在 Redis 一个键里，不设过期，用户要求删除时整份删掉。
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

from core.llm_utils import safe_text

logger = logging.getLogger(__name__)

# 档案字段：补全时模型从这些里选
FACT_SLOTS: Dict[str, str] = {
    "major": "本科专业",
    "school": "本科院校",
    "gpa": "GPA / 均分",
    "language": "语言考试成绩（写明考试名和分数，如 雅思 6.5）",
    "target_country": "目标国家",
    "target_city": "目标城市",
    "intake": "计划入学时间",
    "budget": "预算",
    "graduation": "毕业时间 / 工作年限",
    "experience": "实习、工作或项目经历",
    "other": "其他明确的偏好或约束",
}
# 这两个字段可以同时有好几条（几段经历、几个偏好）；其他字段只有一个当前值
LIST_SLOTS = {"experience", "other"}
LIST_KEEP = 8
LIST_SHOW = 5
HISTORY_KEEP = 5

Profile = Dict[str, Dict[str, Any]]


def label(slot: str) -> str:
    return FACT_SLOTS[slot].split("（")[0]


def apply_candidate(profile: Profile, slot: str, value: str, quote: str, said_at: str, source: str = "user") -> bool:
    """把一条候选事实合并进档案，返回档案有没有变化。"""
    value = safe_text(value).strip()[:80]
    if slot not in FACT_SLOTS or not value:
        return False
    quote = safe_text(quote).strip()[:300]

    if slot in LIST_SLOTS:
        items = profile.setdefault(slot, {"items": []})["items"]
        if any(it["value"] == value for it in items):
            return False
        items.append({"value": value, "quote": quote, "said_at": said_at, "source": source})
        items.sort(key=lambda it: it["said_at"])
        del items[:-LIST_KEEP]
        return True

    cur = profile.get(slot)
    if not cur:
        profile[slot] = {"value": value, "quote": quote, "said_at": said_at, "source": source,
                         "history": [], "pending": []}
        return True
    if cur["source"] == "advisor" and source == "user":
        # 顾问核实过的值不让模型改：用户后来说了不一样的，记下来等顾问看
        if value == cur["value"] or said_at < cur["said_at"] or any(p["value"] == value for p in cur["pending"]):
            return False
        cur["pending"].append({"value": value, "quote": quote, "said_at": said_at})
        del cur["pending"][:-HISTORY_KEEP]
        return True
    if value == cur["value"]:
        if source == "advisor" and (cur["source"] != "advisor" or cur["pending"]):
            cur.update(source="advisor", said_at=max(said_at, cur["said_at"]), pending=[])
            return True
        if said_at > cur["said_at"]:
            cur.update(said_at=said_at, quote=quote or cur["quote"])
            return True
        return False
    if said_at >= cur["said_at"]:
        old = {"value": cur["value"], "said_at": cur["said_at"], "source": cur["source"]}
        cur["history"] = [h for h in cur["history"] if h["value"] != value] + [old]
        del cur["history"][:-HISTORY_KEEP]
        cur.update(value=value, quote=quote, said_at=said_at, source=source)
        if source == "advisor":
            cur["pending"] = []
        return True
    # 比当前值说得早：只是以前的说法
    if any(h["value"] == value for h in cur["history"]):
        return False
    cur["history"].append({"value": value, "said_at": said_at, "source": source})
    cur["history"].sort(key=lambda h: h["said_at"])
    del cur["history"][:-HISTORY_KEEP]
    return True


def render(profile: Profile) -> List[str]:
    """档案放进提示词的样子：一行一个字段，括号里是以前的说法和待核实的说法。"""
    lines = []
    for slot in FACT_SLOTS:
        entry = profile.get(slot)
        if not entry:
            continue
        if slot in LIST_SLOTS:
            values = [it["value"] for it in entry["items"][-LIST_SHOW:]]
            if values:
                lines.append(f"{label(slot)}：{'；'.join(values)}")
            continue
        line = f"{label(slot)}：{entry['value']}"
        if entry["source"] == "advisor":
            line += "（顾问已核实）"
        if entry["history"]:
            line += f"（之前：{'、'.join(h['value'] for h in entry['history'])}）"
        if entry["pending"]:
            line += f"（用户后来说：{'、'.join(p['value'] for p in entry['pending'])}，待顾问核实）"
        lines.append(line)
    return lines


class ProfileStore:
    """档案的读写，Redis 一个用户一个键。"""

    def __init__(self, redis_client: Any):
        self._redis = redis_client

    @staticmethod
    def key(user_id: str) -> str:
        return f"mem:profile:{user_id}"

    async def get(self, user_id: str) -> Profile:
        raw = await self._redis.get(self.key(user_id))
        if not raw:
            return {}
        try:
            data = json.loads(raw)
        except ValueError:
            logger.warning(f"档案数据损坏，按空档案处理: {user_id}")
            return {}
        return data if isinstance(data, dict) else {}

    async def save(self, user_id: str, profile: Profile) -> None:
        await self._redis.set(self.key(user_id), json.dumps(profile, ensure_ascii=False))

    async def delete(self, user_id: str) -> None:
        await self._redis.delete(self.key(user_id))
