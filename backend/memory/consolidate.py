"""
整理：一次模型调用，同时做两件事。
  1. 从新说的原话里抽出用户本人的关键事实候选（一句话可以有几条）和事件时间；
  2. 给停下来的会话写（或更新）咨询记录摘要。
这里只拼提示词、解析输出；候选怎么写进档案由 profile.apply_candidate 决定。
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

from core.llm_utils import safe_text
from memory.profile import FACT_SLOTS

TRANSCRIPT_CHARS = 6000     # 每个会话给模型看的对话最多多少字（取最后的部分）
ASSISTANT_CHARS = 300       # 对话里每条助手回复最多多少字


def parse_json_object(text: str) -> Dict[str, Any]:
    raw = (text or "").strip().strip("`")
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end <= start:
        raise ValueError(f"没有找到 JSON：{raw[:80]!r}")
    return json.loads(raw[start:end + 1])


def transcript_text(messages: List[Dict[str, Any]]) -> str:
    lines = []
    for m in messages:
        content = safe_text(m.get("content") or "")
        if m.get("role") == "assistant" and len(content) > ASSISTANT_CHARS:
            content = content[:ASSISTANT_CHARS].rstrip() + "…"
        lines.append(f"{'用户' if m.get('role') == 'user' else '助手'}：{content}")
    text = "\n".join(lines)
    return text[-TRANSCRIPT_CHARS:]


def build_prompt(profile_lines: List[str], related: List[Dict[str, Any]], new: List[Dict[str, Any]],
                 cases: List[Dict[str, Any]]) -> Tuple[str, Dict[str, str], Dict[str, str]]:
    """返回提示词，以及 M 编号 → 原话 id、C 编号 → 会话 id 的对照。"""
    m_ids = {f"M{k + 1}": r["id"] for k, r in enumerate(new)}
    c_ids = {f"C{k + 1}": c["conv_id"] for k, c in enumerate(cases)}
    slots = "；".join(f"{k}（{v}）" for k, v in FACT_SLOTS.items())
    parts = ["你在维护一位留学咨询用户的客户档案和咨询记录。只返回 JSON。"]
    if new:
        parts.append(
            "【任务一】对下面每条 M（用户新说的话）给出：\n"
            "- event_time：话里的事发生在什么时候，按说话日期换算成 YYYY-MM-DD 或 YYYY-MM（“去年夏天”要换算）；"
            "话里自带日期的以话里为准；说不出就留空；\n"
            f"- facts：这句话说出的用户**本人**的关键事实，可以有多条，每条给 slot（只能是 {slots}）和 value"
            "（简短取值，保留数字和考试名，如“雅思 7.0”“每年 25 万”）。讲别人的事、闲聊、提问、没有明确事实就给空列表；"
            "取值要结合以前的原话才能确定时，可以结合。\n\n"
            "当前档案：\n" + ("\n".join(f"- {line}" for line in profile_lines) or "（空）") + "\n\n"
            "以前说过的相关原话（只用来理解指代，不用给它们输出）：\n"
            + ("\n".join(f"- （说于 {r['ts'][:10]}）{r['text']}" for r in related) or "（无）") + "\n\n"
            "新说的话：\n" + "\n".join(f"{k}（说于 {r['ts'][:10]}）：{r['text']}" for k, r in zip(m_ids, new))
        )
    if cases:
        blocks = []
        for k, c in zip(c_ids, cases):
            prev = f"已有摘要（在它基础上更新）：{json.dumps(c['previous'], ensure_ascii=False)}\n" if c.get("previous") else ""
            blocks.append(f"{k}：\n{prev}对话：\n{c['transcript']}")
        parts.append(
            "【任务二】下面每个会话已经停下来了，给每个 C 写咨询记录：\n"
            "- summary：1–3 句，用户这次想解决什么、聊到了哪、有什么结论；\n"
            "- open_items：还没解决的问题，或用户说要回去考虑、确认的事（没有就空列表）；\n"
            "- next_step：双方约好的下一步，写清时间和谁做什么（没有就空）。\n"
            "只写对话里出现的内容，不要推测。\n\n" + "\n\n".join(blocks)
        )
    parts.append('返回格式：{"notes": [{"m": "M1", "event_time": "", "facts": [{"slot": "", "value": ""}]}], '
                 '"cases": [{"c": "C1", "summary": "", "open_items": [], "next_step": ""}]}')
    return safe_text("\n\n".join(parts)), m_ids, c_ids
