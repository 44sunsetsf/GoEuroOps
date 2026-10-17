"""金额护栏：模型回复里出现的人民币金额，必须能在本轮的依据里找到。

依据包括：本轮所有工具返回的结果、用户自己说的话、背景和知识库片段、命中的 Skill 和系统提示词、
公开价目表。凭空出现的数字（算错的合计、编造的折扣、没调报价工具就报的价）所在的那句话会被替换成
固定提示，而不是原样发给用户。

只管人民币金额（¥、元、RMB、CNY），也认中文数字写的金额（"三千元""一万二千块钱""3千元"）；不管年份、轮次、百分比、折扣率（"9 折"），也不管工作室报价以外
的外币费用（学费、申请费等，来自知识库）。
流式输出时按句放行：一句话确认没有问题才发出去，所以不会出现"先发出错的数字、再撤回"。
"""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Set

logger = logging.getLogger(__name__)

NOTICE = "（这里的金额没有经过报价工具核算，请以报价工具的计算结果或顾问确认为准。）"

_NUM = r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?"
_MONEY_PATTERNS = (
    re.compile(rf"[¥￥]\s*({_NUM})\s*(万)?"),
    re.compile(rf"(?:RMB|CNY|人民币)\s*({_NUM})\s*(万)?", re.IGNORECASE),
    re.compile(rf"({_NUM})\s*(万)?\s*(?:元|块钱|RMB|CNY)", re.IGNORECASE),
)
_ANY_NUMBER = re.compile(rf"({_NUM})\s*(万)?")

# 中文数字金额：第一个字必须是数字（"几千元""上千元"这种约数不算具体金额）
_CN = r"[一二两三四五六七八九十][零〇一二两三四五六七八九十百千万]*"
_CN_MONEY = (
    re.compile(rf"({_CN})\s*(?:元|块钱)"),
    re.compile(rf"(?:人民币)\s*({_CN})"),
)
_DIGIT_QIAN_MONEY = re.compile(r"(\d+(?:\.\d+)?)\s*千\s*(?:元|块钱)")     # "3千元"
_CN_ANY = re.compile(_CN)
_CN_DIGITS = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
_CN_UNITS = {"十": 10, "百": 100, "千": 1000}
_SENTENCE_END = re.compile(r"[。！？!?；;\n]|(?<=[^\d])\.(?=\s)")
_FORCE_FLUSH_CHARS = 600


def _to_float(token: str, wan: str | None) -> float:
    value = float(token.replace(",", ""))
    return round(value * 10000 if wan else value, 2)


def cn_to_number(text: str) -> float:
    """中文数字转数值：三千 → 3000，一万二千五百 → 12500，一千零五 → 1005。

    口语省略末位单位的也认：一万二 → 12000，三千五 → 3500。
    没有任何单位的（"二零二六"）按逐位读。
    """
    if not any(ch in _CN_UNITS or ch == "万" for ch in text):
        return float("".join(str(_CN_DIGITS.get(ch, 0)) for ch in text) or 0)
    total = section = 0
    digit = None
    last_unit = 1
    zero_seen = False
    for ch in text:
        if ch in _CN_DIGITS:
            digit = _CN_DIGITS[ch]
            zero_seen = zero_seen or digit == 0
        elif ch in _CN_UNITS:
            unit = _CN_UNITS[ch]
            section += (1 if digit is None else digit) * unit
            digit, last_unit, zero_seen = None, unit, False
        elif ch == "万":
            total += (section + (digit or 0)) * 10000
            section, digit, last_unit, zero_seen = 0, None, 10000, False
    if digit:
        section += digit * (1 if zero_seen or last_unit == 10 else last_unit // 10)
    return float(total + section)


def extract_amounts(text: str) -> List[float]:
    """抽出文本里所有带人民币标记的金额。"""
    found: List[float] = []
    for pattern in _MONEY_PATTERNS:
        for m in pattern.finditer(text):
            found.append(_to_float(m.group(1), m.group(2)))
    for pattern in _CN_MONEY:
        for m in pattern.finditer(text):
            found.append(round(cn_to_number(m.group(1)), 2))
    for m in _DIGIT_QIAN_MONEY.finditer(text):
        found.append(round(float(m.group(1)) * 1000, 2))
    return found


class AmountGrounding:
    """本轮有依据的数字集合。"""

    def __init__(self) -> None:
        self._values: Set[float] = set()

    def add_text(self, text: Any) -> None:
        if not text:
            return
        for m in _ANY_NUMBER.finditer(str(text)):
            value = float(m.group(1).replace(",", ""))
            self._values.add(round(value, 2))
            if m.group(2):
                self._values.add(round(value * 10000, 2))
        # 用户说"预算三千"，模型回"3,000 元"也算有依据
        for m in _CN_ANY.finditer(str(text)):
            self._values.add(round(cn_to_number(m.group(0)), 2))
        for m in _DIGIT_QIAN_MONEY.finditer(str(text)):
            self._values.add(round(float(m.group(1)) * 1000, 2))

    def add_json(self, obj: Any) -> None:
        self.add_text(json.dumps(obj, ensure_ascii=False, default=str))

    def add_many(self, texts: Iterable[Any]) -> None:
        for text in texts:
            self.add_text(text)

    def has(self, value: float) -> bool:
        return round(value, 2) in self._values


def guard_mode() -> str:
    mode = os.getenv("GOEUROOPS_AMOUNT_GUARD", "enforce").strip().lower()
    return mode if mode in ("enforce", "warn", "off") else "enforce"


@dataclass
class Violation:
    amounts: List[float]
    sentence: str

    def to_dict(self) -> Dict[str, Any]:
        return {"amounts": self.amounts, "sentence": self.sentence[:120]}


class AmountGuard:
    """检查一段回复里的金额；既能处理完整文本，也能处理流式的片段。"""

    def __init__(self, grounding: AmountGrounding, mode: str | None = None) -> None:
        self.grounding = grounding
        self.mode = mode or guard_mode()
        self.violations: List[Violation] = []
        self._seen: Set[str] = set()
        self._buffer = ""

    # ── 单句 ──
    def _check(self, sentence: str) -> str:
        if self.mode == "off":
            return sentence
        bad = [v for v in extract_amounts(sentence) if not self.grounding.has(v)]
        if not bad:
            return sentence
        if sentence not in self._seen:          # 流式放行和最终文本会检查同一句，只记一次
            self._seen.add(sentence)
            self.violations.append(Violation(sorted(set(bad)), sentence.strip()))
            action = "replaced" if self.mode == "enforce" else "warned"
            logger.warning("金额护栏（%s）：回复里的金额 %s 没有依据：%s", action, sorted(set(bad)), sentence.strip()[:80])
            try:
                from core.metrics import AMOUNT_GUARD
                AMOUNT_GUARD.labels(action=action).inc()
            except Exception:                   # noqa: BLE001
                logger.debug("金额护栏指标记录失败", exc_info=True)
        if self.mode == "warn":
            return sentence
        tail = sentence[len(sentence.rstrip()):]          # 保留句末的换行，不破坏排版
        return NOTICE + tail

    @staticmethod
    def _split(text: str) -> List[str]:
        parts: List[str] = []
        start = 0
        for m in _SENTENCE_END.finditer(text):
            parts.append(text[start:m.end()])
            start = m.end()
        if start < len(text):
            parts.append(text[start:])
        return parts

    # ── 完整文本 ──
    def sanitize(self, text: str) -> str:
        if self.mode == "off" or not text:
            return text
        return "".join(self._check(part) for part in self._split(text))

    # ── 流式 ──
    def feed(self, chunk: str) -> str:
        """收到一个文本片段，返回现在可以发给用户的部分（整句确认过才放行）。"""
        if self.mode == "off":
            return chunk
        self._buffer += chunk
        ends = [m.end() for m in _SENTENCE_END.finditer(self._buffer)]
        cut = ends[-1] if ends else 0
        if not cut and len(self._buffer) > _FORCE_FLUSH_CHARS:
            soft = max(self._buffer.rfind(ch) for ch in "，,、 ")
            cut = soft + 1 if soft > 0 else len(self._buffer)
        if not cut:
            return ""
        ready, self._buffer = self._buffer[:cut], self._buffer[cut:]
        return "".join(self._check(part) for part in self._split(ready))

    def flush(self) -> str:
        """一轮文本结束，放行缓冲区里剩下的部分。"""
        rest, self._buffer = self._buffer, ""
        return "".join(self._check(part) for part in self._split(rest)) if rest else ""
