"""
GoEuroOps Skills v2：加载、校验、路由与渐进式披露。

Skill 是一份可热加载的业务规范（话术、流程、边界、升级条件），用来补充
Agent 的 system prompt。设计参照 Anthropic Agent Skills 的"渐进式披露"：

  skills/<skill_id>/
  ├── SKILL.md          front matter + 核心规则，命中后注入 system prompt
  ├── references/*.md   详细资料，只列目录，模型需要时调用 read_skill_reference 读取
  └── evals/cases.json  命中回归用例（该命中 / 不该命中的句子）

与官方做法的区别：本项目的 Agent 是固定的业务角色，对延迟敏感，所以
"是否注入"由确定性的多信号路由决定（意图绑定 + 关键词 + 语义样例 +
会话粘性），不额外花一轮 LLM 调用让模型自己挑 Skill。
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import yaml

from core.text_embedding import ngram_profile, sparse_cosine

logger = logging.getLogger(__name__)

VALID_AGENTS = {"general", "consulting", "billing", "escalation"}
VALID_MODES = {"always", "auto"}

# 路由权重：意图绑定是最强信号（复用已经算好的三路融合意图），
# 单个关键词命中即可触发，语义样例用于覆盖关键词表之外的说法。
W_INTENT = 0.5
W_INTENT_GROUP = 0.25
W_KEYWORD_FIRST = 0.35
W_KEYWORD_EACH = 0.05
W_KEYWORD_MAX = 0.45
W_SEMANTIC = 0.4
SEMANTIC_FULL_AT = 0.6      # 相似度达到该值时语义分拿满
SEMANTIC_MIN = 0.15         # 低于该值视为噪声

PER_SKILL_CHARS = 3200
REFERENCE_MAX_CHARS = 6000
_REFERENCE_NAME = re.compile(r"^[A-Za-z0-9_\-.]+\.md$")


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


@dataclass
class SkillReference:
    file: str
    title: str
    chars: int
    path: str

    def to_dict(self) -> Dict[str, Any]:
        return {"file": self.file, "title": self.title, "chars": self.chars}


@dataclass
class Skill:
    """单个 Skill 的标准化表示。"""
    id: str
    name: str
    description: str
    content: str
    path: str
    version: str = "1.0.0"
    owner: str = ""
    updated_at: str = ""
    keywords: List[str] = field(default_factory=list)
    agents: List[str] = field(default_factory=list)
    intents: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    mode: str = "auto"
    priority: int = 50
    enabled: bool = True
    references: List[SkillReference] = field(default_factory=list)
    eval_cases: List[Dict[str, Any]] = field(default_factory=list)
    content_hash: str = ""
    loaded_at: str = ""
    warnings: List[str] = field(default_factory=list)
    _profiles: List[Counter] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        self.content_hash = hashlib.sha256(self.content.encode("utf-8")).hexdigest()[:12]
        self.loaded_at = datetime.now(timezone.utc).isoformat()
        texts = list(self.examples) + ([self.description] if self.description else [])
        self._profiles = [ngram_profile(text) for text in texts if text.strip()]

    def applies_to(self, agent_type: Optional[str]) -> bool:
        return not self.agents or not agent_type or agent_type.lower() in self.agents

    def keyword_hits(self, text: str) -> List[str]:
        lowered = (text or "").lower()
        hits = []
        for keyword in self.keywords:
            kw = keyword.lower()
            if kw.isascii():
                # 纯 ASCII 关键词按单词边界匹配："cs" 不应命中 "docs"
                if re.search(rf"(?<![a-z0-9]){re.escape(kw)}(?![a-z0-9])", lowered):
                    hits.append(keyword)
            elif kw in lowered:
                hits.append(keyword)
        return hits

    def semantic_similarity(self, text: str) -> float:
        profile = ngram_profile(text)
        return max((sparse_cosine(profile, p) for p in self._profiles), default=0.0)

    def to_prompt_block(self, max_chars: int = PER_SKILL_CHARS) -> str:
        body = self.content.strip()
        if len(body) > max_chars:
            body = body[:max_chars].rstrip() + "\n...（已按预算截断）"
        lines = [f"### {self.name}（v{self.version}）"]
        if self.description:
            lines.append(f"说明: {self.description}")
        lines.append(body)
        if self.references:
            lines.append("\n可按需查阅的参考资料（调用 read_skill_reference 读取，skill 参数填 "
                         f"`{self.id}`）：")
            lines.extend(f"- {ref.file}：{ref.title}" for ref in self.references)
        return "\n".join(lines)

    def to_summary(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "owner": self.owner,
            "updated_at": self.updated_at,
            "path": self.path,
            "keywords": self.keywords,
            "agents": self.agents,
            "intents": self.intents,
            "examples": self.examples,
            "mode": self.mode,
            "priority": self.priority,
            "enabled": self.enabled,
            "content_chars": len(self.content),
            "content_hash": self.content_hash,
            "loaded_at": self.loaded_at,
            "references": [ref.to_dict() for ref in self.references],
            "eval_cases": len(self.eval_cases),
            "warnings": self.warnings,
        }


@dataclass
class SkillMatch:
    skill: Skill
    score: float
    signals: Dict[str, float]
    reasons: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.skill.id,
            "name": self.skill.name,
            "version": self.skill.version,
            "score": round(self.score, 3),
            "signals": {k: round(v, 3) for k, v in self.signals.items()},
            "reasons": self.reasons,
        }


@dataclass
class SkillSelection:
    matches: List[SkillMatch] = field(default_factory=list)
    prompt: str = ""
    skipped_for_budget: List[str] = field(default_factory=list)

    @property
    def skill_ids(self) -> List[str]:
        return [m.skill.id for m in self.matches]

    @property
    def has_references(self) -> bool:
        return any(m.skill.references for m in self.matches)

    def applied(self) -> List[Dict[str, Any]]:
        return [m.to_dict() for m in self.matches]


class SkillManager:
    """发现、校验、路由并统计 Skills。"""

    def __init__(
        self,
        root_dir: str,
        max_prompt_chars: int = 6000,
        threshold: Optional[float] = None,
    ):
        self.root_dir = Path(root_dir).expanduser().resolve()
        self.max_prompt_chars = max_prompt_chars
        self.threshold = threshold if threshold is not None else _env_float("GOEUROOPS_SKILL_MATCH_THRESHOLD", 0.35)
        self._skills: List[Skill] = []
        self._errors: List[str] = []
        self._stats: Dict[str, Dict[str, Any]] = {}
        self._loaded_at: str = ""

    # ── 加载 ──────────────────────────────────────────────────────────────────

    @property
    def skills(self) -> List[Skill]:
        return list(self._skills)

    @property
    def errors(self) -> List[str]:
        return list(self._errors)

    def get(self, skill_id: str) -> Optional[Skill]:
        return next((s for s in self._skills if s.id == skill_id), None)

    def load(self) -> List[Skill]:
        """重新扫描目录。单个 Skill 校验失败只会跳过它自己。"""
        loaded: List[Skill] = []
        errors: List[str] = []

        if not self.root_dir.exists():
            logger.info("Skill 目录不存在，跳过加载: %s", self.root_dir)
            self._skills, self._errors = [], []
            return []

        seen_ids: set = set()
        seen_names: set = set()
        for path in self._discover_files(self.root_dir):
            try:
                skill = self._load_file(path)
                if skill is None:
                    continue
                problems = self._validate(skill)
                if skill.id in seen_ids:
                    problems.append(f"Skill id 重复: {skill.id}")
                if skill.name in seen_names:
                    problems.append(f"Skill name 重复: {skill.name}")
                if problems:
                    raise ValueError("；".join(problems))
                seen_ids.add(skill.id)
                seen_names.add(skill.name)
                loaded.append(skill)
            except Exception as ex:
                msg = f"{path}: {ex}"
                errors.append(msg)
                logger.warning("Skill 加载失败: %s", msg)

        self._skills = loaded
        self._errors = errors
        self._loaded_at = datetime.now(timezone.utc).isoformat()
        # 统计按 id 保留，热加载不清零
        for skill in loaded:
            self._stats.setdefault(skill.id, {"hits": 0, "last_hit_at": None, "by_agent": {}})
        self._log_loaded_skills()
        return self.skills

    def reload(self) -> List[Skill]:
        return self.load()

    def _validate(self, skill: Skill) -> List[str]:
        problems: List[str] = []
        if not skill.name.strip():
            problems.append("缺少 name")
        if not skill.description.strip():
            problems.append("缺少 description")
        unknown_agents = sorted(set(skill.agents) - VALID_AGENTS)
        if unknown_agents:
            problems.append(f"未知 agents: {', '.join(unknown_agents)}（可选 {', '.join(sorted(VALID_AGENTS))}）")
        valid_intents = _valid_intents()
        unknown_intents = sorted(set(skill.intents) - valid_intents) if valid_intents else []
        if unknown_intents:
            problems.append(f"未知 intents: {', '.join(unknown_intents)}")
        if skill.mode not in VALID_MODES:
            problems.append(f"mode 必须是 {', '.join(sorted(VALID_MODES))} 之一")
        if skill.mode == "auto" and not (skill.intents or skill.keywords or skill.examples):
            problems.append("mode=auto 时至少需要配置 intents / keywords / examples 之一，否则永远不会命中")
        if len(skill.content) > PER_SKILL_CHARS:
            skill.warnings.append(
                f"正文 {len(skill.content)} 字超过单 Skill 预算 {PER_SKILL_CHARS}，超出部分注入时会被截断；"
                "建议把细节移到 references/"
            )
        return problems

    def _discover_files(self, root_dir: Path) -> Iterable[Path]:
        """优先读取目录规范 SKILL.md；兼容根目录下的单文件 Skill（.md/.txt/.json）。"""
        skill_md_files = sorted(root_dir.rglob("SKILL.md"))
        yield from skill_md_files

        for path in sorted(root_dir.iterdir()):
            if not path.is_file() or path.name.startswith("."):
                continue
            if path.name.upper() == "README.MD":
                continue
            if path.suffix.lower() in {".md", ".txt", ".json"}:
                yield path

    def _load_file(self, path: Path) -> Optional[Skill]:
        if path.suffix.lower() == ".json":
            raw = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError("JSON Skill 必须是对象格式")
            body = str(raw.get("content") or raw.get("instructions") or "").strip()
            if not body:
                raise ValueError("缺少 content 或 instructions")
            return self._build_skill(path, raw, body, skill_id=path.stem)

        meta, body = self._split_front_matter(path.read_text(encoding="utf-8"))
        body = body.strip()
        if not body:
            return None
        skill_id = path.parent.name if path.name == "SKILL.md" else path.stem
        return self._build_skill(path, meta, body, skill_id=skill_id)

    def _build_skill(self, path: Path, meta: Dict[str, Any], body: str, skill_id: str) -> Skill:
        name = str(meta.get("name") or self._first_heading(body) or skill_id)
        body = self._strip_first_heading(body, name)
        skill_dir = path.parent if path.name == "SKILL.md" else None
        try:
            priority = int(meta.get("priority", 50))
        except (TypeError, ValueError):
            priority = 50
        return Skill(
            id=str(meta.get("id") or skill_id),
            name=name,
            description=str(meta.get("description") or ""),
            content=body,
            path=str(path),
            version=str(meta.get("version") or "1.0.0"),
            owner=str(meta.get("owner") or ""),
            updated_at=str(meta.get("updated_at") or ""),
            keywords=self._as_list(meta.get("keywords")),
            agents=[a.lower() for a in self._as_list(meta.get("agents"))],
            intents=[i.lower() for i in self._as_list(meta.get("intents"))],
            examples=self._as_list(meta.get("examples"), split=False),
            mode=str(meta.get("mode") or "auto").lower(),
            priority=priority,
            enabled=self._as_bool(meta.get("enabled"), default=True),
            references=self._load_references(skill_dir) if skill_dir else [],
            eval_cases=self._load_eval_cases(skill_dir) if skill_dir else [],
        )

    @staticmethod
    def _load_references(skill_dir: Path) -> List[SkillReference]:
        ref_dir = skill_dir / "references"
        if not ref_dir.is_dir():
            return []
        refs = []
        for path in sorted(ref_dir.glob("*.md")):
            text = path.read_text(encoding="utf-8")
            title = next(
                (line.lstrip("#").strip() for line in text.splitlines() if line.strip()),
                path.stem,
            )
            refs.append(SkillReference(file=path.name, title=title[:80], chars=len(text), path=str(path)))
        return refs

    @staticmethod
    def _load_eval_cases(skill_dir: Path) -> List[Dict[str, Any]]:
        path = skill_dir / "evals" / "cases.json"
        if not path.is_file():
            return []
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"{path} 必须是用例数组")
        return data

    # ── 路由 ──────────────────────────────────────────────────────────────────

    def score(
        self,
        skill: Skill,
        message: str,
        *,
        intent: Optional[str] = None,
        intent_group: Optional[str] = None,
        previous_message: str = "",
    ) -> SkillMatch:
        """计算单个 Skill 对当前请求的得分（不含 Agent 过滤）。"""
        if skill.mode == "always":
            return SkillMatch(skill, 1.0, {"always": 1.0}, ["常驻 Skill（mode=always）"])

        signals: Dict[str, float] = {}
        reasons: List[str] = []
        # 具体意图命中给满分；只命中意图大类（如 service_inquiry 属于 study_consult 组）给一半，
        # 单独不足以触发，避免同组的其他 Skill 被顺带注入。
        if intent and intent in skill.intents:
            signals["intent"] = W_INTENT
            reasons.append(f"意图 {intent} 已绑定")
        elif intent_group and intent_group != intent and intent_group in skill.intents:
            signals["intent_group"] = W_INTENT_GROUP
            reasons.append(f"意图大类 {intent_group} 已绑定")

        text_signals, text_reasons = self._text_signals(skill, message)
        signals.update(text_signals)
        reasons.extend(text_reasons)

        score = sum(signals.values())
        if score < self.threshold and previous_message:
            # 会话粘性：上一轮用户消息单独就能命中时，本轮追问（"那奖学金呢"）继承命中。
            # 只回看一轮，话题切换两轮后自然失效。
            prev_signals, _ = self._text_signals(skill, previous_message)
            if sum(prev_signals.values()) >= self.threshold:
                sticky = self.threshold - score
                signals["sticky"] = sticky
                reasons.append("上一轮对话命中该 Skill，本轮继承（会话粘性）")
                score += sticky
        return SkillMatch(skill, score, signals, reasons)

    @staticmethod
    def _text_signals(skill: Skill, text: str) -> tuple[Dict[str, float], List[str]]:
        signals: Dict[str, float] = {}
        reasons: List[str] = []
        hits = skill.keyword_hits(text)
        if hits:
            signals["keywords"] = min(W_KEYWORD_MAX, W_KEYWORD_FIRST + W_KEYWORD_EACH * (len(hits) - 1))
            reasons.append(f"关键词: {', '.join(hits[:6])}")
        sim = skill.semantic_similarity(text)
        if sim >= SEMANTIC_MIN:
            signals["semantic"] = W_SEMANTIC * min(1.0, sim / SEMANTIC_FULL_AT)
            reasons.append(f"语义相似度 {sim:.2f}")
        return signals, reasons

    def select(
        self,
        message: str,
        agent_type: Optional[str] = None,
        *,
        intent: Optional[str] = None,
        intent_group: Optional[str] = None,
        history: Optional[Sequence[Dict[str, str]]] = None,
    ) -> SkillSelection:
        """为一次 Agent 调用挑选 Skill 并拼好 prompt。"""
        previous = _previous_user_message(history, message)
        matched: List[SkillMatch] = []
        for skill in self._skills:
            if not skill.enabled or not skill.applies_to(agent_type):
                continue
            match = self.score(skill, message, intent=intent, intent_group=intent_group, previous_message=previous)
            if match.score >= self.threshold:
                matched.append(match)

        matched.sort(key=lambda m: (m.skill.mode != "always", -m.skill.priority, -m.score))

        selection = SkillSelection()
        blocks: List[str] = []
        remaining = self.max_prompt_chars
        for match in matched:
            block = match.skill.to_prompt_block()
            if len(block) > remaining:
                if remaining < 400:
                    selection.skipped_for_budget.append(match.skill.id)
                    continue
                block = block[:remaining].rstrip() + "\n...（已按总预算截断）"
            blocks.append(block)
            selection.matches.append(match)
            remaining -= len(block)

        if blocks:
            selection.prompt = (
                "以下是当前请求命中的 GoEuroOps Skills（业务规范）。请优先遵循这些规则；"
                "如果与系统角色或安全边界冲突，以系统角色和安全边界为准。\n\n"
                + "\n\n".join(blocks)
            )
            logger.info(
                "Skills 已注入: agent=%s matched=%s message=%r",
                agent_type or "all",
                "; ".join(f"{m.skill.id}({m.score:.2f})" for m in selection.matches),
                (message or "")[:80],
            )
        return selection

    def prompt_for(self, message: str, agent_type: Optional[str] = None) -> str:
        """兼容旧接口：只根据消息文本选择 Skill。"""
        return self.select(message, agent_type).prompt

    def match_report(
        self,
        message: str,
        agent_type: Optional[str] = None,
        *,
        intent: Optional[str] = None,
        intent_group: Optional[str] = None,
        history: Optional[Sequence[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """干跑：列出每个 Skill 的得分明细，供 /skills/match 和前端命中测试使用。"""
        previous = _previous_user_message(history, message)
        rows = []
        for skill in self._skills:
            match = self.score(skill, message, intent=intent, intent_group=intent_group, previous_message=previous)
            if not skill.enabled:
                status = "disabled"
            elif not skill.applies_to(agent_type):
                status = "agent_filtered"
            elif match.score >= self.threshold:
                status = "matched"
            else:
                status = "below_threshold"
            rows.append({**match.to_dict(), "status": status, "agents": skill.agents, "mode": skill.mode})
        rows.sort(key=lambda r: (r["status"] != "matched", -r["score"]))
        selection = self.select(message, agent_type, intent=intent, intent_group=intent_group, history=history)
        return {
            "message": message,
            "agent_type": agent_type,
            "intent": intent,
            "intent_group": intent_group,
            "threshold": self.threshold,
            "results": rows,
            "injected": selection.skill_ids,
            "prompt_chars": len(selection.prompt),
            "skipped_for_budget": selection.skipped_for_budget,
        }

    # ── 渐进式披露 ────────────────────────────────────────────────────────────

    def read_reference(
        self,
        skill_id: str,
        file: str,
        allowed_skill_ids: Optional[Iterable[str]] = None,
    ) -> Dict[str, Any]:
        """读取 Skill 的参考资料。只允许读取本轮已命中 Skill 的已登记文件。"""
        if allowed_skill_ids is not None and skill_id not in set(allowed_skill_ids):
            return {"success": False, "error": f"Skill {skill_id} 未在本轮命中，不能读取其参考资料"}
        skill = self.get(skill_id)
        if skill is None:
            return {"success": False, "error": f"未知 Skill: {skill_id}"}
        if not _REFERENCE_NAME.match(file or ""):
            return {"success": False, "error": "文件名不合法"}
        ref = next((r for r in skill.references if r.file == file), None)
        if ref is None:
            return {
                "success": False,
                "error": f"{skill_id} 没有参考资料 {file}",
                "available": [r.file for r in skill.references],
            }
        text = Path(ref.path).read_text(encoding="utf-8")
        return {
            "success": True,
            "skill": skill_id,
            "file": file,
            "content": text[:REFERENCE_MAX_CHARS],
            "truncated": len(text) > REFERENCE_MAX_CHARS,
        }

    # ── 统计 / 摘要 ───────────────────────────────────────────────────────────

    def record(self, selection: SkillSelection, agent_type: str) -> None:
        if not selection.matches:
            return
        from core.metrics import SKILL_HITS

        now = datetime.now(timezone.utc).isoformat()
        for match in selection.matches:
            stat = self._stats.setdefault(match.skill.id, {"hits": 0, "last_hit_at": None, "by_agent": {}})
            stat["hits"] += 1
            stat["last_hit_at"] = now
            stat["by_agent"][agent_type] = stat["by_agent"].get(agent_type, 0) + 1
            SKILL_HITS.labels(skill=match.skill.id, agent=agent_type).inc()

    def detail(self, skill_id: str) -> Optional[Dict[str, Any]]:
        skill = self.get(skill_id)
        if skill is None:
            return None
        return {**skill.to_summary(), "content": skill.content, "stats": self._stats.get(skill.id, {})}

    def summary(self) -> Dict[str, Any]:
        empty = {"hits": 0, "last_hit_at": None, "by_agent": {}}
        return {
            "root_dir": str(self.root_dir),
            "count": len(self._skills),
            "loaded_at": self._loaded_at,
            "threshold": self.threshold,
            "max_prompt_chars": self.max_prompt_chars,
            "skills": [
                {**skill.to_summary(), "stats": self._stats.get(skill.id, empty)}
                for skill in self._skills
            ],
            "errors": self.errors,
        }

    def eval_cases(self) -> List[Dict[str, Any]]:
        """汇总各 Skill 的命中回归用例，附上所属 Skill id。"""
        return [{"skill": skill.id, **case} for skill in self._skills for case in skill.eval_cases]

    def run_evals(self) -> Dict[str, Any]:
        """跑所有 Skill 的命中用例，返回准确率和失败明细（评测面板使用）。"""
        results = []
        for case in self.eval_cases():
            selection = self.select(
                case["message"],
                case.get("agent"),
                intent=case.get("intent"),
                intent_group=case.get("intent_group"),
                history=case.get("history"),
            )
            hit = case["skill"] in selection.skill_ids
            expected = bool(case.get("expect_hit", True))
            results.append({**case, "hit": hit, "passed": hit == expected})
        total = len(results)
        passed = sum(1 for r in results if r["passed"])
        return {
            "total": total,
            "passed": passed,
            "accuracy": round(passed / total, 4) if total else 0.0,
            "failures": [r for r in results if not r["passed"]],
        }

    def _log_loaded_skills(self) -> None:
        lines = [
            "",
            "================ GoEuroOps Skills Loaded ================",
            f"目录: {self.root_dir}",
            f"数量: {len(self._skills)}  阈值: {self.threshold}",
        ]
        for index, skill in enumerate(self._skills, start=1):
            lines.extend([
                f"{index}. {skill.name}  [{skill.id} v{skill.version}, mode={skill.mode}, hash={skill.content_hash}]",
                f"   agents: {', '.join(skill.agents) or 'all'}  intents: {', '.join(skill.intents) or '-'}",
                f"   keywords: {len(skill.keywords)}  examples: {len(skill.examples)}  "
                f"references: {len(skill.references)}  evals: {len(skill.eval_cases)}",
            ])
            lines.extend(f"   ⚠ {w}" for w in skill.warnings)
        if not self._skills:
            lines.append("未加载任何 Skill。")
        if self._errors:
            lines.append("解析错误:")
            lines.extend(f"  - {error}" for error in self._errors)
        lines.append("========================================================")
        logger.info("\n".join(lines))

    # ── 解析辅助 ──────────────────────────────────────────────────────────────

    @staticmethod
    def _split_front_matter(raw: str) -> tuple[Dict[str, Any], str]:
        """解析 Markdown 顶部 YAML front matter；兼容旧版 `key: a,b,c` 写法。"""
        text = raw.lstrip()
        if not text.startswith("---"):
            return {}, raw
        lines = text.splitlines()
        if not lines or lines[0].strip() != "---":
            return {}, raw
        end_idx = next((i for i, line in enumerate(lines[1:], start=1) if line.strip() == "---"), None)
        if end_idx is None:
            return {}, raw
        try:
            meta = yaml.safe_load("\n".join(lines[1:end_idx])) or {}
        except yaml.YAMLError as ex:
            raise ValueError(f"front matter 不是合法 YAML: {ex}") from ex
        if not isinstance(meta, dict):
            raise ValueError("front matter 必须是键值对")
        return meta, "\n".join(lines[end_idx + 1:])

    @staticmethod
    def _first_heading(body: str) -> Optional[str]:
        for line in body.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                return stripped.lstrip("#").strip() or None
        return None

    @staticmethod
    def _strip_first_heading(body: str, name: str) -> str:
        lines = body.splitlines()
        if lines and lines[0].strip().startswith("#") and lines[0].strip().lstrip("#").strip() == name:
            return "\n".join(lines[1:]).strip()
        return body

    @staticmethod
    def _as_list(value: Any, split: bool = True) -> List[str]:
        """列表原样返回；字符串按中英文逗号切分（examples 这类整句字段不切分）。"""
        if value is None or value == "":
            return []
        if isinstance(value, (list, tuple)):
            return [str(item).strip() for item in value if str(item).strip()]
        if not split:
            return [str(value).strip()]
        return [item.strip() for item in re.split(r"[,，、]", str(value)) if item.strip()]

    @staticmethod
    def _as_bool(value: Any, default: bool = False) -> bool:
        if value is None or value == "":
            return default
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() not in {"0", "false", "no", "off", "disabled"}


def _previous_user_message(history: Optional[Sequence[Dict[str, str]]], current: str) -> str:
    if not history:
        return ""
    for item in reversed(list(history)):
        if item.get("role") == "user":
            content = str(item.get("content", ""))
            if content.strip() and content.strip() != (current or "").strip():
                return content
    return ""


def _valid_intents() -> set:
    try:
        from core.intent_recognizer import IntentCategory
    except Exception:  # pragma: no cover - 意图模块不可用时不做意图校验
        return set()
    return {c.value for c in IntentCategory}
