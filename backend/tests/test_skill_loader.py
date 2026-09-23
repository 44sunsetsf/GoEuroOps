from pathlib import Path

import pytest

from core.skill_loader import PER_SKILL_CHARS, SkillManager

SKILLS_DIR = Path(__file__).resolve().parent.parent / "skills"


def write_skill(root: Path, skill_id: str, front: str, body: str = "规则正文", refs=None) -> Path:
    skill_dir = root / skill_id
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(f"---\n{front}\n---\n{body}", encoding="utf-8")
    for name, text in (refs or {}).items():
        (skill_dir / "references").mkdir(exist_ok=True)
        (skill_dir / "references" / name).write_text(text, encoding="utf-8")
    return skill_dir


def load(root: Path, **kwargs) -> SkillManager:
    manager = SkillManager(str(root), **kwargs)
    manager.load()
    return manager


# ── 解析与校验 ──────────────────────────────────────────────────────────────

def test_yaml_lists_and_legacy_comma_strings_both_parse(tmp_path):
    write_skill(tmp_path, "a", "name: A\ndescription: d\nagents: [consulting]\nkeywords: [瑞典, 德国]")
    write_skill(tmp_path, "b", "name: B\ndescription: d\nagents: billing\nkeywords: 退款，发票、invoice,定金")
    manager = load(tmp_path)

    a, b = manager.get("a"), manager.get("b")
    assert a.keywords == ["瑞典", "德国"]
    # 中文逗号、顿号、英文逗号都能切分（旧版只认英文逗号）
    assert b.keywords == ["退款", "发票", "invoice", "定金"]
    assert b.agents == ["billing"]


def test_invalid_skills_are_skipped_with_errors(tmp_path):
    write_skill(tmp_path, "ok", "name: OK\ndescription: d\nkeywords: [x]")
    write_skill(tmp_path, "bad_agent", "name: Bad\ndescription: d\nagents: [technical]\nkeywords: [x]")
    write_skill(tmp_path, "bad_intent", "name: Bad2\ndescription: d\nintents: [order_status]")
    write_skill(tmp_path, "never_hits", "name: Never\ndescription: d\nmode: auto")
    write_skill(tmp_path, "no_desc", "name: NoDesc\nkeywords: [x]")
    manager = load(tmp_path)

    assert [s.id for s in manager.skills] == ["ok"]
    joined = "\n".join(manager.errors)
    assert "未知 agents: technical" in joined
    assert "未知 intents: order_status" in joined
    assert "永远不会命中" in joined
    assert "缺少 description" in joined


def test_duplicate_names_rejected(tmp_path):
    write_skill(tmp_path, "a", "name: 同名\ndescription: d\nkeywords: [x]")
    write_skill(tmp_path, "b", "name: 同名\ndescription: d\nkeywords: [y]")
    manager = load(tmp_path)
    assert len(manager.skills) == 1
    assert any("重复" in e for e in manager.errors)


def test_oversized_body_warns_and_is_truncated(tmp_path):
    write_skill(tmp_path, "big", "name: Big\ndescription: d\nmode: always", body="规" * (PER_SKILL_CHARS + 500))
    manager = load(tmp_path)
    skill = manager.get("big")
    assert skill.warnings
    assert "已按预算截断" in skill.to_prompt_block()


def test_content_hash_changes_on_edit(tmp_path):
    skill_dir = write_skill(tmp_path, "a", "name: A\ndescription: d\nkeywords: [x]", body="v1")
    manager = load(tmp_path)
    first = manager.get("a").content_hash
    (skill_dir / "SKILL.md").write_text("---\nname: A\ndescription: d\nkeywords: [x]\n---\nv2", encoding="utf-8")
    manager.reload()
    assert manager.get("a").content_hash != first


# ── 路由信号 ────────────────────────────────────────────────────────────────

def test_intent_binding_alone_triggers(tmp_path):
    write_skill(tmp_path, "s", "name: S\ndescription: 退款规范\nintents: [refund]")
    manager = load(tmp_path)
    selection = manager.select("能退吗", "billing", intent="refund")
    assert selection.skill_ids == ["s"]
    assert selection.matches[0].signals == {"intent": 0.5}


def test_ascii_keywords_match_on_word_boundary(tmp_path):
    write_skill(tmp_path, "s", "name: S\ndescription: d\nkeywords: [cs, APS]")
    manager = load(tmp_path)
    assert manager.select("docs 在哪", "general").skill_ids == []
    assert manager.select("maps 怎么用", "general").skill_ids == []
    assert manager.select("CS 硕士", "general").skill_ids == ["s"]
    assert manager.select("aps要多久", "general").skill_ids == ["s"]


def test_semantic_examples_cover_unlisted_phrasings(tmp_path):
    write_skill(
        tmp_path, "s",
        "name: S\ndescription: 留学咨询\nexamples: [想去北欧读计算机研究生]\nkeywords: [雅思]",
    )
    manager = load(tmp_path)
    selection = manager.select("我想去北欧读计算机方向的研究生", "consulting")
    assert selection.skill_ids == ["s"]
    assert "semantic" in selection.matches[0].signals


def test_agent_filter_is_hard(tmp_path):
    write_skill(tmp_path, "s", "name: S\ndescription: d\nagents: [billing]\nmode: always")
    manager = load(tmp_path)
    assert manager.select("任何消息", "consulting").skill_ids == []
    assert manager.select("任何消息", "billing").skill_ids == ["s"]


def test_sticky_previous_turn_keeps_skill(tmp_path):
    write_skill(tmp_path, "s", "name: S\ndescription: d\nkeywords: [学费, 瑞典]")
    manager = load(tmp_path)
    history = [{"role": "user", "content": "瑞典的学费多少"}, {"role": "assistant", "content": "..."}]

    assert manager.select("那奖学金呢", "consulting").skill_ids == []
    selection = manager.select("那奖学金呢", "consulting", history=history)
    assert selection.skill_ids == ["s"]
    assert "sticky" in selection.matches[0].signals


def test_ordering_budget_and_skip(tmp_path):
    write_skill(tmp_path, "always", "name: Always\ndescription: d\nmode: always\npriority: 10", body="A" * 300)
    write_skill(tmp_path, "high", "name: High\ndescription: d\nkeywords: [x]\npriority: 90", body="H" * 300)
    write_skill(tmp_path, "low", "name: Low\ndescription: d\nkeywords: [x]\npriority: 1", body="L" * 300)
    manager = load(tmp_path, max_prompt_chars=750)

    selection = manager.select("x", "general")
    # always 永远在最前，其余按 priority；超出总预算的被跳过
    assert selection.skill_ids == ["always", "high"]
    assert selection.skipped_for_budget == ["low"]


# ── 渐进式披露 ──────────────────────────────────────────────────────────────

def test_reference_catalog_listed_but_not_inlined(tmp_path):
    write_skill(tmp_path, "s", "name: S\ndescription: d\nmode: always",
                refs={"faq.md": "# 常见问题\n很长的正文内容"})
    manager = load(tmp_path)
    prompt = manager.select("hi", "general").prompt
    assert "faq.md：常见问题" in prompt
    assert "很长的正文内容" not in prompt


def test_read_reference_guards(tmp_path):
    write_skill(tmp_path, "s", "name: S\ndescription: d\nmode: always", refs={"faq.md": "# FAQ\n内容"})
    write_skill(tmp_path, "t", "name: T\ndescription: d\nmode: always", refs={"x.md": "# X\n机密"})
    (tmp_path / "secret.md").write_text("不应被读到", encoding="utf-8")
    manager = load(tmp_path)

    assert manager.read_reference("s", "faq.md", allowed_skill_ids=["s"])["content"].endswith("内容")
    assert not manager.read_reference("t", "x.md", allowed_skill_ids=["s"])["success"]
    assert not manager.read_reference("s", "../../secret.md", allowed_skill_ids=["s"])["success"]
    assert not manager.read_reference("s", "missing.md", allowed_skill_ids=["s"])["success"]


def test_record_updates_stats(tmp_path):
    write_skill(tmp_path, "s", "name: S\ndescription: d\nmode: always")
    manager = load(tmp_path)
    manager.record(manager.select("hi", "general"), "general")
    manager.record(manager.select("hi", "consulting"), "consulting")
    stats = manager.summary()["skills"][0]["stats"]
    assert stats["hits"] == 2
    assert stats["by_agent"] == {"general": 1, "consulting": 1}


def test_match_report_explains_every_skill(tmp_path):
    write_skill(tmp_path, "a", "name: A\ndescription: d\nagents: [billing]\nkeywords: [退款]")
    write_skill(tmp_path, "b", "name: B\ndescription: d\nkeywords: [瑞典]")
    manager = load(tmp_path)
    report = manager.match_report("瑞典退款", "consulting")
    status = {row["id"]: row["status"] for row in report["results"]}
    assert status == {"a": "agent_filtered", "b": "matched"}
    assert report["injected"] == ["b"]


# ── 仓库内真实 Skills 的回归用例 ──────────────────────────────────────────────

_REAL = SkillManager(str(SKILLS_DIR))
_REAL.load()


def test_repository_skills_load_without_errors():
    assert _REAL.errors == []
    assert {s.id for s in _REAL.skills} == {
        "studio_brand_voice", "study_consulting", "service_sales", "billing_support", "general_reception",
    }


@pytest.mark.parametrize("case", _REAL.eval_cases(), ids=lambda c: f"{c['skill']}:{c['message']}")
def test_repository_skill_eval_case(case):
    selection = _REAL.select(
        case["message"], case.get("agent"),
        intent=case.get("intent"), intent_group=case.get("intent_group"), history=case.get("history"),
    )
    assert (case["skill"] in selection.skill_ids) == case.get("expect_hit", True), case.get("note", "")


def test_intent_group_alone_does_not_trigger(tmp_path):
    """service_inquiry 属于 study_consult 组，但不应该把只绑定 study_consult 的 Skill 顺带注入。"""
    write_skill(tmp_path, "s", "name: S\ndescription: 留学答疑\nintents: [study_consult]")
    manager = load(tmp_path)
    assert manager.select("全程陪跑多少钱", "consulting", intent="service_inquiry", intent_group="study_consult").skill_ids == []
    report = manager.match_report("全程陪跑多少钱", "consulting", intent="service_inquiry", intent_group="study_consult")
    assert report["results"][0]["signals"] == {"intent_group": 0.25}
    assert manager.select("瑞典CS", "consulting", intent="study_consult", intent_group="study_consult").skill_ids == ["s"]


def test_composite_detection_ignores_service_names_in_refund_questions():
    from agents.agent_orchestrator import AgentOrchestrator, AgentType, Request
    from core.intent_recognizer import IntentCategory

    orchestrator = AgentOrchestrator.__new__(AgentOrchestrator)
    orchestrator._pool = {t: [object()] for t in (AgentType.GENERAL, AgentType.CONSULTING, AgentType.BILLING)}
    req = Request(message="全程陪跑交了定金，还没开始能退吗", user_id="u", conv_id="c",
                  intent=IntentCategory.REFUND, intent_confidence=0.8)
    decision = orchestrator._route_decision(req)
    assert decision.primary_agent is AgentType.BILLING
    assert decision.supporting_agents == []
