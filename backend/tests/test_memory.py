"""分层记忆：原话笔记（A-Mem 补全、链接、取代、只追加的演化）、客户档案的合并规则、咨询记录、整理时机、读取和字数预算、证件号、隐私删除、失败兜底。"""
import asyncio
import json
from collections import defaultdict
from datetime import datetime, timedelta
from types import SimpleNamespace

from agents.base import Request
from agents.tools import build_handoff_summary
from core import chat_log
from evaluation.memory_benchmark import FakeRedis
from memory.notes import NoteStore
from memory.manager import MemoryContext, MemoryManager, Message, MsgRole, has_self_info
from memory.profile import apply_candidate, render


def _match(meta, where):
    if not where:
        return True
    if "$and" in where:
        return all(_match(meta, w) for w in where["$and"])
    for k, v in where.items():
        if isinstance(v, dict) and "$ne" in v:
            if meta.get(k) == v["$ne"]:
                return False
        elif isinstance(v, dict) and "$lt" in v:
            if not meta.get(k, 0) < v["$lt"]:
                return False
        elif meta.get(k) != v:
            return False
    return True


class FakeCollection:
    """ChromaDB 集合的替身：按写入顺序保存；query 不做语义排序，按 where 过滤后返回前 n 条。"""

    def __init__(self):
        self.rows, self.docs = {}, {}
        self.fail_query = False

    def add(self, ids, documents, metadatas):
        for i, d, m in zip(ids, documents, metadatas):
            self.rows[i], self.docs[i] = dict(m), d

    def get(self, ids=None, where=None, include=None):
        keys = [i for i in (ids or self.rows) if i in self.rows and _match(self.rows[i], where)]
        import numpy as np                          # 真实 ChromaDB 返回的向量是 numpy 数组
        return {"ids": keys, "metadatas": [self.rows[i] for i in keys], "documents": [self.docs[i] for i in keys],
                "embeddings": np.array([[1.0 if self.docs[i].startswith("我本科") else 0.0] for i in keys])}

    def query(self, query_embeddings, n_results, where=None, include=None):
        if self.fail_query:
            raise RuntimeError("Cannot return the results in a contigious 2D array. Probably ef or M is too small")
        keys = [i for i in self.rows if _match(self.rows[i], where)][:n_results]
        return {"ids": [keys], "metadatas": [[self.rows[i] for i in keys]]}

    def update(self, ids, metadatas=None, documents=None):
        for k, i in enumerate(ids):
            if metadatas:
                self.rows[i] = dict(metadatas[k])
            if documents:
                self.docs[i] = documents[k]

    def delete(self, where=None):
        for i in [i for i in self.rows if _match(self.rows[i], where)]:
            self.rows.pop(i)
            self.docs.pop(i)


class FakeEmbedder:
    def embed_query(self, text):
        return [1.0]


class ScriptedClient:
    """整理调用按顺序返回预设回复（可以是看得到提示词的函数）。"""

    def __init__(self, replies):
        self.replies, self.calls, self.prompts = list(replies), 0, []
        owner = self

        class Messages:
            async def create(inner, **kwargs):
                owner.calls += 1
                prompt = kwargs["messages"][-1]["content"]
                owner.prompts.append(prompt)
                r = owner.replies.pop(0) if owner.replies else '{"notes": [], "cases": []}'
                text = r(prompt) if callable(r) else r
                return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)])

        self.messages = Messages()


def _mgr(replies=()):
    mgr = MemoryManager.__new__(MemoryManager)          # 不连真实的 Redis、ChromaDB 和模型
    mgr._redis = FakeRedis()
    mgr._redis_timeout = 1.0
    mgr._client = ScriptedClient(replies)
    mgr._model = "m"
    mgr._notes = NoteStore(FakeCollection(), FakeEmbedder())
    mgr._locks = defaultdict(asyncio.Lock)
    mgr._bg_tasks = set()
    mgr._idle_timers = {}
    return mgr


def _rows(mgr):
    return mgr._notes._col.rows


def _by_text(mgr, text):
    return next((i, m) for i, m in _rows(mgr).items() if m["text"] == text)


def _reply(notes=(), cases=(), updates=()):
    return json.dumps({"notes": list(notes), "cases": list(cases), "neighbor_updates": list(updates)}, ensure_ascii=False)


def _note(m, facts=(), event_time="", **kw):
    base = {"m": m, "keywords": [], "tags": [], "context": "", "event_time": event_time,
            "facts": [{"slot": s, "value": v} for s, v in facts], "links": [], "supersedes": []}
    return {**base, **kw}


async def _say(mgr, uid, conv, *texts):
    for t in texts:
        await mgr.add_message(uid, conv, MsgRole.USER, t)
        await mgr.add_message(uid, conv, MsgRole.ASSISTANT, "好的")


# ── 写入 ──────────────────────────────────────────────────────────────────────

def test_has_self_info_only_for_statements_about_the_user():
    assert has_self_info("我本科是计算机专业")
    assert has_self_info("我雅思 6.5")
    assert not has_self_info("瑞典的学费贵吗")
    assert not has_self_info("我想问一下")


def test_user_statements_are_archived_immediately_without_model_calls():
    mgr = _mgr()
    asyncio.run(_say(mgr, "u1", "c1", "我本科是自动化专业", "好的", "瑞典学费多少？"))
    assert [m["text"] for m in _rows(mgr).values()] == ["我本科是自动化专业"]   # 太短的和纯提问不存
    assert mgr._client.calls == 0


def test_sensitive_ids_are_redacted_before_anything_is_stored():
    mgr = _mgr()
    asyncio.run(_say(mgr, "u1", "c1", "我的护照号是 E12345678，身份证 110101199901011234"))
    stored = json.dumps(_rows(mgr), ensure_ascii=False) + json.dumps(mgr._redis.lists, ensure_ascii=False)
    assert "12345678" not in stored and "110101" not in stored
    assert "[证件号已隐去]" in stored


def test_repeated_sentence_is_one_note_marked_for_reconsolidation():
    mgr = _mgr()
    asyncio.run(_say(mgr, "u1", "c1", "我想去荷兰读书"))
    nid, meta = _by_text(mgr, "我想去荷兰读书")
    _rows(mgr)[nid] = {**meta, "consolidated": "1", "ts": "2020-01-01T00:00:00"}
    asyncio.run(_say(mgr, "u1", "c2", "我想去荷兰读书"))
    assert len(_rows(mgr)) == 1
    meta = _rows(mgr)[nid]
    assert meta["consolidated"] == "0" and meta["conv_id"] == "c2" and meta["ts"] > "2020"


def test_window_shows_recent_messages_but_keeps_more_for_the_case_summary():
    mgr = _mgr()

    async def run():
        for i in range(20):
            await mgr.add_message("u1", "c1", MsgRole.USER, f"消息 {i}")
        return await mgr.get_context("u1", "c1")

    ctx = asyncio.run(run())
    assert [m.content for m in ctx.recent_messages] == [f"消息 {i}" for i in range(6, 20)]
    assert len(mgr._redis.lists["wm:u1:c1"]) == 20


# ── 客户档案的合并规则 ──────────────────────────────────────────────────────────

def test_later_statement_wins_and_old_value_goes_to_history():
    p = {}
    apply_candidate(p, "target_country", "荷兰", "我想去荷兰", "2026-01-01")
    apply_candidate(p, "target_country", "瑞典", "改申瑞典", "2026-02-01")
    assert p["target_country"]["value"] == "瑞典" and p["target_country"]["quote"] == "改申瑞典"
    assert render(p) == ["目标国家：瑞典（之前：荷兰）"]


def test_earlier_statement_processed_late_only_goes_to_history():
    p = {}
    apply_candidate(p, "budget", "每年 25 万", "", "2026-03-01")
    apply_candidate(p, "budget", "每年 20 万", "", "2026-01-01")
    assert p["budget"]["value"] == "每年 25 万"
    assert [h["value"] for h in p["budget"]["history"]] == ["每年 20 万"]


def test_switching_back_keeps_history_without_duplicates():
    p = {}
    for v, d in (("荷兰", "01"), ("瑞典", "02"), ("荷兰", "03")):
        apply_candidate(p, "target_country", v, "", f"2026-{d}-01")
    assert render(p) == ["目标国家：荷兰（之前：瑞典）"]


def test_advisor_value_is_not_overwritten_and_user_change_waits_for_review():
    p = {}
    apply_candidate(p, "gpa", "3.4", "", "2026-01-01")
    apply_candidate(p, "gpa", "3.5", "顾问填写", "2026-02-01", source="advisor")
    apply_candidate(p, "gpa", "3.6", "我 GPA 3.6", "2026-03-01")
    assert p["gpa"]["value"] == "3.5" and p["gpa"]["source"] == "advisor"
    assert render(p) == ["GPA / 均分：3.5（顾问已核实）（之前：3.4）（用户后来说：3.6，待顾问核实）"]
    apply_candidate(p, "gpa", "3.6", "顾问填写", "2026-04-01", source="advisor")
    assert p["gpa"]["value"] == "3.6" and p["gpa"]["pending"] == []


def test_advisor_confirming_the_current_value_locks_it():
    p = {}
    apply_candidate(p, "major", "数学", "", "2026-01-01")
    apply_candidate(p, "major", "数学", "顾问填写", "2026-02-01", source="advisor")
    assert p["major"]["source"] == "advisor"
    assert not apply_candidate(p, "major", "统计", "", "2026-01-15")      # 比顾问核实早的说法不算


def test_list_slots_keep_every_value_and_unknown_slots_are_ignored():
    p = {}
    apply_candidate(p, "experience", "后端实习", "", "2026-01-01")
    apply_candidate(p, "experience", "数学建模省二", "", "2026-01-02")
    apply_candidate(p, "experience", "后端实习", "", "2026-01-03")
    assert not apply_candidate(p, "girlfriend", "哥本哈根", "", "2026-01-04")
    assert render(p) == ["实习、工作或项目经历：后端实习；数学建模省二"]


# ── 整理 ──────────────────────────────────────────────────────────────────────

def test_consolidation_fills_profile_event_time_and_writes_case_for_closed_conversations():
    def reply(prompt):
        assert "当前档案：\n（空）" in prompt and "C1：" in prompt and "用户：我本科是自动化专业，GPA 3.4" in prompt
        return _reply([_note("M1", [("major", "自动化"), ("gpa", "3.4")]),
                       _note("M2", [("language", "雅思 6.0")], event_time="2025-07")],
                      [{"c": "C1", "summary": "了解瑞典申请", "open_items": ["定预算"], "next_step": "周四视频"}])

    mgr = _mgr([reply])

    async def run():
        await _say(mgr, "u1", "c1", "我本科是自动化专业，GPA 3.4", "我去年夏天考了雅思 6.0")
        await mgr.consolidate("u1")
        return await mgr.get_context("u1", "c2", query="雅思")

    ctx = asyncio.run(run())
    assert mgr._client.calls == 1
    assert ctx.facts == ["本科专业：自动化", "GPA / 均分：3.4", "语言考试成绩：雅思 6.0"]
    assert ctx.cases[0].endswith("了解瑞典申请；还没解决：定预算；约好的下一步：周四视频")
    assert _by_text(mgr, "我去年夏天考了雅思 6.0")[1]["event_time"] == "2025-07"
    assert all(m["consolidated"] == "1" for m in _rows(mgr).values())
    assert mgr._redis.sets["mem:open:u1"] == set()


def test_case_is_updated_from_previous_summary_with_only_new_messages():
    prompts = []

    def first(prompt):
        return _reply(cases=[{"c": "C1", "summary": "第一次", "open_items": [], "next_step": ""}])

    def second(prompt):
        prompts.append(prompt)
        return _reply(cases=[{"c": "C1", "summary": "第二次", "open_items": [], "next_step": ""}])

    mgr = _mgr([first, second])

    async def run():
        await mgr.add_message("u1", "c1", MsgRole.USER, "旧的话")
        await mgr.consolidate("u1")
        await mgr.consolidate("u1")                     # 没有新消息：不调模型
        await mgr.add_message("u1", "c1", MsgRole.USER, "新的话")
        await mgr.consolidate("u1")

    asyncio.run(run())
    assert mgr._client.calls == 2
    assert '"summary": "第一次"' in prompts[0] and "新的话" in prompts[0] and "旧的话" not in prompts[0]
    cases = [json.loads(c) for c in mgr._redis.lists["mem:cases:u1"]]
    assert [c["summary"] for c in cases] == ["第二次"]


def test_after_turn_waits_until_a_batch_is_full():
    mgr = _mgr([_reply()])

    async def run():
        await _say(mgr, "u1", "c1", *[f"我第 {i} 次说自己的情况" for i in range(MemoryManager.BATCH - 1)])
        await mgr.after_turn("u1", "c1")
        before = mgr._client.calls
        await _say(mgr, "u1", "c1", "我最后再补充一句情况")
        await mgr.after_turn("u1", "c1")
        return before

    assert asyncio.run(run()) == 0
    assert mgr._client.calls == 1
    assert "C1：" not in mgr._client.prompts[0]          # 当前会话还在进行，不写咨询记录


def test_switching_conversation_closes_only_the_stale_one():
    mgr = _mgr([lambda p: _reply(cases=[{"c": "C1", "summary": "旧会话", "open_items": [], "next_step": ""}])])

    async def run():
        await _say(mgr, "u1", "old", "我想去芬兰读书")
        old = (datetime.now() - timedelta(seconds=MemoryManager.IDLE_SECONDS + 5)).isoformat()
        mgr._redis.lists["wm:u1:old"] = [json.dumps({**json.loads(r), "ts": old}) for r in mgr._redis.lists["wm:u1:old"]]
        await _say(mgr, "u1", "new", "继续问问题")
        await mgr.after_turn("u1", "new")

    asyncio.run(run())
    assert mgr._client.calls == 1
    assert json.loads(mgr._redis.lists["mem:cases:u1"][0])["conv_id"] == "old"
    assert mgr._redis.sets["mem:open:u1"] == {"new"}


def test_bad_model_output_keeps_raw_notes_and_open_cases_then_gives_up_on_notes():
    mgr = _mgr(["不是 JSON", "还不是"])

    async def run():
        await _say(mgr, "u1", "c1", "我本科是自动化专业")
        await mgr.consolidate("u1")
        await mgr.consolidate("u1")
        return await mgr.get_context("u1", "c2", query="专业")

    ctx = asyncio.run(run())
    meta = _by_text(mgr, "我本科是自动化专业")[1]
    assert meta["tries"] == 2 and meta["consolidated"] == "1"
    assert any("自动化" in n for n in ctx.notes)          # 原话照样能检索到
    assert mgr._redis.sets["mem:open:u1"] == {"c1"}       # 咨询记录下次再写


def test_old_notes_are_purged_after_retention():
    mgr = _mgr([_reply()])

    async def run():
        await _say(mgr, "u1", "c1", "我本科是自动化专业")
        nid, meta = _by_text(mgr, "我本科是自动化专业")
        _rows(mgr)[nid] = {**meta, "ts_num": (datetime.now() - timedelta(days=200)).timestamp()}
        await _say(mgr, "u1", "c1", "我雅思 6.5 分")
        await mgr.consolidate("u1")

    asyncio.run(run())
    assert [m["text"] for m in _rows(mgr).values()] == ["我雅思 6.5 分"]


def test_idle_user_gets_consolidated_and_new_turns_reset_the_timer():
    mgr = _mgr([_reply()])
    mgr.IDLE_SECONDS = 0.05

    async def run():
        await _say(mgr, "u1", "c1", "我本科是自动化专业")
        mgr.schedule_after_turn("u1", "c1")
        await asyncio.sleep(0.03)
        mgr.schedule_after_turn("u1", "c1")
        await asyncio.sleep(0.03)
        assert mgr._client.calls == 0
        await asyncio.sleep(0.05)
        await mgr.wait_background()

    asyncio.run(run())
    assert mgr._client.calls == 1


def test_schedule_after_turn_keeps_a_reference_until_done():
    mgr = _mgr()

    async def run():
        mgr.schedule_after_turn("u1", "c1")
        assert mgr._bg_tasks
        await mgr.wait_background()
        assert not mgr._bg_tasks
        for t in mgr._idle_timers.values():
            t.cancel()

    asyncio.run(run())


def test_enrichment_adds_keywords_two_way_links_supersedes_and_appends_context_versions():
    mgr = _mgr([_reply([_note("M1", context="想去荷兰")])])

    async def run():
        await _say(mgr, "u1", "c1", "我想去荷兰读计算机硕士")
        await mgr.consolidate("u1")
        old_id, _ = _by_text(mgr, "我想去荷兰读计算机硕士")
        mgr._client.replies = [lambda p: _reply(
            [_note("M1", [("target_country", "瑞典")], keywords=["瑞典", "改申"], context="改申瑞典", links=["N1"], supersedes=["N1"])],
            updates=[{"n": "N1", "context": "最初想去荷兰，后来改了"}])]
        await _say(mgr, "u1", "c1", "我决定不去荷兰了，改申瑞典")
        await mgr.consolidate("u1")
        return old_id, await mgr.get_context("u1", "c2", query="目标国家")

    old_id, ctx = asyncio.run(run())
    new_id, new = _by_text(mgr, "我决定不去荷兰了，改申瑞典")
    old = _rows(mgr)[old_id]
    assert new["keywords"] == "瑞典、改申" and json.loads(new["links"]) == [old_id]
    assert json.loads(old["links"]) == [new_id] and old["superseded_by"] == new_id
    assert old["context"] == "最初想去荷兰，后来改了" and json.loads(old["versions"])[0]["context"] == "想去荷兰"
    assert "关键词：瑞典、改申" in mgr._notes._col.docs[new_id]          # 向量按补全后的内容重算
    assert ctx.facts == ["目标国家：瑞典"]
    assert any("后来有更新" in n for n in ctx.notes)


def test_search_expands_links():
    mgr = _mgr()

    async def run():
        await _say(mgr, "u1", "c1", "我本科是自动化专业", "我女朋友在哥本哈根工作")
        a, _ = _by_text(mgr, "我本科是自动化专业")
        b, meta = _by_text(mgr, "我女朋友在哥本哈根工作")
        _rows(mgr)[a]["links"] = json.dumps([b])
        mgr.TOP_K = 1                                   # 只取到第一条，第二条靠链接带出来
        return await mgr.get_context("u1", "c2", query="专业")

    ctx = asyncio.run(run())
    assert len(ctx.notes) == 2 and "哥本哈根" in ctx.notes[1]


# ── 读取 ──────────────────────────────────────────────────────────────────────

def test_context_excludes_current_conversation_case_and_window_duplicates():
    mgr = _mgr()

    async def run():
        await mgr._cases.upsert("u1", {"conv_id": "c1", "updated_at": "2026-10-01", "summary": "当前会话", "open_items": [], "next_step": ""})
        await mgr._cases.upsert("u1", {"conv_id": "c0", "updated_at": "2026-09-01", "summary": "上次", "open_items": [], "next_step": ""})
        await _say(mgr, "u1", "c1", "我本科是自动化专业")
        return await mgr.get_context("u1", "c1", query="专业")

    ctx = asyncio.run(run())
    assert ctx.cases == ["2026-09-01｜上次"]
    assert ctx.notes == []                                # 已经在最近对话里了


def test_prompt_text_sections_and_budget_trims_oldest_notes_first():
    ctx = MemoryContext(
        recent_messages=[Message(MsgRole.USER, "你好"), Message(MsgRole.ASSISTANT, "长" * 400)],
        notes=["最早的原话" + "x" * 100, "较新的原话"], facts=["目标国家：瑞典"], cases=["2026-10-01｜上次"], max_chars=10_000)
    text = ctx.to_prompt_text()
    assert text.index("[客户档案]") < text.index("[以前的咨询记录]") < text.index("[用户以前说过的相关原话]") < text.index("[最近对话]")
    assert "付款、订单、报价以系统查询结果为准" in text and "…（略）" in text
    ctx.max_chars = len(text) - 50
    trimmed = ctx.to_prompt_text()
    assert "最早的原话" not in trimmed and "较新的原话" in trimmed and "上次" in trimmed


def test_vector_search_error_falls_back_to_exact_search():
    mgr = _mgr()

    async def run():
        await _say(mgr, "u1", "c1", "我本科是自动化专业", "我雅思 6.5 分")
        mgr._notes._col.fail_query = True
        return await mgr.get_context("u1", "c2", query="专业")

    ctx = asyncio.run(run())
    assert ctx.notes and "自动化" in ctx.notes[0]


# ── 顾问和隐私 ──────────────────────────────────────────────────────────────────

def test_advisor_edit_and_privacy_delete_clear_every_layer():
    mgr = _mgr([_reply([_note("M1", [("major", "自动化")])],
                       [{"c": "C1", "summary": "咨询", "open_items": [], "next_step": ""}])])

    async def run():
        await _say(mgr, "u1", "c1", "我本科是自动化专业")
        await _say(mgr, "u2", "c9", "我是别人")
        await mgr._redis.lpush(chat_log.KEY, json.dumps({"user_id": "u1", "message": "x"}))
        await mgr._redis.lpush(chat_log.KEY, json.dumps({"user_id": "u2", "message": "y"}))
        await mgr.consolidate("u1")
        await mgr.set_profile_field("u1", "major", "自动化（已核实）")
        assert (await mgr.get_profile("u1"))["major"]["source"] == "advisor"
        result = await mgr.forget_user("u1")
        return result, await mgr.get_context("u1", "c1", query="专业")

    result, ctx = asyncio.run(run())
    assert result == {"conversations": 1, "chatlog_entries": 1}
    assert ctx.recent_messages == [] and ctx.facts == [] and ctx.cases == [] and ctx.notes == []
    assert all(m["user_id"] == "u2" for m in _rows(mgr).values())
    assert [json.loads(r)["user_id"] for r in mgr._redis.lists[chat_log.KEY]] == ["u2"]
    assert not any(k.endswith(":u1") or ":u1:" in k for k in list(mgr._redis.lists) + list(mgr._redis.kv))


def test_unknown_profile_field_is_rejected():
    mgr = _mgr()
    try:
        asyncio.run(mgr.set_profile_field("u1", "girlfriend", "x"))
    except ValueError as ex:
        assert "未知字段" in str(ex)
    else:
        raise AssertionError("应该拒绝未知字段")


def test_handoff_summary_carries_profile_and_previous_cases():
    req = Request(message="转人工", user_id="u1", conv_id="c1",
                  memory_facts=["目标国家：瑞典"], memory_cases=["2026-10-01｜问了退款"])
    summary = build_handoff_summary(req, "用户要求真人顾问")
    assert summary["background"] == "客户档案（用户自述）：目标国家：瑞典。以前的咨询：2026-10-01｜问了退款"
