"""A-Mem 记忆：原文笔记、去重、批量补全、关键事实、只追加的演化、检索、补全时机、失败兜底。"""
import asyncio
import json
from collections import defaultdict
from types import SimpleNamespace

from evaluation.memory_benchmark import FakeRedis
from memory.amem import MemoryContext, MemoryManager, Message, MsgRole, has_self_info


def _match(meta, where):
    if not where:
        return True
    if "$and" in where:
        return all(_match(meta, w) for w in where["$and"])
    for k, v in where.items():
        if isinstance(v, dict) and "$ne" in v:
            if meta.get(k) == v["$ne"]:
                return False
        elif meta.get(k) != v:
            return False
    return True


class FakeNotes:
    """笔记集合的替身：按写入顺序保存；query 不做语义排序，按 where 过滤后返回前 n 条。"""

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


class FakeEmbedder:
    def embed_query(self, text):
        return [1.0]


class ScriptedClient:
    """补全笔记的调用按顺序返回预设回复（可以是看得到提示词的函数）。"""

    def __init__(self, replies):
        self.replies, self.calls = list(replies), 0
        owner = self

        class Messages:
            async def create(inner, **kwargs):
                owner.calls += 1
                r = owner.replies.pop(0) if owner.replies else '{"notes": []}'
                text = r(kwargs["messages"][-1]["content"]) if callable(r) else r
                return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)])

        self.messages = Messages()


def _mgr(replies=()):
    mgr = MemoryManager.__new__(MemoryManager)          # 不连真实的 Redis、ChromaDB 和模型
    mgr._redis = FakeRedis()
    mgr._redis_timeout = 1.0
    mgr._client = ScriptedClient(replies)
    mgr._model = "m"
    mgr._notes = FakeNotes()
    mgr._embedder = FakeEmbedder()
    mgr._note_locks = defaultdict(asyncio.Lock)
    mgr._bg_tasks = set()
    mgr._idle_timers = {}
    return mgr


def _by_text(mgr, text):
    return next((i, m) for i, m in mgr._notes.rows.items() if m["text"] == text)


def _reply(notes, updates=()):
    return json.dumps({"notes": notes, "neighbor_updates": list(updates)}, ensure_ascii=False)


def _note(m, **kw):
    base = {"m": m, "keywords": [], "tags": [], "context": "", "event_time": "", "slot": "", "value": "",
            "links": [], "supersedes": []}
    return {**base, **kw}


def test_has_self_info_only_for_statements_about_the_user():
    assert has_self_info("我本科是数学专业，GPA 3.6")
    assert has_self_info("我决定不去荷兰了，改申瑞典")
    assert not has_self_info("瑞典的硕士一般读几年？")
    assert not has_self_info("我想问一下")


def test_user_statements_become_raw_notes_immediately_without_model_calls():
    mgr = _mgr()

    async def run():
        await mgr.add_message("u", "c", MsgRole.USER, "我在物流公司实习了四个月，用 Go 写调度服务。")
        await mgr.add_message("u", "c", MsgRole.USER, "好的")                          # 太短
        await mgr.add_message("u", "c", MsgRole.USER, "瑞典的硕士一般读几年？")          # 纯提问
        await mgr.add_message("u", "c", MsgRole.ASSISTANT, "这是助手的回复，不进笔记。")

    asyncio.run(run())
    assert [m["text"] for m in mgr._notes.rows.values()] == ["我在物流公司实习了四个月，用 Go 写调度服务。"]
    assert mgr._notes.rows[next(iter(mgr._notes.rows))]["enriched"] == "0" and mgr._client.calls == 0


def test_same_sentence_twice_is_one_note_with_refreshed_time_and_kept_enrichment():
    mgr = _mgr()

    async def run():
        await mgr.add_message("u", "a", MsgRole.USER, "我想去芬兰读书。")
        nid, meta = _by_text(mgr, "我想去芬兰读书。")
        mgr._notes.rows[nid] = {**meta, "enriched": "1", "keywords": "芬兰"}
        await asyncio.sleep(0.01)
        await mgr.add_message("u", "b", MsgRole.USER, "我想去芬兰读书。")
        return meta["ts"]

    first_ts = asyncio.run(run())
    assert len(mgr._notes.rows) == 1
    _, meta = _by_text(mgr, "我想去芬兰读书。")
    assert meta["ts"] > first_ts and meta["conv_id"] == "b" and meta["keywords"] == "芬兰"


def test_window_keeps_only_the_latest_messages():
    mgr = _mgr()
    mgr.WINDOW = 4

    async def run():
        for i in range(6):
            await mgr.add_message("u", "c", MsgRole.USER, f"第 {i} 句话我说的是这个。")
        return await mgr._get_window("u", "c")

    assert [m.content[:3] for m in asyncio.run(run())] == ["第 2", "第 3", "第 4", "第 5"]


def test_batch_enrichment_adds_keywords_event_time_fact_and_two_way_links():
    reply = _reply([_note("M1", keywords=["暑期项目", "斯德哥尔摩"], tags=["经历"], context="讲暑期项目",
                          event_time="2025-07", slot="experience", value="斯德哥尔摩暑期项目", links=["M2"]),
                    _note("M2", keywords=["KTH"], context="想申 KTH", slot="unknown_slot", value="KTH")])
    mgr = _mgr([reply])

    async def run():
        await mgr.add_message("u", "c", MsgRole.USER, "我去年夏天在斯德哥尔摩参加过一个暑期项目。")
        await mgr.add_message("u", "c", MsgRole.USER, "所以我特别想申请 KTH 的计算机硕士。")
        return await mgr.flush_notes("u")

    assert asyncio.run(run()) == 2 and mgr._client.calls == 1                 # 两句一次调用
    id1, m1 = _by_text(mgr, "我去年夏天在斯德哥尔摩参加过一个暑期项目。")
    id2, m2 = _by_text(mgr, "所以我特别想申请 KTH 的计算机硕士。")
    assert m1["event_time"] == "2025-07" and "斯德哥尔摩" in m1["keywords"] and m1["slot"] == "experience"
    assert m2["slot"] == "" and m2["value"] == ""                               # 不认识的字段丢掉
    assert json.loads(m1["links"]) == [id2] and json.loads(m2["links"]) == [id1]   # 双向
    assert "关键词：暑期项目" in mgr._notes.docs[id1]                         # 向量文本带上补全内容


def test_change_of_mind_appends_versions_marks_superseded_and_facts_keep_history():
    first = _reply([_note("M1", keywords=["荷兰"], context="目标国家是荷兰", slot="target_country", value="荷兰")])

    def second(prompt):
        assert "N1" in prompt and "目标国家是荷兰" in prompt                  # 旧笔记作为候选给了模型
        return _reply([_note("M1", keywords=["瑞典"], context="改申瑞典", slot="target_country", value="瑞典",
                             links=["N1"], supersedes=["N1"])],
                      [{"n": "N1", "context": "最初想去荷兰，后来改成瑞典"}])

    mgr = _mgr([first, second])

    async def run():
        await mgr.add_message("u", "a", MsgRole.USER, "我想去荷兰读计算机硕士。")
        await mgr.flush_notes("u")
        await mgr.add_message("u", "b", MsgRole.USER, "我决定不去荷兰了，改申瑞典。")
        await mgr.flush_notes("u")
        return await mgr.get_context("u", "c", "我现在的目标国家是哪个？")

    ctx = asyncio.run(run())
    old_id, old = _by_text(mgr, "我想去荷兰读计算机硕士。")
    new_id, _ = _by_text(mgr, "我决定不去荷兰了，改申瑞典。")
    assert old["context"] == "最初想去荷兰，后来改成瑞典"
    assert json.loads(old["versions"])[0]["context"] == "目标国家是荷兰"   # 旧背景留档
    assert old["superseded_by"] == new_id and "最初想去荷兰" in mgr._notes.docs[old_id]
    assert ctx.facts == ["目标国家：瑞典（之前：荷兰）"]
    assert "荷兰" in ctx.notes[0] and "后来有更新" in ctx.notes[0]           # 按时间先后，旧的在前


def test_search_expands_links_and_skips_what_is_already_in_the_window():
    mgr = _mgr()

    async def run():
        for t in ["我本科是自动化专业。", "我做过一个机器人项目。", "我想申请控制方向。"]:
            await mgr.add_message("u", "old", MsgRole.USER, t)
        ids = list(mgr._notes.rows)
        mgr.NOTE_TOP_K = 1
        mgr._notes.rows[ids[0]]["links"] = json.dumps([ids[2]])
        return await mgr._search_notes("u", "专业", exclude=["我做过一个机器人项目。"])

    out = asyncio.run(run())
    assert len(out) == 2 and "自动化" in out[0] and "控制方向" in out[1]     # 第 1 条 + 顺链接取到第 3 条
    assert not any("机器人" in t for t in out)


def test_vector_search_error_falls_back_to_exact_search():
    mgr = _mgr()

    async def run():
        await mgr.add_message("u", "c", MsgRole.USER, "我想去芬兰读书，喜欢安静。")
        await mgr.add_message("u", "c", MsgRole.USER, "我本科是自动化专业。")
        mgr._notes.fail_query = True
        mgr.NOTE_TOP_K = 1
        return await mgr._search_notes("u", "专业", exclude=[])

    out = asyncio.run(run())
    assert len(out) == 1 and "自动化" in out[0]                               # 精确计算距离，取最近的一条


def test_enrichment_waits_while_statements_are_still_in_the_window():
    reply = _reply([_note("M1", context="专业")])
    mgr = _mgr([reply])

    async def run():
        await mgr.add_message("u", "a", MsgRole.USER, "我本科是自动化专业。")
        await mgr.after_turn("u", "a")
        calls_same_conv = mgr._client.calls
        await mgr.after_turn("u", "b")                    # 另一个窗口，但会话 a 刚刚还在说话：不触发
        calls_other_fresh = mgr._client.calls
        nid, meta = _by_text(mgr, "我本科是自动化专业。")
        mgr._notes.rows[nid] = {**meta, "ts": "2026-01-01T00:00:00"}   # 会话 a 早就停了
        await mgr.after_turn("u", "b")
        return calls_same_conv, calls_other_fresh

    assert asyncio.run(run()) == (0, 0) and mgr._client.calls == 1
    assert _by_text(mgr, "我本科是自动化专业。")[1]["enriched"] == "1"


def test_enrichment_runs_when_a_batch_is_full():
    mgr = _mgr()
    mgr.NOTE_BATCH = 2

    async def run():
        await mgr.add_message("u", "a", MsgRole.USER, "我本科是自动化专业。")
        await mgr.add_message("u", "a", MsgRole.USER, "我的 GPA 是 3.4。")
        await mgr.after_turn("u", "a")

    asyncio.run(run())
    assert mgr._client.calls == 1


def test_bad_model_output_keeps_raw_notes_and_gives_up_after_retries():
    mgr = _mgr(["抱歉", "还是不行"])

    async def run():
        await mgr.add_message("u", "c", MsgRole.USER, "我本科是自动化专业。")
        await mgr.flush_notes("u")
        first = dict(_by_text(mgr, "我本科是自动化专业。")[1])
        await mgr.flush_notes("u")
        return first, await mgr._search_notes("u", "专业", exclude=[])

    first, out = asyncio.run(run())
    _, last = _by_text(mgr, "我本科是自动化专业。")
    assert first["enriched"] == "0" and first["tries"] == 1
    assert last["enriched"] == "1" and last["tries"] == 2 and last["keywords"] == ""
    assert any("自动化" in t for t in out)                                     # 原文仍然可检索


def test_prompt_text_has_facts_notes_and_truncated_assistant_replies():
    msgs = [Message(MsgRole.USER, "我想去瑞典。"), Message(MsgRole.ASSISTANT, "很长的回复" * 100)]
    text = MemoryContext(recent_messages=msgs, notes=["[2026-10-01] 我本科是数学专业。"], facts=["预算：每年 25 万（之前：每年 20 万）"]).to_prompt_text()
    assert text.index("[用户的关键事实]") < text.index("[用户以前说过的相关原话]") < text.index("[最近对话]")
    assert "user: 我想去瑞典。" in text and "（略）" in text


def test_schedule_after_turn_keeps_a_reference_until_done():
    mgr = _mgr()

    async def run():
        mgr.NOTE_BATCH = 1
        await mgr.add_message("u", "a", MsgRole.USER, "我本科是自动化专业。")
        mgr.schedule_after_turn("u", "a")
        assert len(mgr._bg_tasks) == 1
        await mgr.wait_background()
        return len(mgr._bg_tasks)

    assert asyncio.run(run()) == 0 and mgr._client.calls == 1


def test_list_slots_keep_every_value_instead_of_treating_them_as_changes():
    mgr = _mgr()
    for i, (slot, value) in enumerate([("other", "女朋友在哥本哈根工作"), ("other", "想和女朋友在同一个城市"),
                                       ("budget", "每年 20 万"), ("budget", "每年 25 万")]):
        mgr._notes.add([str(i)], [value], [{"user_id": "u", "slot": slot, "value": value, "ts": f"2026-10-0{i + 1}"}])
    facts = asyncio.run(mgr._facts("u"))
    assert "其他明确的偏好或约束：女朋友在哥本哈根工作；想和女朋友在同一个城市" in facts
    assert "预算：每年 25 万（之前：每年 20 万）" in facts


def test_idle_user_gets_remaining_notes_enriched_and_new_turns_reset_the_timer():
    mgr = _mgr()
    mgr.IDLE_SECONDS = 0.05

    async def run():
        await mgr.add_message("u", "a", MsgRole.USER, "我本科是自动化专业。")
        mgr.schedule_after_turn("u", "a")           # 还在窗口里，不会立刻补全
        await asyncio.sleep(0.03)
        mgr.schedule_after_turn("u", "a")           # 又说了一句：重新计时
        await asyncio.sleep(0.03)
        before = mgr._client.calls
        await asyncio.sleep(0.05)
        return before

    assert asyncio.run(run()) == 0 and mgr._client.calls == 1
    assert mgr._idle_timers == {}
