"""A-Mem 魔改版记忆：原文笔记、批量补全、链接、只追加的演化、检索、失败兜底、实现切换。"""
import asyncio
import json
from collections import defaultdict
from types import SimpleNamespace

from evaluation.memory_benchmark import FakeRedis
from memory.agentic_memory import AgenticMemoryManager
from memory.conversation_memory import MemoryManager, MsgRole, build_memory_manager
from tests.test_memory_v2 import FakeCollection, FakeTurns


def _match(meta, where):
    if not where:
        return True
    if "$and" in where:
        return all(_match(meta, w) for w in where["$and"])
    return all(meta.get(k) == v for k, v in where.items())


class FakeNotes:
    """笔记集合的替身：按写入顺序保存；query 不做语义排序，按 where 过滤后返回前 n 条。记下 update 时写入的 documents。"""

    def __init__(self):
        self.rows = {}
        self.docs = {}

    def add(self, ids, documents, metadatas):
        for i, d, m in zip(ids, documents, metadatas):
            self.rows[i] = dict(m)
            self.docs[i] = d

    def get(self, ids=None, where=None):
        keys = [i for i in (ids or self.rows) if i in self.rows and _match(self.rows[i], where)]
        return {"ids": keys, "metadatas": [self.rows[i] for i in keys], "documents": [self.docs[i] for i in keys]}

    def query(self, query_texts, n_results, where=None):
        keys = [i for i in self.rows if _match(self.rows[i], where)][:n_results]
        return {"ids": [keys], "metadatas": [[self.rows[i] for i in keys]], "documents": [[self.docs[i] for i in keys]]}

    def update(self, ids, metadatas=None, documents=None):
        for k, i in enumerate(ids):
            if metadatas:
                self.rows[i] = dict(metadatas[k])
            if documents:
                self.docs[i] = documents[k]


class ScriptedClient:
    """补全笔记的调用按顺序返回预设回复（函数可以看到提示词）；其他调用返回固定摘要。"""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0
        owner = self

        class Messages:
            async def create(inner, **kwargs):
                owner.calls += 1
                prompt = kwargs["messages"][-1]["content"]
                if "整理一位用户的记忆笔记" in prompt:
                    r = owner.replies.pop(0) if owner.replies else '{"notes": []}'
                    text = r(prompt) if callable(r) else r
                elif "只返回 JSON" in prompt:
                    text = '{"facts": []}'
                else:
                    text = "用户在咨询留学。"
                return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)])

        self.messages = Messages()


def _amem(replies=()):
    mgr = AgenticMemoryManager.__new__(AgenticMemoryManager)
    mgr._redis = FakeRedis()
    mgr._redis_timeout = 1.0
    mgr._client = ScriptedClient(replies)
    mgr._model = "m"
    mgr._profile = FakeCollection()
    mgr._episodic = FakeCollection()
    mgr._turns = FakeTurns()
    mgr._notes = FakeNotes()
    mgr._profile_locks = defaultdict(asyncio.Lock)
    mgr._note_locks = defaultdict(asyncio.Lock)
    mgr._bg_tasks = set()
    return mgr


def _by_text(mgr, text):
    return next((i, m) for i, m in mgr._notes.rows.items() if m["text"] == text)


def test_user_statements_become_raw_notes_immediately():
    mgr = _amem()

    async def run():
        await mgr.add_message("u", "c", MsgRole.USER, "我在物流公司实习了四个月，用 Go 写调度服务。")
        await mgr.add_message("u", "c", MsgRole.USER, "好的")
        await mgr.add_message("u", "c", MsgRole.USER, "瑞典的硕士一般读几年？")
        await mgr.add_message("u", "c", MsgRole.ASSISTANT, "这是助手的回复，不进笔记。")

    asyncio.run(run())
    metas = list(mgr._notes.rows.values())
    assert [m["text"] for m in metas] == ["我在物流公司实习了四个月，用 Go 写调度服务。"]
    assert metas[0]["enriched"] == "0" and mgr._client.calls == 0          # 写入时不调模型
    assert len(mgr._turns.rows) == 1                                         # 基类的逐句索引照常写，切回 v2 不丢数据


def test_batch_enrichment_adds_keywords_event_time_and_two_way_links():
    reply = json.dumps({"notes": [
        {"m": "M1", "keywords": ["暑期项目", "斯德哥尔摩"], "tags": ["经历"], "context": "讲暑期项目",
         "event_time": "2025-07", "links": ["M2"], "supersedes": []},
        {"m": "M2", "keywords": ["KTH"], "tags": ["选校"], "context": "想申 KTH", "event_time": "", "links": [], "supersedes": []},
    ]}, ensure_ascii=False)
    mgr = _amem([reply])

    async def run():
        await mgr.add_message("u", "c", MsgRole.USER, "我去年夏天在斯德哥尔摩参加过一个暑期项目。")
        await mgr.add_message("u", "c", MsgRole.USER, "所以我特别想申请 KTH 的计算机硕士。")
        return await mgr.flush_notes("u")

    assert asyncio.run(run()) == 2 and mgr._client.calls == 1               # 两句一次调用
    id1, m1 = _by_text(mgr, "我去年夏天在斯德哥尔摩参加过一个暑期项目。")
    id2, m2 = _by_text(mgr, "所以我特别想申请 KTH 的计算机硕士。")
    assert m1["enriched"] == "1" and m1["event_time"] == "2025-07" and "斯德哥尔摩" in m1["keywords"]
    assert json.loads(m1["links"]) == [id2] and json.loads(m2["links"]) == [id1]   # 双向
    assert "关键词：暑期项目" in mgr._notes.docs[id1]                       # 向量文本带上了补全的内容


def test_evolution_appends_versions_and_marks_superseded():
    first = json.dumps({"notes": [{"m": "M1", "keywords": ["荷兰"], "tags": [], "context": "目标国家是荷兰",
                                   "event_time": "", "links": [], "supersedes": []}]}, ensure_ascii=False)

    def second(prompt):
        assert "N1" in prompt and "目标国家是荷兰" in prompt                  # 旧笔记作为候选给了模型
        return json.dumps({"notes": [{"m": "M1", "keywords": ["瑞典"], "tags": [], "context": "改申瑞典",
                                      "event_time": "", "links": ["N1"], "supersedes": ["N1"]}],
                           "neighbor_updates": [{"n": "N1", "context": "最初想去荷兰，后来改成瑞典"}]}, ensure_ascii=False)

    mgr = _amem([first, second])

    async def run():
        await mgr.add_message("u", "a", MsgRole.USER, "我想去荷兰读计算机硕士。")
        await mgr.flush_notes("u")
        await mgr.add_message("u", "b", MsgRole.USER, "我决定不去荷兰了，改申瑞典。")
        await mgr.flush_notes("u")
        return await mgr._search_turns("u", "目标国家", exclude=[])

    out = asyncio.run(run())
    old_id, old = _by_text(mgr, "我想去荷兰读计算机硕士。")
    new_id, _ = _by_text(mgr, "我决定不去荷兰了，改申瑞典。")
    assert old["context"] == "最初想去荷兰，后来改成瑞典"
    assert json.loads(old["versions"])[0]["context"] == "目标国家是荷兰"   # 旧背景留档，不覆盖丢失
    assert old["superseded_by"] == new_id and "最初想去荷兰" in mgr._notes.docs[old_id]   # 向量文本随新版更新
    assert any("荷兰" in t and "后来有更新" in t for t in out)


def test_search_expands_links_and_skips_working_memory():
    mgr = _amem()

    async def run():
        for t in ["我本科是自动化专业。", "我做过一个机器人项目。", "我想申请控制方向。"]:
            await mgr.add_message("u", "c", MsgRole.USER, t)
        ids = list(mgr._notes.rows)
        mgr.NOTE_TOP_K = 1
        mgr._notes.rows[ids[0]]["links"] = json.dumps([ids[2]])
        return await mgr._search_turns("u", "专业", exclude=["我做过一个机器人项目。"])

    out = asyncio.run(run())
    assert any("自动化" in t for t in out) and any("控制方向" in t for t in out)   # 第 1 条 + 顺链接取到第 3 条
    assert not any("机器人" in t for t in out)


def test_bad_model_output_keeps_raw_notes_and_gives_up_after_retries():
    mgr = _amem(["抱歉", "还是不行"])

    async def run():
        await mgr.add_message("u", "c", MsgRole.USER, "我的雅思是 6.5，小分最低 6。")
        await mgr.flush_notes("u")
        first = dict(next(iter(mgr._notes.rows.values())))
        await mgr.flush_notes("u")
        return first, await mgr._search_turns("u", "雅思", exclude=[])

    first, out = asyncio.run(run())
    meta = next(iter(mgr._notes.rows.values()))
    assert first["enriched"] == "0" and first["tries"] == 1                 # 第一次失败：留着下次再补
    assert meta["enriched"] == "1" and meta["keywords"] == ""                # 第二次失败：放弃补全
    assert any("6.5" in t for t in out)                                      # 原文照样能检索到


def test_update_profile_enriches_only_after_a_full_batch():
    mgr = _amem()

    async def run():
        await mgr.add_message("u", "c", MsgRole.USER, "我本科读的是软件工程专业。")
        await mgr.update_profile("u", "c")

    asyncio.run(run())
    notes_calls = mgr._client.calls - 1                                      # 1 次是画像提炼
    assert notes_calls == 0 and next(iter(mgr._notes.rows.values()))["enriched"] == "0"


def test_factory_defaults_to_v2_and_switches_by_env(monkeypatch):
    built = []
    monkeypatch.setattr(MemoryManager, "__init__", lambda self, **kw: built.append(type(self).__name__))
    monkeypatch.setattr(AgenticMemoryManager, "__init__", lambda self, **kw: built.append(type(self).__name__))
    monkeypatch.delenv("GOEUROOPS_MEMORY_IMPL", raising=False)
    assert type(build_memory_manager()) is MemoryManager
    monkeypatch.setenv("GOEUROOPS_MEMORY_IMPL", "amem")
    assert type(build_memory_manager()) is AgenticMemoryManager
