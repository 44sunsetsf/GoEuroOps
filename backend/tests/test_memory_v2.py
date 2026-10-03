"""记忆模块 v2：画像只追加带时间、有新信息才提炼、提示词放全部工作记忆、情景记忆检索条件。"""
import asyncio
import json
from types import SimpleNamespace

from evaluation.memory_benchmark import FakeRedis
from memory.conversation_memory import (
    ASSISTANT_MSG_CHARS,
    MemoryContext,
    MemoryManager,
    Message,
    MsgRole,
    has_self_info,
    normalize_profile,
    profile_view,
    render_profile,
)


class FakeCollection:
    """ChromaDB collection 的最小替身：按 id 存文档，记下 query 的 where 条件。"""

    def __init__(self):
        self.docs = {}
        self.queries = []

    def get(self, ids=None, where=None):
        if ids:
            found = [i for i in ids if i in self.docs]
            return {"documents": [self.docs[i][0] for i in found], "metadatas": [self.docs[i][1] for i in found]}
        return {"documents": [], "metadatas": []}

    def add(self, ids, documents, metadatas):
        for i, d, m in zip(ids, documents, metadatas):
            self.docs[i] = (d, m)

    def delete(self, ids):
        for i in ids:
            self.docs.pop(i, None)

    def query(self, query_texts, n_results, where):
        self.queries.append(where)
        return {"documents": [[]]}


class FakeTurns:
    """逐句索引的替身：按写入顺序保存，query 按 user_id 过滤后原样返回前 n 条（不做语义排序，只验证流程）。"""

    def __init__(self):
        self.rows = []

    def add(self, ids, documents, metadatas):
        self.rows.extend(zip(ids, documents, metadatas))

    def query(self, query_texts, n_results, where):
        rows = [r for r in self.rows if r[2]["user_id"] == where["user_id"]][:n_results]
        return {"ids": [[r[0] for r in rows]], "documents": [[r[1] for r in rows]], "metadatas": [[r[2] for r in rows]]}


class FakeClient:
    """提炼画像的调用按顺序返回预设回复；写摘要等其他调用返回一段固定摘要。记下调用次数。"""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0
        owner = self

        class Messages:
            async def create(inner, **kwargs):
                owner.calls += 1
                prompt = kwargs["messages"][-1]["content"]
                if "只返回 JSON" in prompt:
                    text = owner.replies.pop(0) if owner.replies else '{"facts": []}'
                else:
                    text = "用户在咨询留学。"
                return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)])

        self.messages = Messages()


def _manager(replies=()):
    mgr = MemoryManager.__new__(MemoryManager)
    mgr._redis = FakeRedis()
    mgr._redis_timeout = 1.0
    mgr._client = FakeClient(replies)
    mgr._model = "m"
    mgr._profile = FakeCollection()
    mgr._episodic = FakeCollection()
    mgr._turns = FakeTurns()
    from collections import defaultdict
    mgr._profile_locks = defaultdict(asyncio.Lock)
    return mgr


def test_has_self_info_only_for_statements_about_the_user():
    assert has_self_info("我本科是数学专业，GPA 3.6")
    assert has_self_info("我决定不去荷兰了，改申瑞典")
    assert not has_self_info("瑞典的硕士一般读几年？")          # 一般性问题，不是在讲自己
    assert not has_self_info("你们的服务包括哪些？")
    assert not has_self_info("我想问一下")                       # 有“我”，但没有任何信息


def test_profile_keeps_history_and_latest_wins():
    profile = normalize_profile({"facts": [
        {"field": "target_country", "value": "荷兰", "ts": "2026-01-01T00:00:00"},
        {"field": "major", "value": "软件工程", "ts": "2026-01-01T00:00:01"},
        {"field": "target_country", "value": "瑞典", "ts": "2026-01-02T00:00:00"},
    ]})
    view = profile_view(profile)
    assert view["target_country"] == {"value": "瑞典", "history": ["荷兰"]}
    text = render_profile(profile)
    assert "目标国家：瑞典（之前：荷兰）" in text
    assert "\\u" not in text                                       # 中文不转义


def test_legacy_profile_is_kept_not_dropped():
    profile = normalize_profile({"preferences": ["想去北欧"]})
    assert profile["facts"] == [] and profile["legacy"] == {"preferences": ["想去北欧"]}
    assert "想去北欧" in render_profile(profile)


def test_prompt_shows_all_working_memory_and_truncates_only_assistant():
    long_reply = "很" * (ASSISTANT_MSG_CHARS + 50)
    long_user = "我" + "情况" * 200
    msgs = [Message(role=MsgRole.USER, content=f"用户第{i}句") for i in range(13)]
    msgs.insert(0, Message(role=MsgRole.USER, content=long_user))
    msgs.append(Message(role=MsgRole.ASSISTANT, content=long_reply))
    text = MemoryContext(recent_messages=msgs, relevant_history=[], user_profile={}, summary="").to_prompt_text()
    assert "用户第0句" in text                                     # 原来只放最后 8 条，这句会丢
    assert long_user in text                                       # 用户的话不截断
    assert "…（略）" in text and long_reply not in text            # 助手的长回复截断


def test_episodic_search_uses_and_operator():
    mgr = _manager()
    asyncio.run(mgr._search_episodic("u", "c", "瑞典"))
    assert mgr._episodic.queries[0] == {"$and": [{"user_id": "u"}, {"conv_id": "c"}]}


def test_extract_appends_new_facts_and_skips_unchanged():
    mgr = _manager([
        json.dumps({"facts": [{"field": "target_country", "value": "荷兰"}, {"field": "major", "value": "数学"}]}),
        json.dumps({"facts": [{"field": "target_country", "value": "瑞典"}, {"field": "major", "value": "数学"},
                              {"field": "不存在的字段", "value": "x"}]}),
    ])
    msgs = [Message(role=MsgRole.USER, content="我想去荷兰，本科数学")]
    assert asyncio.run(mgr._extract_facts("u", "c", msgs)) == 2
    assert asyncio.run(mgr._extract_facts("u", "c", msgs)) == 1   # 专业没变、未知字段丢弃，只追加了瑞典
    profile = asyncio.run(mgr._get_profile("u"))
    assert [f["value"] for f in profile["facts"]] == ["荷兰", "数学", "瑞典"]
    assert profile_view(profile)["target_country"]["history"] == ["荷兰"]


def test_bad_model_output_keeps_existing_profile():
    mgr = _manager([json.dumps({"facts": [{"field": "major", "value": "数学"}]}), "抱歉，我无法处理"])
    msgs = [Message(role=MsgRole.USER, content="我本科数学")]
    asyncio.run(mgr._extract_facts("u", "c", msgs))
    asyncio.run(mgr._extract_facts("u", "c", msgs))                 # 第二次模型没给 JSON
    assert profile_view(asyncio.run(mgr._get_profile("u")))["major"]["value"] == "数学"


def test_update_profile_skips_model_call_without_self_info():
    mgr = _manager()

    async def run():
        await mgr.add_message("u", "c", MsgRole.USER, "瑞典的硕士一般读几年？")
        await mgr.add_message("u", "c", MsgRole.ASSISTANT, "两年。")
        await mgr.update_profile("u", "c")

    asyncio.run(run())
    assert mgr._client.calls == 0


def test_compression_consolidates_profile_before_summarizing():
    mgr = _manager([json.dumps({"facts": [{"field": "major", "value": "自动化"}]})])

    async def run():
        await mgr.add_message("u", "c", MsgRole.USER, "我本科读的是自动化专业。")
        for i in range(14):
            await mgr.add_message("u", "c", MsgRole.ASSISTANT if i % 2 == 0 else MsgRole.USER, f"第{i}条")

    asyncio.run(run())                                               # 1 + 14 = 15 条，触发压缩
    assert profile_view(asyncio.run(mgr._get_profile("u")))["major"]["value"] == "自动化"
    assert asyncio.run(mgr._redis.llen("wm:u:c")) == MemoryManager.KEEP_RAW


def test_turn_index_keeps_statements_and_skips_noise():
    mgr = _manager()

    async def run():
        await mgr.add_message("u", "c", MsgRole.USER, "我在一家互联网公司做后端开发实习，做了半年。")
        await mgr.add_message("u", "c", MsgRole.USER, "好的")                           # 太短
        await mgr.add_message("u", "c", MsgRole.USER, "瑞典的硕士一般读几年？")           # 纯提问
        await mgr.add_message("u", "c", MsgRole.USER, "我的预算是多少合适，家里给了 15 万？")  # 提问但讲了自己的情况
        await mgr.add_message("u", "c", MsgRole.ASSISTANT, "这是助手的一段很长的回复，不应该建索引。")

    asyncio.run(run())
    docs = [r[1] for r in mgr._turns.rows]
    assert docs == ["我在一家互联网公司做后端开发实习，做了半年。", "我的预算是多少合适，家里给了 15 万？"]


def test_search_turns_formats_dates_and_skips_working_memory():
    mgr = _manager()

    async def run():
        await mgr.add_message("u", "a", MsgRole.USER, "我去年夏天在斯德哥尔摩参加过一个暑期项目。")
        await mgr.add_message("u", "a", MsgRole.USER, "我打算申请的方向是分布式系统。")
        return await mgr._search_turns("u", "暑期项目", exclude=["我打算申请的方向是分布式系统。"])

    out = asyncio.run(run())
    assert len(out) == 1 and out[0].startswith("[20") and "斯德哥尔摩" in out[0]      # 带日期；工作记忆里已有的那句不重复


def test_context_has_turns_from_other_conversations():
    mgr = _manager()

    async def run():
        await mgr.add_message("u", "old", MsgRole.USER, "我本科做过一个基于 PyTorch 的图像分类项目。")
        return await mgr.get_context("u", "new", query="我做过什么项目")

    ctx = asyncio.run(run())
    assert "PyTorch" in ctx.to_prompt_text() and "[用户以前说过的相关原话]" in ctx.to_prompt_text()


def test_turn_index_failure_does_not_break_add_message():
    mgr = _manager()

    class Broken:
        def add(self, **kwargs):
            raise RuntimeError("chroma down")

    mgr._turns = Broken()

    async def run():
        await mgr.add_message("u", "c", MsgRole.USER, "我的雅思是 6.5，小分最低 6。")
        return await mgr._redis.llen("wm:u:c")

    assert asyncio.run(run()) == 1                                   # 索引挂了，消息照样进工作记忆


def test_turn_index_can_be_switched_off():
    mgr = _manager()
    mgr.TURN_INDEX = False

    async def run():
        await mgr.add_message("u", "c", MsgRole.USER, "我的雅思是 6.5，小分最低 6。")
        return await mgr.get_context("u", "d", query="雅思")

    ctx = asyncio.run(run())
    assert mgr._turns.rows == [] and ctx.relevant_turns == []
