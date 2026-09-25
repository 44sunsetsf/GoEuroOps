import asyncio
import json
from types import SimpleNamespace

from core import chat_log


class FakePipe:
    def __init__(self, store):
        self.store, self.ops = store, []

    def lpush(self, key, value):
        self.ops.append(("lpush", key, value))

    def ltrim(self, key, a, b):
        self.ops.append(("ltrim", key, a, b))

    async def execute(self):
        for op in self.ops:
            if op[0] == "lpush":
                self.store.setdefault(op[1], []).insert(0, op[2])
            else:
                self.store[op[1]] = self.store[op[1]][op[2]:op[3] + 1]


class FakeRedis:
    def __init__(self):
        self.store = {}

    def pipeline(self):
        return FakePipe(self.store)


def test_records_question_answer_and_routing():
    r = FakeRedis()
    resp = SimpleNamespace(request_id="r1", response="答" * 2000, intent="application_process", intent_group="study",
                           agent_types=["consulting"], tools_used=["search_knowledge_base"], skills_applied=[],
                           escalated=False, latency_ms=5377.8)
    asyncio.run(chat_log.record("u1", "c1", "瑞典什么时候截止？", resp, r))
    [raw] = r.store[chat_log.KEY]
    e = json.loads(raw)
    assert e["message"] == "瑞典什么时候截止？" and e["conv_id"] == "c1" and e["agents"] == ["consulting"]
    assert len(e["answer"]) == 1500 and e["latency_ms"] == 5378


def test_never_raises():
    class Broken:
        def pipeline(self):
            raise ConnectionError("down")
    asyncio.run(chat_log.record("u", "c", "m", SimpleNamespace(), Broken()))
