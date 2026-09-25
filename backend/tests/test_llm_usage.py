import asyncio
from types import SimpleNamespace

from core import llm_usage


class FakePipe:
    def __init__(self, store):
        self.store, self.ops = store, []

    def hincrby(self, key, field, n):
        self.ops.append((key, field, n))

    hincrbyfloat = hincrby

    def expire(self, key, ttl):
        pass

    async def execute(self):
        for key, field, n in self.ops:
            h = self.store.setdefault(key, {})
            h[field] = h.get(field, 0) + n


class FakeRedis:
    def __init__(self):
        self.store = {}

    def pipeline(self):
        return FakePipe(self.store)


def usage(i, o):
    return SimpleNamespace(input_tokens=i, output_tokens=o, cache_read_input_tokens=0, cache_creation_input_tokens=0)


def test_record_counts_calls_and_tokens_by_source():
    r = FakeRedis()
    asyncio.run(llm_usage.record("intent", usage(100, 20), r))
    asyncio.run(llm_usage.record("agents", usage(50, 5), r))
    [day] = r.store.values()
    assert day["calls"] == 2 and day["in"] == 150 and day["out"] == 25
    assert day["in:intent"] == 100 and day["calls:agents"] == 1
    assert 0 < day["cost"] < 0.01


def test_peak_hours_cost_double():
    from datetime import datetime
    off = datetime(2026, 9, 26, 3, 0, tzinfo=llm_usage._TZ)       # Saturday
    peak = datetime(2026, 9, 25, 10, 0, tzinfo=llm_usage._TZ)     # Friday 10:00
    assert llm_usage.cost_of(1_000_000, 0, 0, off) == 4.5
    assert llm_usage.cost_of(1_000_000, 0, 0, peak) == 9.0
    assert llm_usage.cost_of(0, 1_000_000, 1_000_000, off) == 0.15 + 13.5


def test_record_never_raises():
    class Broken:
        def pipeline(self):
            raise ConnectionError("down")
    asyncio.run(llm_usage.record("intent", usage(1, 1), Broken()))


def test_track_wraps_create_and_stream(monkeypatch):
    seen = []

    async def fake_record(source, u, redis_client=None):
        seen.append((source, u.input_tokens, u.output_tokens))
    monkeypatch.setattr(llm_usage, "record", fake_record)

    class Stream:
        current_message_snapshot = SimpleNamespace(usage=usage(7, 3))

    class Manager:
        async def __aenter__(self):
            return Stream()

        async def __aexit__(self, *exc):
            return False

    class Messages:
        async def create(self, **kw):
            return SimpleNamespace(usage=usage(10, 2), content=[])

        def stream(self, **kw):
            return Manager()

    client = llm_usage.track(SimpleNamespace(messages=Messages()), "agents")
    assert llm_usage.track(client, "agents") is client          # wrapping twice is harmless

    async def run():
        await client.messages.create(model="m")
        async with client.messages.stream(model="m"):
            pass
        await asyncio.sleep(0)                                   # let the background records run
    asyncio.run(run())
    assert seen == [("agents", 10, 2), ("agents", 7, 3)]
