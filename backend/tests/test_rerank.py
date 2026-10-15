"""检索重排：cross-encoder 打分排序、阈值、失败退回混合分、熔断，以及追问补全。"""
import asyncio

import httpx

from core.rag_gate import RagGate, retrieval_query
from retrieval.manager import RetrievalManager, Tool
from retrieval.rerank import ApiReranker, passes_threshold

ITEMS = [
    {"doc_id": "a", "title": "瑞典概览", "content": "瑞典申请费 SEK 900", "score": 0.62},
    {"doc_id": "b", "title": "退款政策", "content": "服务未启动全额退", "score": 0.48},
]


def _reranker(handler) -> ApiReranker:
    reranker = ApiReranker("bge-reranker", "key", "https://rerank.test/v1")
    transport = httpx.MockTransport(handler)
    real = httpx.AsyncClient
    reranker_client = lambda **kw: real(transport=transport, **kw)  # noqa: E731
    httpx_patch = {"AsyncClient": reranker_client}
    reranker._patch = httpx_patch
    return reranker


def _run(reranker, monkeypatch, coro_fn):
    monkeypatch.setattr(httpx, "AsyncClient", reranker._patch["AsyncClient"])
    return asyncio.run(coro_fn())


def test_rerank_orders_by_cross_encoder(monkeypatch):
    seen = {}

    def handler(request):
        body = request.read().decode()
        seen["body"] = body
        return httpx.Response(200, json={"results": [{"index": 1, "relevance_score": 0.9},
                                                     {"index": 0, "relevance_score": 0.01}]})

    reranker = _reranker(handler)
    ranked = _run(reranker, monkeypatch, lambda: reranker.rerank("定金能退吗", ITEMS))
    assert [r["doc_id"] for r in ranked] == ["b", "a"]
    assert ranked[0]["reranked"] and ranked[0]["hybrid_score"] == 0.48 and ranked[0]["score"] == 0.9
    assert "退款政策\\n服务未启动全额退" in seen["body"]


def test_rerank_failure_returns_none_and_breaker_opens(monkeypatch):
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(500)

    reranker = _reranker(handler)
    for _ in range(3):
        assert _run(reranker, monkeypatch, lambda: reranker.rerank("q", ITEMS)) is None
    assert _run(reranker, monkeypatch, lambda: reranker.rerank("q", ITEMS)) is None
    assert len(calls) == 3                        # 第 4 次熔断，不再请求


def test_thresholds_depend_on_whether_reranked():
    assert passes_threshold({"score": 0.45}, rerank_min=0.1, hybrid_min=0.4)
    assert not passes_threshold({"score": 0.35}, rerank_min=0.1, hybrid_min=0.4)
    assert passes_threshold({"score": 0.2, "reranked": True}, rerank_min=0.1, hybrid_min=0.4)
    assert not passes_threshold({"score": 0.05, "reranked": True}, rerank_min=0.1, hybrid_min=0.4)


class _FakeReranker:
    def __init__(self, scores):
        self.scores_by_doc = scores
        self.queries = []

    async def rerank(self, query, items):
        self.queries.append(query)
        if self.scores_by_doc is None:
            return None
        ranked = [{**i, "hybrid_score": i["score"], "score": self.scores_by_doc[i["doc_id"]], "reranked": True}
                  for i in items]
        return sorted(ranked, key=lambda i: i["score"], reverse=True)


def _manager(reranker) -> RetrievalManager:
    manager = RetrievalManager(api_key="test", reranker=reranker)

    async def handler(params, context):
        return [dict(i) for i in ITEMS]

    manager.register(Tool(name="knowledge_search", description="", handler=handler,
                          schema={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}))
    return manager


def test_prefetch_reranks_and_gate_filters_on_rerank_score():
    manager = _manager(_FakeReranker({"a": 0.02, "b": 0.8}))

    async def search(query, top_k, domain):
        return await manager.search_fast("knowledge_search", query, top_k=top_k, domain=domain)

    gate = RagGate(search, enabled=True, min_rerank_score=0.1)

    async def prefetch():
        return await gate.resolve(gate.speculate("定金能退吗"), intent="refund", confidence=0.9,
                                  domain="billing", message="定金能退吗")

    decision = asyncio.run(prefetch())
    assert [i["doc_id"] for i in decision.items] == ["b"]          # 混合分更高的 a 被重排分数过滤掉


def test_prefetch_falls_back_to_hybrid_score_when_rerank_fails():
    manager = _manager(_FakeReranker(None))
    items = asyncio.run(manager.search_fast("knowledge_search", "瑞典申请费", top_k=3))
    assert [i["doc_id"] for i in items] == ["a", "b"] and not any(i.get("reranked") for i in items)


def test_deep_search_reranks_completed_followup_query():
    reranker = _FakeReranker({"a": 0.7, "b": 0.01})
    manager = _manager(reranker)
    prompts = []

    async def rewrite(query, n=3, previous=""):
        prompts.append(previous)
        return [query]

    manager.rewrite_query = rewrite
    history = [{"role": "user", "content": "瑞典申请费多少"}, {"role": "assistant", "content": "SEK 900"}]
    result = asyncio.run(manager.search_with_rewrite("knowledge_search", "那德国呢", top_k=3, history=history))
    assert result.success and result.reranked and [i["doc_id"] for i in result.data] == ["a"]
    assert prompts == ["瑞典申请费多少"] and reranker.queries == ["瑞典申请费多少 那德国呢"]


def test_retrieval_query_only_completes_short_followups():
    history = [{"role": "user", "content": "瑞典什么时候截止"}, {"role": "assistant", "content": "1 月中旬"}]
    assert retrieval_query("那德国呢", history) == "瑞典什么时候截止 那德国呢"
    assert retrieval_query("这个多少钱", history) == "瑞典什么时候截止 这个多少钱"
    assert retrieval_query("能开发票吗", history) == "能开发票吗"                      # 换了话题，不拼
    assert retrieval_query("那德国的申请费和学费分别是多少呢", history) == "那德国的申请费和学费分别是多少呢"  # 不是短句
    assert retrieval_query("那德国呢", None) == "那德国呢"
