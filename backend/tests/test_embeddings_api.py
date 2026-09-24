import json

import httpx

from mcp.embeddings import BGE_ZH_QUERY_PREFIX, ApiEmbedFunction


def _fake_api(calls):
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        calls.append(body["input"])
        data = [{"index": i, "embedding": [float(len(t)), 1.0]} for i, t in enumerate(body["input"])]
        return httpx.Response(200, json={"data": list(reversed(data))})  # 乱序返回，按 index 还原
    return httpx.MockTransport(handler)


def test_queries_get_bge_prefix_and_are_cached():
    calls = []
    fn = ApiEmbedFunction("BAAI/bge-large-zh-v1.5", "k", transport=_fake_api(calls))
    v1 = fn.embed_query("瑞典")
    v2 = fn.embed_query("瑞典")
    assert v1 == v2 == [float(len(BGE_ZH_QUERY_PREFIX + "瑞典")), 1.0]
    assert calls == [[BGE_ZH_QUERY_PREFIX + "瑞典"]]  # 第二次命中缓存


def test_batch_queries_dedupe_and_keep_order():
    calls = []
    fn = ApiEmbedFunction("BAAI/bge-large-zh-v1.5", "k", transport=_fake_api(calls))
    out = fn.embed_queries(["a", "bb", "a"])
    assert [v[0] for v in out] == [len(BGE_ZH_QUERY_PREFIX) + 1, len(BGE_ZH_QUERY_PREFIX) + 2, len(BGE_ZH_QUERY_PREFIX) + 1]
    assert len(calls) == 1 and len(calls[0]) == 2


def test_documents_have_no_prefix_and_are_batched():
    calls = []
    fn = ApiEmbedFunction("BAAI/bge-large-zh-v1.5", "k", transport=_fake_api(calls))
    docs = [f"doc{i}" for i in range(40)]
    vecs = fn(docs)
    assert len(vecs) == 40 and vecs[0][0] == len("doc0")
    assert [len(c) for c in calls] == [32, 8]


def test_collection_suffix_differs_from_local_model():
    fn = ApiEmbedFunction("BAAI/bge-large-zh-v1.5", "k", transport=_fake_api([]))
    assert fn.collection_suffix == "baai_bge_large_zh_v1_5"
