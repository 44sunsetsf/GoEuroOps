import asyncio
import json

from business.catalog import default_prices, get_catalog, set_service_prices
from business.kb_seed import build_seed_documents
from core.rag_gate import RagGate
from retrieval.wiki import WikiKnowledge, build_pages
from retrieval.wiki_compile import lint


class SelectClient:
    """假的选页模型：按问题返回预设的页面 id，记录调用次数。"""

    def __init__(self, picks=None, delay=0.0, error=None):
        self.picks = picks or {}
        self.delay = delay
        self.error = error
        self.calls = []
        owner = self

        class Messages:
            async def create(inner, **kwargs):
                owner.calls.append(kwargs)
                if owner.delay:
                    await asyncio.sleep(owner.delay)
                if owner.error:
                    raise owner.error
                question = kwargs["messages"][-1]["content"].rsplit("用户问题：", 1)[-1]
                text = json.dumps({"pages": owner.picks.get(question, [])})
                return type("R", (), {"content": [type("T", (), {"type": "text", "text": text})()]})()

        self.messages = Messages()


def test_compiled_wiki_passes_lint():
    # 目录和术语表是 LLM 编译后提交的，这里保证它们和当前代码、原始资料对得上
    assert lint()["errors"] == []


def test_pages_cover_every_seed_document():
    _, docs = build_seed_documents()
    covered = {s for p in build_pages() for s in p.sources}
    assert {d["title"] for d in docs} <= covered


def test_price_pages_follow_price_changes_without_recompiling():
    sku, price = next(iter(default_prices().items()))
    wiki = WikiKnowledge(SelectClient(), "m")
    name = next(s.name for s in get_catalog().services if s.sku == sku)
    page = next(p for p in wiki.pages if any(name in t for t in p.sources))
    set_service_prices({sku: price + 7})
    asyncio.run(wiki.search_fast("knowledge_search", "随便问问"))   # 每次检索前都会检查目录是否换过
    assert f"{price + 7:,}" in wiki.page(page.id).content


def test_select_returns_whole_pages_and_drops_unknown_ids():
    client = SelectClient({"瑞典什么时候截止": ["country-sweden", "no-such-page", "timeline", "country-sweden"]})
    wiki = WikiKnowledge(client, "m")
    items = asyncio.run(wiki.search_fast("knowledge_search", "瑞典什么时候截止", top_k=3, domain="billing"))
    assert [i["page_id"] for i in items] == ["country-sweden", "timeline"]
    assert items[0]["content"] == wiki.page("country-sweden").content
    assert items[0]["score"] > items[1]["score"] >= 0.4        # 能过门控的分数下限
    assert "目录" in client.calls[0]["system"] and "country-sweden" in client.calls[0]["system"]


def test_speculative_prefetch_shares_one_select_call_and_survives_cancellation():
    client = SelectClient({"德国 APS 要多久": ["germany-aps"]}, delay=0.05)
    wiki = WikiKnowledge(client, "m")

    async def run():
        gate = RagGate(lambda q, k, d: wiki.search_fast("knowledge_search", q, top_k=k, domain=d), enabled=True)
        speculative = gate.speculate("德国 APS 要多久")             # 三个 domain 同时预取
        return await gate.resolve(speculative, intent="application_process", confidence=0.9,
                                  domain="consulting", message="德国 APS 要多久")   # 会取消另外两个

    decision = asyncio.run(run())
    assert len(client.calls) == 1
    assert [i["page_id"] for i in decision.items] == ["germany-aps"]


def test_no_relevant_page_injects_nothing():
    wiki = WikiKnowledge(SelectClient(), "m")
    assert asyncio.run(wiki.search_fast("knowledge_search", "今天天气怎么样")) == []


def test_select_timeout_falls_back_to_local_question_matching():
    wiki = WikiKnowledge(SelectClient(delay=1.0), "m", timeout_s=0.05)
    question = wiki.page("refund-invoice").questions[0]
    ids = asyncio.run(wiki.select(question))
    assert "refund-invoice" in ids
    assert wiki.get_stats()["local_fallback"] == 1


def test_select_results_are_cached():
    client = SelectClient({"能开发票吗": ["refund-invoice"]})
    wiki = WikiKnowledge(client, "m")
    asyncio.run(wiki.select("能开发票吗"))
    asyncio.run(wiki.select("能开发票吗"))
    assert len(client.calls) == 1


def test_uploaded_documents_become_pages_and_reload_on_change():
    uploads = [{"doc_id": "d1", "title": "冰岛 CS 硕士简介", "domain": "consulting", "content": "冰岛大学的 CS 硕士……"}]
    wiki = WikiKnowledge(SelectClient(), "m", extra_docs=lambda: list(uploads))
    assert wiki.page("upload-d1").title == "冰岛 CS 硕士简介"
    uploads.clear()
    wiki.invalidate_cache("knowledge_search")
    assert wiki.page("upload-d1") is None


def test_tool_path_returns_retrieval_result_shape():
    wiki = WikiKnowledge(SelectClient({"退款": ["refund-invoice"]}), "m")
    result = asyncio.run(wiki.search_with_rewrite("knowledge_search", "退款", top_k=5, history=[]))
    assert result.success is True and result.reranked is False
    assert [i["page_id"] for i in result.data] == ["refund-invoice"]


def test_agent_search_tool_reads_wiki_pages():
    from agents.agent_orchestrator import GeneralAgent, Request, build_shared_rag_tools

    wiki = WikiKnowledge(SelectClient({"德国要办 APS 吗": ["germany-aps"]}), "m")

    class ToolThenText:
        def __init__(self):
            self.calls = []
            owner = self

            class Messages:
                async def create(inner, **kwargs):
                    owner.calls.append(kwargs)
                    if len(owner.calls) == 1:
                        block = type("B", (), {"type": "tool_use", "id": "t1", "name": "search_knowledge_base",
                                               "input": {"query": "德国要办 APS 吗"}})()
                    else:
                        block = type("T", (), {"type": "text", "text": "要办。"})()
                    return type("R", (), {"content": [block]})()

            self.messages = Messages()

    client = ToolThenText()
    agent = GeneralAgent(client, "m")
    agent.set_shared_tools(build_shared_rag_tools(wiki, domain="general"))
    response = asyncio.run(agent.handle(Request(message="德国要办 APS 吗", user_id="u", conv_id="c")))

    assert response.tools_used == ["search_knowledge_base"]
    tool_result = json.loads(client.calls[1]["messages"][-1]["content"][0]["content"])
    assert [r["page_id"] for r in tool_result["results"]] == ["germany-aps"]
