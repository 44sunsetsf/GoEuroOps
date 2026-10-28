import asyncio
import json

from business.catalog import default_prices, get_catalog, set_service_prices
from business.kb_seed import build_seed_documents
from core.rag_gate import RagGate
from retrieval.wiki import WikiKnowledge, build_pages
from retrieval.wiki_compile import article_consistency_errors, catalog_consistency_errors, lint


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


class FakeEmbed:
    """假的向量模型：按关键词给出固定方向的向量，记录调用次数。"""

    AXES = ["退款", "签证", "APS"]

    def __init__(self, fail=False):
        self.fail = fail
        self.calls = 0

    def __call__(self, texts):
        self.calls += 1
        if self.fail:
            raise RuntimeError("embedding down")
        return [[1.0 if axis in t else 0.0 for axis in self.AXES] + [0.01] for t in texts]


def test_llm_failure_uses_embedding_match_before_ngram():
    embed = FakeEmbed()
    wiki = WikiKnowledge(SelectClient(error=RuntimeError("deepseek down")), "m", embed_fn=embed)
    asyncio.run(wiki.prepare_embeddings())
    ids = asyncio.run(wiki.select("德国 APS 怎么办"))
    assert ids and all("APS" in " ".join([wiki.page(i).title, *wiki.page(i).questions]) for i in ids)
    stats = wiki.get_stats()
    assert stats["embed_fallback"] == 1 and stats["local_fallback"] == 0 and stats["embed_fallback_ready"]


def test_embedding_unavailable_falls_back_to_ngram():
    wiki = WikiKnowledge(SelectClient(error=RuntimeError("deepseek down")), "m", embed_fn=FakeEmbed(fail=True))
    question = wiki.page("refund-invoice").questions[0]
    ids = asyncio.run(wiki.select(question))
    assert "refund-invoice" in ids
    assert wiki.get_stats()["local_fallback"] == 1


def test_embeddings_not_ready_yet_falls_back_to_ngram_and_starts_preparing():
    embed = FakeEmbed()
    wiki = WikiKnowledge(SelectClient(error=RuntimeError("deepseek down")), "m", embed_fn=embed)

    async def run():
        question = wiki.page("refund-invoice").questions[0]
        ids = await wiki.select(question)
        await wiki._embed_prep                      # 第一次兜底时开始在后台准备向量
        return ids

    assert "refund-invoice" in asyncio.run(run())
    assert wiki.get_stats()["local_fallback"] == 1 and wiki.get_stats()["embed_fallback_ready"]


def test_successful_llm_select_ignores_embedding():
    embed = FakeEmbed()
    wiki = WikiKnowledge(SelectClient({"能退款吗": ["refund-invoice"]}), "m", embed_fn=embed)
    asyncio.run(wiki.prepare_embeddings())
    assert asyncio.run(wiki.select("能退款吗")) == ["refund-invoice"]
    assert wiki.get_stats()["embed_fallback"] == 0


def test_index_change_invalidates_phrase_vectors():
    uploads = []
    wiki = WikiKnowledge(SelectClient(), "m", embed_fn=FakeEmbed(), extra_docs=lambda: list(uploads))
    asyncio.run(wiki.prepare_embeddings())
    assert wiki.get_stats()["embed_fallback_ready"]
    uploads.append({"doc_id": "d2", "title": "新文档", "content": "内容"})
    wiki.invalidate_cache()
    assert not wiki.get_stats()["embed_fallback_ready"]


class IndexClient:
    """假的编译模型：给任意页面返回固定的摘要和问法（其中一条带金额，应被去掉）。"""

    def __init__(self):
        self.calls = 0
        owner = self

        class Messages:
            async def create(inner, **kwargs):
                owner.calls += 1
                text = json.dumps({"summary": "冰岛 CS 硕士的申请要求", "questions": ["冰岛好申吗", "冰岛读研要 ¥5000 吗"]})
                return type("R", (), {"content": [type("T", (), {"type": "text", "text": text})()]})()

        self.messages = Messages()


def test_long_upload_is_split_into_pages_without_losing_content():
    body = "\n\n".join(f"## 第{i}节\n" + "内容" * 1500 for i in range(3))     # 每节约 3000 字
    uploads = [{"doc_id": "d3", "title": "长文档", "content": body}]
    wiki = WikiKnowledge(SelectClient(), "m", extra_docs=lambda: list(uploads))
    pages = [p for p in wiki.pages if p.kind == "upload"]
    assert len(pages) >= 2 and all(len(p.content) <= 4000 for p in pages)
    assert sum(p.content.count("内容") for p in pages) == 4500          # 以前超过 4000 字直接截断
    assert pages[0].id == "upload-d3-1" and "第0节" in pages[0].title


def test_uploaded_pages_get_compiled_index_entries_and_keep_them(tmp_path):
    uploads = [{"doc_id": "d1", "title": "冰岛 CS 硕士简介", "content": "冰岛大学的 CS 硕士……"}]
    client = IndexClient()
    wiki = WikiKnowledge(client, "m", extra_docs=lambda: list(uploads), data_dir=tmp_path)
    assert wiki.page("upload-d1").questions == ["冰岛 CS 硕士简介"]        # 编好之前只有标题
    assert asyncio.run(wiki.compile_uploads()) == 1
    page = wiki.page("upload-d1")
    assert page.summary == "冰岛 CS 硕士的申请要求" and page.questions == ["冰岛好申吗"]
    assert asyncio.run(wiki.compile_uploads()) == 0 and client.calls == 1   # 编过的不重编
    again = WikiKnowledge(IndexClient(), "m", extra_docs=lambda: list(uploads), data_dir=tmp_path)
    assert again.page("upload-d1").questions == ["冰岛好申吗"]              # 重启后从数据目录读回


def test_empty_and_failed_selects_are_recorded_for_review(tmp_path):
    from retrieval.wiki_compile import read_misses

    wiki = WikiKnowledge(SelectClient({"能退款吗": ["refund-invoice"]}), "m", data_dir=tmp_path)
    asyncio.run(wiki.select("能退款吗"))
    asyncio.run(wiki.select("有没有学生价"))
    assert [m["query"] for m in wiki.misses] == ["有没有学生价"] and wiki.misses[0]["reason"] == "empty"
    failing = WikiKnowledge(SelectClient(error=RuntimeError("down")), "m", data_dir=tmp_path)
    asyncio.run(failing.select("学生价怎么算"))
    assert failing.misses[0]["reason"] == "fallback"
    assert read_misses(tmp_path / "misses.jsonl") == ["有没有学生价", "学生价怎么算"]


def test_accepted_extra_questions_join_the_page(monkeypatch, tmp_path):
    import retrieval.wiki as wiki_mod
    import retrieval.wiki_compile as compile_mod

    extra = tmp_path / "extra_questions.yaml"
    suggestions = tmp_path / "miss_suggestions.yaml"
    monkeypatch.setattr(wiki_mod, "EXTRA_QUESTIONS_PATH", extra)
    monkeypatch.setattr(compile_mod, "EXTRA_QUESTIONS_PATH", extra)
    monkeypatch.setattr(compile_mod, "SUGGESTIONS_PATH", suggestions)
    suggestions.write_text("pages:\n  discounts-payment: [有没有学生价]\n", encoding="utf-8")
    assert compile_mod.accept_suggestions() == 1 and not suggestions.exists()
    page = next(p for p in build_pages() if p.id == "discounts-payment")
    assert "有没有学生价" in page.questions


def test_lint_catches_rule_text_that_contradicts_catalog_fields():
    from business.catalog import Catalog

    catalog = get_catalog()
    assert catalog_consistency_errors(catalog) == []
    data = catalog.model_dump(mode="json")
    data["discounts"]["early_bird"]["rate"] = 0.85                         # 改了字段，规则文字还写 9 折
    data["payment"]["deposit_ratio"] = 0.2                                  # 规则文字还写 30%
    data["services"][1]["upgrade_note"] = "30 天内升级可抵扣 ¥199。"         # 写死金额，改价后过期
    errors = " ".join(catalog_consistency_errors(Catalog.model_validate(data)))
    assert "early_bird" in errors and "payment.rule" in errors and "upgrade_note" in errors


def test_lint_catches_articles_that_contradict_catalog():
    catalog = get_catalog()
    assert article_consistency_errors("ok.md", "套餐签约时付 30% 定金，早鸟 9 折。", catalog) == []
    errors = article_consistency_errors("bad.md", "签约时付 20% 定金，早鸟 8 折，单次咨询 ¥199。", catalog)
    assert len(errors) == 3


def test_review_misses_only_suggests_existing_pages(monkeypatch, tmp_path):
    import retrieval.wiki_compile as compile_mod

    monkeypatch.setattr(compile_mod, "SUGGESTIONS_PATH", tmp_path / "miss_suggestions.yaml")
    misses = tmp_path / "misses.jsonl"
    misses.write_text("\n".join(json.dumps({"query": q}) for q in ["有没有学生价", "今天天气", "有没有学生价", "x"]),
                      encoding="utf-8")

    class ReviewClient:
        class messages:
            @staticmethod
            async def create(**kwargs):
                text = json.dumps({"有没有学生价": "discounts-payment", "今天天气": None, "x": "no-such-page"})
                return type("R", (), {"content": [type("T", (), {"type": "text", "text": text})()]})()

    out = asyncio.run(compile_mod.review_misses(ReviewClient(), "m", misses))
    assert out == {"discounts-payment": ["有没有学生价"]}
    assert (tmp_path / "miss_suggestions.yaml").is_file()
