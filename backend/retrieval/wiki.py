"""LLM Wiki 知识库：按主题组织的整页知识 + LLM 读目录选页，替代向量检索。

和 Karpathy 的 LLM Wiki 一样分三层，但按本项目的场景改造过：

  原始资料   catalog.yaml / countries.yaml / articles/*.md（人维护，唯一事实来源）
  wiki 页面  按主题把原始资料组织成整页：
               - 代码页：服务、价格、优惠、退款、国家概览、五国对比，每次从 YAML 现场生成，
                 后台改价立即生效，也不存在 LLM 抄错数字的问题
               - 编译页：术语表（business/wiki/glossary.md），由 LLM 离线从原始资料编译，
                 提交前经过 lint（页面里出现的数字必须能在原始资料里找到）
  目录       business/wiki/index.yaml：每页的摘要和"用户会怎么问"，由 LLM 离线编译，
             lint 禁止出现数字，所以改价不会让目录过期

查询时不做向量召回：一次 LLM 调用读目录、选出最多 3 页，整页注入。
这一步和意图识别并行（RagGate 的投机预取），通常不增加首字延迟。

对外接口和 RetrievalManager 一致（search_fast / search_with_rewrite），
编排器、门控、检索工具都不用改。编译和 lint 见 retrieval/wiki_compile.py。
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional

import yaml

from business.catalog import get_catalog, get_countries
from business.kb_seed import build_seed_documents
from core.config import env_float, env_int
from core.llm_utils import NO_THINKING_KWARGS, extract_text_content, safe_text
from core.text_embedding import ngram_profile, sparse_cosine

logger = logging.getLogger(__name__)

WIKI_DIR = Path(__file__).resolve().parent.parent / "business" / "wiki"
INDEX_PATH = WIKI_DIR / "index.yaml"
GLOSSARY_PATH = WIKI_DIR / "glossary.md"

# 页面 id → (标题, domain, 组成它的种子文档标题前缀)。种子文档由 business/kb_seed.py 从 YAML 生成。
PAGE_LAYOUT: List[tuple] = [
    ("studio", "关于指北工作室：介绍、常见问题与预约流程", "shared",
     ["关于指北：工作室介绍", "常见问题（关于指北工作室）", "预约咨询流程"]),
    ("services-selection", "选校类服务：免费初步沟通、单次选校咨询、选校定位全案", "consulting",
     ["服务介绍：免费初步沟通", "服务介绍：单次选校咨询", "服务介绍：选校定位全案"]),
    ("services-writing", "文书类服务：PS/动机信、简历、推荐信、文书套餐", "consulting",
     ["服务介绍：PS/动机信单篇辅导", "服务介绍：学术简历（CV）优化", "服务介绍：推荐信指导",
      "服务介绍：文书额外修改轮", "服务介绍：文书套餐 · 3 个项目", "服务介绍：文书套餐 · 5 个项目"]),
    ("services-fullservice", "全程陪跑与单项服务：全程陪跑、Plus、网申审核、模拟面试、签证指导", "consulting",
     ["服务介绍：全程陪跑（", "服务介绍：全程陪跑 Plus", "服务介绍：网申材料审核",
      "服务介绍：模拟面试", "服务介绍：签证/居留材料指导"]),
    ("packages", "套餐对比、怎么选与全部价目一览", "consulting", ["套餐对比与怎么选"]),
    ("discounts-payment", "优惠规则、付款方式与定金", "shared", ["优惠规则（早鸟、团报、老带新）", "付款方式与定金规则"]),
    ("refund-invoice", "退款政策与发票", "billing", ["退款政策", "发票说明"]),
    ("country-sweden", "瑞典（Sweden）CS 硕士申请概览", "consulting", ["瑞典（Sweden）"]),
    ("country-germany", "德国（Germany）CS 硕士申请概览", "consulting", ["德国（Germany）"]),
    ("country-netherlands", "荷兰（Netherlands）CS 硕士申请概览", "consulting", ["荷兰（Netherlands）"]),
    ("country-finland", "芬兰（Finland）CS 硕士申请概览", "consulting", ["芬兰（Finland）"]),
    ("country-denmark", "丹麦（Denmark）CS 硕士申请概览", "consulting", ["丹麦（Denmark）"]),
    ("materials", "CS 硕士申请材料清单与准备建议", "consulting", ["CS 硕士申请材料清单"]),
    ("germany-aps", "德国 APS 审核与 uni-assist 申请说明", "consulting", ["德国 APS 审核"]),
    ("school-selection", "选校基本思路：冲刺、匹配、保底", "consulting", ["选校基本思路"]),
    ("timeline", "2027 年秋季入学申请时间线（五国通用参考）", "consulting", ["2027 年秋季入学申请时间线"]),
]
COMPARE_PAGE = "countries-compare"
GLOSSARY_PAGE = "glossary"

DEFAULT_MAX_PAGES = 3
# 本地兜底：选页模型超时或出错时，用字符 n-gram 把问题和目录里的常见问法比对
LOCAL_MIN_SCORE = env_float("GOEUROOPS_WIKI_LOCAL_MIN_SCORE", 0.25)
LOCAL_MAX_PAGES = 2
UPLOAD_PAGE_MAX_CHARS = 4000


@dataclass
class WikiPage:
    id: str
    title: str
    domain: str
    content: str
    sources: List[str] = field(default_factory=list)   # 组成这一页的原始资料（种子文档标题）
    summary: str = ""
    questions: List[str] = field(default_factory=list)
    kind: str = "code"                                  # code 代码现场生成 / compiled LLM 编译

    def as_item(self, score: float = 1.0) -> Dict[str, Any]:
        """转成和向量检索结果一样的结构，门控和 Agent 不用区分来源。"""
        return {"title": self.title, "content": self.content, "score": score, "reranked": False,
                "domain": self.domain, "source": "wiki", "page_id": self.id, "sources": list(self.sources)}


def _match(title: str, prefixes: List[str]) -> bool:
    return any(title.startswith(p) for p in prefixes)


def _compare_page(seed_titles: List[str]) -> WikiPage:
    book = get_countries()
    lines = [f"五国英语授课 CS 硕士横向对比（{book.last_verified}，面向{book.target_intake}，请以各国官网为准）。"]
    fields = [("申请平台", "portal"), ("申请时间", "application_window"), ("申请费", "application_fee"),
              ("非欧盟学费", "tuition_non_eu"), ("学制", "duration"), ("语言要求", "language"), ("特点", "highlights")]
    for label, attr in fields:
        lines.append(f"\n【{label}】")
        lines += [f"- {ct.key}：{getattr(ct, attr)}" for ct in book.countries]
    return WikiPage(
        id=COMPARE_PAGE, title="五国横向对比：申请时间、申请费、学费、语言要求、学制", domain="consulting",
        content="\n".join(lines), sources=[t for t in seed_titles if "CS 硕士申请概览" in t],
    )


def _price_list(seed_docs: List[Dict[str, str]]) -> str:
    rows = [d["title"].removeprefix("服务介绍：") for d in seed_docs if d["title"].startswith("服务介绍：")]
    return "全部服务价目一览：" + "；".join(rows) + "。"


def load_glossary() -> Optional[WikiPage]:
    if not GLOSSARY_PATH.is_file():
        return None
    raw = GLOSSARY_PATH.read_text(encoding="utf-8")
    meta: Dict[str, Any] = {}
    body = raw
    if raw.startswith("---"):
        _, front, body = raw.split("---", 2)
        meta = yaml.safe_load(front) or {}
    return WikiPage(id=GLOSSARY_PAGE, title=str(meta.get("title") or "术语表"), domain="consulting",
                    content=body.strip(), sources=list(meta.get("sources") or []), kind="compiled")


def load_index() -> Dict[str, Dict[str, Any]]:
    if not INDEX_PATH.is_file():
        return {}
    data = yaml.safe_load(INDEX_PATH.read_text(encoding="utf-8")) or {}
    return dict(data.get("pages") or {})


def build_pages(*, with_index: bool = True) -> List[WikiPage]:
    """从当前的 YAML 和文章现场组装全部页面（价格随后台改价实时变化）。"""
    _, seed_docs = build_seed_documents()
    titles = [d["title"] for d in seed_docs]
    pages: List[WikiPage] = []
    for page_id, title, domain, prefixes in PAGE_LAYOUT:
        docs = [d for d in seed_docs if _match(d["title"], prefixes)]
        parts = [f"## {d['title']}\n{d['content']}" for d in docs]
        if page_id == "packages":
            parts.append(_price_list(seed_docs))
        pages.append(WikiPage(id=page_id, title=title, domain=domain, content="\n\n".join(parts),
                              sources=[d["title"] for d in docs]))
    pages.append(_compare_page(titles))
    glossary = load_glossary()
    if glossary is not None:
        pages.append(glossary)
    if with_index:
        index = load_index()
        for page in pages:
            entry = index.get(page.id) or {}
            page.summary = str(entry.get("summary") or "")
            page.questions = [str(q) for q in entry.get("questions") or []]
    return pages


def render_index(pages: List[WikiPage]) -> str:
    """给选页模型看的目录：id、标题、摘要、常见问法。"""
    blocks = []
    for p in pages:
        qs = "；".join(p.questions)
        blocks.append(f"- id: {p.id}\n  标题: {p.title}\n  摘要: {p.summary or '（无）'}" + (f"\n  常见问法: {qs}" if qs else ""))
    return "\n".join(blocks)


SELECT_SYSTEM = """你是"指北"留学工作室知识库的导航员。工作室只做瑞典、德国、荷兰、芬兰、丹麦五国英语授课 CS 硕士的申请咨询和文书辅导。

下面是知识库 wiki 的目录。根据用户的问题，选出回答它真正需要读的页面：
- 最多 {max_pages} 页，按重要性排序；一页就够时只选一页。
- 问题涉及多个国家或"哪个更好/有什么区别"时，优先选五国横向对比页。
- 问具体某项服务或价格时，选包含该服务的服务页；问"有哪些服务/怎么选"时选套餐对比页。
- 目录里没有任何页面能回答时（闲聊、问候、五国以外的国家、本科/博士、与留学无关），返回空列表，不要硬选。

只输出 JSON，不要解释：{{"pages": ["页面 id", ...]}}

目录：
{index}"""


class WikiKnowledge:
    """LLM Wiki 检索：读目录选页。接口对齐 RetrievalManager，可直接作为编排器的共享检索工具。"""

    def __init__(self, client: Any, model: str, *, max_pages: Optional[int] = None,
                 timeout_s: Optional[float] = None, cache_ttl_s: float = 300.0, cache_size: int = 512,
                 extra_docs: Optional[Callable[[], List[Dict[str, str]]]] = None):
        self._client = client
        self._extra_docs = extra_docs
        self._model = model
        self.max_pages = max_pages or env_int("GOEUROOPS_WIKI_MAX_PAGES", DEFAULT_MAX_PAGES)
        self.timeout_s = timeout_s or env_float("GOEUROOPS_WIKI_SELECT_TIMEOUT", 4.0)
        self._cache_ttl = cache_ttl_s
        self._cache_size = cache_size
        self._cache: "OrderedDict[str, tuple]" = OrderedDict()
        self._inflight: Dict[str, "asyncio.Task[List[str]]"] = {}
        self.stats = {"selects": 0, "cache_hits": 0, "errors": 0, "empty": 0, "total_ms": 0.0}
        self._pages: List[WikiPage] = []
        self._by_id: Dict[str, WikiPage] = {}
        self._version = ""
        self.reload()

    # ── 页面 ──────────────────────────────────────────────────────────────
    def reload(self) -> None:
        """重新组装页面（后台上传/删除文档后调用）；目录变了就清掉选页缓存。"""
        self._catalog = get_catalog()
        self._pages = build_pages() + self._upload_pages()
        self._by_id = {p.id: p for p in self._pages}
        self._profiles = {p.id: [ngram_profile(t) for t in [p.title, *p.questions] if t.strip()] for p in self._pages}
        version = hashlib.sha256(render_index(self._pages).encode("utf-8")).hexdigest()[:10]
        if version != self._version:
            self._cache.clear()
        self._version = version

    def _refresh_if_catalog_changed(self) -> None:
        # 改价后 get_catalog() 的缓存会被清掉、换成新对象，页面内容要跟着变；
        # 目录不含数字，所以选页缓存仍然有效
        catalog = get_catalog()
        if catalog is not self._catalog:
            self._catalog = catalog
            uploads = [p for p in self._pages if p.kind == "upload"]
            self._pages = build_pages() + uploads
            self._by_id = {p.id: p for p in self._pages}

    def _upload_pages(self) -> List[WikiPage]:
        """后台上传的文档各自成页。没有经过离线编译，目录里用标题和开头一段当摘要。"""
        if self._extra_docs is None:
            return []
        try:
            docs = self._extra_docs()
        except Exception as ex:
            logger.warning("读取上传文档失败，wiki 只用内置页面：%s", ex)
            return []
        pages = []
        for d in docs:
            content = str(d.get("content", "")).strip()
            if not content:
                continue
            title = str(d.get("title") or "未命名文档")
            pages.append(WikiPage(
                id=f"upload-{d.get('doc_id') or hashlib.md5(title.encode()).hexdigest()[:8]}", title=title,
                domain=str(d.get("domain") or "shared"), content=content[:UPLOAD_PAGE_MAX_CHARS], sources=[title],
                summary=re.sub(r"\s+", " ", content)[:80], questions=[title], kind="upload"))
        return pages

    @property
    def pages(self) -> List[WikiPage]:
        return list(self._pages)

    def page(self, page_id: str) -> Optional[WikiPage]:
        return self._by_id.get(page_id)

    # ── 选页 ──────────────────────────────────────────────────────────────
    async def select(self, query: str, history: Optional[List[Dict[str, str]]] = None) -> List[str]:
        """返回选中的页面 id。同一个问题并发调用只发一次请求，结果缓存一段时间。"""
        query = (query or "").strip()
        if not query:
            return []
        key = self._cache_key(query, history)
        hit = self._cache.get(key)
        if hit and time.monotonic() - hit[0] < self._cache_ttl:
            self._cache.move_to_end(key)
            self.stats["cache_hits"] += 1
            return list(hit[1])
        task = self._inflight.get(key)
        if task is None:
            task = asyncio.create_task(self._select_uncached(query, history))
            self._inflight[key] = task
            task.add_done_callback(lambda _t, k=key: self._inflight.pop(k, None))
        # shield：门控会取消没用上的投机任务，不能把共享的选页请求一起取消
        ids = await asyncio.shield(task)
        self._cache[key] = (time.monotonic(), list(ids))
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return ids

    @staticmethod
    def _cache_key(query: str, history: Optional[List[Dict[str, str]]]) -> str:
        prev = ""
        for turn in reversed(history or []):
            if turn.get("role") == "user" and str(turn.get("content", "")).strip() != query:
                prev = str(turn.get("content", ""))
                break
        return f"{prev}\x00{query}"

    async def _select_uncached(self, query: str, history: Optional[List[Dict[str, str]]]) -> List[str]:
        t0 = time.monotonic()
        self.stats["selects"] += 1
        system = SELECT_SYSTEM.format(max_pages=self.max_pages, index=render_index(self._pages))
        prev = [str(t.get("content", "")) for t in (history or [])[-4:] if t.get("role") == "user"]
        user = (f"此前用户说过：{' / '.join(prev[-2:])}\n" if prev else "") + f"用户问题：{query}"
        try:
            resp = await asyncio.wait_for(self._client.messages.create(
                model=self._model, max_tokens=80, temperature=0, system=system,
                messages=[{"role": "user", "content": safe_text(user)}], **NO_THINKING_KWARGS,
            ), timeout=self.timeout_s)
            ids = self._parse(extract_text_content(resp.content))
        except Exception as ex:
            self.stats["errors"] += 1
            self.stats["local_fallback"] = self.stats.get("local_fallback", 0) + 1
            logger.warning("wiki 选页失败（%s），改用本地问法匹配：%r", type(ex).__name__, ex)
            ids = self.local_select(query)
        if not ids:
            self.stats["empty"] += 1
        self.stats["total_ms"] += (time.monotonic() - t0) * 1000
        return ids

    def local_select(self, query: str) -> List[str]:
        """不调模型的选页：问题和每页"常见问法"的字符 n-gram 余弦相似度，取最像的一两页。"""
        q = ngram_profile(query)
        scored = sorted(((max((sparse_cosine(q, prof) for prof in profs), default=0.0), pid)
                         for pid, profs in self._profiles.items()), reverse=True)
        return [pid for score, pid in scored[:LOCAL_MAX_PAGES] if score >= LOCAL_MIN_SCORE]

    def _parse(self, text: str) -> List[str]:
        m = re.search(r"\{.*\}", text or "", re.S)
        try:
            raw = json.loads(m.group(0)).get("pages", []) if m else []
        except (json.JSONDecodeError, AttributeError):
            raw = []
        ids = [str(i) for i in raw if str(i) in self._by_id]
        return list(dict.fromkeys(ids))[: self.max_pages]

    # ── 对齐 RetrievalManager 的接口 ─────────────────────────────────────
    async def search_fast(self, tool_name: str, query: str, top_k: int = 3, domain: Optional[str] = None,
                          history: Optional[List[Dict[str, str]]] = None) -> List[Dict[str, Any]]:
        """门控预取用。domain 不过滤：wiki 页面按主题组织，跨领域问题需要跨领域的页。"""
        self._refresh_if_catalog_changed()
        ids = await self.select(query, history)
        n = len(ids)
        return [self._by_id[i].as_item(score=1.0 - 0.05 * rank) for rank, i in enumerate(ids[: max(top_k, 1)]) if n]

    async def search_with_rewrite(self, tool_name: str, query: str, top_k: int = 3, domain: Optional[str] = None,
                                  history: Optional[List[Dict[str, str]]] = None) -> Any:
        """模型主动调用 search_knowledge_base 时走这里，返回整页。"""
        items = await self.search_fast(tool_name, query, top_k=min(top_k, self.max_pages), domain=domain, history=history)
        return SimpleNamespace(success=True, data=items, reranked=False, error="")

    def invalidate_cache(self, tool_name: Optional[str] = None) -> None:
        """和 RetrievalManager.invalidate_cache 同名：知识库内容变了，重组页面并清掉选页缓存。"""
        self.reload()
        self._cache.clear()

    async def warmup(self) -> None:
        """启动时发一次选页请求，让服务商把目录前缀缓存起来，第一位用户不用等冷启动。"""
        t0 = time.monotonic()
        ids = await self._select_uncached("瑞典 CS 硕士什么时候截止申请", None)
        logger.info("wiki 选页预热完成：%.0f ms，选中 %s", (time.monotonic() - t0) * 1000, ids)

    def get_stats(self) -> Dict[str, Any]:
        n = max(1, self.stats["selects"])
        return {"backend": "wiki", "pages": len(self._pages), "index_version": self._version,
                **{k: v for k, v in self.stats.items() if k != "total_ms"},
                "avg_select_ms": round(self.stats["total_ms"] / n, 1)}


def knowledge_backend() -> str:
    """GOEUROOPS_KNOWLEDGE_BACKEND=wiki|rag，默认 wiki（评测结果见 docs/ARCHITECTURE.md 第 12 节）。"""
    return os.getenv("GOEUROOPS_KNOWLEDGE_BACKEND", "wiki").strip().lower()
