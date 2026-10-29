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

线上闭环（原版是"好的回答存回 wiki"，客服的回答不一定对，所以改成人工确认）：
  选页为空或失败的问题记进 misses.jsonl → wiki_compile --review-misses 让模型建议归属页 →
  人工删掉不对的 → --accept 并入 business/wiki/extra_questions.yaml，加载时并入对应页的常见问法。

后台上传的文档：按标题拆成不超过 UPLOAD_PAGE_MAX_CHARS 的几页（不截断），上传后在后台
用同一个编译提示词补目录条目（摘要 + 常见问法），结果按内容哈希存在数据目录，编好之前先用标题和开头一段。

查询时不做向量召回：一次 LLM 调用读目录、选出最多 3 页，整页注入。
这一步和意图识别并行（RagGate 的投机预取），通常不增加首字延迟。

选页失败（超时、出错）时三层兜底，都是拿问题去比对目录里每页的"常见问法"：
  1. embedding：和 LLM 选页同时把问题向量化，LLM 失败时直接用，不额外等待
     （线上 bge-large，阈值 0.6 时命中 93.5%、无答案误注入 15.4%）
  2. 字符 n-gram：不依赖任何网络，向量服务也不可用时的最后一层（命中约 71%）
阈值由 evaluation/wiki_fallback_calibrate.py 校准。

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
from collections import OrderedDict, deque
from datetime import datetime, timezone
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
EXTRA_QUESTIONS_PATH = WIKI_DIR / "extra_questions.yaml"
MISS_LOG_KEEP = 200                      # 内存里留最近多少条选不中的问题
MISS_LOG_MAX_BYTES = 2 * 1024 * 1024     # 落盘文件上限，到了就不再追加（等人工处理后清空）


def wiki_data_dir() -> Path:
    """运行时数据（上传文档的目录条目、选不中的问题）；和改价覆盖文件放在同一个数据卷。"""
    return Path(os.getenv("GOEUROOPS_WIKI_DATA_DIR", "/app/data/wiki")).expanduser()

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
# 兜底选页（LLM 选页超时或出错时）：问题和目录里的常见问法比相似度，取最像的一两页
EMBED_MIN_SCORE = env_float("GOEUROOPS_WIKI_EMBED_MIN_SCORE", 0.6)
EMBED_TIMEOUT_S = env_float("GOEUROOPS_WIKI_EMBED_TIMEOUT", 2.0)
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


def load_extra_questions() -> Dict[str, List[str]]:
    """人工确认过的补充问法 {页面 id: [问法]}。"""
    if not EXTRA_QUESTIONS_PATH.is_file():
        return {}
    data = yaml.safe_load(EXTRA_QUESTIONS_PATH.read_text(encoding="utf-8")) or {}
    return {str(k): [str(q) for q in v or []] for k, v in (data.get("pages") or {}).items()}


def split_upload(content: str, max_chars: int) -> List[tuple]:
    """上传文档按 Markdown 标题拆成 (小标题, 正文) 段；一段太长再按空行、最后按字数切，不丢内容。"""
    sections: List[List[str]] = [["", ""]]
    for line in content.splitlines(keepends=True):
        m = re.match(r"#{1,3}\s+(.+)", line)
        if m and sections[-1][1].strip():
            sections.append([m.group(1).strip(), ""])
        elif m:
            sections[-1][0] = m.group(1).strip()
        sections[-1][1] += line
    out: List[tuple] = []
    for heading, body in sections:
        chunk = ""
        for para in re.split(r"(?<=\n)\s*\n", body):
            while len(para) > max_chars:                 # 单段超长，硬切
                out.append((heading, para[:max_chars]))
                para = para[max_chars:]
            if chunk and len(chunk) + len(para) > max_chars:
                out.append((heading, chunk))
                chunk = ""
            chunk += para
        if chunk.strip():
            out.append((heading, chunk))
    # 太短的相邻段合并，免得一篇短文被拆成很多页
    merged: List[tuple] = []
    for heading, body in out:
        if merged and len(merged[-1][1]) + len(body) <= max_chars // 2:
            merged[-1] = (merged[-1][0] or heading, merged[-1][1] + body)
        else:
            merged.append((heading, body))
    return [(h, b.strip()) for h, b in merged if b.strip()]


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
        for page in pages:
            page.questions += [q for q in load_extra_questions().get(page.id, []) if q not in page.questions]
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
                 extra_docs: Optional[Callable[[], List[Dict[str, str]]]] = None,
                 embed_fn: Optional[Callable[[List[str]], List[Any]]] = None,
                 data_dir: Optional[Path] = None):
        self._client = client
        self._data_dir = data_dir or wiki_data_dir()
        self._upload_index: Dict[str, Dict[str, Any]] = self._load_upload_index()
        self._compile_lock = asyncio.Lock()
        self.misses: "deque[Dict[str, Any]]" = deque(maxlen=MISS_LOG_KEEP)
        self._extra_docs = extra_docs
        self._embed_fn = embed_fn
        self._phrase_ids: List[str] = []
        self._phrase_vecs: Any = None          # 常见问法的单位向量矩阵；None 表示还没准备好
        self._embed_prep: Optional["asyncio.Task[None]"] = None
        self._model = model
        self.max_pages = max_pages or env_int("GOEUROOPS_WIKI_MAX_PAGES", DEFAULT_MAX_PAGES)
        self.timeout_s = timeout_s or env_float("GOEUROOPS_WIKI_SELECT_TIMEOUT", 4.0)
        self._cache_ttl = cache_ttl_s
        self._cache_size = cache_size
        self._cache: "OrderedDict[str, tuple]" = OrderedDict()
        self._inflight: Dict[str, "asyncio.Task[List[str]]"] = {}
        self.stats = {"selects": 0, "cache_hits": 0, "errors": 0, "empty": 0, "embed_fallback": 0,
                      "local_fallback": 0, "upload_compiled": 0, "total_ms": 0.0}
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
            self._phrase_vecs = None           # 目录变了，问法向量要重算
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
        """后台上传的文档按标题拆成几页。目录条目上传后在后台编译，编好之前用标题和开头一段。"""
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
            base = f"upload-{d.get('doc_id') or hashlib.md5(title.encode()).hexdigest()[:8]}"
            parts = split_upload(content, UPLOAD_PAGE_MAX_CHARS)
            for n, (heading, body) in enumerate(parts, 1):
                page_title = title if len(parts) == 1 else f"{title}（{n}/{len(parts)}{'：' + heading if heading else ''}）"
                entry = self._upload_index.get(self._upload_key(page_title, body)) or {}
                pages.append(WikiPage(
                    id=base if len(parts) == 1 else f"{base}-{n}", title=page_title,
                    domain=str(d.get("domain") or "shared"), content=body, sources=[title],
                    summary=entry.get("summary") or re.sub(r"\s+", " ", body)[:80],
                    questions=list(entry.get("questions") or [page_title]), kind="upload"))
        return pages

    @staticmethod
    def _upload_key(title: str, content: str) -> str:
        return hashlib.sha256(f"{title}\x00{content}".encode("utf-8")).hexdigest()[:16]

    def _load_upload_index(self) -> Dict[str, Dict[str, Any]]:
        path = self._data_dir / "upload_index.json"
        try:
            return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        except (OSError, json.JSONDecodeError) as ex:
            logger.warning("读取上传文档目录条目失败，先用标题和开头一段：%r", ex)
            return {}

    def _save_upload_index(self) -> None:
        live = {self._upload_key(p.title, p.content) for p in self._pages if p.kind == "upload"}
        self._upload_index = {k: v for k, v in self._upload_index.items() if k in live}   # 删掉的文档顺手清掉
        try:
            self._data_dir.mkdir(parents=True, exist_ok=True)
            tmp = self._data_dir / "upload_index.json.tmp"
            tmp.write_text(json.dumps(self._upload_index, ensure_ascii=False, indent=1), encoding="utf-8")
            tmp.replace(self._data_dir / "upload_index.json")
        except OSError as ex:
            logger.warning("保存上传文档目录条目失败（本次进程内仍生效）：%r", ex)

    async def compile_uploads(self) -> int:
        """给还没有目录条目的上传页编条目（和内置页同一个提示词、同样去掉金额），返回编好的页数。"""
        from retrieval.wiki_compile import compile_entry

        async with self._compile_lock:
            todo = [p for p in self._pages if p.kind == "upload"
                    and self._upload_key(p.title, p.content) not in self._upload_index]
            done = 0
            for page in todo:
                try:
                    entry = await compile_entry(self._client, self._model, page)
                except Exception as ex:
                    logger.warning("上传文档 %s 编目录条目失败，继续用标题和开头一段：%r", page.id, ex)
                    continue
                if entry.get("summary") and entry.get("questions"):
                    self._upload_index[self._upload_key(page.title, page.content)] = entry
                    done += 1
            if done or todo:
                self._save_upload_index()
            if done:
                await asyncio.to_thread(self.reload)
                self.stats["upload_compiled"] += done
                logger.info("上传文档目录条目已编译 %d 页", done)
            return done

    def _record_miss(self, query: str, reason: str, ids: List[str]) -> None:
        """选页为空或 LLM 失败的问题：留在内存给后台看，并追加到数据目录，供 wiki_compile --review-misses 处理。"""
        row = {"at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "query": query[:200],
               "reason": reason, "pages": ids}
        self.misses.append(row)
        path = self._data_dir / "misses.jsonl"
        try:
            if path.is_file() and path.stat().st_size > MISS_LOG_MAX_BYTES:
                return
            self._data_dir.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        except OSError as ex:
            logger.warning("记录选不中的问题失败：%r", ex)

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
        # 和 LLM 选页同时把问题向量化：LLM 失败时直接用，不用再等
        embed_task = asyncio.create_task(self.embed_select(query)) if self._embed_fn is not None else None
        try:
            resp = await asyncio.wait_for(self._client.messages.create(
                model=self._model, max_tokens=80, temperature=0, system=system,
                messages=[{"role": "user", "content": safe_text(user)}], **NO_THINKING_KWARGS,
            ), timeout=self.timeout_s)
            ids = self._parse(extract_text_content(resp.content))
            if embed_task is not None:
                embed_task.cancel()
        except Exception as ex:
            self.stats["errors"] += 1
            ids = await self._fallback(query, embed_task, ex)
            self._record_miss(query, "fallback", ids)
        else:
            if not ids:
                self._record_miss(query, "empty", ids)
        if not ids:
            self.stats["empty"] += 1
        self.stats["total_ms"] += (time.monotonic() - t0) * 1000
        return ids

    async def _fallback(self, query: str, embed_task: Optional["asyncio.Task[Optional[List[str]]]"],
                        ex: BaseException) -> List[str]:
        embedded = None
        if embed_task is not None:
            try:
                embedded = await embed_task
            except Exception as embed_ex:
                logger.warning("wiki 兜底向量化失败：%r", embed_ex)
        if embedded is not None:
            self.stats["embed_fallback"] += 1
            logger.warning("wiki 选页失败（%s），改用向量匹配：%s", type(ex).__name__, embedded)
            return embedded
        self.stats["local_fallback"] += 1
        logger.warning("wiki 选页失败（%s），向量也不可用，改用本地问法匹配：%r", type(ex).__name__, ex)
        return self.local_select(query)

    # ── 兜底：向量匹配 ────────────────────────────────────────────────────
    def _prepare_embeddings_sync(self) -> None:
        import numpy as np

        phrases = [(p.id, t) for p in self._pages for t in [p.title, *p.questions] if t.strip()]
        version = self._version
        vecs = np.asarray(self._embed_fn([t for _, t in phrases]), dtype=np.float32)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
        if version == self._version:          # 编码期间目录又变了就作废，下次再算
            self._phrase_ids, self._phrase_vecs = [pid for pid, _ in phrases], vecs

    async def prepare_embeddings(self) -> None:
        """把目录里所有常见问法向量化（线上约 450 条、5 秒左右），放在后台做，不阻塞请求。"""
        if self._embed_fn is None or self._phrase_vecs is not None:
            return
        t0 = time.monotonic()
        try:
            await asyncio.to_thread(self._prepare_embeddings_sync)
            logger.info("wiki 兜底问法向量已就绪：%d 条，%.0f ms", len(self._phrase_ids), (time.monotonic() - t0) * 1000)
        except Exception as ex:
            logger.warning("wiki 兜底问法向量化失败，兜底先用本地问法匹配：%r", ex)

    def _ensure_embeddings(self) -> None:
        if self._phrase_vecs is None and (self._embed_prep is None or self._embed_prep.done()):
            self._embed_prep = asyncio.create_task(self.prepare_embeddings())

    async def embed_select(self, query: str) -> Optional[List[str]]:
        """向量兜底：返回选中的页（可能为空列表）；向量还没准备好或服务不可用时返回 None。"""
        import numpy as np

        if self._embed_fn is None:
            return None
        if self._phrase_vecs is None:
            self._ensure_embeddings()
            return None
        ids, vecs = self._phrase_ids, self._phrase_vecs
        q = await asyncio.wait_for(asyncio.to_thread(self._embed_fn, [query]), timeout=EMBED_TIMEOUT_S)
        qv = np.asarray(q[0], dtype=np.float32)
        sims = vecs @ (qv / (np.linalg.norm(qv) + 1e-9))
        best: Dict[str, float] = {}
        for pid, score in zip(ids, sims.tolist()):
            best[pid] = max(best.get(pid, -1.0), score)
        ranked = sorted(best.items(), key=lambda x: -x[1])[:LOCAL_MAX_PAGES]
        return [pid for pid, score in ranked if score >= EMBED_MIN_SCORE and pid in self._by_id]

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
        """启动时发一次选页请求，让服务商把目录前缀缓存起来，第一位用户不用等冷启动；同时准备兜底向量。"""
        self._ensure_embeddings()
        t0 = time.monotonic()
        ids = await self._select_uncached("瑞典 CS 硕士什么时候截止申请", None)
        logger.info("wiki 选页预热完成：%.0f ms，选中 %s", (time.monotonic() - t0) * 1000, ids)

    def get_stats(self) -> Dict[str, Any]:
        n = max(1, self.stats["selects"])
        return {"backend": "wiki", "pages": len(self._pages), "index_version": self._version,
                "embed_fallback_ready": self._phrase_vecs is not None,
                **{k: v for k, v in self.stats.items() if k != "total_ms"},
                "avg_select_ms": round(self.stats["total_ms"] / n, 1)}


def knowledge_backend() -> str:
    """GOEUROOPS_KNOWLEDGE_BACKEND=wiki|rag，默认 wiki（评测结果见 docs/ARCHITECTURE.md 第 12 节）。"""
    return os.getenv("GOEUROOPS_KNOWLEDGE_BACKEND", "wiki").strip().lower()
