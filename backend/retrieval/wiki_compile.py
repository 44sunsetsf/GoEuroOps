"""LLM Wiki 的离线编译和 lint。

用法（在 backend/ 目录下）：
    python -m retrieval.wiki_compile              # 编译术语表和目录，然后 lint
    python -m retrieval.wiki_compile --only index # 只重编目录（新增/调整页面后）
    python -m retrieval.wiki_compile --lint       # 只检查，不调模型（CI 里由 tests/test_wiki.py 执行）
    python -m retrieval.wiki_compile --review-misses misses.jsonl   # 线上选不中的问题 → 建议归到哪页
    python -m retrieval.wiki_compile --accept     # 人工删掉不对的建议后，并入补充问法

编译产物提交进 git，人工看过 diff 再合并：
  business/wiki/glossary.md   术语表：只从原始资料里摘取，标注来源
  business/wiki/index.yaml    目录：每页的摘要和用户常见问法
  business/wiki/extra_questions.yaml  补充问法：线上选不中、人工确认过的说法（见 --review-misses）

编译时模型只看原始资料，不看评测集，避免把评测题"背"进目录。

lint 规则：
  - 每个页面都有目录条目，目录里没有多余的页面
  - 目录（摘要、常见问法）不含金额和折扣：价格在后台随时会改，目录必须不随之过期
  - 术语表里出现的每个数字都能在原始资料里找到；标注的来源必须是真实存在的种子文档
  - 术语表记录了编译时原始资料的哈希，原始资料改了必须重编
  - 补充问法只能挂在已有页面上，也不含金额
  - 手写文字和结构化数据不能矛盾：catalog 里的规则文字（几折、定金比例、保底比例）要和对应字段一致，
    除老带新外不写死金额；文章里不写金额，提到的折扣和定金比例要和 catalog 一致
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from business.kb_seed import build_seed_documents
from core.llm_utils import NO_THINKING_KWARGS, extract_text_content
from business.catalog import Catalog, get_catalog
from retrieval.wiki import (EXTRA_QUESTIONS_PATH, GLOSSARY_PAGE, GLOSSARY_PATH, INDEX_PATH, WIKI_DIR, WikiPage,
                            build_pages, load_extra_questions, render_index)

BUSINESS = Path(__file__).resolve().parent.parent / "business"
_NUM = re.compile(r"\d+(?:[.,]\d+)*")
# 目录里禁止的是金额和折扣（后台随时会改价），雅思分数、学分这类出自静态资料的数字可以出现
_PRICE = re.compile(r"[¥￥]\s*\d|\d[\d,]*\s*(?:元|块|折|rmb|RMB)|\d+\s*%")
_AMOUNT = re.compile(r"[¥￥]\s*(\d[\d,]*)|(\d[\d,]*)\s*(?:元|块)")
_ZHE = re.compile(r"(\d+(?:\.\d+)?)\s*折")
_PERCENT = re.compile(r"(\d+(?:\.\d+)?)\s*%")
SUGGESTIONS_PATH = WIKI_DIR / "miss_suggestions.yaml"


def raw_sources_text() -> str:
    """术语表的原始资料：国家资料和文章（不含价目，术语表不写价格）。"""
    parts = [(BUSINESS / "countries.yaml").read_text(encoding="utf-8")]
    parts += [p.read_text(encoding="utf-8") for p in sorted((BUSINESS / "articles").glob("*.md"))]
    return "\n\n".join(parts)


def sources_hash() -> str:
    return hashlib.sha256(raw_sources_text().encode("utf-8")).hexdigest()[:12]


def numbers_in(text: str) -> set:
    return {n.replace(",", "") for n in _NUM.findall(text or "")}


# ── 编译 ──────────────────────────────────────────────────────────────────

GLOSSARY_PROMPT = """下面是一家留学咨询工作室的原始资料（五国英语授课 CS 硕士申请）。请从中整理一份"术语表"wiki 页面，帮助学生看懂申请中遇到的专有名词。

要求：
1. 只收录资料里实际出现的术语：申请平台和系统、审核与认证、证件与签证、学分与语言标准、选校概念等，大约 20–30 条。
2. 每条格式：`- **术语**（所属国家或"通用"）：一两句话解释它是什么、什么时候需要。来源：《文档标题》`
   文档标题只能从这个列表里选：{titles}
3. 只用资料里的信息，不补充资料外的知识；资料里的数字可以照抄，但不要写资料里没有的数字；不写工作室的服务价格。
4. 按主题分组，用 `### 分组名` 作小标题。只输出页面正文，不要前言。

原始资料：
{sources}"""

INDEX_PROMPT = """你在为一家留学咨询工作室的知识库编写 wiki 目录条目。下面是其中一页的全文。

请输出 JSON：{{"summary": "...", "questions": ["...", ...]}}
- summary：一句话（40 字以内）说清这一页能回答哪类问题。
- questions：学生在聊天框里可能怎么问、而这一页能回答的问题，12–16 条。要覆盖：
  正式问法、口语化问法、只提专有名词或缩写的问法、拼写不规范的问法、追问式的短句。
- 摘要和问题里不写任何价格、金额、折扣（价格会变，目录不能跟着过期），用"多少钱""优惠多少"这类说法；雅思分数、学分这类资料里的数字可以出现。
- 只写这一页真正能回答的问题。

页面标题：{title}
页面全文：
{content}"""


async def _ask(client: Any, model: str, prompt: str, max_tokens: int) -> str:
    resp = await client.messages.create(model=model, max_tokens=max_tokens, temperature=0.2,
                                        messages=[{"role": "user", "content": prompt}], **NO_THINKING_KWARGS)
    return extract_text_content(resp.content).strip()


async def compile_glossary(client: Any, model: str) -> None:
    _, docs = build_seed_documents()
    titles = "、".join(f"《{d['title']}》" for d in docs if "服务介绍" not in d["title"])
    body = await _ask(client, model, GLOSSARY_PROMPT.format(titles=titles, sources=raw_sources_text()), 4000)
    body = re.sub(r"^```\w*\n|\n```$", "", body).strip()
    cited = sorted({t for t in re.findall(r"《([^》]+)》", body) if any(d["title"] == t for d in docs)})
    front = {"title": "术语表：申请平台、审核认证、签证与常见概念", "kind": "compiled",
             "compiled_from": sources_hash(), "sources": cited}
    WIKI_DIR.mkdir(parents=True, exist_ok=True)
    GLOSSARY_PATH.write_text("---\n" + yaml.safe_dump(front, allow_unicode=True, sort_keys=False) + "---\n" + body + "\n",
                             encoding="utf-8")
    print(f"术语表已写入 {GLOSSARY_PATH}（引用 {len(cited)} 篇来源）")


async def compile_entry(client: Any, model: str, page: WikiPage,
                        sem: Optional[asyncio.Semaphore] = None) -> Dict[str, Any]:
    """给一页编目录条目（摘要 + 常见问法）。离线编译和后台上传文档共用；编不出来返回空摘要。"""
    data: Dict[str, Any] = {}
    for _ in range(2):                        # 输出被截断或 JSON 不合法时重试一次
        if sem is None:
            text = await _ask(client, model, INDEX_PROMPT.format(title=page.title, content=page.content), 3000)
        else:
            async with sem:
                text = await _ask(client, model, INDEX_PROMPT.format(title=page.title, content=page.content), 3000)
        m = re.search(r"\{.*\}", text, re.S)
        try:
            data = json.loads(m.group(0)) if m else {}
        except json.JSONDecodeError:
            data = {}
        if data.get("summary") and data.get("questions"):
            break
    # 兜底：模型偶尔还是会写金额，删掉含金额的问法
    questions = [str(q) for q in data.get("questions", []) if not _PRICE.search(str(q))]
    summary = str(data.get("summary", "")).strip()
    if _PRICE.search(summary):
        summary = ""
    return {"summary": summary, "questions": questions[:24]}


async def compile_index(client: Any, model: str, missing_only: bool = False) -> None:
    pages = build_pages(with_index=False)
    old = (yaml.safe_load(INDEX_PATH.read_text(encoding="utf-8")) or {}).get("pages", {}) if INDEX_PATH.is_file() else {}
    if missing_only:
        pages = [p for p in pages if not (old.get(p.id) or {}).get("questions")]
    sem = asyncio.Semaphore(4)

    async def one(p: WikiPage) -> tuple:
        return p.id, {"title": p.title, **await compile_entry(client, model, p, sem)}

    fresh = dict(await asyncio.gather(*(one(p) for p in pages)))
    ids = [p.id for p in build_pages(with_index=False)]
    entries = {}
    for pid in ids:
        new = fresh.get(pid)
        # 这一页这次没编出来（或没参与编译）就保留旧条目
        entries[pid] = new if new and new["summary"] and new["questions"] else old.get(pid, new or {})
    out = {"note": "由 python -m retrieval.wiki_compile 生成，改动前先改原始资料或编译提示词，再重编", "pages": entries}
    INDEX_PATH.write_text(yaml.safe_dump(out, allow_unicode=True, sort_keys=False, width=200), encoding="utf-8")
    print(f"目录已写入 {INDEX_PATH}（{len(entries)} 页，本次编译 {len(pages)} 页）")


# ── 线上选不中的问题 → 补充问法 ──────────────────────────────────────────────

REVIEW_PROMPT = """下面是一家留学咨询工作室知识库 wiki 的目录，以及线上用户问过、但当时没有选中任何页面（或选页失败）的问题。
逐条判断：目录里有没有哪一页其实能回答它。能回答就给出页面 id；闲聊、问候、五国以外、资料里确实没有的，给 null。

只输出 JSON 对象，键是问题原文，值是页面 id 或 null，不要解释。

目录：
{index}

问题：
{queries}"""


def read_misses(path: Path) -> List[str]:
    """线上记录的选不中问题（JSONL），去重保序。"""
    seen: Dict[str, None] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            q = str(json.loads(line).get("query", "")).strip()
        except (json.JSONDecodeError, AttributeError):
            continue
        if q and not _PRICE.search(q):
            seen.setdefault(q, None)
    return list(seen)


async def review_misses(client: Any, model: str, path: Path) -> Dict[str, List[str]]:
    """让模型给每个选不中的问题建议归属页，写进 miss_suggestions.yaml 等人工确认，不直接改目录。"""
    pages = build_pages()
    ids = {p.id for p in pages}
    queries = read_misses(path)
    suggestions: Dict[str, List[str]] = {}
    for start in range(0, len(queries), 30):
        batch = queries[start:start + 30]
        text = await _ask(client, model, REVIEW_PROMPT.format(index=render_index(pages), queries="\n".join(batch)), 3000)
        m = re.search(r"\{.*\}", text, re.S)
        try:
            picks = json.loads(m.group(0)) if m else {}
        except json.JSONDecodeError:
            picks = {}
        for q in batch:
            pid = picks.get(q)
            if pid in ids and q not in next(p for p in pages if p.id == pid).questions:
                suggestions.setdefault(pid, []).append(q)
    out = {"note": "模型建议的归属页。删掉不对的，再运行 python -m retrieval.wiki_compile --accept", "pages": suggestions}
    SUGGESTIONS_PATH.write_text(yaml.safe_dump(out, allow_unicode=True, sort_keys=False, width=200), encoding="utf-8")
    print(f"{len(queries)} 个问题，{sum(map(len, suggestions.values()))} 个有建议，已写入 {SUGGESTIONS_PATH}")
    return suggestions


def accept_suggestions() -> int:
    """把人工确认过的建议并入 extra_questions.yaml，删掉建议文件。返回新增条数。"""
    if not SUGGESTIONS_PATH.is_file():
        print("没有待确认的建议")
        return 0
    picked = (yaml.safe_load(SUGGESTIONS_PATH.read_text(encoding="utf-8")) or {}).get("pages") or {}
    extra = load_extra_questions()
    added = 0
    for pid, qs in picked.items():
        bucket = extra.setdefault(str(pid), [])
        for q in qs or []:
            if str(q) not in bucket:
                bucket.append(str(q))
                added += 1
    out = {"note": "线上选不中、人工确认过的用户说法，加载时并入对应页的常见问法", "pages": extra}
    EXTRA_QUESTIONS_PATH.write_text(yaml.safe_dump(out, allow_unicode=True, sort_keys=False, width=200), encoding="utf-8")
    SUGGESTIONS_PATH.unlink()
    print(f"并入 {added} 条补充问法")
    return added


# ── 手写文字和结构化数据的一致性 ───────────────────────────────────────────────

def _zhe(rate: float) -> str:
    """0.9 → "9"，0.95 → "95"（中文的几折写法）。"""
    r = round(rate * 100)
    return str(r // 10) if r % 10 == 0 else str(r)


def _pct(ratio: float) -> str:
    return f"{ratio * 100:g}"


def _walk_text(obj: Any, path: str = ""):
    if isinstance(obj, str):
        yield path, obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from _walk_text(v, f"{path}.{k}" if path else str(k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _walk_text(v, f"{path}[{i}]")


def catalog_consistency_errors(catalog: Catalog) -> List[str]:
    """catalog 里的规则文字要和结构化字段一致。价格可以在后台改，写死在文字里的金额不会跟着变。"""
    errors: List[str] = []
    d, pay = catalog.discounts, catalog.payment
    for key, rate in (("early_bird", d.early_bird.rate), ("group", d.group.rate)):
        rule = getattr(d, key).rule
        found = _ZHE.findall(rule)
        if found != [_zhe(rate)] * len(found) or not found:
            errors.append(f"discounts.{key}.rule 写的是 {found} 折，rate 是 {rate}（应为 {_zhe(rate)} 折）")
    for path, text, ratio in (("discounts.floor_rule", d.floor_rule, d.floor_ratio), ("payment.rule", pay.rule, pay.deposit_ratio)):
        found = _PERCENT.findall(text)
        if not found or any(f"{float(x):g}" != _pct(ratio) for x in found):
            errors.append(f"{path} 写的是 {found}%，字段是 {ratio}（应为 {_pct(ratio)}%）")
    allowed = {"discounts.referral.rule": {d.referral.amount}}
    for path, text in _walk_text(catalog.model_dump(mode="json")):
        amounts = {int((a or b).replace(",", "")) for a, b in _AMOUNT.findall(text)}
        if amounts and not amounts <= allowed.get(path, set()):
            errors.append(f"catalog 的 {path} 写死了金额 {sorted(amounts)}：改价后它不会跟着变")
    return errors


def article_consistency_errors(name: str, text: str, catalog: Catalog) -> List[str]:
    """文章不写金额；提到的几折、定金比例要和 catalog 一致。"""
    errors: List[str] = []
    if _AMOUNT.search(text):
        errors.append(f"文章 {name} 写了金额 {_AMOUNT.findall(text)}：价格以 catalog 为准，文章里不要写")
    rates = {_zhe(catalog.discounts.early_bird.rate), _zhe(catalog.discounts.group.rate)}
    bad = sorted(set(_ZHE.findall(text)) - rates)
    if bad:
        errors.append(f"文章 {name} 写了 {bad} 折，catalog 里的折扣是 {sorted(rates)} 折")
    deposit = _pct(catalog.payment.deposit_ratio)
    for sentence in re.split(r"[。；\n]", text):
        if "定金" in sentence:
            wrong = [x for x in _PERCENT.findall(sentence) if f"{float(x):g}" != deposit]
            if wrong:
                errors.append(f"文章 {name} 写定金 {wrong}%，catalog 是 {deposit}%：{sentence.strip()}")
    return errors


# ── lint ─────────────────────────────────────────────────────────────────

def lint() -> Dict[str, List[str]]:
    errors: List[str] = []
    warnings: List[str] = []
    pages = build_pages()
    index = (yaml.safe_load(INDEX_PATH.read_text(encoding="utf-8")) or {}).get("pages", {}) if INDEX_PATH.is_file() else {}
    ids = {p.id for p in pages}
    for p in pages:
        if p.id not in index:
            errors.append(f"页面 {p.id} 没有目录条目")
        elif not p.questions:
            errors.append(f"页面 {p.id} 的目录条目没有常见问法")
    errors += [f"目录里有不存在的页面 {i}" for i in index if i not in ids]
    for pid, entry in index.items():
        text = str(entry.get("summary", "")) + " ".join(map(str, entry.get("questions", [])))
        if _PRICE.search(text):
            errors.append(f"目录条目 {pid} 含金额或折扣：{_PRICE.findall(text)}")
        if not entry.get("summary"):
            errors.append(f"目录条目 {pid} 没有摘要")
    _, docs = build_seed_documents()
    glossary = next((p for p in pages if p.id == GLOSSARY_PAGE), None)
    if glossary is None:
        warnings.append("没有术语表页面")
    else:
        source_text = raw_sources_text() + "\n".join(d["content"] for d in docs)
        unknown = numbers_in(glossary.content) - numbers_in(source_text)
        if unknown:
            errors.append(f"术语表里有原始资料中找不到的数字：{sorted(unknown)}")
        titles = {d["title"] for d in docs}
        bad = sorted({t for t in re.findall(r"《([^》]+)》", glossary.content) if t not in titles})
        if bad:
            errors.append(f"术语表引用了不存在的来源：{bad}")
        front = yaml.safe_load(GLOSSARY_PATH.read_text(encoding="utf-8").split("---", 2)[1]) or {}
        if front.get("compiled_from") != sources_hash():
            # 过期的术语表会把旧说法当知识注入，必须重编（以前只警告，CI 照样通过）
            errors.append("原始资料在术语表编译后改过，请重编：python -m retrieval.wiki_compile --only glossary")
    for pid, qs in load_extra_questions().items():
        if pid not in index:
            errors.append(f"补充问法挂在不存在的页面 {pid} 上")
        bad = [q for q in qs if _PRICE.search(q)]
        if bad:
            errors.append(f"补充问法 {pid} 含金额或折扣：{bad}")
    catalog = get_catalog()
    errors += catalog_consistency_errors(catalog)
    for art in sorted((BUSINESS / "articles").glob("*.md")):
        errors += article_consistency_errors(art.name, art.read_text(encoding="utf-8"), catalog)
    return {"errors": errors, "warnings": warnings}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", choices=["index", "glossary"])
    parser.add_argument("--lint", action="store_true", help="只检查，不调模型")
    parser.add_argument("--missing", action="store_true", help="目录只补还没有条目的页面")
    parser.add_argument("--review-misses", type=Path, metavar="JSONL", help="线上选不中的问题，生成归属建议")
    parser.add_argument("--accept", action="store_true", help="把人工确认过的建议并入补充问法")
    parser.add_argument("--model", default=os.getenv("GOEUROOPS_WIKI_COMPILE_MODEL", "deepseek-v4-pro"))
    args = parser.parse_args()
    if args.accept:
        accept_suggestions()
    elif not args.lint:
        from evaluation.intent_benchmark import load_env_file
        from core.llm_utils import make_client

        load_env_file()
        client = make_client(os.environ["ANTHROPIC_API_KEY"], os.getenv("ANTHROPIC_BASE_URL"), "wiki_compile")

        async def run() -> None:
            if args.review_misses:
                await review_misses(client, args.model, args.review_misses)
                return
            if args.only in (None, "glossary"):
                await compile_glossary(client, args.model)
            if args.only in (None, "index"):
                await compile_index(client, args.model, missing_only=args.missing)

        asyncio.run(run())
    result = lint()
    for w in result["warnings"]:
        print("警告：", w)
    for e in result["errors"]:
        print("错误：", e)
    print("lint 通过" if not result["errors"] else f"lint 失败：{len(result['errors'])} 个错误")
    return 1 if result["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
