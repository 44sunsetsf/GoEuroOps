"""LLM Wiki 的离线编译和 lint。

用法（在 backend/ 目录下）：
    python -m retrieval.wiki_compile              # 编译术语表和目录，然后 lint
    python -m retrieval.wiki_compile --only index # 只重编目录（新增/调整页面后）
    python -m retrieval.wiki_compile --lint       # 只检查，不调模型（CI 里由 tests/test_wiki.py 执行）

编译产物提交进 git，人工看过 diff 再合并：
  business/wiki/glossary.md   术语表：只从原始资料里摘取，标注来源
  business/wiki/index.yaml    目录：每页的摘要和用户常见问法

编译时模型只看原始资料，不看评测集，避免把评测题"背"进目录。

lint 规则：
  - 每个页面都有目录条目，目录里没有多余的页面
  - 目录（摘要、常见问法）不含金额和折扣：价格在后台随时会改，目录必须不随之过期
  - 术语表里出现的每个数字都能在原始资料里找到；标注的来源必须是真实存在的种子文档
  - 术语表记录了编译时原始资料的哈希，原始资料改了要重编（只警告，不算错误）
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
from typing import Any, Dict, List

import yaml

from business.kb_seed import build_seed_documents
from core.llm_utils import NO_THINKING_KWARGS, extract_text_content
from retrieval.wiki import GLOSSARY_PAGE, GLOSSARY_PATH, INDEX_PATH, WIKI_DIR, WikiPage, build_pages

BUSINESS = Path(__file__).resolve().parent.parent / "business"
_NUM = re.compile(r"\d+(?:[.,]\d+)*")
# 目录里禁止的是金额和折扣（后台随时会改价），雅思分数、学分这类出自静态资料的数字可以出现
_PRICE = re.compile(r"[¥￥]\s*\d|\d[\d,]*\s*(?:元|块|折|rmb|RMB)|\d+\s*%")


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


async def compile_index(client: Any, model: str, missing_only: bool = False) -> None:
    pages = build_pages(with_index=False)
    old = (yaml.safe_load(INDEX_PATH.read_text(encoding="utf-8")) or {}).get("pages", {}) if INDEX_PATH.is_file() else {}
    if missing_only:
        pages = [p for p in pages if not (old.get(p.id) or {}).get("questions")]
    sem = asyncio.Semaphore(4)

    async def one(p: WikiPage) -> tuple:
        data: Dict[str, Any] = {}
        for _ in range(2):                    # 输出被截断或 JSON 不合法时重试一次
            async with sem:
                text = await _ask(client, model, INDEX_PROMPT.format(title=p.title, content=p.content), 3000)
            m = re.search(r"\{.*\}", text, re.S)
            try:
                data = json.loads(m.group(0)) if m else {}
            except json.JSONDecodeError:
                data = {}
            if data.get("summary") and data.get("questions"):
                break
        # 兜底：模型偶尔还是会写金额，删掉含金额的问法
        questions = [q for q in data.get("questions", []) if not _PRICE.search(q)]
        summary = str(data.get("summary", "")).strip()
        return p.id, {"title": p.title, "summary": summary, "questions": questions[:24]}

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
            warnings.append("原始资料在术语表编译后改过，建议重编：python -m retrieval.wiki_compile --only glossary")
    return {"errors": errors, "warnings": warnings}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", choices=["index", "glossary"])
    parser.add_argument("--lint", action="store_true", help="只检查，不调模型")
    parser.add_argument("--missing", action="store_true", help="目录只补还没有条目的页面")
    parser.add_argument("--model", default=os.getenv("GOEUROOPS_WIKI_COMPILE_MODEL", "deepseek-v4-pro"))
    args = parser.parse_args()
    if not args.lint:
        from evaluation.intent_benchmark import load_env_file
        from core.llm_utils import make_client

        load_env_file()
        client = make_client(os.environ["ANTHROPIC_API_KEY"], os.getenv("ANTHROPIC_BASE_URL"), "wiki_compile")

        async def run() -> None:
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
