"""知识库检索离线基准：在固定标注集上量化预取链路（和可选的深查链路）。

用法（在 backend/ 目录下）：
    python -m evaluation.retrieval_benchmark                 # 预取链路，不调大模型
    python -m evaluation.retrieval_benchmark --deep          # 另跑深查链路（查询改写 + 重排，会调模型）

每次运行都在临时目录里新建一个知识库，按当前代码写入种子文档，不读写线上数据。
向量模型跟随环境变量（默认本地 bge-small-zh；设置 GOEUROOPS_EMBEDDING_PROVIDER=api 时用远端 API）。

指标（文档级：同一文档的多个片段只算一次）：
  hit@1 / hit@3 / mrr   检索排序本身（不看分数阈值），只统计有答案的样本
  injected_hit          经过门控阈值后，真正注入给 Agent 的片段里有没有应命中的文档
  false_inject          知识库里没有答案的问题，被注入了片段的比例（越低越好）
  injected_chars        平均每次注入的字数（上下文成本）

报告写到 evaluation/reports/retrieval_<数据集>_<时间>.{json,md}。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import tempfile
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.rag_gate import RagGate, retrieval_query
from evaluation.intent_benchmark import load_env_file
from retrieval.knowledge_base import KnowledgeBase
from retrieval.manager import RetrievalManager, Tool
from retrieval.rerank import load_reranker

HERE = Path(__file__).parent
DATASETS = HERE / "datasets"
REPORTS = HERE / "reports"
RANK_DEPTH = 10


def load_dataset(name: str) -> List[Dict[str, Any]]:
    path = DATASETS / f"{name}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_manager(chroma_path: str) -> tuple[RetrievalManager, KnowledgeBase]:
    """和 api/main.py 一样注册 knowledge_search，只是知识库放在临时目录。"""
    manager = RetrievalManager(
        api_key=os.getenv("ANTHROPIC_API_KEY", "benchmark"),
        base_url=os.getenv("ANTHROPIC_BASE_URL") or None,
        model=os.getenv("ANTHROPIC_MODEL", "") or "benchmark",
        reranker=load_reranker(),
    )
    kb = KnowledgeBase(chroma_host="", chroma_path=chroma_path)
    manager.register(Tool(
        name="knowledge_search", description="benchmark", handler=kb.search_handler,
        schema={"type": "object", "properties": {"query": {"type": "string"}, "top_k": {"type": "integer"}},
                "required": ["query"]},
        supports_rerank=True,
    ))
    return manager, kb


def matches(title: str, expected: List[str]) -> bool:
    return any(title.startswith(e) for e in expected)


def doc_titles(items: List[Dict[str, Any]]) -> List[str]:
    return list(dict.fromkeys(str(i.get("title", "")) for i in items))


def history_of(row: Dict[str, Any]) -> List[Dict[str, str]]:
    return [{"role": "user", "content": h} for h in row.get("history") or []]


async def prefetch(gate: RagGate, row: Dict[str, Any], query: str) -> List[Dict[str, Any]]:
    """走线上的门控链路：投机召回 → 按路由到的领域取结果 → 分数阈值 → Top-K。"""
    speculative = gate.speculate(query)
    decision = await gate.resolve(speculative, intent="study_consult", confidence=1.0,
                                  domain=row["domain"], message=query)
    return decision.items


async def run_one(manager: RetrievalManager, gate: RagGate, row: Dict[str, Any], deep: bool,
                  followup: bool) -> Dict[str, Any]:
    t0 = time.monotonic()
    # 和编排器一样：很短的追问拼上上一轮用户的话再检索（--no-followup 时只用原话，做对照）
    query = retrieval_query(row["query"], history_of(row)) if followup else row["query"]
    ranked = await manager.search_fast("knowledge_search", query, top_k=RANK_DEPTH, domain=row["domain"])
    injected = await prefetch(gate, row, query)
    out: Dict[str, Any] = {
        **row,
        "ranked": doc_titles(ranked),
        "injected": doc_titles(injected),
        "injected_chars": sum(len(str(i.get("content", ""))) for i in injected),
        "top_score": round(float(ranked[0]["score"]), 4) if ranked else 0.0,
        "top_hybrid": round(float(ranked[0].get("hybrid_score", ranked[0]["score"])), 4) if ranked else 0.0,
        "ms": (time.monotonic() - t0) * 1000,
    }
    if deep:
        result = await manager.search_with_rewrite("knowledge_search", row["query"], top_k=3, domain=row["domain"],
                                                   history=history_of(row) if followup else None)
        out["deep"] = doc_titles(result.data) if result.success else []
    return out


def rank_of(titles: List[str], expected: List[str]) -> Optional[int]:
    return next((i + 1 for i, t in enumerate(titles) if matches(t, expected)), None)


def summarize(results: List[Dict[str, Any]], deep: bool) -> Dict[str, Any]:
    answerable = [r for r in results if r["expected"]]
    unanswerable = [r for r in results if not r["expected"]]

    def block(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not rows:
            return {"n": 0}
        ranks = [rank_of(r["ranked"], r["expected"]) for r in rows]
        stats = {
            "n": len(rows),
            "hit@1": round(sum(1 for k in ranks if k == 1) / len(rows), 4),
            "hit@3": round(sum(1 for k in ranks if k and k <= 3) / len(rows), 4),
            "mrr": round(sum(1 / k for k in ranks if k) / len(rows), 4),
            "injected_hit": round(sum(1 for r in rows if rank_of(r["injected"], r["expected"])) / len(rows), 4),
        }
        if deep:
            stats["deep_hit@3"] = round(sum(1 for r in rows if rank_of(r.get("deep", []), r["expected"])) / len(rows), 4)
        return stats

    by_tag: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in answerable:
        for tag in r["tags"]:
            by_tag[tag].append(r)
    summary = {
        "overall": block(answerable),
        "by_tag": {t: block(rows) for t, rows in sorted(by_tag.items())},
        "by_domain": {d: block([r for r in answerable if r["domain"] == d])
                      for d in sorted({r["domain"] for r in answerable})},
        "no_answer": {
            "n": len(unanswerable),
            "false_inject": round(sum(1 for r in unanswerable if r["injected"]) / max(1, len(unanswerable)), 4),
        },
        "injected_chars_avg": round(sum(r["injected_chars"] for r in results) / max(1, len(results))),
        "latency_ms_avg": round(sum(r["ms"] for r in results) / max(1, len(results)), 1),
    }
    if deep:
        summary["no_answer"]["deep_false_answer"] = round(
            sum(1 for r in unanswerable if r.get("deep")) / max(1, len(unanswerable)), 4)
    return summary


async def main(dataset: str, deep: bool, label: str, followup: bool = True) -> Path:
    load_env_file()
    rows = load_dataset(dataset)
    with tempfile.TemporaryDirectory(prefix="kb-bench-") as tmp:
        manager, kb = build_manager(tmp)

        async def search_fn(query: str, top_k: int, domain: Optional[str]) -> List[Dict[str, Any]]:
            return await manager.search_fast("knowledge_search", query, top_k=top_k, domain=domain)

        gate = RagGate(search_fn, enabled=True)
        results = [await run_one(manager, gate, r, deep, followup) for r in rows]
        stats = kb.stats()

    report = {
        "dataset": dataset,
        "label": label,
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "embedding_model": stats["embedding_model"],
        "chunks": stats["total_chunks"],
        "gate": {"top_k": gate.top_k, "min_score": gate.min_score, "min_rerank_score": gate.min_rerank_score},
        "reranker": manager._reranker.model_name if manager._reranker else None,
        "followup_completion": followup,
        "summary": summarize(results, deep),
        # 每条的最高分（重排后是重排分数），用来校准阈值
        "rows": [{"id": r["id"], "tags": r["tags"], "top_score": r["top_score"], "top_hybrid": r["top_hybrid"],
                  "top_correct": bool(r["expected"]) and rank_of(r["ranked"], r["expected"]) == 1,
                  "hit": bool(rank_of(r["injected"], r["expected"])) if r["expected"] else not r["injected"]}
                 for r in results],
        "misses": [
            {"id": r["id"], "query": r["query"], "expected": r["expected"], "ranked": r["ranked"][:3],
             "injected": r["injected"], "top_score": r["top_score"], "tags": r["tags"]}
            for r in results
            if (r["expected"] and not rank_of(r["injected"], r["expected"])) or (not r["expected"] and r["injected"])
        ],
    }
    REPORTS.mkdir(exist_ok=True)
    stem = f"{dataset}_{label + '_' if label else ''}{datetime.now().strftime('%Y%m%d-%H%M')}"
    (REPORTS / f"{stem}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (REPORTS / f"{stem}.md").write_text(render_markdown(report), encoding="utf-8")
    print(render_markdown(report))
    return REPORTS / f"{stem}.md"


def render_markdown(r: Dict[str, Any]) -> str:
    s = r["summary"]
    deep = "deep_hit@3" in s["overall"]
    head = "| 分组 | 样本 | Hit@1 | Hit@3 | MRR | 注入命中 |" + (" 深查 Hit@3 |" if deep else "")
    sep = "|---|---|---|---|---|---|" + ("---|" if deep else "")

    def row(name: str, b: Dict[str, Any]) -> str:
        line = f"| {name} | {b['n']} | {b['hit@1']:.1%} | {b['hit@3']:.1%} | {b['mrr']:.3f} | {b['injected_hit']:.1%} |"
        return line + (f" {b['deep_hit@3']:.1%} |" if deep else "")

    lines = [
        f"# 知识库检索基准：{r['dataset']}{'（' + r['label'] + '）' if r['label'] else ''}",
        "",
        f"- 运行于 {r['run_at']}；向量模型 `{r['embedding_model']}`；知识库 {r['chunks']} 个片段",
        f"- 门控：注入 Top-{r['gate']['top_k']}，分数下限：混合分 {r['gate']['min_score']}"
        + (f"，重排分 {r['gate']['min_rerank_score']}（重排模型 `{r['reranker']}`）" if r.get("reranker") else "（未启用重排）"),
        f"- 无答案问题 {s['no_answer']['n']} 条，误注入率 **{s['no_answer']['false_inject']:.1%}**"
        + (f"，深查误答率 {s['no_answer']['deep_false_answer']:.1%}" if deep else ""),
        f"- 平均每次注入 {s['injected_chars_avg']} 字；平均耗时 {s['latency_ms_avg']} ms",
        "",
        "## 有答案的问题",
        "",
        head,
        sep,
        row("全部", s["overall"]),
    ]
    lines += [row(f"标签 {t}", b) for t, b in s["by_tag"].items()]
    lines += [row(f"领域 {d}", b) for d, b in s["by_domain"].items()]
    lines += ["", "## 未命中和误注入", "", "| id | 问题 | 应命中 | 排序前 3 | 实际注入 | 最高分 |", "|---|---|---|---|---|---|"]
    lines += [f"| {m['id']} | {m['query']} | {'、'.join(m['expected']) or '（无答案）'} | {'、'.join(m['ranked'])} | "
              f"{'、'.join(m['injected']) or '—'} | {m['top_score']} |" for m in r["misses"]]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="retrieval_v1")
    parser.add_argument("--deep", action="store_true", help="另跑深查链路（查询改写 + 重排，会调模型）")
    parser.add_argument("--label", default="", help="写进报告文件名，例如 baseline")
    parser.add_argument("--no-followup", action="store_true", help="不做追问补全（对照用）")
    args = parser.parse_args()
    asyncio.run(main(args.dataset, args.deep, args.label, followup=not args.no_followup))
