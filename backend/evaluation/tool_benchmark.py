"""工具选择离线基准：给定用户消息，Agent 有没有调对工具。

用法（在 backend/ 目录下）：
    python -m evaluation.tool_benchmark                      # 默认数据集 tool_selection_v1
    python -m evaluation.tool_benchmark --limit 10 --concurrency 2

每条样本走完整的线上链路（意图识别 → 路由 → 门控 → Agent 工具循环），会调大模型。
知识库建在临时目录，线索和交接单写进内存，不读写线上数据。

标注（datasets/tool_selection_v1.jsonl）：
  must    必须调用的工具（空表示不该调任何业务工具）
  allow   可调可不调的工具；search_knowledge_base 对所有样本都允许
  forbid  调了就算错的工具（如用户没同意就登记线索）

指标：
  accuracy      完全正确：must 都调了，且没调 must/allow 以外的工具
  must_recall   must 里的工具被调到的比例（漏调率 = 1 - 它）
  extra_rate    调了 must/allow 以外工具的样本比例（多调）
  forbid_hits   调了 forbid 工具的样本数（应为 0）
  avg_calls     平均每条的工具调用次数（按 trace 计，同一工具调两次算两次）

报告写到 evaluation/reports/tools_<数据集>_<时间>.{json,md}。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import pathlib
import tempfile
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from evaluation.intent_benchmark import load_env_file
from evaluation.retrieval_benchmark import build_manager

HERE = Path(__file__).parent
DATASETS = HERE / "datasets"
REPORTS = HERE / "reports"
ALWAYS_ALLOWED = {"search_knowledge_base"}
# trace 里这些是事件记录，不是工具调用
_NOT_TOOLS = {"amount_guard", "agent_fallback"}


def load_dataset(name: str) -> List[Dict[str, Any]]:
    path = DATASETS / f"{name}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def judge(row: Dict[str, Any], used: List[str]) -> Dict[str, Any]:
    """按标注判断一条样本。used 是按调用顺序的工具名（可重复）。"""
    must, allow, forbid = set(row.get("must") or []), set(row.get("allow") or []), set(row.get("forbid") or [])
    called = set(used)
    missing = sorted(must - called)
    extra = sorted(called - must - allow - ALWAYS_ALLOWED)
    forbidden = sorted(called & forbid)
    return {
        "missing": missing,
        "extra": extra,
        "forbidden": forbidden,
        "correct": not missing and not extra and not forbidden,
    }


def summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(results)
    must_total = sum(len(r["must"]) for r in results)
    must_hit = sum(len(r["must"]) - len(r["missing"]) for r in results)
    return {
        "samples": n,
        "accuracy": round(sum(r["correct"] for r in results) / n, 4),
        "must_recall": round(must_hit / must_total, 4) if must_total else None,
        "extra_rate": round(sum(bool(r["extra"]) for r in results) / n, 4),
        "forbid_hits": sum(bool(r["forbidden"]) for r in results),
        "avg_calls": round(sum(len(r["used"]) for r in results) / n, 2),
        "search_rate": round(sum("search_knowledge_base" in r["used"] for r in results) / n, 4),
        "errors": sum(bool(r.get("error")) for r in results),
    }


async def run_one(orchestrator: Any, row: Dict[str, Any], sem: asyncio.Semaphore) -> Dict[str, Any]:
    from agents.agent_orchestrator import Request

    history = [{"role": "user", "content": h} for h in row.get("history") or []]
    req = Request(message=row["message"], user_id="bench_tools", conv_id=f"tools_{row['id']}", history=history or None)
    async with sem:
        t0 = time.monotonic()
        try:
            result = await orchestrator.run(req)
        except Exception as ex:  # noqa: BLE001 — 一条失败不影响整轮，记为错例
            used, agent, intent, error = [], None, None, str(ex)
        else:
            used = [t["tool_name"] for t in result.tool_traces if t.get("tool_name") not in _NOT_TOOLS]
            agent = result.primary_agent.value if result.primary_agent else None
            intent = result.intent.value if result.intent else None
            error = ""
        ms = (time.monotonic() - t0) * 1000
    return {
        "id": row["id"], "message": row["message"], "tags": row.get("tags") or [],
        "must": row.get("must") or [], "used": used, "agent": agent, "intent": intent,
        "latency_ms": round(ms), "error": error, **judge(row, used),
    }


async def main(dataset: str, limit: Optional[int], concurrency: int) -> Path:
    load_env_file()
    from agents.agent_orchestrator import AgentOrchestrator
    from business.lead_store import LeadStore
    from core.skill_loader import SkillManager

    rows = load_dataset(dataset)[:limit] if limit else load_dataset(dataset)
    model = os.getenv("ANTHROPIC_MODEL", "")
    skills = SkillManager(root_dir=str(pathlib.Path(__file__).parent.parent / "skills"))
    skills.load()
    with tempfile.TemporaryDirectory(prefix="tools-bench-") as tmp:
        manager, _ = build_manager(tmp)
        orchestrator = AgentOrchestrator(
            api_key=os.environ["ANTHROPIC_API_KEY"],
            base_url=os.getenv("ANTHROPIC_BASE_URL") or None,
            model=model,
            skill_manager=skills,
            lead_store=LeadStore(),
        )
        orchestrator.set_shared_tools(manager)
        sem = asyncio.Semaphore(concurrency)
        t0 = time.monotonic()
        results = await asyncio.gather(*(run_one(orchestrator, r, sem) for r in rows))
        wall_s = time.monotonic() - t0

    by_tag: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in results:
        for tag in r["tags"] or ["plain"]:
            by_tag[tag].append(r)
    report = {
        "dataset": dataset,
        "model": model,
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "wall_seconds": round(wall_s, 1),
        "summary": summarize(results),
        "by_tag": {t: summarize(rs) for t, rs in sorted(by_tag.items())},
        "wrong": [r for r in results if not r["correct"]],
        "rows": results,
    }
    REPORTS.mkdir(exist_ok=True)
    stem = f"tools_{dataset}_{datetime.now().strftime('%Y%m%d-%H%M')}"
    (REPORTS / f"{stem}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (REPORTS / f"{stem}.md").write_text(render_markdown(report), encoding="utf-8")
    print(render_markdown(report))
    return REPORTS / f"{stem}.md"


def render_markdown(r: Dict[str, Any]) -> str:
    s = r["summary"]
    lines = [
        f"# 工具选择基准：{r['dataset']}",
        "",
        f"- 样本 {s['samples']} 条；模型 `{r['model']}`；运行于 {r['run_at']}，耗时 {r['wall_seconds']} 秒；运行出错 {s['errors']} 条",
        "",
        "| 完全正确 | 必调召回 | 多调样本 | 误调禁用工具 | 平均调用次数 | 调了检索的比例 |",
        "|---|---|---|---|---|---|",
        f"| {s['accuracy']:.1%} | {s['must_recall']:.1%} | {s['extra_rate']:.1%} | {s['forbid_hits']} | {s['avg_calls']} | {s['search_rate']:.1%} |",
        "",
        "## 按标签",
        "",
        "| 标签 | 样本 | 完全正确 | 必调召回 | 多调 |",
        "|---|---|---|---|---|",
    ]
    for tag, t in r["by_tag"].items():
        recall = "-" if t["must_recall"] is None else f"{t['must_recall']:.0%}"
        lines.append(f"| {tag} | {t['samples']} | {t['accuracy']:.0%} | {recall} | {t['extra_rate']:.0%} |")
    lines += ["", "## 错例", "", "| id | 消息 | 路由 | 期望必调 | 实际 | 漏调 | 多调 | 禁用 |", "|---|---|---|---|---|---|---|---|"]
    for e in r["wrong"]:
        lines.append(
            f"| {e['id']} | {e['message'][:28]} | {e['agent']}/{e['intent']} | {', '.join(e['must']) or '（无）'} | "
            f"{', '.join(e['used']) or '（无）'} | {', '.join(e['missing'])} | {', '.join(e['extra'])} | {', '.join(e['forbidden'])} |"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="tool_selection_v1")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()
    asyncio.run(main(args.dataset, args.limit, args.concurrency))
