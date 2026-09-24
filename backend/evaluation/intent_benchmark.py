"""意图识别离线基准：在固定标注集上跑三路融合，并输出消融对比。

用法（在 backend/ 目录下）：
    python -m evaluation.intent_benchmark                       # 默认数据集 v1
    python -m evaluation.intent_benchmark --limit 40 --concurrency 4

每条样本只调用一次 LLM：三路结果分别保存，再在本地计算以下几种"识别器"的预测：
  fusion       当前线上使用的两层投票 + 分歧检测（IntentRecognizer._vote）
  fusion_v3_0  修复前的逐意图投票，用来量化 v3.1 修复的影响
  llm_only     只用 LLM 那一路（失败时判 OTHER）
  embedding    只用字符 n-gram 模板相似度
  pattern      只用关键词规则
  degraded     模拟 LLM 调用失败时的降级结果（向量优先，其次关键词）

报告写到 evaluation/reports/intent_<数据集>_<时间>.{json,md}。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.intent_recognizer import (
    _GENERIC_INTENTS,
    _SPECIFIC_INTENTS,
    _TEMPLATES,
    IntentCategory,
    IntentRecognizer,
    _group_of,
)

HERE = Path(__file__).parent
DATASETS = HERE / "datasets"
REPORTS = HERE / "reports"
LABELS = [c.value for c in IntentCategory]


# ── 数据与配置 ────────────────────────────────────────────────────────────────

def load_dataset(name: str) -> List[Dict[str, Any]]:
    path = DATASETS / f"{name}.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    unknown = {r["label"] for r in rows} - set(LABELS)
    if unknown:
        raise ValueError(f"数据集中有未知标签: {unknown}")
    return rows


def check_leakage(rows: List[Dict[str, Any]]) -> List[str]:
    """测试样本不能和识别器内置的 few-shot / 向量模板重复，否则分数虚高。"""
    templates = {t for tpls in _TEMPLATES.values() for t in tpls}
    return [r["text"] for r in rows if r["text"] in templates]


def load_env_file() -> None:
    """未设置环境变量时，从仓库根目录 .env 读取模型配置（不打印任何值）。"""
    env = HERE.parent.parent / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


# ── 修复前的投票逻辑（仅用于对比） ────────────────────────────────────────────

def vote_v3_0(llm: Dict, emb: Dict, pat: Dict, threshold: float = 0.5) -> str:
    if llm.get("failed"):
        for r in (emb, pat):
            if r.get("intent") != IntentCategory.OTHER and r.get("confidence", 0) > 0:
                return r["intent"].value
        return "other"
    scores: Dict[IntentCategory, float] = defaultdict(float)
    for r, w in ((llm, 0.7), (emb, 0.2), (pat, 0.1)):
        scores[r.get("intent", IntentCategory.OTHER)] += w * float(r.get("confidence", 0) or 0)
    best = max(scores, key=scores.get)
    pat_intent, pat_conf = pat.get("intent"), float(pat.get("confidence", 0) or 0)
    if best in _GENERIC_INTENTS and pat_intent in _SPECIFIC_INTENTS and pat_conf >= 0.5 and scores[best] < 0.8:
        return pat_intent.value
    return best.value if scores[best] >= threshold else "other"


# ── 指标 ──────────────────────────────────────────────────────────────────────

def classification_metrics(gold: List[str], pred: List[str]) -> Dict[str, Any]:
    per_class = {}
    for label in LABELS:
        tp = sum(1 for g, p in zip(gold, pred) if g == label and p == label)
        fp = sum(1 for g, p in zip(gold, pred) if g != label and p == label)
        fn = sum(1 for g, p in zip(gold, pred) if g == label and p != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4), "support": tp + fn}
    present = [l for l in LABELS if per_class[l]["support"]]
    group = lambda v: _group_of(IntentCategory(v)).value  # noqa: E731
    return {
        "accuracy": round(sum(g == p for g, p in zip(gold, pred)) / len(gold), 4),
        "macro_f1": round(statistics.mean(per_class[l]["f1"] for l in present), 4),
        # 领域准确率：大类和细类属于同一领域时算对，这个口径直接决定路由到哪个 Agent
        "group_accuracy": round(sum(group(g) == group(p) for g, p in zip(gold, pred)) / len(gold), 4),
        "per_class": per_class,
    }


def confusion_pairs(gold: List[str], pred: List[str], top: int = 12) -> List[Dict[str, Any]]:
    pairs = Counter((g, p) for g, p in zip(gold, pred) if g != p)
    return [{"gold": g, "pred": p, "count": c} for (g, p), c in pairs.most_common(top)]


# ── 运行 ──────────────────────────────────────────────────────────────────────

async def run_one(rec: IntentRecognizer, row: Dict[str, Any], sem: asyncio.Semaphore) -> Dict[str, Any]:
    async with sem:
        t0 = time.monotonic()
        llm = await rec._llm_recognize(row["text"], row.get("history") or None)
        llm_ms = (time.monotonic() - t0) * 1000
    emb = await rec._embedding_recognize(row["text"])
    pat = rec._pattern_recognize(row["text"])
    fused, confidence, sources = rec._vote(llm, emb, pat)
    return {
        **row,
        "pred": {
            "fusion": fused.value,
            "fusion_v3_0": vote_v3_0(llm, emb, pat, rec.threshold),
            "llm_only": "other" if llm.get("failed") else llm["intent"].value,
            "embedding": emb["intent"].value,
            "pattern": pat["intent"].value,
            "degraded": rec._vote({"failed": True}, emb, pat)[0].value,
        },
        "confidence": round(confidence, 4),
        "llm_confidence": float(llm.get("confidence", 0) or 0),
        "llm_failed": bool(llm.get("failed")),
        "conflict": bool(sources.get("conflict")),
        "refined": bool(sources.get("refined_by_pattern")),
        "llm_ms": round(llm_ms, 1),
    }


async def main(dataset: str, limit: Optional[int], concurrency: int) -> Path:
    load_env_file()
    rows = load_dataset(dataset)
    if limit:
        rows = rows[:limit]
    leaked = check_leakage(rows)
    if leaked:
        raise SystemExit(f"测试样本与内置模板重复，先修改数据集: {leaked}")

    model = os.getenv("GOEUROOPS_INTENT_MODEL") or os.getenv("ANTHROPIC_MODEL", "")
    rec = IntentRecognizer(api_key=os.environ["ANTHROPIC_API_KEY"], base_url=os.getenv("ANTHROPIC_BASE_URL") or None, model=model)
    sem = asyncio.Semaphore(concurrency)
    t0 = time.monotonic()
    results = await asyncio.gather(*(run_one(rec, r, sem) for r in rows))
    wall_s = time.monotonic() - t0

    gold = [r["label"] for r in results]
    systems = ["fusion", "fusion_v3_0", "llm_only", "embedding", "pattern", "degraded"]
    metrics = {s: classification_metrics(gold, [r["pred"][s] for r in results]) for s in systems}
    fused = [r["pred"]["fusion"] for r in results]

    by_tag: Dict[str, List[bool]] = defaultdict(list)
    for r in results:
        for tag in r["tags"] or ["plain"]:
            by_tag[tag].append(r["pred"]["fusion"] == r["label"])
    latencies = sorted(r["llm_ms"] for r in results)

    report = {
        "dataset": dataset,
        "samples": len(results),
        "model": model,
        "intent_embedding": "bge" if rec._semantic_embedder() is not None else "ngram",
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "wall_seconds": round(wall_s, 1),
        "llm_failures": sum(r["llm_failed"] for r in results),
        "llm_latency_ms": {
            "p50": latencies[len(latencies) // 2],
            "p95": latencies[int(len(latencies) * 0.95) - 1],
        },
        "metrics": {s: {k: v for k, v in m.items() if k != "per_class"} for s, m in metrics.items()},
        "per_class": metrics["fusion"]["per_class"],
        "by_tag": {t: {"n": len(v), "accuracy": round(sum(v) / len(v), 4)} for t, v in sorted(by_tag.items())},
        "confusions": confusion_pairs(gold, fused),
        "conflict_triggered": sum(r["conflict"] for r in results),
        "fixed_by_v3_1": [r["text"] for r in results if r["pred"]["fusion"] == r["label"] != r["pred"]["fusion_v3_0"]],
        "broken_by_v3_1": [r["text"] for r in results if r["pred"]["fusion_v3_0"] == r["label"] != r["pred"]["fusion"]],
        "errors": [
            {"text": r["text"], "gold": r["label"], "pred": r["pred"]["fusion"], "llm": r["pred"]["llm_only"],
             "confidence": r["confidence"], "tags": r["tags"]}
            for r in results if r["pred"]["fusion"] != r["label"]
        ],
    }

    REPORTS.mkdir(exist_ok=True)
    stem = f"intent_{dataset}_{datetime.now().strftime('%Y%m%d-%H%M')}"
    (REPORTS / f"{stem}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (REPORTS / f"{stem}.md").write_text(render_markdown(report), encoding="utf-8")
    print(render_markdown(report))
    return REPORTS / f"{stem}.md"


def render_markdown(r: Dict[str, Any]) -> str:
    names = {"fusion": "三路融合（v3.1，线上）", "fusion_v3_0": "三路融合（v3.0，修复前）",
             "llm_only": "仅 LLM", "embedding": "仅向量模板", "pattern": "仅关键词",
             "degraded": "LLM 故障时的降级"}
    lines = [
        f"# 意图识别基准：{r['dataset']}",
        "",
        f"- 样本 {r['samples']} 条，19 类；模型 `{r['model']}`；向量路 `{r.get('intent_embedding', 'ngram')}`；运行于 {r['run_at']}，耗时 {r['wall_seconds']} 秒",
        f"- LLM 调用失败 {r['llm_failures']} 次；LLM 单次延迟 p50 {r['llm_latency_ms']['p50']:.0f} ms，p95 {r['llm_latency_ms']['p95']:.0f} ms",
        f"- 分歧检测触发 {r['conflict_triggered']} 次",
        "",
        "## 总体与消融",
        "",
        "| 识别器 | 准确率 | Macro-F1 | 领域准确率（决定路由） |",
        "|---|---|---|---|",
    ]
    for key, m in r["metrics"].items():
        lines.append(f"| {names[key]} | {m['accuracy']:.1%} | {m['macro_f1']:.3f} | {m['group_accuracy']:.1%} |")
    lines += ["", "## 各类别（三路融合）", "", "| 意图 | Precision | Recall | F1 | 样本 |", "|---|---|---|---|---|"]
    for label, m in sorted(r["per_class"].items(), key=lambda kv: kv[1]["f1"]):
        if m["support"]:
            lines.append(f"| {label} | {m['precision']:.2f} | {m['recall']:.2f} | {m['f1']:.2f} | {m['support']} |")
    lines += ["", "## 按难点标签", "", "| 标签 | 样本 | 准确率 |", "|---|---|---|"]
    lines += [f"| {t} | {v['n']} | {v['accuracy']:.1%} |" for t, v in r["by_tag"].items()]
    lines += ["", "## 最常见的混淆", "", "| 标注 | 预测 | 次数 |", "|---|---|---|"]
    lines += [f"| {c['gold']} | {c['pred']} | {c['count']} |" for c in r["confusions"]]
    lines += ["", f"## v3.1 修复的影响", "",
              f"- 修复前错、修复后对：{len(r['fixed_by_v3_1'])} 条 {r['fixed_by_v3_1'][:8]}",
              f"- 修复前对、修复后错：{len(r['broken_by_v3_1'])} 条 {r['broken_by_v3_1'][:8]}",
              "", "## 错例", "", "| 文本 | 标注 | 预测 | LLM | 置信度 |", "|---|---|---|---|---|"]
    lines += [f"| {e['text']} | {e['gold']} | {e['pred']} | {e['llm']} | {e['confidence']:.2f} |" for e in r["errors"]]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="intent_eval_v1")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--concurrency", type=int, default=6)
    args = parser.parse_args()
    asyncio.run(main(args.dataset, args.limit, args.concurrency))
