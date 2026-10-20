"""知识库方案对比：RAG（向量检索 + 重排）vs LLM Wiki（读目录选整页）。

用法（在 backend/ 目录下）：
    python -m evaluation.knowledge_compare --retrieval-only   # 只比检索层，wiki 每条一次选页调用
    python -m evaluation.knowledge_compare                    # 检索层 + 端到端回答盲评

RAG 组用 evaluation/datasets/rag_prefetch_retrieval_v1.json：线上容器里按线上配置
（bge-large-zh + bge-reranker-v2-m3，门控 Top-3）导出的每条问题实际注入的片段，
导出方法见 evaluation/rag_prefetch_dump.py。

两层评测：
  检索层   沿用 retrieval_benchmark 的口径（retrieval_v1，106 条，文档级）：
           注入命中（注入内容里有没有应命中的原始文档）、无答案误注入、注入字数、检索耗时。
           wiki 页面由哪些原始文档组成记录在 page.sources 里，按它判断命中。
  端到端   同一条问题，分别把两组的注入内容交给线上同款 Agent（同一模型、Skill、工具，
           不给检索工具，只比"首轮注入的知识"本身）生成回答；
           再由更强的模型做盲评：随机打乱 A/B 顺序，以全部原始资料为参考，
           给每个回答打正确性（0–2）、完整度（1–5）、是否编造资料外的具体事实，并判胜负。

报告写到 evaluation/reports/knowledge_compare_<时间>.{json,md}。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import statistics
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from agents.agent_orchestrator import BillingAgent, ConsultingAgent, GeneralAgent, Request
from business.kb_seed import build_seed_documents
from core.intent_recognizer import IntentCategory, UrgencyLevel
from core.llm_utils import NO_THINKING_KWARGS, extract_text_content, make_client
from core.rag_gate import RagGate, retrieval_query
from core.skill_loader import SkillManager
from evaluation.intent_benchmark import load_env_file
from evaluation.retrieval_benchmark import history_of, load_dataset, matches
from retrieval.wiki import WikiKnowledge

HERE = Path(__file__).parent
REPORTS = HERE / "reports"
RAG_DUMP = HERE / "datasets" / "rag_prefetch_retrieval_v1.json"
SKILLS_DIR = HERE.parent / "skills"

AGENTS = {"consulting": ConsultingAgent, "billing": BillingAgent, "general": GeneralAgent}
INTENTS = {
    "consulting": (IntentCategory.APPLICATION_PROCESS, "study_consult"),
    "billing": (IntentCategory.REFUND, "billing"),
    "general": (IntentCategory.QUERY, "query"),
}


def hit(sources: List[str], expected: List[str]) -> bool:
    return any(matches(s, expected) for s in sources)


def item_sources(item: Dict[str, Any]) -> List[str]:
    return list(item.get("sources") or [item.get("title", "")])


# ── 检索层 ────────────────────────────────────────────────────────────────

async def wiki_retrieve(wiki: WikiKnowledge, gate: RagGate, row: Dict[str, Any]) -> Dict[str, Any]:
    query = retrieval_query(row["query"], history_of(row))
    t0 = time.monotonic()
    items = await wiki.search_fast("knowledge_search", query, top_k=gate.top_k, domain=row["domain"])
    ms = (time.monotonic() - t0) * 1000
    return {"ms": round(ms, 1), "items": items}


def retrieval_block(rows: List[Dict[str, Any]], arm: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    answerable = [r for r in rows if r["expected"]]
    unanswerable = [r for r in rows if not r["expected"]]

    def inj_hit(rs: List[Dict[str, Any]]) -> float:
        return round(sum(1 for r in rs if hit([s for i in arm[r["id"]]["items"] for s in item_sources(i)], r["expected"]))
                     / max(1, len(rs)), 4)

    by_tag: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in answerable:
        for t in r["tags"]:
            by_tag[t].append(r)
    ms = [arm[r["id"]]["ms"] for r in rows]
    return {
        "injected_hit": inj_hit(answerable),
        "by_tag": {t: {"n": len(rs), "injected_hit": inj_hit(rs)} for t, rs in sorted(by_tag.items())},
        "by_domain": {d: {"n": len(rs), "injected_hit": inj_hit(rs)}
                      for d in sorted({r["domain"] for r in answerable})
                      for rs in [[r for r in answerable if r["domain"] == d]]},
        "false_inject": round(sum(1 for r in unanswerable if arm[r["id"]]["items"]) / max(1, len(unanswerable)), 4),
        "injected_chars_avg": round(statistics.mean(sum(len(i.get("content", "")) for i in arm[r["id"]]["items"]) for r in rows)),
        "ms_p50": round(statistics.median(ms), 1),
        "ms_p90": round(sorted(ms)[int(len(ms) * 0.9) - 1], 1),
    }


# ── 端到端 ────────────────────────────────────────────────────────────────

def make_request(row: Dict[str, Any], items: List[Dict[str, Any]]) -> Request:
    intent, group = INTENTS[row["domain"]]
    return Request(
        message=row["query"], user_id="knowledge-compare", conv_id=row["id"], history=history_of(row) or None,
        intent=intent, intent_group=group, urgency=UrgencyLevel.LOW, intent_confidence=0.9,
        rag_mode="prefetch", knowledge=[{"title": i.get("title", ""), "content": i.get("content", "")} for i in items],
    )


JUDGE_PROMPT = """你是留学咨询工作室客服回答的评审。下面给出工作室的全部原始资料、一个学生的问题，以及两个 AI 助手的回答 A 和 B。

评审标准（以原始资料为唯一事实依据，资料外的常识性礼貌用语不算问题）：
- correct（0–2）：2 = 涉及的事实、数字都与资料一致；1 = 有小错或不够准确；0 = 有明显错误或与资料矛盾。
- complete（1–5）：是否回答到了问题的要点；资料里有答案却没答出来要扣分。
- fabricated（true/false）：是否给出了资料里没有的具体事实（日期、金额、分数、政策、院校要求等）。
{no_answer_note}
最后判断哪个回答整体更好（winner：A、B 或 tie），差距很小就判 tie。

只输出 JSON：{{"A": {{"correct": 0, "complete": 1, "fabricated": false}}, "B": {{...}}, "winner": "A", "reason": "一句话"}}

【原始资料】
{reference}

【对话】
{history}学生：{question}

【回答 A】
{a}

【回答 B】
{b}"""

NO_ANSWER_NOTE = "- 注意：原始资料没有覆盖这个问题。理想的回答是如实说明资料没有或不在服务范围、建议以官网为准或咨询顾问，而不是编造答案；这种情况下编造具体事实应判 correct=0。"


async def judge(client: Any, model: str, reference: str, row: Dict[str, Any], a: str, b: str) -> Dict[str, Any]:
    history = "".join(f"学生（上一轮）：{h}\n" for h in row.get("history") or [])
    prompt = JUDGE_PROMPT.format(no_answer_note=NO_ANSWER_NOTE if not row["expected"] else "", reference=reference,
                                 history=history, question=row["query"], a=a, b=b)
    for _ in range(3):
        resp = await client.messages.create(model=model, max_tokens=400, temperature=0,
                                            messages=[{"role": "user", "content": prompt}], **NO_THINKING_KWARGS)
        m = re.search(r"\{.*\}", extract_text_content(resp.content), re.S)
        try:
            data = json.loads(m.group(0)) if m else {}
            if data.get("winner") in {"A", "B", "tie"} and "A" in data and "B" in data:
                return data
        except json.JSONDecodeError:
            pass
    return {}


async def end_to_end(rows: List[Dict[str, Any]], arms: Dict[str, Dict[str, Dict[str, Any]]], model: str,
                     judge_model: str, concurrency: int, seed: int) -> List[Dict[str, Any]]:
    client = make_client(os.environ["ANTHROPIC_API_KEY"], os.getenv("ANTHROPIC_BASE_URL"), "knowledge_compare")
    skills = SkillManager(root_dir=str(SKILLS_DIR))
    skills.load()
    agents = {d: cls(client, model, skills) for d, cls in AGENTS.items()}
    _, docs = build_seed_documents()
    reference = "\n\n".join(f"《{d['title']}》\n{d['content']}" for d in docs)
    sem = asyncio.Semaphore(concurrency)

    async def one(row: Dict[str, Any]) -> Dict[str, Any]:
        async with sem:
            answers = {}
            for name in ("rag", "wiki"):
                resp = await agents[row["domain"]].handle(make_request(row, arms[name][row["id"]]["items"]))
                answers[name] = {"text": resp.content, "ms": round(resp.latency_ms, 1), "ok": resp.success,
                                 "tools": resp.tools_used}
            wiki_first = random.Random(f"{seed}-{row['id']}").random() < 0.5
            a, b = ("wiki", "rag") if wiki_first else ("rag", "wiki")
            verdict = await judge(client, judge_model, reference, row, answers[a]["text"], answers[b]["text"])
        out = {"id": row["id"], "query": row["query"], "tags": row["tags"], "domain": row["domain"],
               "answerable": bool(row["expected"]), "answers": answers}
        if verdict:
            out["scores"] = {a: verdict["A"], b: verdict["B"]}
            out["winner"] = {"A": a, "B": b, "tie": "tie"}[verdict["winner"]]
            out["reason"] = verdict.get("reason", "")
        return out

    return list(await asyncio.gather(*(one(r) for r in rows)))


def e2e_block(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    judged = [r for r in results if "scores" in r]

    def arm_stats(rs: List[Dict[str, Any]], arm: str) -> Dict[str, Any]:
        if not rs:
            return {}
        sc = [r["scores"][arm] for r in rs]
        return {
            "correct_avg": round(statistics.mean(float(s.get("correct", 0)) for s in sc), 3),
            "fully_correct": round(sum(1 for s in sc if float(s.get("correct", 0)) >= 2) / len(sc), 4),
            "complete_avg": round(statistics.mean(float(s.get("complete", 1)) for s in sc), 3),
            "fabricated": round(sum(1 for s in sc if s.get("fabricated")) / len(sc), 4),
            "latency_ms_p50": round(statistics.median(r["answers"][arm]["ms"] for r in rs), 1),
        }

    def group(rs: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {"n": len(rs), "wiki_win": sum(1 for r in rs if r["winner"] == "wiki"),
                "rag_win": sum(1 for r in rs if r["winner"] == "rag"), "tie": sum(1 for r in rs if r["winner"] == "tie"),
                "rag": arm_stats(rs, "rag"), "wiki": arm_stats(rs, "wiki")}

    by_tag: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in judged:
        for t in r["tags"]:
            by_tag[t].append(r)
    return {"judged": len(judged), "failed_judgements": len(results) - len(judged), "overall": group(judged),
            "answerable": group([r for r in judged if r["answerable"]]),
            "no_answer": group([r for r in judged if not r["answerable"]]),
            "by_tag": {t: group(rs) for t, rs in sorted(by_tag.items())}}


# ── 报告 ──────────────────────────────────────────────────────────────────

def render(report: Dict[str, Any]) -> str:
    rr, rw = report["retrieval"]["rag"], report["retrieval"]["wiki"]
    lines = [
        "# 知识库方案对比：RAG vs LLM Wiki", "",
        f"- 运行于 {report['run_at']}；数据集 retrieval_v1（{report['n']} 条）；Agent 模型 `{report['model']}`；"
        f"评审模型 `{report.get('judge_model', '—')}`",
        f"- RAG：{report['rag_config']}", f"- Wiki：{report['wiki_config']}", "",
        "## 检索层", "", "| 指标 | RAG | Wiki |", "|---|---|---|",
        f"| 注入命中（有答案 {report['n_answerable']} 条） | {rr['injected_hit']:.1%} | {rw['injected_hit']:.1%} |",
        f"| 无答案误注入（{report['n'] - report['n_answerable']} 条，越低越好） | {rr['false_inject']:.1%} | {rw['false_inject']:.1%} |",
        f"| 平均注入字数 | {rr['injected_chars_avg']} | {rw['injected_chars_avg']} |",
        f"| 检索耗时 p50 / p90（ms，和意图识别并行） | {rr['ms_p50']} / {rr['ms_p90']} | {rw['ms_p50']} / {rw['ms_p90']} |",
    ]
    lines += [f"| 注入命中 · {t}（{b['n']}） | {b['injected_hit']:.1%} | {rw['by_tag'][t]['injected_hit']:.1%} |"
              for t, b in rr["by_tag"].items()]
    lines += [f"| 注入命中 · 领域 {d}（{b['n']}） | {b['injected_hit']:.1%} | {rw['by_domain'][d]['injected_hit']:.1%} |"
              for d, b in rr["by_domain"].items()]
    e = report.get("e2e")
    if e:
        def arm_rows(name: str, g: Dict[str, Any]) -> List[str]:
            if not g.get("n"):
                return []
            ra, wa = g["rag"], g["wiki"]
            return [
                f"| {name}（{g['n']}） | Wiki 胜 {g['wiki_win']} / RAG 胜 {g['rag_win']} / 平 {g['tie']} | "
                f"{ra['correct_avg']} / {wa['correct_avg']} | {ra['fully_correct']:.1%} / {wa['fully_correct']:.1%} | "
                f"{ra['complete_avg']} / {wa['complete_avg']} | {ra['fabricated']:.1%} / {wa['fabricated']:.1%} | "
                f"{ra['latency_ms_p50']} / {wa['latency_ms_p50']} |"]
        lines += ["", "## 端到端回答（盲评，A/B 顺序随机）", "",
                  f"评审成功 {e['judged']} 条，失败 {e['failed_judgements']} 条。每格为 RAG / Wiki。", "",
                  "| 分组 | 胜负 | 正确性均分（0–2） | 完全正确率 | 完整度（1–5） | 编造率 | Agent 耗时 p50（ms） |",
                  "|---|---|---|---|---|---|---|"]
        lines += arm_rows("全部", e["overall"]) + arm_rows("有答案", e["answerable"]) + arm_rows("无答案", e["no_answer"])
        for t, g in e["by_tag"].items():
            lines += arm_rows(f"标签 {t}", g)
        losses = [r for r in report["e2e_rows"] if r.get("winner") == "rag"]
        if losses:
            lines += ["", "## Wiki 输掉的题", "", "| id | 问题 | 评审理由 |", "|---|---|---|"]
            lines += [f"| {r['id']} | {r['query']} | {r.get('reason', '')} |" for r in losses]
    return "\n".join(lines) + "\n"


async def main(args: argparse.Namespace) -> Path:
    load_env_file()
    rows = load_dataset(args.dataset)
    dump = json.loads(RAG_DUMP.read_text(encoding="utf-8"))
    rag_arm = dump["rows"]
    client = make_client(os.environ["ANTHROPIC_API_KEY"], os.getenv("ANTHROPIC_BASE_URL"), "knowledge_compare")
    wiki = WikiKnowledge(client, args.model)
    gate = RagGate(enabled=True)
    sem = asyncio.Semaphore(args.concurrency)

    async def retrieve(row: Dict[str, Any]) -> tuple:
        async with sem:
            return row["id"], await wiki_retrieve(wiki, gate, row)

    wiki_arm = dict(await asyncio.gather(*(retrieve(r) for r in rows)))
    report: Dict[str, Any] = {
        "run_at": datetime.now().isoformat(timespec="seconds"), "n": len(rows),
        "n_answerable": sum(1 for r in rows if r["expected"]), "model": args.model,
        "rag_config": f"{dump['embedding_model']} + {dump['reranker']}，{dump['chunks']} 个片段，Top-{gate.top_k}，线上容器导出",
        "wiki_config": f"{len(wiki.pages)} 页，读目录选最多 {wiki.max_pages} 页，选页模型 `{args.model}`",
        "retrieval": {"rag": retrieval_block(rows, rag_arm), "wiki": retrieval_block(rows, wiki_arm)},
        "wiki_selection": {r["id"]: [i["page_id"] for i in wiki_arm[r["id"]]["items"]] for r in rows},
        "wiki_stats": wiki.get_stats(),
    }
    if not args.retrieval_only:
        results = await end_to_end(rows, {"rag": rag_arm, "wiki": wiki_arm}, args.model, args.judge_model,
                                   args.concurrency, args.seed)
        report["judge_model"] = args.judge_model
        report["e2e"] = e2e_block(results)
        report["e2e_rows"] = results
    REPORTS.mkdir(exist_ok=True)
    stem = f"knowledge_compare_{args.label + '_' if args.label else ''}{datetime.now().strftime('%Y%m%d-%H%M')}"
    (REPORTS / f"{stem}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (REPORTS / f"{stem}.md").write_text(render(report), encoding="utf-8")
    print(render(report))
    return REPORTS / f"{stem}.md"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="retrieval_v1")
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument("--model", default="deepseek-v4-flash", help="Agent 和选页用的模型（线上同款）")
    parser.add_argument("--judge-model", default="deepseek-v4-pro")
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--label", default="")
    asyncio.run(main(parser.parse_args()))
