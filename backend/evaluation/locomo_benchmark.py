"""用公开数据集 LoCoMo（A-Mem、Mem0 等论文都用它）测情景记忆的跨会话检索。

LoCoMo：两个人几十个会话的长对话，附带问答（多跳、时间、开放、单跳四类，另有对抗类这里不测）。
数据：https://github.com/snap-research/locomo 的 data/locomo10.json，用 --data 指定本地路径。

怎么跑：
  - 每个会话一个 conv_id，每句话作为一条用户消息写入（不做画像提炼：画像字段是留学场景的，和这份数据无关）；
  - 每个会话结束时把剩下的消息也压进情景记忆（见 MemoryManager._compress 的 force）；
  - 提问放在一个全新的会话里，所以只能靠情景记忆回答，这正是要比较的部分；
  - 两种条件：旧版（只有摘要式情景记忆）/ 逐句索引（用户每句话单独建向量，提问时取回原话）；同一次写入，提问时切换开关；
  - 读者模型回答，再由同一个模型当裁判判对错（和 Mem0 论文的做法一致）。

用法：python -m evaluation.locomo_benchmark --data locomo10.json --per-cat 8
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.llm_utils import NO_THINKING_KWARGS, extract_text_content, make_client  # noqa: E402
from evaluation.memory_benchmark import Counter, FakeRedis  # noqa: E402
from memory.conversation_memory import MemoryManager, MsgRole  # noqa: E402

REPORT_DIR = ROOT / "evaluation" / "reports"
CATS = {1: "multi_hop", 2: "temporal", 3: "open_domain", 4: "single_hop"}
READER = ("Answer the question using only the memory below, which comes from earlier conversations between two people. "
          "Reply with a short phrase, no explanation. Dates may be written as in the memory. "
          "If the memory does not contain the answer, reply 'unknown'.")
JUDGE = ("You grade an answer against a gold answer. Reply with exactly CORRECT or WRONG. "
         "CORRECT if the answer contains the same key information as the gold answer (dates may be formatted differently "
         "or be approximate by a day; extra detail is fine). WRONG otherwise, including 'unknown'.")


def load(data_path: str, conv_idx: int, per_cat: int):
    d = json.load(open(data_path, encoding="utf-8"))[conv_idx]
    conv = d["conversation"]
    sessions = []
    k = 1
    while f"session_{k}" in conv:
        date = conv.get(f"session_{k}_date_time", "")
        sessions.append((f"s{k}", [f"[{date}] {t['speaker']}: {t['text']}" for t in conv[f"session_{k}"]]))
        k += 1
    qs = []
    for cat in CATS:
        pool = [q for q in d["qa"] if q["category"] == cat and q.get("answer") is not None]
        step = max(1, len(pool) // per_cat)
        qs += [dict(q, category=cat) for q in pool[::step][:per_cat]]
    for q in qs:                                          # 证据 D3:5 表示第 3 个会话的第 5 句
        q["ev_sessions"] = sorted({f"s{e.split(':')[0][1:]}" for e in q.get("evidence", []) if e.startswith("D") and ":" in e})
    return sessions, qs


async def ingest(mgr: MemoryManager, uid: str, sessions) -> None:
    async def one(conv: str, turns: List[str]) -> None:
        for t in turns:
            await mgr.add_message(uid, f"{uid}-{conv}", MsgRole.USER, t)
        await mgr._compress(uid, f"{uid}-{conv}", force=True)
    sem = asyncio.Semaphore(4)

    async def guarded(c, t):
        async with sem:
            await one(c, t)
    await asyncio.gather(*(guarded(c, t) for c, t in sessions))


async def ask(mgr: MemoryManager, reader: Any, model: str, uid: str, q: Dict[str, Any]) -> Dict[str, Any]:
    ctx = await mgr.get_context(uid, f"{uid}-probe", query=q["question"])
    text = ctx.to_prompt_text()
    r = await reader.messages.create(model=model, max_tokens=64, temperature=0.0, system=READER,
                                     messages=[{"role": "user", "content": f"[Memory]\n{text or '(none)'}\n\nQuestion: {q['question']}"}],
                                     **NO_THINKING_KWARGS)
    ans = extract_text_content(r.content).strip()
    where = {"user_id": uid}
    if mgr.TURN_INDEX:
        res = await asyncio.to_thread(mgr._turns.query, query_texts=[q["question"]], n_results=mgr.TURN_TOP_K, where=where)
    else:
        res = await asyncio.to_thread(mgr._episodic.query, query_texts=[q["question"]], n_results=3, where=where)
    got = {m.get("conv_id", "").split("-")[-1] for m in (res.get("metadatas") or [[]])[0]}
    ev = q.get("ev_sessions") or []
    j = await reader.messages.create(model=model, max_tokens=8, temperature=0.0, system=JUDGE,
                                     messages=[{"role": "user", "content": f"Question: {q['question']}\nGold: {q['answer']}\nAnswer: {ans}"}],
                                     **NO_THINKING_KWARGS)
    return {"cat": CATS[q["category"]], "q": q["question"], "gold": str(q["answer"]), "answer": ans,
            "correct": "CORRECT" in extract_text_content(j.content).upper() and "WRONG" not in extract_text_content(j.content).upper(),
            "ctx_chars": len(text), "ev_hit": bool(ev) and all(e in got for e in ev), "ev_any": bool(ev) and any(e in got for e in ev)}


async def run_both(sessions, qs, model: str, key: str, base_url):
    mgr = MemoryManager(redis_url="redis://unused:1/0", chroma_host="", chroma_path=tempfile.mkdtemp(prefix="lc-"),
                        api_key=key, base_url=base_url, model=model)
    mgr._redis = FakeRedis()
    mgr.TURN_INDEX = True                                 # 写入时建逐句索引；提问时再按条件开关

    async def no_profile(*a, **k):
        return 0
    mgr._extract_facts = no_profile                      # 画像字段是留学场景的，这份数据不测
    maint = Counter()
    maint.wrap(mgr._client)
    t0 = time.time()
    await ingest(mgr, "lc", sessions)
    write = {"calls": maint.calls, "tokens": maint.tokens_in + maint.tokens_out, "seconds": round(time.time() - t0, 1)}
    reader = make_client(key, base_url, "eval")
    out = {}
    for label, on in (("old", False), ("turn_index", True)):
        mgr.TURN_INDEX = on
        sem = asyncio.Semaphore(6)

        async def go(q):
            async with sem:
                return await ask(mgr, reader, model, "lc", q)
        out[label] = await asyncio.gather(*(go(q) for q in qs))
    return out, write


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--conv", type=int, default=0)
    ap.add_argument("--per-cat", type=int, default=8)
    ap.add_argument("--label", default="locomo")
    a = ap.parse_args()
    key = os.environ["ANTHROPIC_API_KEY"]
    base_url = os.environ.get("ANTHROPIC_BASE_URL") or None
    model = os.environ.get("ANTHROPIC_MODEL", "deepseek-v4-flash")
    sessions, qs = load(a.data, a.conv, a.per_cat)
    print(f"会话 {len(sessions)} 个，共 {sum(len(t) for _, t in sessions)} 句；问题 {len(qs)} 个；模型 {model}")
    res, write = await run_both(sessions, qs, model, key, base_url)
    lines = [f"# LoCoMo 情景记忆对比（{a.label}）", "", f"- 第 {a.conv} 段对话，{len(sessions)} 个会话；{len(qs)} 题（每类最多 {a.per_cat}）；模型 `{model}`；单次运行",
             f"- 写入时的模型调用（摘要、合并）：{write['calls']} 次 / {write['tokens']} token（逐句索引不增加调用）", "",
             "| 条件 | 总体 | " + " | ".join(CATS.values()) + " | 提示词中记忆平均字数 | 证据会话全部被取回 | 至少取回一个 |", "|---|---|" + "---|" * len(CATS) + "---|---|---|"]
    for label, rows in res.items():
        per = [sum(r["correct"] for r in rows if r["cat"] == c) / max(1, sum(1 for r in rows if r["cat"] == c)) for c in CATS.values()]
        lines.append(f"| {label} | {sum(r['correct'] for r in rows) / len(rows):.0%} | " + " | ".join(f"{p:.0%}" for p in per)
                     + f" | {sum(r['ctx_chars'] for r in rows) / len(rows):.0f} | {sum(r['ev_hit'] for r in rows) / len(rows):.0%} | {sum(r['ev_any'] for r in rows) / len(rows):.0%} |")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    base = REPORT_DIR / f"locomo_{a.label}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    base.with_suffix(".json").write_text(json.dumps({"results": res, "write": write}, ensure_ascii=False, indent=1), encoding="utf-8")
    base.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print("report:", base.with_suffix(".md"))


if __name__ == "__main__":
    asyncio.run(main())
