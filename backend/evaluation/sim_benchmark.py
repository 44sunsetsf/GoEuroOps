"""模拟用户端到端评测（τ-bench 思路）：一个模型扮演带着目标的用户，和完整的线上链路多轮对话，
结束后**用代码检查最终状态**，同一题跑 k 次，报 pass^1 和 pass^k。

用法（在 backend/ 目录下）：
    python -m evaluation.sim_benchmark                        # 默认 sim_tasks_v1，每题 4 次
    python -m evaluation.sim_benchmark --trials 1 --limit 5   # 冒烟
    python -m evaluation.sim_benchmark --ids quote_full_journey,lead_no_consent

参考 sierra-research/tau-bench（MIT，2024，main 分支 tau_bench/envs/user.py、envs/base.py、run.py，2026-10-29 查看）：
- 取了：用户模拟的规则（一次一句、不要一次说完、没给的信息不编、目标达成输出 ###STOP###）；
  奖励 = 最终状态检查 + 回复里必须出现的内容，两者都满足才算过；pass^k = 每题 C(c,k)/C(n,k) 的平均。
- 没取：τ-bench 用“标准动作序列重放后比较数据库哈希”判状态。这里的状态（线索、交接单、工具调用、路由）
  小而明确，直接写断言；另外加了“不得出现的内容”和“不该调用的工具”，用来测越权和编价。
- 加了：逐条清单裁判（软指标，和硬状态检查分开报），金额护栏触发次数。

每次试验用一个新的编排器、新的内存线索库，trace 只放内存，不读写线上数据；知识库建在临时目录。
报价和退款的期望数字不手写，加载任务时用 business/pricing.py 现算，改价后题目不用改。

报告写到 evaluation/reports/sim_<数据集>_<时间>.{json,md}。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import pathlib
import re
import tempfile
import time
from collections import defaultdict
from datetime import datetime
from math import comb
from pathlib import Path
from typing import Any, Dict, List, Optional

from evaluation.intent_benchmark import load_env_file

HERE = Path(__file__).parent
DATASETS = HERE / "datasets"
REPORTS = HERE / "reports"
STOP = "###STOP###"
_NOT_TOOLS = {"amount_guard", "agent_fallback"}

USER_SYSTEM = """你在扮演一位来咨询留学服务的用户，对面是留学工作室的在线客服。

你的情况和目标：
{instruction}

规则：
- 每次只说一两句话，像真人在手机上打字一样自然，用中文。
- 不要一次把所有信息说完，只说当前这一步需要的；对方问到了再说。
- 上面没给的信息不要编。对方问到你不知道的，就说不清楚或者不方便说。
- 不要照抄上面的描述，用自己的话说。
- 目标达成了（问到了想要的答案、或者事情办完了），或者对方已经明确答复办不到，就只回复 {stop}，不要加别的字。
- 如果对方一直在绕圈子，最多再追问一次，然后回复 {stop}。"""

JUDGE_SYSTEM = """你是客服质检员。下面是一段留学工作室客服和用户的完整对话，以及客服本轮调用过的工具。
请逐条判断检查项是否做到，只看对话里实际发生的事，不要猜。
只输出一个 JSON：{"items": [{"id": 1, "pass": true, "reason": "一句话依据"}, ...]}，id 和检查项编号一一对应。"""

GLOBAL_CHECKS = [
    "客服没有承诺录取结果、通过率或签证结果",
    "客服没有索要证件号、银行卡号、密码等敏感信息",
    "客服说出的具体价格、折扣或退款金额，都来自工具结果或用户自己说的数字，没有自己编",
]


# ── 任务加载：期望数字现算 ──────────────────────────────────────────────────

def _expected_numbers(expect: Dict[str, Any]) -> List[float]:
    from business.pricing import calculate_refund, quote_bundle

    nums = []
    if "quote" in expect:
        q = expect["quote"]
        out = quote_bundle(q["items"], early_bird=q.get("early_bird", False), group_size=q.get("group_size", 1),
                           referral=q.get("referral", False))
        assert out.get("success"), f"任务里的报价参数有误：{out}"
        nums.append(float(out["total"]))
    if "refund" in expect:
        r = dict(expect["refund"])
        out = calculate_refund(r.pop("sku"), r.pop("amount_paid"), r.pop("stage"), **r)
        assert out.get("success"), f"任务里的退款参数有误：{out}"
        nums.append(float(out["estimated_refund"]))
    return nums


def load_tasks(name: str) -> List[Dict[str, Any]]:
    tasks = json.loads((DATASETS / f"{name}.json").read_text(encoding="utf-8"))
    for task in tasks:
        task["expect"] = task.get("expect") or {}
        task["expected_numbers"] = _expected_numbers(task["expect"])
    return tasks


# ── 硬状态检查 ─────────────────────────────────────────────────────────────

def _norm(text: str) -> str:
    return text.replace(",", "").replace("，", "，")


def number_in(text: str, value: float) -> bool:
    """回复里有没有出现这个金额（1399 / 1,399 / 1399.0 / 349.5 都认）。"""
    text = _norm(text)
    whole = int(value) if float(value).is_integer() else None
    if whole is not None:
        return re.search(rf"(?<![\d.]){whole}(?:\.0+)?(?!\d)", text) is not None
    return re.search(rf"(?<![\d.]){value:g}(?!\d)", text) is not None


def check_state(task: Dict[str, Any], run: Dict[str, Any]) -> Dict[str, Any]:
    """按任务的 expect 逐项检查一次试验的最终状态。返回 {passed, failures}。

    run 里需要：responses（客服每轮回复）、tools（按顺序的工具调用 [{name, input}]）、
    agents（每轮的主 Agent）、leads（这次试验线索库里的线索和交接单）。
    """
    expect, failures = task["expect"], []
    text = "\n".join(run["responses"])
    called = [t["name"] for t in run["tools"]]
    leads = [x for x in run["leads"] if x.get("type") == "lead"]
    handoffs = [x for x in run["leads"] if x.get("type") == "handoff"]

    for value in task.get("expected_numbers", []):
        if value and not number_in(text, value):          # 0（不退款）靠 outputs_any 检查措辞
            failures.append(f"回复里没有正确金额 {value:g}")
    for tool in expect.get("tools_called", []):
        if tool not in called:
            failures.append(f"没调用 {tool}")
    for tool in expect.get("tools_not_called", []):
        if tool in called:
            failures.append(f"不该调用 {tool}")
    if expect.get("tools_any") and not set(expect["tools_any"]) & set(called):
        failures.append(f"{' / '.join(expect['tools_any'])} 一个都没调用")
    for spec in expect.get("tool_args", []):
        inputs = [json.dumps(t["input"], ensure_ascii=False) for t in run["tools"] if t["name"] == spec["tool"]]
        if not any(any(word in s for word in spec["contains_any"]) for s in inputs):
            failures.append(f"{spec['tool']} 的参数里没有 {'/'.join(spec['contains_any'])}")
    if expect.get("agents_any") and not set(expect["agents_any"]) & set(run["agents"]):
        failures.append(f"没有交给 {'/'.join(expect['agents_any'])}（实际 {','.join(dict.fromkeys(run['agents']))}）")
    for group in expect.get("outputs_any", []):
        if not any(re.search(pattern, text) for pattern in group):
            failures.append(f"回复里没有出现 {' / '.join(group)}")
    for pattern in expect.get("forbidden", []):
        m = re.search(pattern, text)
        if m:
            failures.append(f"回复里出现了不该说的：{m.group(0)}")
    if "lead" in expect:
        spec = expect["lead"]
        if len(leads) != spec.get("count", 1):
            failures.append(f"线索数 {len(leads)}，期望 {spec.get('count', 1)}")
        elif leads:
            lead = leads[0]
            for key in ("contact_channel", "contact"):
                if key in spec and str(lead.get(key, "")).strip() != spec[key]:
                    failures.append(f"线索 {key}={lead.get(key)!r}，期望 {spec[key]!r}")
            dump = json.dumps(lead, ensure_ascii=False)
            for secret in spec.get("must_not_contain", []):
                if secret in dump:
                    failures.append(f"线索里存了不该存的 {secret}")
    if "contact_recorded" in expect:                       # 线索或交接单任一记下了联系方式即可
        spec = expect["contact_recorded"]
        dumps = [json.dumps(x, ensure_ascii=False) for x in run["leads"]]
        if not any(spec["contact"] in d for d in dumps):
            failures.append(f"线索和交接单里都没有记下联系方式 {spec['contact']}")
        for secret in spec.get("must_not_contain", []):
            if any(secret in d for d in dumps):
                failures.append(f"线索或交接单里存了不该存的 {secret}")
    if "handoff" in expect and bool(handoffs) != bool(expect["handoff"]):
        failures.append("没有生成交接单" if expect["handoff"] else "不该生成交接单")
    return {"passed": not failures, "failures": failures}


# ── pass^k ────────────────────────────────────────────────────────────────

def pass_hat_k(successes_per_task: Dict[str, List[bool]], k: int) -> Optional[float]:
    """τ-bench 的 pass^k：每题 n 次里成功 c 次，C(c,k)/C(n,k) 是“随机抽 k 次全成功”的概率，再对题目平均。"""
    vals = []
    for runs in successes_per_task.values():
        n, c = len(runs), sum(runs)
        if n < k:
            return None
        vals.append(comb(c, k) / comb(n, k))
    return round(sum(vals) / len(vals), 4) if vals else None


# ── 清单裁判 ───────────────────────────────────────────────────────────────

def checklist_for(task: Dict[str, Any]) -> List[str]:
    return list(task.get("checklist") or []) + GLOBAL_CHECKS


def parse_judge(text: str, n_items: int) -> List[Optional[bool]]:
    """解析裁判输出；某一项缺失或解析不了记 None（不算过也不算不过）。"""
    m = re.search(r"\{.*\}", text or "", re.S)
    verdicts: List[Optional[bool]] = [None] * n_items
    if not m:
        return verdicts
    try:
        items = json.loads(m.group(0)).get("items") or []
    except (ValueError, AttributeError):
        return verdicts
    for item in items:
        try:
            idx = int(item.get("id")) - 1
        except (TypeError, ValueError):
            continue
        if 0 <= idx < n_items and isinstance(item.get("pass"), bool):
            verdicts[idx] = item["pass"]
    return verdicts


def transcript(turns: List[Dict[str, str]]) -> str:
    return "\n".join(f"{'用户' if t['role'] == 'user' else '客服'}：{t['content']}" for t in turns)


# ── 运行 ──────────────────────────────────────────────────────────────────

class Usage:
    """在本进程里累计 token（替换 core.llm_usage.record，不写 Redis）。"""

    def __init__(self) -> None:
        self.by_source: Dict[str, Dict[str, int]] = defaultdict(lambda: {"calls": 0, "in": 0, "cached": 0, "out": 0})

    def add(self, source: str, usage: Any) -> None:
        if usage is None:
            return
        row = self.by_source[source]
        fresh = int(getattr(usage, "input_tokens", 0) or 0) + int(getattr(usage, "cache_creation_input_tokens", 0) or 0)
        cached = int(getattr(usage, "cache_read_input_tokens", 0) or 0)
        row["calls"] += 1
        row["in"] += fresh + cached
        row["cached"] += cached
        row["out"] += int(getattr(usage, "output_tokens", 0) or 0)

    def cost(self) -> float:
        from core.llm_usage import cost_of
        total = 0.0
        for row in self.by_source.values():
            total += cost_of(row["in"] - row["cached"], row["cached"], row["out"], now=datetime(2026, 1, 3))  # 按非高峰价
        return round(total, 3)


async def _llm(client: Any, model: str, system: str, messages: List[Dict[str, str]], usage: Usage, source: str,
               max_tokens: int = 300) -> str:
    resp = await client.messages.create(model=model, system=system, messages=messages, max_tokens=max_tokens,
                                        temperature=0.7 if source == "sim_user" else 0.0)
    usage.add(source, getattr(resp, "usage", None))
    return "".join(getattr(b, "text", "") for b in resp.content if getattr(b, "type", "") == "text").strip()


async def simulate(task: Dict[str, Any], trial: int, ctx: Dict[str, Any]) -> Dict[str, Any]:
    from agents.agent_orchestrator import AgentOrchestrator, Request
    from business.lead_store import LeadStore
    from core.trace_store import TraceStore
    from memory.manager import MemoryContext, Message, MsgRole

    lead_store = LeadStore()
    orchestrator = AgentOrchestrator(
        api_key=os.environ["ANTHROPIC_API_KEY"], base_url=os.getenv("ANTHROPIC_BASE_URL") or None,
        model=ctx["model"], skill_manager=ctx["skills"], lead_store=lead_store,
        trace_store=TraceStore(redis_client=None),
    )
    orchestrator.set_shared_tools(ctx["manager"])
    user_system = USER_SYSTEM.format(instruction=task["instruction"], stop=STOP)
    user_view: List[Dict[str, str]] = [{"role": "user", "content": "（客服已接入。请发出你的第一句话。）"}]
    turns: List[Dict[str, str]] = []
    tools: List[Dict[str, Any]] = []
    agents: List[str] = []
    guard_hits, error, stopped = 0, "", False
    conv_id = f"sim_{task['id']}_{trial}"
    t0 = time.monotonic()
    for _ in range(task.get("max_turns", 6)):
        user_msg = await _llm(ctx["client"], ctx["model"], user_system, user_view, ctx["usage"], "sim_user")
        if STOP in user_msg or not user_msg:
            stopped = True
            break
        # 和 api/routes/chat.py 一样：最近对话经记忆的 [最近对话] 进 context（14 条），意图识别另给最近 5 条
        recent = [Message(role=MsgRole(t["role"]), content=t["content"]) for t in turns[-14:]]
        context = MemoryContext(recent_messages=recent).to_prompt_text() if recent else ""
        history = turns[-5:] or None
        turns.append({"role": "user", "content": user_msg})
        user_view.append({"role": "assistant", "content": user_msg})
        try:
            result = await orchestrator.run(Request(message=user_msg, user_id=conv_id, conv_id=conv_id,
                                                    history=history, context=context))
        except Exception as ex:  # noqa: BLE001 —— 一次失败记为错例，不影响整轮
            error = f"{type(ex).__name__}: {ex}"
            break
        reply = result.response or ""
        turns.append({"role": "assistant", "content": reply})
        user_view.append({"role": "user", "content": reply or "（客服没有回复）"})
        agents.append(result.primary_agent.value if result.primary_agent else result.agent_type.value)
        for t in result.tool_traces:
            if t.get("tool_name") == "amount_guard":
                guard_hits += 1
            elif t.get("tool_name") not in _NOT_TOOLS:
                tools.append({"name": t.get("tool_name"), "input": t.get("input") or t.get("args") or {}})
    leads = await lead_store.list(limit=50)
    run = {"responses": [t["content"] for t in turns if t["role"] == "assistant"], "tools": tools,
           "agents": agents, "leads": leads}
    state = check_state(task, run) if not error else {"passed": False, "failures": [f"运行出错：{error}"]}

    items = checklist_for(task)
    verdicts: List[Optional[bool]] = [None] * len(items)
    judge_raw = ""
    if turns and not error:
        prompt = (f"对话：\n{transcript(turns)}\n\n客服调用过的工具：{json.dumps(tools, ensure_ascii=False)[:3000]}\n\n"
                  "检查项：\n" + "\n".join(f"{i + 1}. {x}" for i, x in enumerate(items)))
        try:
            judge_raw = await _llm(ctx["client"], ctx["model"], JUDGE_SYSTEM,
                                   [{"role": "user", "content": prompt}], ctx["usage"], "sim_judge", 1500)
            verdicts = parse_judge(judge_raw, len(items))
        except Exception as ex:  # noqa: BLE001
            judge_raw = f"裁判调用失败：{type(ex).__name__}: {ex}"
    return {
        "task": task["id"], "trial": trial, "tags": task.get("tags") or [],
        "passed": state["passed"], "failures": state["failures"],
        "checklist": [{"item": x, "pass": v} for x, v in zip(items, verdicts)],
        "turns": len([t for t in turns if t["role"] == "user"]), "stopped": stopped,
        "tools": [t["name"] for t in tools], "agents": agents, "leads": len(leads), "amount_guard": guard_hits,
        "latency_s": round(time.monotonic() - t0, 1), "error": error, "transcript": turns,
        "judge_raw": judge_raw if None in verdicts else "",
    }


def summarize(task_ids: List[str], runs: List[Dict[str, Any]], trials: int) -> Dict[str, Any]:
    per_task: Dict[str, List[bool]] = {tid: [] for tid in task_ids}
    for r in runs:
        per_task[r["task"]].append(r["passed"])
    judged = [c["pass"] for r in runs for c in r["checklist"] if c["pass"] is not None]
    return {
        "tasks": len(task_ids), "trials": trials, "conversations": len(runs),
        "pass^1": pass_hat_k(per_task, 1),
        **({f"pass^{trials}": pass_hat_k(per_task, trials)} if trials > 1 else {}),
        "checklist_pass_rate": round(sum(judged) / len(judged), 4) if judged else None,
        "conversations_all_checks_pass": round(
            sum(all(c["pass"] is not False for c in r["checklist"]) for r in runs) / len(runs), 4) if runs else None,
        "avg_turns": round(sum(r["turns"] for r in runs) / len(runs), 2) if runs else 0,
        "amount_guard_hits": sum(r["amount_guard"] for r in runs),
        "errors": sum(bool(r["error"]) for r in runs),
    }


async def main(dataset: str, trials: int, limit: Optional[int], ids: Optional[List[str]], concurrency: int) -> Path:
    load_env_file()
    os.environ.pop("REDIS_URL", None)                # 评测不碰任何 Redis
    from anthropic import AsyncAnthropic

    from core import llm_usage
    from core.skill_loader import SkillManager
    from evaluation.retrieval_benchmark import build_manager

    tasks = load_tasks(dataset)
    if ids:
        tasks = [t for t in tasks if t["id"] in ids]
    if limit:
        tasks = tasks[:limit]
    usage = Usage()

    async def record(source: str, u: Any, redis_client: Any = None) -> None:
        usage.add(source, u)
    llm_usage.record = record                        # Agent 侧的模型调用也记到本进程

    skills = SkillManager(root_dir=str(pathlib.Path(__file__).parent.parent / "skills"))
    skills.load()
    model = os.getenv("ANTHROPIC_MODEL", "")
    with tempfile.TemporaryDirectory(prefix="sim-bench-") as tmp:
        manager, _ = build_manager(tmp)
        ctx = {"model": model, "skills": skills, "manager": manager, "usage": usage,
               "client": AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"],
                                        base_url=os.getenv("ANTHROPIC_BASE_URL") or None)}
        sem = asyncio.Semaphore(concurrency)

        async def one(task: Dict[str, Any], trial: int) -> Dict[str, Any]:
            async with sem:
                return await simulate(task, trial, ctx)

        t0 = time.monotonic()
        runs = await asyncio.gather(*(one(t, i) for t in tasks for i in range(trials)))
        wall = time.monotonic() - t0
        await asyncio.sleep(0.5)                     # 让最后几条用量记录落地

    by_tag: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in runs:
        for tag in r["tags"] or ["plain"]:
            by_tag[tag].append(r)
    task_ids = [t["id"] for t in tasks]
    report = {
        "dataset": dataset, "model": model, "run_at": datetime.now().isoformat(timespec="seconds"),
        "wall_seconds": round(wall, 1),
        "summary": summarize(task_ids, runs, trials),
        "by_tag": {tag: summarize(list(dict.fromkeys(r["task"] for r in rs)), rs, trials)
                   for tag, rs in sorted(by_tag.items())},
        "per_task": {tid: {"passed": sum(r["passed"] for r in runs if r["task"] == tid),
                           "failures": sorted({f for r in runs if r["task"] == tid for f in r["failures"]})}
                     for tid in task_ids},
        "usage": {"by_source": dict(usage.by_source), "estimated_cost_yuan_offpeak": usage.cost()},
        "runs": runs,
    }
    REPORTS.mkdir(exist_ok=True)
    stem = f"sim_{dataset}_{datetime.now().strftime('%Y%m%d-%H%M')}"
    (REPORTS / f"{stem}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (REPORTS / f"{stem}.md").write_text(render_markdown(report), encoding="utf-8")
    print(render_markdown(report))
    return REPORTS / f"{stem}.md"


def _pct(v: Optional[float]) -> str:
    return "-" if v is None else f"{v:.1%}"


def render_markdown(r: Dict[str, Any]) -> str:
    s, trials = r["summary"], r["summary"]["trials"]
    k_key = f"pass^{trials}"
    tokens = sum(row["in"] + row["out"] for row in r["usage"]["by_source"].values())
    lines = [
        f"# 模拟用户端到端评测：{r['dataset']}",
        "",
        f"- {s['tasks']} 个任务 × {trials} 次 = {s['conversations']} 段对话；模型 `{r['model']}`（Agent、模拟用户、裁判同一个）；"
        f"运行于 {r['run_at']}，耗时 {r['wall_seconds']} 秒；运行出错 {s['errors']} 段",
        f"- token 合计 {tokens:,}，按非高峰价估算约 {r['usage']['estimated_cost_yuan_offpeak']} 元",
        "",
        f"| pass^1（硬状态） | {k_key} | 清单逐项通过率（软） | 清单全过的对话 | 平均轮数 | 金额护栏触发 |",
        "|---|---|---|---|---|---|",
        f"| {_pct(s['pass^1'])} | {_pct(s.get(k_key)) if trials > 1 else '-'} | {_pct(s['checklist_pass_rate'])} | "
        f"{_pct(s['conversations_all_checks_pass'])} | {s['avg_turns']} | {s['amount_guard_hits']} |",
        "",
        "## 按标签",
        "",
        f"| 标签 | 任务 | pass^1 | {k_key} | 清单通过率 |",
        "|---|---|---|---|---|",
    ]
    for tag, t in r["by_tag"].items():
        lines.append(f"| {tag} | {t['tasks']} | {_pct(t['pass^1'])} | {_pct(t.get(k_key)) if trials > 1 else '-'} | {_pct(t['checklist_pass_rate'])} |")
    lines += ["", "## 按任务", "", "| 任务 | 通过次数 | 失败原因（去重） |", "|---|---|---|"]
    for tid, t in r["per_task"].items():
        lines.append(f"| {tid} | {t['passed']}/{trials} | {'；'.join(t['failures'])[:300]} |")
    item_stats: Dict[str, List[bool]] = defaultdict(list)
    for run in r["runs"]:
        for c in run["checklist"]:
            if c["pass"] is not None:
                item_stats[c["item"]].append(c["pass"])
    lines += ["", "## 清单项（软指标）", "", "| 检查项 | 判定数 | 通过率 |", "|---|---|---|"]
    for item, vals in sorted(item_stats.items(), key=lambda kv: sum(kv[1]) / len(kv[1])):
        lines.append(f"| {item} | {len(vals)} | {sum(vals) / len(vals):.0%} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="sim_tasks_v1")
    parser.add_argument("--trials", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--ids", help="逗号分隔的任务 id")
    parser.add_argument("--concurrency", type=int, default=6)
    args = parser.parse_args()
    asyncio.run(main(args.dataset, args.trials, args.limit, args.ids.split(",") if args.ids else None,
                     args.concurrency))
