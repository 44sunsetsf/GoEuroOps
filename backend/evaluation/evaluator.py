"""
亮点：端到端 Agent 评测框架

核心问题：如何评测端到端 Agent？

评测维度：
  1. 意图识别准确率 —— 预测意图 vs 标注意图，计算 Accuracy / F1
  2. 响应质量评分 —— 用 LLM 作为评判者（LLM-as-Judge），
     从相关性、准确性、完整性、有用性四个维度打分
  3. 端到端对话评测 —— 模拟完整多轮对话，评估整体体验
  4. 回归测试 —— 与历史基线对比，防止性能退化
  5. Skill 路由准确率 —— 跑各 Skill 自带的命中用例（skills/*/evals/cases.json）
  6. 意图门控 RAG 对照实验（可选）—— 同一批知识类问题分别走门控和"模型自行检索"，
     对比延迟和检索工具调用次数

评分口径按场景校准：每个对话用例可以写 expected_behavior（期望行为），
例如"不给录取结论、引导预约顾问"。完整性按"是否完成了这个场景下助手该做的部分"
打分，正确地转交顾问不再被当成"没解决问题"扣分。

LLM-as-Judge 是评测 Agent 质量的关键技术：
  人工标注成本高、主观性强；用 LLM 评判可以规模化、可重复。
"""
import asyncio
import json
import logging
import os
import pathlib
import statistics
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from anthropic import AsyncAnthropic
from core.llm_usage import track

from core.llm_utils import NO_THINKING_KWARGS, extract_text_content

from core.intent_recognizer import IntentCategory, IntentRecognizer

logger = logging.getLogger(__name__)


# ── 数据结构 ──────────────────────────────────────────────────────────────────

@dataclass
class IntentTestCase:
    message:          str
    expected_intent:  str
    context:          Optional[Dict[str, Any]] = None


@dataclass
class QualityScores:
    """LLM-as-Judge 评分结果。"""
    relevance:    float   # 相关性：回答是否针对问题
    accuracy:     float   # 准确性：信息是否正确
    completeness: float   # 完整性：是否完成了该场景下助手应完成的部分
    helpfulness:  float   # 有用性：用户是否能据此行动
    compliance:   float = 1.0  # 边界合规：不承诺录取、不编造价格、不代写、不索要敏感信息
    judge_failed: bool = False
    error: Optional[str] = None
    comment: str = ""

    @property
    def overall(self) -> float:
        return statistics.mean([self.relevance, self.accuracy, self.completeness, self.helpfulness, self.compliance])


@dataclass
class EvalResult:
    test_id:    str
    passed:     bool
    scores:     Dict[str, float]
    detail:     str = ""
    metadata:   Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalReport:
    """评测报告。"""
    timestamp:        str
    total:            int
    passed:           int
    pass_rate:        float
    avg_scores:       Dict[str, float]
    regressions:      List[str]          # 相比基线退化的指标
    recommendations:  List[str]
    results:          List[EvalResult]


# ── LLM-as-Judge ─────────────────────────────────────────────────────────────

class LLMJudge:
    """
    用 LLM 评判 Agent 响应质量。

    为什么用 LLM 而不是人工？
    - 可规模化：数千条测试用例自动评测
    - 可重复：相同输入得到稳定评分
    - 多维度：同时评估相关性、准确性等多个维度

    注意：LLM Judge 本身也有偏差，建议定期用人工标注校准。
    """

    JUDGE_PROMPT = """你是留学咨询工作室「指北」的服务质量评估员。工作室的 AI 助手负责：回答五国英语授课 CS 硕士的公开知识、介绍服务并用工具报价、登记线索、解释付款退款政策；
它不能替顾问做个性化选校结论、不能写改文书、不能承诺录取、不能给公开规则以外的折扣，这些情况下正确做法是说明原因并引导预约顾问。

用户问题: {question}
助手回复: {response}
{context_section}{expected_section}
请按 0.0–1.0 评分，返回 JSON：
- relevance: 是否直接回应了用户的问题
- accuracy: 信息是否准确、没有编造（参考信息注明以官网为准不扣分）
- completeness: 是否完成了**这个场景下助手应该完成的部分**（包括正确地追问必要信息、转交顾问并说明下一步）；不要因为助手没有替代顾问做决定而扣分
- helpfulness: 用户看完能否知道下一步做什么
- compliance: 是否守住边界（不承诺录取、不编造价格或折扣、不代写文书、不索要证件号/银行卡/密码）
- comment: 一句话说明最主要的扣分原因（没有就写"无"）

只返回 JSON，例如: {{"relevance": 0.9, "accuracy": 0.8, "completeness": 0.85, "helpfulness": 0.85, "compliance": 1.0, "comment": "无"}}"""

    def __init__(self, client: AsyncAnthropic, model: str):
        self._client = client
        self._model  = model

    async def judge(
        self,
        question: str,
        response: str,
        context: Optional[str] = None,
        expected_behavior: Optional[str] = None,
    ) -> QualityScores:
        ctx_section = f"背景信息: {context}\n" if context else ""
        expected_section = f"该场景的期望行为: {expected_behavior}\n" if expected_behavior else ""
        prompt = self.JUDGE_PROMPT.format(
            question=question,
            response=response,
            context_section=ctx_section,
            expected_section=expected_section,
        )
        prompt = self._clean_text(prompt)
        try:
            resp = await self._client.messages.create(
                model=self._model, max_tokens=400, temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
                **NO_THINKING_KWARGS,
            )
            raw = extract_text_content(resp.content)
            s, e = raw.find("{"), raw.rfind("}") + 1
            data = json.loads(raw[s:e])
            return QualityScores(
                relevance=float(data.get("relevance", 0.5)),
                accuracy=float(data.get("accuracy", 0.5)),
                completeness=float(data.get("completeness", 0.5)),
                helpfulness=float(data.get("helpfulness", 0.5)),
                compliance=float(data.get("compliance", 0.5)),
                comment=str(data.get("comment", ""))[:200],
            )
        except Exception as ex:
            logger.warning(f"LLM Judge 失败: {ex}")
            return QualityScores(
                0.5, 0.5, 0.5, 0.5, 0.5,
                judge_failed=True,
                error=str(ex),
            )

    @staticmethod
    def _clean_text(value: Any) -> str:
        """移除 Unicode 代理字符，避免 LLM 请求编码失败。"""
        if value is None:
            return ""
        if not isinstance(value, str):
            value = str(value)
        return value.encode("utf-8", errors="ignore").decode("utf-8")


# ── 意图识别评测 ──────────────────────────────────────────────────────────────

class IntentEvaluator:
    """评测意图识别的准确率和 F1。"""

    def __init__(self, recognizer: IntentRecognizer):
        self._recognizer = recognizer

    async def evaluate(self, cases: List[IntentTestCase]) -> Dict[str, Any]:
        predictions, ground_truth = [], []
        case_details: List[Dict[str, Any]] = []

        for case in cases:
            result = await self._recognizer.recognize(case.message)
            predicted = result.intent.value
            predictions.append(predicted)
            ground_truth.append(case.expected_intent)
            case_details.append({
                "message": case.message,
                "expected": case.expected_intent,
                "predicted": predicted,
                "confidence": result.confidence,
                "reasoning": result.reasoning,
            })

        # 纯 Python 计算指标
        correct = sum(p == g for p, g in zip(predictions, ground_truth))
        accuracy = correct / len(predictions) if predictions else 0.0

        # 每类 F1
        labels = sorted(set(ground_truth + predictions))
        per_class: Dict[str, Dict[str, float]] = {}
        for label in labels:
            tp = sum(p == label and g == label for p, g in zip(predictions, ground_truth))
            fp = sum(p == label and g != label for p, g in zip(predictions, ground_truth))
            fn = sum(p != label and g == label for p, g in zip(predictions, ground_truth))
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec  = tp / (tp + fn) if (tp + fn) else 0.0
            f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
            per_class[label] = {"precision": prec, "recall": rec, "f1": f1}

        macro_f1 = statistics.mean(v["f1"] for v in per_class.values()) if per_class else 0.0

        return {
            "accuracy":   round(accuracy, 4),
            "macro_f1":   round(macro_f1, 4),
            "per_class":  per_class,
            "total":      len(cases),
            "correct":    correct,
            "cases":      case_details,
        }


# ── 端到端评测器 ──────────────────────────────────────────────────────────────

class EndToEndEvaluator:
    """
    端到端 Agent 评测。

    评测流程：
      1. 运行意图识别评测（准确率/F1）
      2. 运行对话质量评测（LLM-as-Judge）
      3. 与历史基线对比（回归检测）
      4. 生成可操作的优化建议
    """

    # 质量及格线（整体）与单维度建议阈值
    PASS_THRESHOLD = 0.75
    DIMENSIONS = ("relevance", "accuracy", "completeness", "helpfulness", "compliance")

    def __init__(
        self,
        orchestrator,
        recognizer: IntentRecognizer,
        api_key:  str,
        base_url: Optional[str] = None,
        model:    str = "claude-3-5-sonnet-20241022",
        baseline_path: Optional[str] = None,
        skill_manager: Optional[Any] = None,
    ):
        kwargs: Dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        client = track(AsyncAnthropic(**kwargs), "evaluation")

        self._orchestrator     = orchestrator
        self._judge            = LLMJudge(client, model)
        self._intent_evaluator = IntentEvaluator(recognizer)
        self._history:         List[EvalReport] = []
        self._baseline_path = pathlib.Path(baseline_path) if baseline_path else None
        self._baseline: Optional[EvalReport] = self._load_baseline()
        self._skill_manager = skill_manager
        try:
            self.dim_threshold = float(os.getenv("GOEUROOPS_EVAL_DIM_THRESHOLD", "0.75"))
        except ValueError:
            self.dim_threshold = 0.75

    async def run(
        self,
        intent_cases:    Optional[List[IntentTestCase]] = None,
        dialog_cases:    Optional[List[Dict[str, Any]]] = None,
        include_skill_evals: bool = True,
        compare_rag_gate: bool = False,
    ) -> EvalReport:
        """
        运行完整评测。

        intent_cases: 意图识别测试用例
        dialog_cases:
          - 单轮: [{"question": "..."}]
          - 多轮: [{"turns": ["第一轮", "第二轮", ...]}]
        """
        results: List[EvalResult] = []
        all_scores: Dict[str, List[float]] = {k: [] for k in self.DIMENSIONS}

        # 1. 意图识别评测
        intent_metrics: Dict[str, Any] = {}
        if intent_cases:
            intent_metrics = await self._intent_evaluator.evaluate(intent_cases)
            passed = intent_metrics["accuracy"] >= self.PASS_THRESHOLD
            results.append(EvalResult(
                test_id="intent_recognition",
                passed=passed,
                scores={"accuracy": intent_metrics["accuracy"], "macro_f1": intent_metrics["macro_f1"]},
                detail=f"准确率 {intent_metrics['accuracy']:.1%}，Macro-F1 {intent_metrics['macro_f1']:.3f}",
                metadata={
                    "total": intent_metrics.get("total", 0),
                    "correct": intent_metrics.get("correct", 0),
                    "cases": intent_metrics.get("cases", []),
                },
            ))

        # 2. 对话质量评测（调用 orchestrator 产出回复，再用 LLM Judge 评分）
        if dialog_cases:
            for i, case in enumerate(dialog_cases):
                case_results = await self._evaluate_dialog_case(case, i)
                results.extend(case_results)
                for r in case_results:
                    for k in all_scores:
                        if k in r.scores:
                            all_scores[k].append(r.scores[k])

        # 3. Skill 路由准确率（确定性，不调用 LLM）
        skill_metrics: Dict[str, Any] = {}
        if include_skill_evals and self._skill_manager is not None and hasattr(self._skill_manager, "run_evals"):
            skill_metrics = self._skill_manager.run_evals()
            if skill_metrics.get("total"):
                results.append(EvalResult(
                    test_id="skill_routing",
                    passed=skill_metrics["accuracy"] >= 0.9,
                    scores={"accuracy": skill_metrics["accuracy"]},
                    detail=f"Skill 命中用例 {skill_metrics['passed']}/{skill_metrics['total']}",
                    metadata={"failures": skill_metrics["failures"]},
                ))

        # 4. 意图门控 RAG 对照实验（可选，会额外调用 LLM）
        rag_comparison: Dict[str, Any] = {}
        if compare_rag_gate and dialog_cases:
            rag_comparison = await self._compare_rag_gate(dialog_cases)
            if rag_comparison.get("cases"):
                results.append(EvalResult(
                    test_id="rag_gate_ab",
                    passed=True,
                    scores={
                        "gated_avg_latency_ms": rag_comparison["gated"]["avg_latency_ms"],
                        "baseline_avg_latency_ms": rag_comparison["baseline"]["avg_latency_ms"],
                        "gated_avg_search_calls": rag_comparison["gated"]["avg_search_calls"],
                        "baseline_avg_search_calls": rag_comparison["baseline"]["avg_search_calls"],
                    },
                    detail=rag_comparison["summary"],
                    metadata=rag_comparison,
                ))

        # 5. 汇总
        avg_scores = {
            k: round(statistics.mean(v), 4) for k, v in all_scores.items() if v
        }
        if intent_metrics:
            avg_scores["intent_accuracy"] = intent_metrics["accuracy"]
        if skill_metrics.get("total"):
            avg_scores["skill_routing_accuracy"] = skill_metrics["accuracy"]
        tool_checks = [r.scores["tool_check"] for r in results if "tool_check" in r.scores]
        if tool_checks:
            avg_scores["tool_accuracy"] = round(statistics.mean(tool_checks), 4)

        passed_count = sum(1 for r in results if r.passed)
        pass_rate    = passed_count / len(results) if results else 0.0

        # 6. 回归检测
        regressions = self._detect_regressions(avg_scores)

        # 7. 优化建议（基于失败样本，而不是固定模板）
        recommendations = self._recommendations(avg_scores, intent_metrics, results, skill_metrics)

        report = EvalReport(
            timestamp=datetime.now().isoformat(),
            total=len(results),
            passed=passed_count,
            pass_rate=round(pass_rate, 4),
            avg_scores=avg_scores,
            regressions=regressions,
            recommendations=recommendations,
            results=results,
        )
        self._history.append(report)
        self._save_baseline(report)
        return report

    async def _evaluate_dialog_case(self, case: Dict[str, Any], case_idx: int) -> List[EvalResult]:
        """评测单轮或多轮对话用例。"""
        from agents.agent_orchestrator import Request as OrcReq

        questions = self._dialog_turns(case)
        if not questions:
            return []

        conv_id = str(case.get("conv_id") or f"eval_{case_idx}")
        user_id = str(case.get("user_id") or "eval_user")
        history: List[Dict[str, str]] = []
        results: List[EvalResult] = []

        for turn_idx, question in enumerate(questions):
            context = self._history_context(history)
            orch_req = OrcReq(
                message=question,
                user_id=user_id,
                conv_id=conv_id,
                context=context,
                history=history[-6:] if history else None,
            )
            orch_result = await self._orchestrator.run(orch_req)
            actual_answer = orch_result.response

            is_last_turn = turn_idx == len(questions) - 1
            expected_behavior = case.get("expected_behavior") if is_last_turn else None
            scores = await self._judge.judge(
                question, actual_answer, context=context or None, expected_behavior=expected_behavior
            )
            passed = scores.overall >= self.PASS_THRESHOLD

            # 期望工具：确定性检查，不依赖 LLM Judge
            expected_tools = list(case.get("expected_tools") or []) if is_last_turn else []
            missing_tools = [t for t in expected_tools if t not in orch_result.tools_used]
            if expected_tools and missing_tools:
                passed = False

            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": actual_answer})

            test_id = f"dialog_{case_idx}" if len(questions) == 1 else f"dialog_{case_idx}_turn_{turn_idx}"
            results.append(EvalResult(
                test_id=test_id,
                passed=passed,
                scores={
                    "relevance": scores.relevance,
                    "accuracy": scores.accuracy,
                    "completeness": scores.completeness,
                    "helpfulness": scores.helpfulness,
                    "compliance": scores.compliance,
                    "overall": scores.overall,
                    **({"tool_check": 0.0 if missing_tools else 1.0} if expected_tools else {}),
                },
                detail=f"Q: {question[:30]}... → 综合评分 {scores.overall:.3f}"
                       + (f"，缺少工具调用 {', '.join(missing_tools)}" if missing_tools else ""),
                metadata={
                    "question": question,
                    "response": actual_answer,
                    "agent_type": orch_result.agent_type.value,
                    "intent": orch_result.intent.value if orch_result.intent else None,
                    "turn": turn_idx,
                    "conv_id": conv_id,
                    "expected_behavior": expected_behavior,
                    "expected_tools": expected_tools,
                    "tools_used": list(orch_result.tools_used),
                    "skills_applied": [s.get("id") for s in orch_result.skills_applied],
                    "rag_gate": orch_result.rag_gate,
                    "judge_comment": scores.comment,
                    "judge_failed": scores.judge_failed,
                    "judge_error": scores.error,
                },
            ))

        return results

    @staticmethod
    def _dialog_turns(case: Dict[str, Any]) -> List[str]:
        turns = case.get("turns")
        if isinstance(turns, list):
            return [str(t) for t in turns if str(t).strip()]
        question = case.get("question")
        return [str(question)] if question else []

    @staticmethod
    def _history_context(history: List[Dict[str, str]]) -> str:
        if not history:
            return ""
        lines = [f"{m['role']}: {m['content']}" for m in history[-8:]]
        return "[评测多轮历史]\n" + "\n".join(lines)

    def _detect_regressions(self, current: Dict[str, float]) -> List[str]:
        """与上一次评测对比，找出退化超过 5% 的指标。"""
        prev_report = self._history[-1] if self._history else self._baseline
        if prev_report is None:
            return []
        prev = prev_report.avg_scores
        regressions = []
        for metric, value in current.items():
            if metric in prev and prev[metric] > 0:
                delta = (value - prev[metric]) / prev[metric]
                if delta < -0.05:
                    regressions.append(
                        f"{metric}: {prev[metric]:.3f} → {value:.3f} (退化 {abs(delta):.1%})"
                    )
        return regressions

    _DIMENSION_ADVICE = {
        "relevance": ("相关性", "回答没有扣住问题。检查对应 Agent 的 system_prompt 和命中的 Skill 是否把话题带偏，或路由是否选错了 Agent"),
        "accuracy": ("准确性", "存在不准确或编造的信息。检查知识库和 business/*.yaml 是否覆盖了这些问题，并在 Skill 里强调金额只能来自工具"),
        "completeness": ("完整性", "没有完成该场景下应做的部分。对照下面样本的期望行为，补充对应 Skill 的处理流程（例如转顾问时要给出下一步和时效）"),
        "helpfulness": ("有用性", "用户看完不知道下一步做什么。在 Skill 的回复模板里加上明确的下一步（继续问 / 查看服务 / 预约）"),
        "compliance": ("边界合规", "出现越界表述（承诺录取、自创折扣、代写、索要敏感信息）。优先修复，检查 studio_brand_voice 和对应 Skill 的禁止事项"),
    }

    def _recommendations(
        self,
        scores: Dict[str, float],
        intent_metrics: Dict[str, Any],
        results: Optional[List[EvalResult]] = None,
        skill_metrics: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """
        按失败样本生成建议。

        旧版用固定阈值 + 固定文案（完整性 < 0.75 就提示"过早结束回答"），而通用客服口径的
        Judge 天然会给"转顾问"类回答打低完整性分，于是每次都出现同一条建议。这里改成：
        只有当某维度均分低于阈值、或超过 30% 的样本低于阈值时才提示，并附上最差的样本。
        """
        results = results or []
        recs: List[str] = []
        threshold = self.dim_threshold

        if intent_metrics and scores.get("intent_accuracy", 1.0) < 0.9:
            wrong = [c for c in intent_metrics.get("cases", []) if c["expected"] != c["predicted"]]
            sample = "；".join(f"「{c['message']}」期望 {c['expected']} 实际 {c['predicted']}" for c in wrong[:3])
            recs.append(f"意图识别准确率 {scores['intent_accuracy']:.0%}：为这些样本补充 few-shot 模板或关键词。{sample}")

        dialog = [r for r in results if "overall" in r.scores and not r.metadata.get("judge_failed")]
        for dim, (label, advice) in self._DIMENSION_ADVICE.items():
            values = [r.scores[dim] for r in dialog if dim in r.scores]
            if not values:
                continue
            low = sorted((r for r in dialog if r.scores.get(dim, 1.0) < threshold), key=lambda r: r.scores[dim])
            avg = statistics.mean(values)
            if avg >= threshold and len(low) / len(values) <= 0.3:
                continue
            samples = "；".join(
                f"「{r.metadata.get('question', '')[:24]}」{r.scores[dim]:.2f}"
                + (f"（{r.metadata['judge_comment']}）" if r.metadata.get("judge_comment") not in (None, "", "无") else "")
                for r in low[:3]
            )
            recs.append(f"{label}均分 {avg:.2f}，{len(low)}/{len(values)} 个样本低于 {threshold}：{advice}。最差样本：{samples}")

        missing_tool = [r for r in results if r.scores.get("tool_check") == 0.0]
        if missing_tool:
            samples = "；".join(
                f"「{r.metadata.get('question', '')[:24]}」期望 {r.metadata.get('expected_tools')} 实际 {r.metadata.get('tools_used')}"
                for r in missing_tool[:3]
            )
            recs.append(f"有 {len(missing_tool)} 个用例没有调用期望的工具：检查工具描述和 Skill 里的强制调用规则。{samples}")

        if skill_metrics and skill_metrics.get("failures"):
            samples = "；".join(
                f"{f['skill']}「{f['message']}」期望{'命中' if f.get('expect_hit', True) else '不命中'}"
                for f in skill_metrics["failures"][:3]
            )
            recs.append(f"Skill 路由有 {len(skill_metrics['failures'])} 个用例未通过：调整对应 Skill 的 keywords / examples / intents。{samples}")

        judge_failed = [r for r in results if r.metadata.get("judge_failed")]
        if judge_failed:
            recs.append(f"{len(judge_failed)} 个样本 LLM Judge 调用失败（按 0.5 计分，未计入建议），请检查模型配置或重试")

        if not recs:
            recs.append("所有指标均达标，继续保持")
        return recs

    async def _compare_rag_gate(self, dialog_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """对知识类单轮用例，分别用门控（默认）和"模型自行检索"跑一遍，对比延迟与检索调用次数。"""
        from agents.agent_orchestrator import Request as OrcReq

        questions = [c["question"] for c in dialog_cases if c.get("knowledge") and c.get("question")]
        rows = []
        for idx, question in enumerate(questions):
            row: Dict[str, Any] = {"question": question}
            # 先预热意图缓存，两组都命中缓存，差异只来自检索策略本身（公平对比）
            await self._orchestrator.recognize_intent(question)
            for label, rag_mode in (("baseline", "on_demand"), ("gated", None)):
                req = OrcReq(message=question, user_id="eval_rag_ab", conv_id=f"rag_ab_{idx}_{label}", rag_mode=rag_mode)
                t0 = time.monotonic()
                result = await self._orchestrator.run(req)
                row[label] = {
                    "latency_ms": round((time.monotonic() - t0) * 1000, 1),
                    "search_calls": sum(1 for t in result.tool_traces if t.get("tool_name") == "search_knowledge_base"),
                    "tool_calls": len(result.tool_traces),
                    "rag_mode": result.rag_gate.get("mode"),
                    "prefetched": result.rag_gate.get("prefetched", 0),
                }
            rows.append(row)
        if not rows:
            return {}

        def agg(label: str) -> Dict[str, float]:
            return {
                "avg_latency_ms": round(statistics.mean(r[label]["latency_ms"] for r in rows), 1),
                "avg_search_calls": round(statistics.mean(r[label]["search_calls"] for r in rows), 2),
                "avg_tool_calls": round(statistics.mean(r[label]["tool_calls"] for r in rows), 2),
            }

        gated, baseline = agg("gated"), agg("baseline")
        saved = baseline["avg_latency_ms"] - gated["avg_latency_ms"]
        summary = (
            f"{len(rows)} 个知识类问题：门控平均 {gated['avg_latency_ms']:.0f}ms / 工具调用 {gated['avg_tool_calls']} 次，"
            f"模型自行检索平均 {baseline['avg_latency_ms']:.0f}ms / 工具调用 {baseline['avg_tool_calls']} 次，"
            f"{'节省' if saved >= 0 else '增加'} {abs(saved):.0f}ms（两组均命中意图缓存，差异来自检索策略）"
        )
        return {"cases": rows, "gated": gated, "baseline": baseline, "summary": summary}

    @property
    def history(self) -> List[EvalReport]:
        return self._history

    def _load_baseline(self) -> Optional[EvalReport]:
        if not self._baseline_path or not self._baseline_path.exists():
            return None
        try:
            data = json.loads(self._baseline_path.read_text(encoding="utf-8"))
            return self._report_from_dict(data)
        except Exception as ex:
            logger.warning(f"读取评测基线失败: {ex}")
            return None

    def _save_baseline(self, report: EvalReport) -> None:
        if not self._baseline_path:
            return
        try:
            self._baseline_path.parent.mkdir(parents=True, exist_ok=True)
            self._baseline_path.write_text(
                json.dumps(asdict(report), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            self._baseline = report
        except Exception as ex:
            logger.warning(f"保存评测基线失败: {ex}")

    @staticmethod
    def _report_from_dict(data: Dict[str, Any]) -> EvalReport:
        return EvalReport(
            timestamp=data.get("timestamp", ""),
            total=int(data.get("total", 0)),
            passed=int(data.get("passed", 0)),
            pass_rate=float(data.get("pass_rate", 0.0)),
            avg_scores=dict(data.get("avg_scores", {})),
            regressions=list(data.get("regressions", [])),
            recommendations=list(data.get("recommendations", [])),
            results=[
                EvalResult(
                    test_id=r.get("test_id", ""),
                    passed=bool(r.get("passed", False)),
                    scores=dict(r.get("scores", {})),
                    detail=r.get("detail", ""),
                    metadata=dict(r.get("metadata", {})),
                )
                for r in data.get("results", [])
            ],
        )


# ── 内置测试用例（开箱即用）──────────────────────────────────────────────────

DEFAULT_INTENT_CASES: List[IntentTestCase] = [
    IntentTestCase("你好",                              "greeting"),
    IntentTestCase("瑞典有哪些英语授课的CS硕士项目？",     "study_consult"),
    IntentTestCase("德国申请CS硕士需要APS吗？",           "application_process"),
    IntentTestCase("雅思一般要考到多少分？",              "application_process"),
    IntentTestCase("全程陪跑多少钱？",                    "service_inquiry"),
    IntentTestCase("能帮我约一次咨询吗？",                "booking"),
    IntentTestCase("我的PS改到第几轮了？",                "service_progress"),
    IntentTestCase("定金交了还能退吗？",                  "refund"),
    IntentTestCase("可以开发票吗？",                      "invoice"),
    IntentTestCase("尾款一直付不了",                      "payment_issue"),
    IntentTestCase("请删除我留下的个人信息",              "data_privacy"),
    IntentTestCase("我要找真人顾问",                      "human_handoff"),
    IntentTestCase("文书返回太慢了，很不满意",            "complaint"),
    IntentTestCase("顾问很专业，谢谢！",                  "feedback"),
    IntentTestCase("我换了微信号，麻烦更新一下",          "account"),
]

DEFAULT_DIALOG_CASES: List[Dict[str, Any]] = [
    {
        "question": "瑞典的计算机硕士怎么申请？什么时候截止？",
        "expected_behavior": "基于资料说明申请平台和时间窗口，提示以官网为准",
        "knowledge": True,
    },
    {
        "question": "德国CS硕士申请要准备哪些材料？APS要提前多久办？",
        "expected_behavior": "列出主要材料，说明 APS 需提前数月办理，提示以官网为准",
        "knowledge": True,
    },
    {
        "question": "芬兰读CS硕士有奖学金吗？",
        "expected_behavior": "说明芬兰学费减免奖学金普遍、通常与申请同步，提示以官网为准",
        "knowledge": True,
    },
    {
        "question": "我想要选校全案加3个项目的文书套餐，9月底前签约，一共多少钱？",
        "expected_behavior": "调用报价工具，按早鸟规则给出明细和签约时应付金额，不自行计算",
        "expected_tools": ["quote_service_bundle"],
    },
    {
        "question": "我均分85、雅思7，你直接告诉我能不能上KTH",
        "expected_behavior": "不给录取结论，说明需要顾问评估完整背景，介绍对应服务或免费初步沟通",
    },
    {
        "question": "定金交了但服务还没开始，可以退吗？",
        "expected_behavior": "说明服务未启动可全额退，收集合同号等信息，说明由顾问核验后处理",
    },
    {
        "turns": ["你好", "你们的全程陪跑包括什么？", "能便宜点吗？"],
        "expected_behavior": "不自行让价，只说明公开优惠规则（早鸟、团报、老带新），可建议更小的服务组合",
    },
    {
        "question": "请删除我之前留下的个人资料",
        "expected_behavior": "转交顾问处理隐私请求，给出交接单编号和响应时效，不索要证件信息",
    },
]
