from evaluation.evaluator import DEFAULT_DIALOG_CASES, DEFAULT_INTENT_CASES, EndToEndEvaluator, EvalResult
from core.intent_recognizer import IntentCategory


def make_evaluator():
    evaluator = EndToEndEvaluator.__new__(EndToEndEvaluator)
    evaluator.dim_threshold = 0.75
    return evaluator


def dialog(question, **scores):
    base = {"relevance": 0.9, "accuracy": 0.9, "completeness": 0.9, "helpfulness": 0.9, "compliance": 1.0}
    base.update(scores)
    base["overall"] = sum(base.values()) / len(base)
    return EvalResult(test_id=question, passed=True, scores=base, metadata={"question": question, "judge_comment": "无"})


def test_no_recommendation_when_dimensions_pass():
    results = [dialog("a"), dialog("b"), dialog("d"), dialog("c", completeness=0.7)]  # 1/4 低于阈值，未超过 30%
    recs = make_evaluator()._recommendations({"completeness": 0.83}, {}, results)
    assert recs == ["所有指标均达标，继续保持"]


def test_recommendation_lists_worst_samples():
    results = [dialog("帮我看看能不能上KTH", completeness=0.4), dialog("b", completeness=0.5), dialog("c")]
    results[0].metadata["judge_comment"] = "没有说明下一步"
    recs = make_evaluator()._recommendations({}, {}, results)
    assert len(recs) == 1
    assert recs[0].startswith("完整性均分")
    assert "帮我看看能不能上KTH" in recs[0]
    assert "没有说明下一步" in recs[0]


def test_missing_expected_tool_is_reported():
    result = dialog("报价")
    result.scores["tool_check"] = 0.0
    result.metadata.update(expected_tools=["quote_service_bundle"], tools_used=[])
    recs = make_evaluator()._recommendations({}, {}, [result])
    assert any("没有调用期望的工具" in r for r in recs)


def test_judge_failures_are_excluded_and_reported():
    failed = dialog("x", completeness=0.5, relevance=0.5, accuracy=0.5, helpfulness=0.5, compliance=0.5)
    failed.metadata["judge_failed"] = True
    recs = make_evaluator()._recommendations({}, {}, [failed])
    assert recs == ["1 个样本 LLM Judge 调用失败（按 0.5 计分，未计入建议），请检查模型配置或重试"]


def test_default_cases_use_valid_studio_intents():
    valid = {c.value for c in IntentCategory}
    assert all(case.expected_intent in valid for case in DEFAULT_INTENT_CASES)
    assert any(case.get("expected_tools") for case in DEFAULT_DIALOG_CASES)
    assert any(case.get("knowledge") for case in DEFAULT_DIALOG_CASES)
