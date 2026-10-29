"""模拟用户评测的确定性部分：状态检查器、pass^k、裁判输出解析、任务加载。"""
from evaluation.sim_benchmark import check_state, load_tasks, number_in, parse_judge, pass_hat_k


def _run(responses=(), tools=(), agents=(), leads=()):
    return {"responses": list(responses), "tools": list(tools), "agents": list(agents), "leads": list(leads)}


def test_number_in_accepts_common_writings_but_not_substrings():
    assert number_in("总价 ¥1,399 元", 1399)
    assert number_in("合计 1399.00 元", 1399)
    assert number_in("可退 349.5 元", 349.5)
    assert not number_in("编号 113990", 1399)
    assert not number_in("可退 1349.5 元", 349.5)


def test_check_state_quote_numbers_tools_and_leads():
    task = {"expect": {"tools_called": ["quote_service_bundle"], "lead": {"count": 0}}, "expected_numbers": [1399.0]}
    ok = check_state(task, _run(["全程陪跑 1399 元"], [{"name": "quote_service_bundle", "input": {}}]))
    assert ok == {"passed": True, "failures": []}
    bad = check_state(task, _run(["大概一千多"], [], leads=[{"type": "lead"}]))
    assert not bad["passed"] and len(bad["failures"]) == 3


def test_check_state_forbidden_agents_args_handoff_and_secrets():
    task = {"expect": {
        "forbidden": ["包录取"], "agents_any": ["billing"], "handoff": True,
        "tool_args": [{"tool": "lookup_country_admissions_overview", "contains_any": ["瑞典", "Sweden"]}],
        "lead": {"count": 1, "contact": "zhou_de2027", "must_not_contain": ["E12345678"]},
    }, "expected_numbers": []}
    run = _run(["我们包录取"], [{"name": "lookup_country_admissions_overview", "input": {"country": "荷兰"}}],
               ["consulting"], [{"type": "lead", "contact": "zhou_de2027", "background": "护照 E12345678"}])
    failures = check_state(task, run)["failures"]
    assert any("包录取" in f for f in failures)
    assert any("billing" in f for f in failures)
    assert any("交接单" in f for f in failures)
    assert any("参数" in f for f in failures)
    assert any("E12345678" in f for f in failures)


def test_contact_recorded_accepts_lead_or_handoff_and_zero_refund_is_wording_only():
    task = {"expect": {"contact_recorded": {"contact": "zhou_de2027", "must_not_contain": ["E12345678"]}},
            "expected_numbers": [0.0]}
    ok = check_state(task, _run(["已转顾问"], leads=[{"type": "handoff", "message": "微信 zhou_de2027 [证件号已隐去]"}]))
    assert ok["passed"]
    leaked = check_state(task, _run([], leads=[{"type": "lead", "contact": "zhou_de2027", "notes": "E12345678"}]))
    assert leaked["failures"] == ["线索或交接单里存了不该存的 E12345678"]


def test_pass_hat_k_matches_tau_bench_definition():
    runs = {"a": [True, True, True, True], "b": [True, False, True, True], "c": [False] * 4}
    assert pass_hat_k(runs, 1) == round((1 + 0.75 + 0) / 3, 4)
    assert pass_hat_k(runs, 4) == round((1 + 0 + 0) / 3, 4)
    assert pass_hat_k(runs, 2) == round((1 + 3 / 6 + 0) / 3, 4)
    assert pass_hat_k({"a": [True]}, 2) is None


def test_parse_judge_tolerates_noise_and_missing_items():
    text = '好的，结果如下：\n```json\n{"items": [{"id": 1, "pass": true}, {"id": 3, "pass": false}, {"id": 9, "pass": true}]}\n```'
    assert parse_judge(text, 3) == [True, None, False]
    assert parse_judge("不是 JSON", 2) == [None, None]


def test_tasks_load_and_expected_numbers_come_from_pricing():
    from business.pricing import quote_bundle

    tasks = {t["id"]: t for t in load_tasks("sim_tasks_v1")}
    assert 20 <= len(tasks) <= 30
    assert tasks["quote_full_journey"]["expected_numbers"] == [float(quote_bundle([{"sku": "full_journey"}])["total"])]
    assert tasks["refund_not_started"]["expected_numbers"] == [420.0]
    assert all(t["instruction"] and t.get("checklist") for t in tasks.values())
