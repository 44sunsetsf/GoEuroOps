from core.config import env_float, env_int
from core.llm_utils import safe_text


def test_env_helpers_fall_back_on_missing_or_invalid(monkeypatch):
    monkeypatch.delenv("GOEUROOPS_TEST_X", raising=False)
    assert env_float("GOEUROOPS_TEST_X", 0.5) == 0.5
    monkeypatch.setenv("GOEUROOPS_TEST_X", "")
    assert env_int("GOEUROOPS_TEST_X", 3) == 3
    monkeypatch.setenv("GOEUROOPS_TEST_X", "abc")
    assert env_float("GOEUROOPS_TEST_X", 0.5) == 0.5
    assert env_int("GOEUROOPS_TEST_X", 3) == 3
    monkeypatch.setenv("GOEUROOPS_TEST_X", "0.8")
    assert env_float("GOEUROOPS_TEST_X", 0.5) == 0.8


def test_safe_text_drops_surrogates_and_stringifies():
    assert safe_text(None) == ""
    assert safe_text(42) == "42"
    assert safe_text("ok\ud800") == "ok"
