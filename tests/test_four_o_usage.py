from __future__ import annotations

import importlib
import sys
from datetime import date, timedelta

import logging
import pytest


@pytest.fixture
def load_main(monkeypatch):
    original_module = sys.modules.get("main")

    def _loader(limit: int):
        monkeypatch.setenv("FOUR_O_DAILY_TOKEN_LIMIT", str(limit))
        sys.modules.pop("main", None)
        module = importlib.import_module("main")
        return module

    yield _loader
    sys.modules.pop("main", None)
    if original_module is not None:
        sys.modules["main"] = original_module


def test_four_o_usage_resets_on_new_day(load_main, monkeypatch):
    main = load_main(100)
    day_one = date(2024, 5, 1)
    day_two = day_one + timedelta(days=1)
    monkeypatch.setattr(main, "_current_utc_date", lambda: day_one)
    remaining = main._record_four_o_usage("test", "gpt-4o", 10, 5, 15)
    assert remaining == 85
    assert main._four_o_usage_state["used"] == 15
    monkeypatch.setattr(main, "_current_utc_date", lambda: day_two)
    remaining = main._record_four_o_usage("test", "gpt-4o", 2, 3, 5)
    assert remaining == 95
    assert main._four_o_usage_state["used"] == 5


def test_four_o_usage_remaining_is_never_negative(load_main, monkeypatch, caplog):
    main = load_main(50)
    today = date(2024, 5, 2)
    monkeypatch.setattr(main, "_current_utc_date", lambda: today)
    caplog.set_level(logging.INFO)
    caplog.clear()
    remaining = main._record_four_o_usage("ask", "gpt-4o", 30, 10, 40)
    assert remaining == 10
    remaining = main._record_four_o_usage("ask", "gpt-4o", 30, 20, 50)
    assert remaining == 0
    assert main._four_o_usage_state["used"] == 50
    assert "remaining=0" in caplog.records[-1].message
