from __future__ import annotations

import importlib
import sys
from datetime import date, timedelta

import logging
import pytest
from aiogram import types


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
    remaining = main._record_four_o_usage(
        "test",
        "gpt-4o",
        {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    )
    assert remaining == 85
    assert main._four_o_usage_state["used"] == 15
    assert main._four_o_usage_state["total"] == 15
    assert main._four_o_usage_state["models"]["gpt-4o"] == 15
    monkeypatch.setattr(main, "_current_utc_date", lambda: day_two)
    remaining = main._record_four_o_usage(
        "test",
        "gpt-4o",
        {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
    )
    assert remaining == 95
    assert main._four_o_usage_state["used"] == 5
    assert main._four_o_usage_state["total"] == 5
    assert main._four_o_usage_state["models"]["gpt-4o"] == 5


def test_four_o_usage_remaining_is_never_negative(load_main, monkeypatch, caplog):
    main = load_main(50)
    today = date(2024, 5, 2)
    monkeypatch.setattr(main, "_current_utc_date", lambda: today)
    caplog.set_level(logging.INFO)
    caplog.clear()
    remaining = main._record_four_o_usage(
        "ask",
        "gpt-4o",
        {"prompt_tokens": 30, "completion_tokens": 10, "total_tokens": 40},
    )
    assert remaining == 10
    remaining = main._record_four_o_usage(
        "ask",
        "gpt-4o",
        {"prompt_tokens": 30, "completion_tokens": 20, "total_tokens": 50},
    )
    assert remaining == 0
    assert main._four_o_usage_state["used"] == 50
    assert main._four_o_usage_state["total"] == 90
    assert main._four_o_usage_state["models"]["gpt-4o"] == 90
    assert "remaining=0" in caplog.records[-1].message


@pytest.mark.asyncio
async def test_stats_reports_four_o_usage(load_main, tmp_path, monkeypatch):
    main = load_main(1000)
    db = main.Database(str(tmp_path / "db.sqlite"))
    await db.init()

    class DummyBot:
        def __init__(self):
            self.messages = []

        async def send_message(self, chat_id, text, **kwargs):
            self.messages.append((chat_id, text, kwargs))

    bot = DummyBot()

    async with db.get_session() as session:
        session.add(main.User(user_id=1, is_superadmin=True))
        await session.commit()

    today = date(2024, 5, 4)
    monkeypatch.setattr(main, "_current_utc_date", lambda: today)
    main._record_four_o_usage(
        "stats",
        "gpt-4o",
        {"prompt_tokens": 10, "completion_tokens": 0, "total_tokens": 10},
    )
    main._record_four_o_usage(
        "stats",
        "gpt-4o-mini",
        {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
    )

    msg = types.Message.model_validate(
        {
            "message_id": 1,
            "date": 0,
            "chat": {"id": 1, "type": "private"},
            "from": {"id": 1, "is_bot": False, "first_name": "Admin"},
            "text": "/stats",
        }
    )

    await main.handle_stats(msg, db, bot)

    assert bot.messages
    lines = bot.messages[-1][1].splitlines()
    assert lines[-2] == "Tokens gpt-4o: 10"
    assert lines[-1] == "Tokens gpt-4o-mini: 5"


@pytest.mark.asyncio
async def test_stats_reports_four_o_usage_with_input_output_tokens(
    load_main, tmp_path, monkeypatch
):
    main = load_main(1000)
    db = main.Database(str(tmp_path / "db.sqlite"))
    await db.init()

    class DummyBot:
        def __init__(self):
            self.messages = []

        async def send_message(self, chat_id, text, **kwargs):
            self.messages.append((chat_id, text, kwargs))

    bot = DummyBot()

    async with db.get_session() as session:
        session.add(main.User(user_id=1, is_superadmin=True))
        await session.commit()

    today = date(2024, 5, 5)
    monkeypatch.setattr(main, "_current_utc_date", lambda: today)
    main._record_four_o_usage(
        "stats",
        "gpt-4o",
        {"input_tokens": 12, "output_tokens": 8, "cache_creation_input_tokens": 3},
    )

    msg = types.Message.model_validate(
        {
            "message_id": 1,
            "date": 0,
            "chat": {"id": 1, "type": "private"},
            "from": {"id": 1, "is_bot": False, "first_name": "Admin"},
            "text": "/stats",
        }
    )

    await main.handle_stats(msg, db, bot)

    assert bot.messages
    lines = bot.messages[-1][1].splitlines()
    assert any(line == "Tokens gpt-4o: 23" for line in lines)
