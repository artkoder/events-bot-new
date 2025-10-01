import asyncio
import logging
from pathlib import Path

import pytest

import main
from main import Database


@pytest.mark.asyncio
async def test_add_events_from_text_logs_timeout(tmp_path: Path, monkeypatch, caplog):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    async def fake_parse(*args, **kwargs):
        exc = asyncio.TimeoutError("timed out")
        setattr(
            exc,
            "_four_o_call_meta",
            {
                "elapsed": 5.0,
                "semaphore_acquired": False,
                "semaphore_wait": None,
            },
        )
        raise exc

    monkeypatch.setattr(main, "parse_event_via_4o", fake_parse)

    with caplog.at_level(logging.ERROR):
        results = await main.add_events_from_text(db, "text", None, None, None)

    assert results == []
    error_logs = [rec for rec in caplog.records if "LLM error" in rec.getMessage()]
    assert error_logs, "Expected LLM error log to be emitted"
    record = error_logs[0]
    message = record.getMessage()
    assert "TimeoutError" in message
    assert "total_elapsed=" in message
    assert "call_elapsed=5.00s" in message
    assert "semaphore_acquired=False" in message
    assert "semaphore_wait=None" in message
    assert record.exc_info is not None
    assert record.exc_info[0] is asyncio.TimeoutError


@pytest.mark.asyncio
async def test_add_events_from_text_logs_other_exception(
    tmp_path: Path, monkeypatch, caplog
):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    async def fake_parse(*args, **kwargs):
        exc = ValueError("boom")
        setattr(
            exc,
            "_four_o_call_meta",
            {
                "elapsed": 0.25,
                "semaphore_acquired": True,
                "semaphore_wait": 0.5,
            },
        )
        raise exc

    monkeypatch.setattr(main, "parse_event_via_4o", fake_parse)

    with caplog.at_level(logging.ERROR):
        results = await main.add_events_from_text(db, "text", None, None, None)

    assert results == []
    error_logs = [rec for rec in caplog.records if "LLM error" in rec.getMessage()]
    assert error_logs, "Expected LLM error log to be emitted"
    record = error_logs[0]
    message = record.getMessage()
    assert "ValueError" in message
    assert "call_elapsed=0.25s" in message
    assert "semaphore_acquired=True" in message
    assert "semaphore_wait=0.50s" in message
    assert record.exc_info is not None
    assert record.exc_info[0] is ValueError
