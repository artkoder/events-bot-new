from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import main
from main import Database, Event, JobTask
import source_parsing.handlers as handlers
from source_parsing.parser import TheatreEvent


@pytest.mark.asyncio
async def test_process_parsing_files_enqueues_month_and_weekend_deferred(
    tmp_path, monkeypatch
):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    async with db.get_session() as session:
        ev = Event(
            title="Weekend show",
            description="desc",
            date="2025-09-06",
            time="19:00",
            location_name="Hall",
            source_text="src",
        )
        session.add(ev)
        await session.commit()

    fake_event = TheatreEvent(
        title="Weekend show",
        date_raw="06.09.2025",
        ticket_status="available",
        url="https://example.com/event",
        source_type="dramteatr",
        parsed_date="2025-09-06",
        parsed_time="19:00",
    )

    monkeypatch.setattr(
        handlers,
        "parse_theatre_json",
        lambda _raw, _source: [fake_event],
    )

    async def _fake_process_source_events(*_args, **_kwargs):
        return handlers.SourceParsingStats(source="dramteatr"), None

    monkeypatch.setattr(handlers, "process_source_events", _fake_process_source_events)

    enqueued: list[dict] = []
    dirty_marks: list[str] = []

    async def _fake_enqueue_job(
        _db,
        event_id,
        task,
        payload=None,
        *,
        coalesce_key=None,
        depends_on=None,
        next_run_at=None,
    ):
        enqueued.append(
            {
                "event_id": event_id,
                "task": task,
                "coalesce_key": coalesce_key,
                "next_run_at": next_run_at,
            }
        )
        return "new"

    async def _fake_mark_pages_dirty(_db, key):
        dirty_marks.append(str(key))

    monkeypatch.setattr(main, "enqueue_job", _fake_enqueue_job)
    monkeypatch.setattr(main, "mark_pages_dirty", _fake_mark_pages_dirty)

    theatre_file = Path(tmp_path / "dramteatr.json")
    theatre_file.write_text("{}", encoding="utf-8")

    result = handlers.SourceParsingResult()
    now_before = datetime.now(timezone.utc)
    await handlers._process_parsing_files(
        db,
        bot=None,
        chat_id=None,
        theatre_files=[str(theatre_file)],
        phil_files=[],
        qtickets_files=[],
        result=result,
        only_sources=None,
        date_from=None,
        date_to=None,
    )
    now_after = datetime.now(timezone.utc)

    month_job = next(
        j
        for j in enqueued
        if j["task"] == JobTask.month_pages
        and j["coalesce_key"] == "month_pages:2025-09"
    )
    weekend_job = next(
        j
        for j in enqueued
        if j["task"] == JobTask.weekend_pages
        and j["coalesce_key"] == "weekend_pages:2025-09-06"
    )

    month_run = month_job["next_run_at"]
    weekend_run = weekend_job["next_run_at"]
    assert month_run is not None
    assert weekend_run is not None

    expected_low = now_before + timedelta(minutes=14)
    expected_high = now_after + timedelta(minutes=16)
    assert expected_low <= month_run <= expected_high
    assert expected_low <= weekend_run <= expected_high
    assert abs((month_run - weekend_run).total_seconds()) < 2

    assert "2025-09" in dirty_marks
    assert "weekend:2025-09-06" in dirty_marks
