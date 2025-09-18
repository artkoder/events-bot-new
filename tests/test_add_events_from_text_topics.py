from datetime import date, timedelta
from pathlib import Path

import pytest

import main
from main import Database, Event


FUTURE_DATE = (date.today() + timedelta(days=7)).isoformat()


@pytest.mark.asyncio
async def test_add_events_from_text_assigns_topics(tmp_path: Path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    async def fake_parse(text: str, source_channel: str | None = None):
        return [
            {
                "title": "Concert",
                "short_description": "Great music night",
                "date": FUTURE_DATE,
                "time": "19:00",
                "location_name": "Hall",
                "city": "Калининград",
            }
        ]

    async def fake_schedule_event_update_tasks(db_obj, event_obj, drain_nav=True):
        return {}

    async def fake_create_source_page(*args, **kwargs):
        return "https://t.me/test", "path", "", 0

    async def fake_classify(event: Event):
        return ["CONCERTS", "PARTIES"]

    monkeypatch.setattr(main, "parse_event_via_4o", fake_parse)
    monkeypatch.setattr(main, "schedule_event_update_tasks", fake_schedule_event_update_tasks)
    monkeypatch.setattr(main, "create_source_page", fake_create_source_page)
    monkeypatch.setattr(main, "classify_event_topics", fake_classify)

    results = await main.add_events_from_text(db, "t", None, None, None)
    saved_events = [item[0] for item in results if isinstance(item[0], Event)]
    assert saved_events

    async with db.get_session() as session:
        stored = await session.get(Event, saved_events[0].id)

    assert stored.topics == ["CONCERTS", "PARTIES"]
    assert stored.topics_manual is False


@pytest.mark.asyncio
async def test_add_events_from_text_multiday_inherits_topics(tmp_path: Path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    start_day = date.today() + timedelta(days=5)
    end_day = start_day + timedelta(days=1)
    calls = {"classify": 0}

    async def fake_parse(text: str, source_channel: str | None = None):
        return [
            {
                "title": "Festival",
                "short_description": "Two days",
                "date": start_day.isoformat(),
                "end_date": end_day.isoformat(),
                "time": "12:00",
                "location_name": "Park",
                "event_type": "концерт",
                "city": "Калининград",
            }
        ]

    async def fake_classify(event: Event):
        calls["classify"] += 1
        return ["PARTIES"]

    async def fake_schedule_event_update_tasks(db_obj, event_obj, drain_nav=True):
        return {}

    async def fake_create_source_page(*args, **kwargs):
        return "https://t.me/test", "path", "", 0

    monkeypatch.setattr(main, "parse_event_via_4o", fake_parse)
    monkeypatch.setattr(main, "classify_event_topics", fake_classify)
    monkeypatch.setattr(main, "schedule_event_update_tasks", fake_schedule_event_update_tasks)
    monkeypatch.setattr(main, "create_source_page", fake_create_source_page)

    results = await main.add_events_from_text(db, "t", None, None, None)
    event_ids = [saved.id for saved, *_ in results if isinstance(saved, Event)]
    assert event_ids

    async with db.get_session() as session:
        stored_events = [await session.get(Event, eid) for eid in event_ids]

    assert calls["classify"] == 1
    assert len(stored_events) == 2
    dates = sorted(ev.date for ev in stored_events)
    assert dates == [
        start_day.isoformat(),
        (start_day + timedelta(days=1)).isoformat(),
    ]
    for ev in stored_events:
        assert ev.topics == ["PARTIES"]
        assert ev.topics_manual is False
