import pytest
from sqlalchemy import select

from db import Database
from models import Event
from smart_event_update import EventCandidate, smart_event_update


@pytest.mark.asyncio
async def test_smart_event_update_skips_past_event_candidates(tmp_path, monkeypatch):
    monkeypatch.setenv("SMART_UPDATE_SKIP_PAST_EVENTS", "1")

    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    candidate = EventCandidate(
        source_type="vk",
        source_url="https://vk.com/wall-1_1",
        source_text="Анонс события.",
        raw_excerpt="",
        title="Событие из прошлого",
        date="2000-01-01",
        time="19:00",
        location_name="Место",
        city="Калининград",
    )

    res = await smart_event_update(db, candidate, check_source_url=True, schedule_tasks=False)
    assert res.status == "skipped_past_event"
    assert res.reason == "past_event"
    assert res.event_id is None

    async with db.get_session() as session:
        saved = (await session.execute(select(Event))).scalars().all()
        assert saved == []
