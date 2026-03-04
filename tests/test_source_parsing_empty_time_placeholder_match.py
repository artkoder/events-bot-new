import pytest

from db import Database
from models import Event
from source_parsing.parser import find_existing_event


@pytest.mark.asyncio
async def test_find_existing_event_matches_empty_time_placeholder(tmp_path) -> None:
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    async with db.get_session() as session:
        ev = Event(
            title="Гараж",
            description="desc",
            date="2026-04-04",
            time="",  # unknown time placeholder from text sources
            location_name="Драматический театр",
            city="Калининград",
            source_text="src",
        )
        session.add(ev)
        await session.commit()
        await session.refresh(ev)

    event_id, needs_full_update = await find_existing_event(
        db,
        location_name="Драматический театр",
        event_date="2026-04-04",
        event_time="18:00",
        title="Гараж",
    )
    assert event_id == ev.id
    assert needs_full_update is True

