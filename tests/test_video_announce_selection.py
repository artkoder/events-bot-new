from datetime import date, timezone

import pytest

from db import Database
from models import Event
import main
from video_announce import selection
from video_announce.types import SelectionContext


@pytest.mark.asyncio
async def test_fetch_candidates_includes_fair_and_schedule_text(tmp_path):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    async with db.get_session() as session:
        fair = Event(
            title="Fair",
            description="d",
            source_text="s",
            date="2025-12-25",
            end_date="2026-01-10",
            time="10:00..17:30",
            location_name="Market",
            event_type="ярмарка",
            photo_urls=["http://example.com/a.jpg"],
            photo_count=1,
        )
        session.add(fair)
        await session.commit()
        await session.refresh(fair)
        fair_id = fair.id

    ctx = SelectionContext(
        tz=timezone.utc,
        target_date=date(2026, 1, 3),
    )
    events, schedule_map, _ = await selection.fetch_candidates(db, ctx)
    assert any(e.id == fair_id for e in events)
    expected = f"по {main.format_day_pretty(date(2026, 1, 10))} с 10:00 до 17:30"
    assert schedule_map[fair_id] == expected
