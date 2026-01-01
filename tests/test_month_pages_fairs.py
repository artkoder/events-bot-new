from datetime import date

import pytest

from db import Database
import main


@pytest.mark.asyncio
async def test_get_month_data_excludes_fairs(tmp_path):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    async with db.get_session() as session:
        session.add(
            main.Event(
                title="Fair",
                description="d",
                source_text="s",
                date="2026-01-03",
                end_date="2026-01-10",
                time="10:00",
                location_name="Market",
                event_type="ярмарка",
            )
        )
        session.add(
            main.Event(
                title="Concert",
                description="d",
                source_text="s",
                date="2026-01-05",
                time="19:00",
                location_name="Hall",
            )
        )
        await session.commit()

    events, _ = await main.get_month_data(db, "2026-01", fallback=False)
    assert any(e.title == "Concert" for e in events)
    assert all((e.event_type or "").casefold() != "ярмарка" for e in events)
