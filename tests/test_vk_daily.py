from datetime import date, datetime, timezone
from pathlib import Path

import pytest

import main
from main import (
    Database,
    WeekendPage,
    WeekPage,
    Event,
    format_weekend_range,
    month_name_nominative,
)


@pytest.mark.asyncio
async def test_format_event_vk_no_ticket_link_with_vk_source():
    e = Event(
        title="T",
        description="d",
        source_text="s",
        date="2025-07-07",
        time="10:00",
        location_name="Club",
        ticket_link="https://example.com",
        source_vk_post_url="https://vk.com/wall-1_1",
    )
    msg = main.format_event_vk(e)
    assert "https://example.com" not in msg


@pytest.mark.asyncio
async def test_build_daily_sections_vk_links(tmp_path: Path):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    today = date(2025, 7, 10)
    w_start = date(2025, 7, 12)

    async with db.get_session() as session:
        session.add(
            WeekendPage(
                start=w_start.isoformat(),
                url="u1",
                path="p1",
                vk_post_url="https://vk.com/wall-1_2",
            )
        )
        session.add(WeekPage(start=date(2025, 7, 7).isoformat(), vk_post_url="https://vk.com/wall-1_3"))
        session.add(WeekPage(start=date(2025, 8, 4).isoformat(), vk_post_url="https://vk.com/wall-1_4"))
        session.add(
            Event(
                title="Party",
                description="d",
                source_text="s",
                date=today.isoformat(),
                time="10:00",
                location_name="Club",
                source_vk_post_url="https://vk.com/wall-1_1",
            )
        )
        await session.commit()

    sec1, _ = await main.build_daily_sections_vk(
        db, timezone.utc, now=datetime(2025, 7, 10, tzinfo=timezone.utc)
    )
    label_weekend = f"выходные {format_weekend_range(w_start)}"
    assert f"[https://vk.com/wall-1_2|{label_weekend}]" in sec1
    assert (
        f"[https://vk.com/wall-1_3|{month_name_nominative('2025-07')}]" in sec1
    )
    assert (
        f"[https://vk.com/wall-1_4|{month_name_nominative('2025-08')}]" in sec1
    )
    assert "u1" not in sec1
