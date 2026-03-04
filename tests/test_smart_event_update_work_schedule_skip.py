from __future__ import annotations

import pytest
from sqlalchemy import select

from db import Database
from models import Event
from smart_event_update import EventCandidate, smart_event_update


@pytest.mark.asyncio
async def test_smart_update_skips_work_schedule_notice_from_vk(tmp_path) -> None:
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    candidate = EventCandidate(
        source_type="vk",
        source_url="https://vk.com/wall-212931419_2224",
        source_text=(
            "Музей Курортной Моды публикует график работы в праздничные дни.\n"
            "21.02 музей работает с 11:00 до 19:00.\n"
            "07.03 музей работает с 11:00 до 20:00.\n"
            "8 марта — выходной день."
        ),
        raw_excerpt="График работы музея в праздничные дни.",
        title="Музей Курортной Моды: праздничные дни",
        date="2026-02-21",
        time="11:00",
        location_name="Музей Курортной Моды",
        city="Зеленоградск",
        trust_level="medium",
    )

    result = await smart_event_update(
        db,
        candidate,
        check_source_url=True,
        schedule_tasks=False,
    )
    assert result.status == "skipped_non_event"
    assert result.reason == "work_schedule"

    async with db.get_session() as session:
        rows = (await session.execute(select(Event))).scalars().all()
        assert rows == []
