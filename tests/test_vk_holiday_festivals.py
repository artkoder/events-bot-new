import asyncio
from datetime import date
from pathlib import Path

import pytest
from sqlalchemy import select

import main
import vk_intake
from db import Database
from models import Event, Festival


@pytest.mark.asyncio
async def test_persist_event_creates_and_reuses_holiday(tmp_path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    # reset holiday caches to ensure tests reflect docs
    main._read_holidays.cache_clear()
    main._holiday_record_map.cache_clear()

    async def fake_assign(event: Event):
        return [], len(event.description or ""), "", False

    scheduled: list[str] = []

    async def fake_schedule(db_obj, event_obj, drain_nav: bool = True, skip_vk_sync: bool = False):
        scheduled.append(event_obj.festival)
        return {}

    async def fake_rebuild(*_args, **_kwargs):
        return False

    sync_calls: list[str] = []

    async def fake_sync_page(db_obj, name: str):
        sync_calls.append(name)

    monkeypatch.setattr(main, "assign_event_topics", fake_assign)
    monkeypatch.setattr(main, "schedule_event_update_tasks", fake_schedule)
    monkeypatch.setattr(main, "rebuild_fest_nav_if_changed", fake_rebuild)
    monkeypatch.setattr(main, "sync_festival_page", fake_sync_page)

    halloween_desc = None
    for line in Path("docs/HOLIDAYS.md").read_text(encoding="utf-8").splitlines():
        if "|" not in line:
            continue
        parts = [part.strip() for part in line.split("|")]
        if len(parts) >= 5 and parts[2] == "Хеллоуин":
            halloween_desc = parts[4]
            break
    assert halloween_desc

    draft1 = vk_intake.EventDraft(
        title="Хэллоуинская вечеринка",
        date="2025-10-30",
        time="22:00",
        festival="Halloween",
        source_text="Spooky",
    )

    result1 = await vk_intake.persist_event_and_pages(draft1, [], db)
    await asyncio.sleep(0)

    async with db.get_session() as session:
        events = (await session.execute(select(Event))).scalars().all()
        festivals = (await session.execute(select(Festival))).scalars().all()

    assert len(events) == 1
    assert len(festivals) == 1

    saved_event = events[0]
    halloween = festivals[0]

    assert result1.event_id == saved_event.id
    assert saved_event.festival == "Хеллоуин"
    assert halloween.name == "Хеллоуин"
    assert halloween.description == halloween_desc
    assert halloween.source_text == halloween_desc
    assert halloween.aliases == ["хеллоуин", "хэллоуин", "halloween"]

    current_year = date.today().year
    expected_date = f"{current_year}-10-31"
    assert halloween.start_date == expected_date
    assert halloween.end_date == expected_date

    assert scheduled == ["Хеллоуин"]
    assert sync_calls == ["Хеллоуин"]

    draft2 = vk_intake.EventDraft(
        title="Вечеринка 2",
        date="2025-10-31",
        festival="хэллоуин",
        source_text="Spooky again",
    )

    result2 = await vk_intake.persist_event_and_pages(draft2, [], db)
    await asyncio.sleep(0)

    async with db.get_session() as session:
        events = (await session.execute(select(Event))).scalars().all()
        festivals = (await session.execute(select(Festival))).scalars().all()

    assert result2.event_id != result1.event_id
    assert len(events) == 2
    assert all(event.festival == "Хеллоуин" for event in events)
    assert len(festivals) == 1
    assert festivals[0].description == halloween_desc
    assert festivals[0].aliases == ["хеллоуин", "хэллоуин", "halloween"]

    assert scheduled == ["Хеллоуин", "Хеллоуин"]
    assert sync_calls == ["Хеллоуин"]
