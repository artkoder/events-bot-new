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
    main.settings_cache.clear()

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

    target_date_token = main._normalize_holiday_date_token("31.10")
    halloween_doc_row = None
    for raw_line in Path("docs/reference/holidays.md").read_text(encoding="utf-8").splitlines():
        if "|" not in raw_line:
            continue
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = [part.strip() for part in raw_line.split("|")]
        if not parts or parts[0].casefold() == "date_or_range":
            continue
        date_token = main._normalize_holiday_date_token(parts[0])
        if date_token != target_date_token:
            continue

        tolerance_token = parts[1] if len(parts) > 1 else ""
        canonical_name = parts[2] if len(parts) > 2 else ""
        alias_field = parts[3] if len(parts) > 3 else ""
        description_field = "|".join(parts[4:]).strip() if len(parts) > 4 else ""

        tolerance_value = tolerance_token.strip()
        if not tolerance_value:
            tolerance_days = None
        elif tolerance_value.casefold() in {"none", "null"}:
            tolerance_days = None
        else:
            tolerance_days = int(tolerance_value)

        aliases_from_doc = [
            alias.strip()
            for alias in alias_field.split(",")
            if alias.strip()
        ]

        halloween_doc_row = {
            "canonical_name": canonical_name,
            "description": description_field,
            "aliases": aliases_from_doc,
            "tolerance_days": tolerance_days,
        }
        break

    assert halloween_doc_row is not None
    assert halloween_doc_row["canonical_name"] == "Хеллоуин"
    assert halloween_doc_row["aliases"] == ["хэллоуин", "halloween"]
    tolerance_days = halloween_doc_row["tolerance_days"]
    assert tolerance_days is not None
    halloween_desc = halloween_doc_row["description"]

    photo_url = "https://example.com/halloween.jpg"

    draft1 = vk_intake.EventDraft(
        title="Хэллоуинская вечеринка",
        date="2025-10-30",
        time="22:00",
        festival="Halloween",
        source_text="Spooky",
    )

    result1 = await vk_intake.persist_event_and_pages(draft1, [photo_url], db)
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
    assert saved_event.photo_urls == [photo_url]
    assert halloween.photo_url == photo_url
    assert halloween.photo_urls == [photo_url]

    stored_html: dict[str, str] = {}

    class DummyTelegraph:
        def __init__(self, *args, **kwargs):
            pass

        def create_page(self, title, html_content, **_):
            stored_html["html"] = html_content
            return {"url": "https://telegra.ph/fests", "path": "fests"}

        def edit_page(self, path, title, html_content, **kwargs):
            stored_html["edited"] = html_content
            return {}

        def get_page(self, path, return_html=True):
            return {"content_html": stored_html.get("html", "")}

    async def fake_telegraph_call(func, *args, **kwargs):
        return func(*args, **kwargs)

    async def fake_create_page(tg, *args, **kwargs):
        return tg.create_page(*args, **kwargs)

    async def fake_edit_page(tg, *args, **kwargs):
        return tg.edit_page(*args, **kwargs)

    monkeypatch.setattr(main, "Telegraph", DummyTelegraph)
    monkeypatch.setattr(main, "telegraph_call", fake_telegraph_call)
    monkeypatch.setattr(main, "telegraph_create_page", fake_create_page)
    monkeypatch.setattr(main, "telegraph_edit_page", fake_edit_page)
    monkeypatch.setattr(main, "get_telegraph_token", lambda: "token")

    await main.sync_festivals_index_page(db)
    main.settings_cache.clear()

    html = stored_html["html"]
    assert photo_url in html
    assert "Хеллоуин" in html

    halloween_record = main.get_holiday_record("Хеллоуин")
    assert halloween_record is not None
    assert halloween_record.date == target_date_token
    assert halloween_record.tolerance_days == tolerance_days
    assert halloween_record.description == halloween_desc
    assert list(halloween_record.aliases) == ["хэллоуин", "halloween"]

    current_year = date.today().year
    start_iso, end_iso = vk_intake._holiday_date_range(halloween_record, current_year)
    assert start_iso == f"{current_year}-10-31"
    assert end_iso == f"{current_year}-10-31"

    assert halloween.start_date == start_iso
    assert halloween.end_date == end_iso

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
