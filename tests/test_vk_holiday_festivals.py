from pathlib import Path

import pytest
from sqlalchemy import select

import main
import smart_event_update as su
import vk_intake
from db import Database
from models import Event, Festival


@pytest.mark.asyncio
async def test_smart_update_creates_and_reuses_holiday(tmp_path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    monkeypatch.setenv("REGION_FILTER_ENABLED", "0")
    monkeypatch.setattr(su, "SMART_UPDATE_LLM_DISABLED", True)

    # Avoid Telegraph/index side effects during unit tests.
    async def fake_rebuild(*_args, **_kwargs):
        return False

    monkeypatch.setattr(main, "rebuild_fest_nav_if_changed", fake_rebuild)

    # reset holiday caches to ensure tests reflect docs
    main._read_holidays.cache_clear()
    main._holiday_record_map.cache_clear()
    main.settings_cache.clear()

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

        aliases_from_doc = [alias.strip() for alias in alias_field.split(",") if alias.strip()]

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

    candidate1 = su.EventCandidate(
        source_type="telegram",
        source_url="test://holiday/halloween/1",
        source_text="Spooky",
        raw_excerpt="Spooky",
        title="Хэллоуинская вечеринка",
        date="2025-10-30",
        time="22:00",
        location_name="Калининград",
        city="Калининград",
        festival="Halloween",
        posters=[su.PosterCandidate(catbox_url=photo_url)],
    )

    result1 = await su.smart_event_update(
        db,
        candidate1,
        check_source_url=False,
        schedule_tasks=False,
    )
    assert result1.status == "created"

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

    halloween_record = main.get_holiday_record("Хеллоуин")
    assert halloween_record is not None
    assert halloween_record.date == target_date_token
    assert halloween_record.tolerance_days == tolerance_days
    assert halloween_record.description == halloween_desc
    assert list(halloween_record.aliases) == ["хэллоуин", "halloween"]

    start_iso, end_iso = vk_intake._holiday_date_range(halloween_record, 2025)
    assert start_iso == "2025-10-31"
    assert end_iso == "2025-10-31"
    assert halloween.start_date == start_iso
    assert halloween.end_date == end_iso

    candidate2 = su.EventCandidate(
        source_type="telegram",
        source_url="test://holiday/halloween/2",
        source_text="Spooky again",
        raw_excerpt="Spooky again",
        title="Вечеринка 2",
        date="2025-10-31",
        time="19:00",
        location_name="Калининград",
        city="Калининград",
        festival="хэллоуин",
    )

    result2 = await su.smart_event_update(
        db,
        candidate2,
        check_source_url=False,
        schedule_tasks=False,
    )
    assert result2.status == "created"

    async with db.get_session() as session:
        events = (await session.execute(select(Event))).scalars().all()
        festivals = (await session.execute(select(Festival))).scalars().all()

    assert result2.event_id != result1.event_id
    assert len(events) == 2
    assert all(event.festival == "Хеллоуин" for event in events)
    assert len(festivals) == 1
    assert festivals[0].description == halloween_desc
    assert festivals[0].aliases == ["хеллоуин", "хэллоуин", "halloween"]

