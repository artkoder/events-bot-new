from datetime import date, timezone

import pytest

from db import Database
from models import Event, EventPoster
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


@pytest.mark.asyncio
async def test_build_selection_random_order_requires_poster_ocr(monkeypatch, tmp_path):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    async def _boom(*args, **kwargs):
        raise AssertionError("ask_4o should not be called for random_order selection")

    monkeypatch.setattr(selection, "ask_4o", _boom)

    async with db.get_session() as session:
        ev_with_title = Event(
            title="One",
            description="d",
            source_text="s",
            date="2026-01-01",
            time="19:00",
            location_name="Loc",
            city="City",
            photo_urls=["https://example.com/1.jpg"],
            photo_count=1,
        )
        ev_without_title = Event(
            title="Two",
            description="d",
            source_text="s",
            date="2026-01-01",
            time="20:00",
            location_name="Loc",
            city="City",
            photo_urls=["https://example.com/2.jpg"],
            photo_count=1,
        )
        ev_no_ocr = Event(
            title="Three",
            description="d",
            source_text="s",
            date="2026-01-01",
            time="21:00",
            location_name="Loc",
            city="City",
            photo_urls=["https://example.com/3.jpg"],
            photo_count=1,
        )
        session.add(ev_with_title)
        session.add(ev_without_title)
        session.add(ev_no_ocr)
        await session.commit()
        await session.refresh(ev_with_title)
        await session.refresh(ev_without_title)
        await session.refresh(ev_no_ocr)

        session.add(
            EventPoster(
                event_id=ev_with_title.id,
                poster_hash="h1",
                ocr_text="TEXT",
                ocr_title="OCR TITLE",
            )
        )
        session.add(
            EventPoster(
                event_id=ev_without_title.id,
                poster_hash="h2",
                ocr_text="TEXT2",
                ocr_title=None,
            )
        )
        await session.commit()

    ctx = SelectionContext(
        tz=timezone.utc,
        target_date=date(2026, 1, 1),
        random_order=True,
        candidate_limit=80,
        default_selected_min=1,
        default_selected_max=12,
    )
    result = await selection.build_selection(
        db,
        ctx,
        candidates=[ev_with_title, ev_without_title, ev_no_ocr],
    )

    candidate_ids = {ev.id for ev in result.candidates}
    assert candidate_ids == {ev_with_title.id, ev_without_title.id}
    assert result.default_ready_ids == {ev_with_title.id, ev_without_title.id}

    ranked_first = result.ranked[0]
    assert ranked_first.event.id == ev_with_title.id
    assert ranked_first.poster_ocr_text is not None
    assert ranked_first.poster_ocr_title == "OCR TITLE"
    assert ranked_first.about
