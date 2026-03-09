from datetime import date, timedelta, timezone
import json
from types import SimpleNamespace

import pytest

from db import Database
from models import Event, EventPoster
import main
from video_announce import selection
from video_announce.custom_types import RenderPayload
from video_announce.custom_types import SelectionContext


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
async def test_fetch_candidates_skips_fair_with_inferred_end_date_before_target(tmp_path):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    async with db.get_session() as session:
        inferred_fair = Event(
            title="Weekend Fair",
            description="d",
            source_text="Каждую субботу ярмарка",
            date="2026-01-03",
            end_date="2026-02-03",
            end_date_is_inferred=True,
            time="08:00..15:00",
            location_name="Market",
            event_type="ярмарка",
            photo_urls=["http://example.com/a.jpg"],
            photo_count=1,
        )
        session.add(inferred_fair)
        await session.commit()
        await session.refresh(inferred_fair)
        fair_id = inferred_fair.id

    ctx = SelectionContext(
        tz=timezone.utc,
        target_date=date(2026, 1, 10),
    )
    events, schedule_map, _ = await selection.fetch_candidates(db, ctx)

    assert all(e.id != fair_id for e in events)
    assert fair_id not in schedule_map


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


@pytest.mark.asyncio
async def test_build_selection_auto_expands_for_min_posters(tmp_path):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    base_date = date(2026, 1, 1)
    offsets = [0, 0, 1, 1, 1, 2, 2]

    async with db.get_session() as session:
        events = []
        for idx, offset in enumerate(offsets, start=1):
            ev = Event(
                title=f"Event {idx}",
                description="d",
                source_text="s",
                date=(base_date + timedelta(days=offset)).isoformat(),
                time="19:00",
                location_name=f"Loc {idx}",
                city="City",
                photo_urls=[f"https://example.com/{idx}.jpg"],
                photo_count=1,
            )
            session.add(ev)
            events.append(ev)
        await session.commit()

        for idx, ev in enumerate(events, start=1):
            await session.refresh(ev)
            session.add(
                EventPoster(
                    event_id=ev.id,
                    poster_hash=f"h{idx}",
                    ocr_text="TEXT",
                    ocr_title="TITLE",
                )
            )
        await session.commit()

    ctx = SelectionContext(
        tz=timezone.utc,
        target_date=base_date,
        random_order=True,
        candidate_limit=80,
        default_selected_min=1,
        default_selected_max=8,
        primary_window_days=0,
        fallback_window_days=0,
    )
    result = await selection.build_selection(
        db,
        ctx,
        auto_expand_min_posters=7,
        auto_expand_step_days=1,
        auto_expand_max_days=2,
    )

    assert len(result.candidates) == 7
    max_date = max(date.fromisoformat(ev.date) for ev in result.candidates)
    assert max_date == base_date + timedelta(days=2)


def test_payload_as_json_clamps_ongoing_longrun_intro_to_target_date():
    fair = Event(
        id=1,
        title="Long Fair",
        description="d",
        source_text="s",
        date="2026-03-07",
        end_date="2026-04-07",
        time="10:00..18:00",
        location_name="Market",
        city="Калининград",
        event_type="ярмарка",
        photo_urls=["https://example.com/fair.jpg"],
        photo_count=1,
        is_free=True,
    )
    lecture = Event(
        id=2,
        title="Lecture",
        description="d",
        source_text="s",
        date="2026-03-12",
        time="19:00",
        location_name="Hall",
        city="Калининград",
        photo_urls=["https://example.com/lecture.jpg"],
        photo_count=1,
        is_free=False,
    )
    session = SimpleNamespace(
        selection_params={
            "target_date": "2026-03-10",
            "dedup_schedule": {
                "1": "по 7 апреля с 10:00 до 18:00",
            },
        }
    )
    items = [
        SimpleNamespace(
            session_id=1,
            event_id=1,
            position=1,
            final_about="About fair",
            final_description="",
            poster_text=None,
            final_title=None,
        ),
        SimpleNamespace(
            session_id=1,
            event_id=2,
            position=2,
            final_about="About lecture",
            final_description="",
            poster_text=None,
            final_title=None,
        ),
    ]
    payload = RenderPayload(session=session, items=items, events=[fair, lecture], scores={})

    data = json.loads(selection.payload_as_json(payload, timezone.utc))

    assert data["intro"]["date_start"] == "2026-03-10"
    assert data["intro"]["date_end"] == "2026-03-12"
    assert data["intro"]["date"] == "10 МАРТА - 12 МАРТА"
