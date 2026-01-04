from datetime import date, datetime, timezone, timedelta
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


@pytest.mark.asyncio
async def test_build_daily_sections_vk_prefers_repost_for_non_partner(tmp_path: Path):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    today = date(2025, 7, 10)

    async with db.get_session() as session:
        session.add(
            Event(
                title="Party",
                description="d",
                source_text="s",
                date=today.isoformat(),
                time="10:00",
                location_name="Club",
                creator_id=404,
                vk_repost_url="https://vk.com/wall-1_7",
                added_at=datetime.now(timezone.utc) - timedelta(days=2),
            )
        )
        await session.commit()

    sec1, _ = await main.build_daily_sections_vk(
        db, timezone.utc, now=datetime(2025, 7, 10, tzinfo=timezone.utc)
    )
    event_line = sec1.splitlines()[4]
    assert event_line.startswith("👉 [https://vk.com/wall-1_7|")


@pytest.mark.asyncio
async def test_build_daily_sections_vk_keeps_partner_source_link(tmp_path: Path):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    today = date(2025, 7, 10)

    async with db.get_session() as session:
        session.add(
            main.User(user_id=505, is_partner=True),
        )
        session.add(
            Event(
                title="Party",
                description="d",
                source_text="s",
                date=today.isoformat(),
                time="10:00",
                location_name="Club",
                creator_id=505,
                source_post_url="https://vk.com/wall-1_8",
                vk_repost_url="https://vk.com/wall-1_9",
                added_at=datetime.now(timezone.utc) - timedelta(days=2),
            )
        )
        await session.commit()

    sec1, _ = await main.build_daily_sections_vk(
        db, timezone.utc, now=datetime(2025, 7, 10, tzinfo=timezone.utc)
    )
    event_line = sec1.splitlines()[4]
    assert event_line.startswith("👉 [https://vk.com/wall-1_8|")


@pytest.mark.asyncio
async def test_build_daily_sections_vk_includes_fair_when_few_events(tmp_path: Path):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    now = datetime(2026, 1, 3, 12, 0, tzinfo=timezone.utc)
    async with db.get_session() as session:
        session.add(
            Event(
                title="Fair",
                description="d",
                source_text="s",
                date="2025-12-25",
                end_date="2026-01-10",
                time="10:00..17:30",
                location_name="Market",
                event_type="ярмарка",
            )
        )
        await session.commit()

    sec1, _ = await main.build_daily_sections_vk(
        db, timezone.utc, now=now
    )
    assert "FAIR" in sec1
    assert main.format_day_pretty(date(2026, 1, 3)) in sec1


@pytest.mark.asyncio
async def test_build_daily_sections_vk_short_link_reuse(tmp_path: Path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    today = date(2025, 7, 10)

    call_ids: list[int] = []

    async def fake_helper(event, db, **kwargs):
        if event.id is not None:
            call_ids.append(event.id)
        event.vk_ticket_short_url = "https://vk.cc/short"
        event.vk_ticket_short_key = "short"
        if event.id is not None:
            async with db.get_session() as session:
                stored = await session.get(Event, event.id)
                if stored:
                    stored.vk_ticket_short_url = "https://vk.cc/short"
                    stored.vk_ticket_short_key = "short"
                    await session.commit()
        return "https://vk.cc/short", "short"

    monkeypatch.setattr(main, "ensure_vk_short_ticket_link", fake_helper)

    async with db.get_session() as session:
        event = Event(
            title="Concert",
            description="d",
            source_text="s",
            date=today.isoformat(),
            time="19:00",
            location_name="Club",
            ticket_link="https://tickets",
            added_at=datetime.now(timezone.utc) - timedelta(hours=1),
        )
        session.add(event)
        await session.commit()
        await session.refresh(event)
        event_id = event.id

    sec1, _ = await main.build_daily_sections_vk(
        db, timezone.utc, now=datetime(2025, 7, 10, tzinfo=timezone.utc)
    )

    async with db.get_session() as session:
        stored = await session.get(Event, event_id)
        assert stored.vk_ticket_short_url == "https://vk.cc/short", call_ids
        assert stored.vk_ticket_short_key == "short"

    assert "vk.cc/short" in sec1
    assert call_ids == [event_id]

    async def fail_helper(*args, **kwargs):  # pragma: no cover - ensure reuse
        raise AssertionError("short link should be reused")

    monkeypatch.setattr(main, "ensure_vk_short_ticket_link", fail_helper)

    sec1_again, _ = await main.build_daily_sections_vk(
        db, timezone.utc, now=datetime(2025, 7, 10, tzinfo=timezone.utc)
    )
    assert "vk.cc/short" in sec1_again


def test_build_vk_source_header_uses_short_ticket_link():
    event = main.Event(
        title="Concert",
        description="desc",
        source_text="src",
        date="2025-07-07",
        time="19:00",
        location_name="Club",
        ticket_link="https://tickets",
        is_free=True,
    )
    event.vk_ticket_short_url = "https://vk.cc/short"

    lines = main.build_vk_source_header(event)

    registration_line = next(
        line for line in lines if "по регистрации" in line
    )

    assert "vk.cc/short" in registration_line
    assert "https://vk.cc/short" not in registration_line


@pytest.mark.asyncio
async def test_sync_vk_source_post_updates_short_link(monkeypatch):
    event = main.Event(
        id=123,
        title="Concert",
        description="desc",
        source_text="src",
        date="2025-07-07",
        time="19:00",
        location_name="Club",
        ticket_link="https://tickets",
        source_vk_post_url="https://vk.com/wall-1_1",
    )

    calls: list[int | None] = []

    async def fake_helper(ev, db, *, bot=None, vk_api_fn=None):
        calls.append(ev.id)
        ev.vk_ticket_short_url = "https://vk.cc/fake"
        ev.vk_ticket_short_key = "fake"
        return "https://vk.cc/fake", "fake"

    async def fake_vk_api(method: str, **params):
        assert method == "wall.getById"
        return {"response": [{"text": ""}]}

    captured: dict[str, str] = {}

    async def fake_edit(post_url, message, db=None, bot=None, attachments=None):
        captured["post_url"] = post_url
        captured["message"] = message
        return True

    monkeypatch.setattr(main, "VK_AFISHA_GROUP_ID", "-1")
    monkeypatch.setattr(main, "ensure_vk_short_ticket_link", fake_helper)
    monkeypatch.setattr(main, "vk_api", fake_vk_api)
    monkeypatch.setattr(main, "edit_vk_post", fake_edit)

    url = await main.sync_vk_source_post(event, event.source_text, db=None, bot=None)

    assert url == event.source_vk_post_url
    assert calls == [event.id]
    assert event.vk_ticket_short_url == "https://vk.cc/fake"
    assert "message" in captured
    assert "vk.cc/fake" in captured["message"]
    assert "https://vk.cc/fake" not in captured["message"]
