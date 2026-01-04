from datetime import date, datetime, timezone, timedelta

import pytest

from db import Database

import main


def make_event(**kwargs: object) -> main.Event:
    base = {
        "title": "Event",
        "description": "Описание",
        "source_text": "source",
        "date": date(2024, 1, 1).isoformat(),
        "time": "18:00",
        "location_name": "Place",
    }
    base.update(kwargs)
    return main.Event(**base)


def test_format_event_daily_uses_partner_vk_link() -> None:
    event = make_event(
        source_post_url="https://vk.com/wall-1_1",
        creator_id=123,
    )

    rendered = main.format_event_daily(event, partner_creator_ids={123})

    assert '<a href="https://vk.com/wall-1_1">' in rendered


def test_format_event_daily_prefers_telegraph_for_vk_queue() -> None:
    event = make_event(
        source_vk_post_url="https://vk.com/wall-1_2",
        telegraph_url="https://telegra.ph/test",
    )

    rendered = main.format_event_daily(event)

    assert '<a href="https://telegra.ph/test">' in rendered


def test_format_event_daily_prefers_telegraph_for_vk_source_url() -> None:
    event = make_event(
        source_post_url="https://vk.com/wall-1_3",
        telegraph_url="https://telegra.ph/source",
    )

    rendered = main.format_event_daily(event)

    assert '<a href="https://telegra.ph/source">' in rendered


def test_format_event_daily_handles_timezone_aware_added_at() -> None:
    event = make_event(
        added_at=datetime(2024, 1, 2, 12, tzinfo=timezone.utc),
    )

    rendered = main.format_event_daily(event)

    assert isinstance(rendered, str)


@pytest.mark.asyncio
async def test_build_daily_posts_lists_recent_festivals(tmp_path):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    now = datetime(2025, 7, 15, 12, 0, tzinfo=timezone.utc)

    async with db.get_session() as session:
        session.add(
            main.Festival(
                name="Fest",
                telegraph_path="Fest",
                created_at=now,
            )
        )
        session.add(
            main.Event(
                title="New Event",
                description="Desc",
                source_text="source",
                date=(now.date() + timedelta(days=1)).isoformat(),
                time="18:00",
                location_name="Place",
                added_at=now,
            )
        )
        await session.commit()

    posts = await main.build_daily_posts(db, timezone.utc, now)
    text = posts[0][0]

    assert "ФЕСТИВАЛИ" in text
    assert "Fest-https://telegra.ph/Fest" in text


@pytest.mark.asyncio
async def test_build_daily_posts_includes_fair_when_few_events(tmp_path):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    now = datetime(2026, 1, 3, 12, 0, tzinfo=timezone.utc)

    async with db.get_session() as session:
        session.add(
            main.Event(
                title="Fair",
                description="Desc",
                source_text="source",
                date="2025-12-25",
                end_date="2026-01-10",
                time="10:00..17:30",
                location_name="Market",
                event_type="ярмарка",
            )
        )
        await session.commit()

    posts = await main.build_daily_posts(db, timezone.utc, now)
    combined = "\n".join(p[0] for p in posts)
    assert "Fair" in combined
    assert main.format_day_pretty(now.date()) in combined
