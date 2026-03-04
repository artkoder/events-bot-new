from datetime import date, datetime, timezone, timedelta
import re

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

    assert "<a href=" not in rendered


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


def test_format_event_daily_uses_telegraph_path_when_url_missing() -> None:
    event = make_event(
        telegraph_url=None,
        telegraph_path="Some-Event-02-20",
        source_post_url="https://t.me/somechannel/123",
    )

    rendered = main.format_event_daily(event)

    assert '<a href="https://telegra.ph/Some-Event-02-20">' in rendered


def test_format_event_daily_inline_links_to_telegraph_only() -> None:
    event = make_event(
        telegraph_url="https://telegra.ph/test",
        source_post_url="https://t.me/somechannel/123",
        date="2024-01-02",
    )

    rendered = main.format_event_daily_inline(event)

    assert '<a href="https://telegra.ph/test">' in rendered
    assert "t.me" not in rendered


def test_format_event_daily_handles_timezone_aware_added_at() -> None:
    event = make_event(
        added_at=datetime(2024, 1, 2, 12, tzinfo=timezone.utc),
    )

    rendered = main.format_event_daily(event)

    assert isinstance(rendered, str)


def test_format_event_daily_prefers_search_digest_over_full_description() -> None:
    event = make_event(
        search_digest="Короткое описание в 1 предложение.",
        description="### О встрече\nОчень длинное описание.\n\n### Формат\nЕщё текст.",
    )

    rendered = main.format_event_daily(event)

    assert "Короткое описание в 1 предложение." in rendered
    assert "### О встрече" not in rendered


def test_format_event_daily_falls_back_to_first_paragraph_when_no_search_digest() -> None:
    event = make_event(
        search_digest=None,
        description="### О встрече\nЭто лекция про мхи и их роль в природе.\n\n### Детали\nРегистрация по ссылке.",
    )

    rendered = main.format_event_daily(event)

    assert "Это лекция про мхи и их роль в природе." in rendered
    assert "### О встрече" not in rendered


def test_format_event_daily_limits_digest_to_16_words() -> None:
    long_digest = (
        "В калининградском пространстве Терка пройдет выставка Набросочная с работами местных художников,"
        " открытой сессией рисунка и лекцией о пластике человеческого тела."
    )
    event = make_event(search_digest=long_digest, description="Полное описание события.")

    rendered = main.format_event_daily(event)
    lines = rendered.splitlines()
    digest_line = re.sub(r"<[^>]+>", " ", lines[1] if len(lines) > 1 else "")
    words = re.findall(r"[0-9A-Za-zА-Яа-яЁё]+", digest_line)
    assert len(words) <= 16


def test_format_event_md_limits_digest_to_16_words() -> None:
    long_digest = " ".join(f"слово{i}" for i in range(1, 22))
    event = make_event(search_digest=long_digest, description="Полное описание события.")

    rendered = main.format_event_md(event)
    digest_line = rendered.splitlines()[1]
    words = re.findall(r"[0-9A-Za-zА-Яа-яЁё]+", digest_line)
    assert len(words) <= 16


def test_format_event_md_does_not_emit_trailing_ellipsis_for_short_description() -> None:
    long_short = (
        'Концертная программа "Песни русской рати" расскажет о ратных подвигах наших солдат. '
        "В программе концерта будут представлены рекрутские баллады и народные песни."
    )
    event = make_event(
        short_description=long_short,
        search_digest=None,
        description="Полное описание события.",
    )

    rendered = main.format_event_md(event)
    digest_line = rendered.splitlines()[1]
    assert not digest_line.endswith("…")
    assert not digest_line.endswith("...")
    assert digest_line.endswith(".")


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
    assert 'href="https://telegra.ph/Fest"' in text
    assert "✨ Fest" in text


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
