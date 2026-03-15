from datetime import date

import pytest

from db import Database
from handlers import popular_posts_cmd as popular
from handlers.popular_posts_cmd import _load_top_items
from models import Event, EventSource


@pytest.mark.asyncio
async def test_popular_posts_skips_vk_on_minimal_db(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_INIT_MINIMAL", "1")
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    try:
        items, dbg = await _load_top_items(db, window_days=3, age_day=2, limit=10)
        assert items == []
        assert dbg.get("tg_available") is True
        assert dbg.get("vk_available") is False
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_popular_posts_report_escapes_debug_counter_html(monkeypatch):
    class _Message:
        def __init__(self) -> None:
            self.chunks: list[str] = []

        async def answer(self, text, **kwargs):  # noqa: ANN003
            self.chunks.append(str(text))

    async def _fake_load_top_items(_db, *, window_days, age_day, limit):  # noqa: ANN001
        return (
            [],
            {
                "tg_available": True,
                "vk_available": True,
                "tg_total": 0,
                "vk_total": 0,
                "skipped_small_sample": 0,
                "skipped_missing_median": 0,
                "skipped_missing_metrics": 0,
                "skipped_not_above_median": 3,
            },
        )

    async def _fake_resolve_telegraph_map(_db, *, source_urls):  # noqa: ANN001
        return {}, set()

    monkeypatch.setattr(popular, "_load_top_items", _fake_load_top_items)
    monkeypatch.setattr(popular, "_resolve_telegraph_map", _fake_resolve_telegraph_map)

    message = _Message()
    await popular._send_popular_posts_report(message, object(), limit=10)

    assert message.chunks
    text = "\n".join(message.chunks)
    assert "skip(<=median)=" not in text
    assert "skip(&lt;=median)=3" in text


@pytest.mark.asyncio
async def test_popular_posts_report_shows_above_median_breakdown(monkeypatch):
    class _Message:
        def __init__(self) -> None:
            self.chunks: list[str] = []

        async def answer(self, text, **kwargs):  # noqa: ANN003
            self.chunks.append(str(text))

    async def _fake_load_top_items(_db, *, window_days, age_day, limit):  # noqa: ANN001
        return (
            [],
            {
                "tg_available": True,
                "vk_available": True,
                "tg_total": 0,
                "vk_total": 0,
                "skipped_small_sample": 0,
                "skipped_missing_median": 0,
                "skipped_missing_metrics": 0,
                "skipped_not_above_median": 7,
                "checked_posts": 7,
                "above_median_views": 2,
                "above_median_likes": 3,
                "above_median_both": 0,
            },
        )

    async def _fake_resolve_telegraph_map(_db, *, source_urls):  # noqa: ANN001
        return {}, set()

    monkeypatch.setattr(popular, "_load_top_items", _fake_load_top_items)
    monkeypatch.setattr(popular, "_resolve_telegraph_map", _fake_resolve_telegraph_map)

    message = _Message()
    await popular._send_popular_posts_report(message, object(), limit=10)

    assert message.chunks
    text = "\n".join(message.chunks)
    assert "Выше медианы (после фильтров): views=2, likes=3, оба=0 (из 7)." in text


@pytest.mark.asyncio
async def test_popular_posts_resolves_event_links_with_tg_url_variants(tmp_path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    monkeypatch.setattr(popular, "_local_today", lambda: date(2026, 3, 1))
    try:
        async with db.get_session() as session:
            ev = Event(
                title="Acid Nymphs",
                description="desc",
                date="2026-03-03",
                time="19:00",
                location_name="Loc",
                source_text="src",
                telegraph_path="Acid-Nymphs-03-03",
            )
            session.add(ev)
            await session.flush()
            session.add(
                EventSource(
                    event_id=int(ev.id),
                    source_type="telegram",
                    source_url="https://t.me/meowafisha/6823?single",
                    source_chat_username="meowafisha",
                    source_message_id=6823,
                )
            )
            await session.commit()

        links, matched_urls = await popular._resolve_telegraph_map(
            db,
            source_urls=[
                "https://t.me/meowafisha/6823",
                "https://t.me/meowafisha/6823?single",
            ],
        )
        assert matched_urls == {"https://t.me/meowafisha/6823", "https://t.me/meowafisha/6823?single"}
        assert links["https://t.me/meowafisha/6823"].total == 1
        assert links["https://t.me/meowafisha/6823"].events
        assert links["https://t.me/meowafisha/6823"].events[0].event_id == int(ev.id)
        assert links["https://t.me/meowafisha/6823"].events[0].telegraph_url == "https://telegra.ph/Acid-Nymphs-03-03"

        assert links["https://t.me/meowafisha/6823?single"].total == 1
        assert links["https://t.me/meowafisha/6823?single"].events
        assert links["https://t.me/meowafisha/6823?single"].events[0].event_id == int(ev.id)
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_popular_posts_hides_past_only_events(tmp_path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    monkeypatch.setattr(popular, "_local_today", lambda: date(2026, 3, 12))

    past_url = "https://t.me/source/1"
    future_url = "https://t.me/source/2"
    ongoing_url = "https://t.me/source/3"

    async with db.get_session() as session:
        past_event = Event(
            title="Past event",
            description="desc",
            date="2026-03-10",
            time="19:00",
            location_name="Hall",
            source_text="src",
        )
        future_event = Event(
            title="Future event",
            description="desc",
            date="2026-03-15",
            time="19:00",
            location_name="Hall",
            source_text="src",
        )
        ongoing_event = Event(
            title="Ongoing event",
            description="desc",
            date="2026-03-01",
            end_date="2026-03-20",
            time="19:00",
            location_name="Hall",
            source_text="src",
        )
        session.add_all([past_event, future_event, ongoing_event])
        await session.commit()
        await session.refresh(past_event)
        await session.refresh(future_event)
        await session.refresh(ongoing_event)

        session.add_all(
            [
                EventSource(
                    event_id=past_event.id,
                    source_type="telegram",
                    source_url=past_url,
                    source_chat_username="source",
                    source_message_id=1,
                ),
                EventSource(
                    event_id=future_event.id,
                    source_type="telegram",
                    source_url=future_url,
                    source_chat_username="source",
                    source_message_id=2,
                ),
                EventSource(
                    event_id=ongoing_event.id,
                    source_type="telegram",
                    source_url=ongoing_url,
                    source_chat_username="source",
                    source_message_id=3,
                ),
            ]
        )
        await session.commit()

    telegraph_map, matched_urls = await popular._resolve_telegraph_map(
        db,
        source_urls=[past_url, future_url, ongoing_url],
    )

    assert matched_urls == {past_url, future_url, ongoing_url}
    assert past_url not in telegraph_map
    assert future_url in telegraph_map
    assert ongoing_url in telegraph_map

    baseline = popular._Baseline(median_views=100, median_likes=10, sample=2)
    items = [
        popular._PostItem(
            platform="tg",
            source_key=1,
            source_label="Past",
            post_id=1,
            post_url=past_url,
            published_ts=1,
            views=200,
            likes=20,
            baseline=baseline,
            popularity="⭐👍",
            score=2.0,
        ),
        popular._PostItem(
            platform="tg",
            source_key=1,
            source_label="Future",
            post_id=2,
            post_url=future_url,
            published_ts=2,
            views=210,
            likes=21,
            baseline=baseline,
            popularity="⭐👍",
            score=2.1,
        ),
        popular._PostItem(
            platform="tg",
            source_key=1,
            source_label="Ongoing",
            post_id=3,
            post_url=ongoing_url,
            published_ts=3,
            views=220,
            likes=22,
            baseline=baseline,
            popularity="⭐👍",
            score=2.2,
        ),
    ]
    dbg = {}

    kept = popular._prune_stale_only_items(
        items,
        telegraph_map=telegraph_map,
        matched_urls=matched_urls,
        dbg=dbg,
    )

    assert [item.post_url for item in kept] == [future_url, ongoing_url]
    assert dbg["skipped_past_event_only"] == 1
