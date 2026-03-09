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
        return {}

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
        return {}

    monkeypatch.setattr(popular, "_load_top_items", _fake_load_top_items)
    monkeypatch.setattr(popular, "_resolve_telegraph_map", _fake_resolve_telegraph_map)

    message = _Message()
    await popular._send_popular_posts_report(message, object(), limit=10)

    assert message.chunks
    text = "\n".join(message.chunks)
    assert "Выше медианы (после фильтров): views=2, likes=3, оба=0 (из 7)." in text


@pytest.mark.asyncio
async def test_popular_posts_resolves_event_links_with_tg_url_variants(tmp_path):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
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

        links = await popular._resolve_telegraph_map(
            db,
            source_urls=[
                "https://t.me/meowafisha/6823",
                "https://t.me/meowafisha/6823?single",
            ],
        )
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
async def test_popular_posts_three_day_uses_latest_available_snapshot(tmp_path, monkeypatch):
    monkeypatch.setenv("POST_POPULARITY_MIN_SAMPLE", "2")
    monkeypatch.setattr(popular, "_utc_now_ts", lambda: 1_800_000_000)

    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    try:
        async with db.raw_conn() as conn:
            await conn.execute(
                "INSERT INTO telegram_source(username, enabled) VALUES(?,1)",
                ("late_snapshots",),
            )
            cur = await conn.execute(
                "SELECT id FROM telegram_source WHERE username=? LIMIT 1",
                ("late_snapshots",),
            )
            row = await cur.fetchone()
            assert row and row[0]
            source_id = int(row[0])
            published_1 = 1_800_000_000 - 2 * 86400
            published_2 = 1_800_000_000 - 86400
            await conn.execute(
                """
                INSERT INTO telegram_scanned_message(
                    source_id, message_id, status, events_extracted, events_imported
                ) VALUES(?,?,?,?,?)
                """,
                (source_id, 101, "done", 1, 1),
            )
            await conn.execute(
                """
                INSERT INTO telegram_scanned_message(
                    source_id, message_id, status, events_extracted, events_imported
                ) VALUES(?,?,?,?,?)
                """,
                (source_id, 102, "done", 1, 1),
            )
            await conn.execute(
                """
                INSERT INTO telegram_post_metric(
                    source_id, message_id, age_day, source_url, message_ts, collected_ts, views, likes
                ) VALUES(?,?,?,?,?,?,?,?)
                """,
                (source_id, 101, 0, "https://t.me/late_snapshots/101", published_1, published_1 + 3600, 100, 10),
            )
            await conn.execute(
                """
                INSERT INTO telegram_post_metric(
                    source_id, message_id, age_day, source_url, message_ts, collected_ts, views, likes
                ) VALUES(?,?,?,?,?,?,?,?)
                """,
                (source_id, 101, 1, "https://t.me/late_snapshots/101", published_1, published_1 + 86400, 150, 15),
            )
            await conn.execute(
                """
                INSERT INTO telegram_post_metric(
                    source_id, message_id, age_day, source_url, message_ts, collected_ts, views, likes
                ) VALUES(?,?,?,?,?,?,?,?)
                """,
                (source_id, 102, 0, "https://t.me/late_snapshots/102", published_2, published_2 + 3600, 200, 20),
            )
            await conn.commit()

        items, dbg = await _load_top_items(db, window_days=3, age_day=2, limit=10)

        assert dbg["snapshot_mode"] == "latest_available"
        assert dbg["tg_metrics_total"] == 2
        assert dbg["tg_total"] == 2
        assert len(items) == 1
        assert items[0].platform == "tg"
        assert items[0].post_id == 102
        assert items[0].baseline.sample == 2
        assert items[0].baseline.median_views == 175
        assert items[0].baseline.median_likes == 17
    finally:
        await db.close()
