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
async def test_popular_posts_includes_post_above_only_one_median(tmp_path, monkeypatch):
    monkeypatch.setenv("POST_POPULARITY_MIN_SAMPLE", "2")
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    try:
        async with db.raw_conn() as conn:
            await conn.execute(
                "INSERT INTO vk_source(group_id, screen_name, name) VALUES(?, ?, ?)",
                (101, "club101", "Club 101"),
            )
            await conn.execute(
                "INSERT INTO vk_inbox(id, group_id, post_id, source_url, status) VALUES(?, ?, ?, ?, ?)",
                (1, 101, 1001, "https://vk.com/wall-101_1001", "imported"),
            )
            await conn.execute(
                "INSERT INTO vk_inbox(id, group_id, post_id, source_url, status) VALUES(?, ?, ?, ?, ?)",
                (2, 101, 1002, "https://vk.com/wall-101_1002", "imported"),
            )
            await conn.execute(
                "INSERT INTO vk_inbox_import_event(id, inbox_id, event_id, created_at) VALUES(?, ?, ?, CURRENT_TIMESTAMP)",
                (1, 1, 1),
            )
            await conn.execute(
                "INSERT INTO vk_inbox_import_event(id, inbox_id, event_id, created_at) VALUES(?, ?, ?, CURRENT_TIMESTAMP)",
                (2, 2, 2),
            )
            await conn.execute(
                "INSERT INTO vk_post_metric(group_id, post_id, age_day, source_url, post_ts, collected_ts, views, likes) "
                "VALUES(?, ?, ?, ?, ?, ?, ?, ?)",
                (101, 1001, 0, "https://vk.com/wall-101_1001", 2_000_000_000, 2_000_000_100, 100, 10),
            )
            await conn.execute(
                "INSERT INTO vk_post_metric(group_id, post_id, age_day, source_url, post_ts, collected_ts, views, likes) "
                "VALUES(?, ?, ?, ?, ?, ?, ?, ?)",
                (101, 1002, 0, "https://vk.com/wall-101_1002", 2_000_000_010, 2_000_000_100, 110, 5),
            )
            await conn.commit()

        monkeypatch.setattr(popular, "_utc_now_ts", lambda: 2_000_000_100)
        items, dbg = await _load_top_items(db, window_days=1, age_day=0, limit=10)

        assert len(items) == 1
        assert items[0].platform == "vk"
        assert items[0].post_id == 1002
        assert dbg["checked_posts"] == 2
        assert dbg["above_median_views"] == 1
        assert dbg["above_median_likes"] == 1
        assert dbg["above_median_both"] == 0
        assert dbg["skipped_not_above_median"] == 1
    finally:
        await db.close()
