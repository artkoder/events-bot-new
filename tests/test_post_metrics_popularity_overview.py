from __future__ import annotations

import pytest

from db import Database
from source_parsing.post_metrics import load_telegram_popularity_overview


async def _create_source(db: Database, username: str) -> int:
    async with db.raw_conn() as conn:
        await conn.execute(
            "INSERT INTO telegram_source(username, enabled) VALUES(?,1)",
            (username,),
        )
        cur = await conn.execute(
            "SELECT id FROM telegram_source WHERE username=? LIMIT 1",
            (username,),
        )
        row = await cur.fetchone()
        assert row and row[0]
        await conn.commit()
        return int(row[0])


async def _seed_message(
    db: Database,
    *,
    source_id: int,
    message_id: int,
    age_day: int,
    message_ts: int,
    views: int,
    likes: int,
) -> None:
    async with db.raw_conn() as conn:
        await conn.execute(
            """
            INSERT INTO telegram_scanned_message(
                source_id, message_id, status, events_extracted, events_imported
            ) VALUES(?,?,?,?,?)
            """,
            (int(source_id), int(message_id), "imported", 1, 1),
        )
        await conn.execute(
            """
            INSERT INTO telegram_post_metric(
                source_id, message_id, age_day, source_url, message_ts, collected_ts, views, likes
            ) VALUES(?,?,?,?,?,?,?,?)
            """,
            (
                int(source_id),
                int(message_id),
                int(age_day),
                f"https://t.me/test/{int(message_id)}",
                int(message_ts),
                int(message_ts) + 3600,
                int(views),
                int(likes),
            ),
        )
        await conn.commit()


@pytest.mark.asyncio
async def test_load_telegram_popularity_overview_basic(tmp_path, monkeypatch):
    monkeypatch.setenv("POST_POPULARITY_MIN_SAMPLE", "2")
    monkeypatch.setenv("POST_POPULARITY_MAX_AGE_DAY", "2")
    monkeypatch.setenv("POST_POPULARITY_HORIZON_DAYS", "90")

    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    source_id = await _create_source(db, "chan_overview_basic")

    # 2 distinct publish days, 3 posts in total (age_day=0).
    now_ts = 1_700_000_000
    day0 = now_ts - 3 * 86400 + 12 * 3600
    day1 = now_ts - 2 * 86400 + 12 * 3600
    await _seed_message(db, source_id=source_id, message_id=1, age_day=0, message_ts=day0, views=100, likes=10)
    await _seed_message(db, source_id=source_id, message_id=2, age_day=0, message_ts=day0 + 60, views=200, likes=20)
    await _seed_message(db, source_id=source_id, message_id=3, age_day=0, message_ts=day1, views=300, likes=30)

    overview = await load_telegram_popularity_overview(
        db,
        source_id=source_id,
        age_day=0,
        horizon_days=90,
        now_ts=now_ts,
    )
    assert overview.used_fallback is False
    assert overview.days_covered == 2
    assert overview.baseline.sample == 3
    assert overview.baseline.median_views == 200
    assert overview.baseline.median_likes == 20


@pytest.mark.asyncio
async def test_load_telegram_popularity_overview_fallback_when_sparse(tmp_path, monkeypatch):
    monkeypatch.setenv("POST_POPULARITY_MIN_SAMPLE", "2")
    monkeypatch.setenv("POST_POPULARITY_MAX_AGE_DAY", "2")
    monkeypatch.setenv("POST_POPULARITY_HORIZON_DAYS", "90")

    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    source_id = await _create_source(db, "chan_overview_fallback")

    now_ts = 1_700_000_000
    day0 = now_ts - 3 * 86400 + 12 * 3600
    day1 = now_ts - 2 * 86400 + 12 * 3600

    # Only one post in the exact age_day bucket -> fallback to age_day<=max_age.
    await _seed_message(db, source_id=source_id, message_id=1, age_day=0, message_ts=day0, views=100, likes=10)
    await _seed_message(db, source_id=source_id, message_id=2, age_day=1, message_ts=day0 + 60, views=200, likes=20)
    await _seed_message(db, source_id=source_id, message_id=3, age_day=2, message_ts=day1, views=300, likes=30)

    overview = await load_telegram_popularity_overview(
        db,
        source_id=source_id,
        age_day=0,
        horizon_days=90,
        now_ts=now_ts,
    )
    assert overview.used_fallback is True
    assert overview.days_covered == 2
    assert overview.baseline.sample == 3
    assert overview.baseline.median_views == 200
    assert overview.baseline.median_likes == 20

