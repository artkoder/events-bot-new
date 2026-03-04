from __future__ import annotations

import time

import pytest

from db import Database
from source_parsing.post_metrics import cleanup_post_metrics


@pytest.mark.asyncio
async def test_cleanup_post_metrics_deletes_rows_older_than_retention(tmp_path):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    now = int(time.time())
    old = now - 200 * 86400

    async with db.raw_conn() as conn:
        await conn.execute(
            "INSERT INTO telegram_source(username, enabled) VALUES(?,1)",
            ("chan_cleanup_test",),
        )
        cur = await conn.execute(
            "SELECT id FROM telegram_source WHERE username=? LIMIT 1",
            ("chan_cleanup_test",),
        )
        row = await cur.fetchone()
        assert row and row[0]
        source_id = int(row[0])
        await conn.execute(
            """
            INSERT INTO telegram_post_metric(
                source_id, message_id, age_day, source_url, message_ts, collected_ts, views, likes
            ) VALUES(?,?,?,?,?,?,?,?)
            """,
            (source_id, 10, 0, "https://t.me/chan_cleanup_test/10", old, old, 1, 1),
        )
        await conn.execute(
            """
            INSERT INTO vk_post_metric(
                group_id, post_id, age_day, source_url, post_ts, collected_ts, views, likes
            ) VALUES(?,?,?,?,?,?,?,?)
            """,
            (123, 456, 0, "https://vk.com/wall-123_456", old, old, 1, 1),
        )
        await conn.commit()

    deleted = await cleanup_post_metrics(db, retention_days=90, now_ts=now)
    assert deleted["telegram_post_metric"] == 1
    assert deleted["vk_post_metric"] == 1

    async with db.raw_conn() as conn:
        cur = await conn.execute("SELECT COUNT(1) FROM telegram_post_metric")
        (cnt_tg,) = await cur.fetchone()
        cur = await conn.execute("SELECT COUNT(1) FROM vk_post_metric")
        (cnt_vk,) = await cur.fetchone()
    assert int(cnt_tg) == 0
    assert int(cnt_vk) == 0
