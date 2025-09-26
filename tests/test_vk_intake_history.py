import os
import sys
import time

import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import main
import vk_intake
from db import Database


@pytest.mark.asyncio
async def test_crawl_enqueues_historical_posts(tmp_path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    async with db.raw_conn() as conn:
        await conn.execute(
            "INSERT INTO vk_source(group_id, screen_name, name, location, default_time) VALUES(?,?,?,?,?)",
            (1, "g", "Group", "", None),
        )
        await conn.commit()

    post_text = "В 1945 году в Кёнигсберге пройдет фестиваль памяти"
    posts = [
        {
            "date": int(time.time()),
            "post_id": 10,
            "text": post_text,
            "photos": [],
        }
    ]

    async def fake_wall_since(gid, since, count, offset=0):
        return posts if offset == 0 else []

    monkeypatch.setattr(main, "vk_wall_since", fake_wall_since)

    async def no_sleep(_):
        pass

    monkeypatch.setattr(vk_intake.asyncio, "sleep", no_sleep)

    stats = await vk_intake.crawl_once(db)
    assert stats["added"] == 1

    async with db.raw_conn() as conn:
        cur = await conn.execute(
            "SELECT text, matched_kw, has_date, event_ts_hint, status FROM vk_inbox WHERE post_id=?",
            (10,),
        )
        row = await cur.fetchone()

    assert row == (
        post_text,
        vk_intake.HISTORY_MATCHED_KEYWORD,
        0,
        None,
        "pending",
    )
