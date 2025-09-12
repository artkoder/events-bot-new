import os
import sys
import time
from datetime import datetime

import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import main
import vk_intake
from db import Database


@pytest.mark.asyncio
async def test_crawl_skips_past_events(tmp_path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    async with db.raw_conn() as conn:
        await conn.execute(
            "INSERT INTO vk_source(group_id, screen_name, name, location, default_time) VALUES(?,?,?,?,?)",
            (1, 'g', 'Group', '', None),
        )
        await conn.commit()

    now = datetime.now()
    past_year = now.year - 1
    future_year = now.year + 1
    posts = [
        {"date": int(time.time()), "post_id": 1, "text": f"31 августа {past_year} концерты"},
        {"date": int(time.time()) + 1, "post_id": 2, "text": f"31 августа {future_year} концерты"},
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
        cur = await conn.execute("SELECT post_id FROM vk_inbox")
        rows = await cur.fetchall()
    assert rows == [(2,)]
