import logging
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
            "INSERT INTO vk_source(group_id, screen_name, name, location, default_time, default_ticket_link) VALUES(?,?,?,?,?,?)",
            (1, 'g', 'Group', '', None, None),
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


@pytest.mark.asyncio
async def test_crawl_inserts_blank_single_photo_post(tmp_path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    async with db.raw_conn() as conn:
        await conn.execute(
            "INSERT INTO vk_source(group_id, screen_name, name, location, default_time, default_ticket_link) VALUES(?,?,?,?,?,?)",
            (1, "g", "Group", "", None, None),
        )
        await conn.commit()

    post_ts = int(time.time()) + 10
    posts = [
        {
            "date": post_ts,
            "post_id": 3,
            "text": "   ",
            "photos": [{"id": 1, "sizes": []}],
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
            "SELECT text, matched_kw, has_date, event_ts_hint FROM vk_inbox WHERE post_id=?",
            (3,),
        )
        row = await cur.fetchone()
    assert row == ("   ", vk_intake.OCR_PENDING_SENTINEL, 0, None)


@pytest.mark.asyncio
async def test_forced_backfill_respects_clamped_horizon(
    tmp_path, monkeypatch
):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    async with db.raw_conn() as conn:
        await conn.execute(
            "INSERT INTO vk_source(group_id, screen_name, name, location, default_time, default_ticket_link) VALUES(?,?,?,?,?,?)",
            (1, "g", "Group", "", None, None),
        )
        await conn.commit()

    monkeypatch.setattr(vk_intake, "VK_CRAWL_BACKFILL_OVERRIDE_MAX_DAYS", 5)
    monkeypatch.setattr(vk_intake, "match_keywords", lambda text: (True, ["test"]))
    monkeypatch.setattr(vk_intake, "detect_date", lambda text: True)
    monkeypatch.setattr(
        vk_intake,
        "extract_event_ts_hint",
        lambda text, default_time=None, *, tz=None: int(time.time()) + 86400,
    )

    async def no_sleep(_):
        pass

    monkeypatch.setattr(vk_intake.asyncio, "sleep", no_sleep)

    now_ts = int(time.time())
    horizon_days = vk_intake.VK_CRAWL_BACKFILL_OVERRIDE_MAX_DAYS
    recent_post = {
        "date": now_ts - 86400,
        "post_id": 101,
        "text": "концерт 01.01.2099",
        "photos": [],
    }
    stale_post = {
        "date": now_ts - (horizon_days + 1) * 86400,
        "post_id": 99,
        "text": "концерт 01.01.2099",
        "photos": [],
    }

    calls: list[tuple[int, int, int, int]] = []

    async def fake_wall_since(gid, since, count, offset=0):
        calls.append((gid, since, count, offset))
        if offset == 0:
            return [recent_post, stale_post]
        return []

    monkeypatch.setattr(main, "vk_wall_since", fake_wall_since)

    stats = await vk_intake.crawl_once(
        db,
        force_backfill=True,
        backfill_days=10,
    )

    assert stats["forced_backfill"] is True
    assert stats["backfill_days_requested"] == 10
    assert stats["backfill_days_used"] == horizon_days
    assert all(call[1] == 0 for call in calls)

    async with db.raw_conn() as conn:
        cur = await conn.execute(
            "SELECT post_id FROM vk_inbox ORDER BY post_id"
        )
        rows = await cur.fetchall()

    assert rows == [(101,)]


@pytest.mark.asyncio
async def test_incremental_pagination_processes_full_backlog(
    tmp_path, monkeypatch, caplog
):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    async with db.raw_conn() as conn:
        await conn.execute(
            "INSERT INTO vk_source(group_id, screen_name, name, location, default_time, default_ticket_link) VALUES(?,?,?,?,?,?)",
            (1, "g", "Group", "", None, None),
        )
        last_seen_ts = int(time.time()) - 1000
        await conn.execute(
            "INSERT INTO vk_crawl_cursor(group_id, last_seen_ts, last_post_id) VALUES(?,?,?)",
            (1, last_seen_ts, 0),
        )
        await conn.commit()

    monkeypatch.setattr(vk_intake, "VK_CRAWL_PAGE_SIZE", 2)
    monkeypatch.setattr(vk_intake, "VK_CRAWL_MAX_PAGES_INC", 1)
    monkeypatch.setattr(vk_intake, "VK_CRAWL_OVERLAP_SEC", 5)
    monkeypatch.setattr(vk_intake, "match_keywords", lambda text: (True, ["test"]))
    monkeypatch.setattr(vk_intake, "detect_date", lambda text: True)
    monkeypatch.setattr(
        vk_intake,
        "extract_event_ts_hint",
        lambda text, default_time=None, *, tz=None: int(time.time()) + 86400,
    )

    base_ts = int(time.time()) - 900
    total_posts = 6
    posts = [
        {
            "date": base_ts + i * 10,
            "post_id": 100 + i,
            "text": "концерт 01.01.2099",
            "photos": [],
        }
        for i in range(total_posts)
    ]

    async def fake_wall_since(gid, since, count, offset=0):
        filtered = [p for p in posts if p["date"] >= since]
        filtered.sort(key=lambda p: (p["date"], p["post_id"]), reverse=True)
        start = offset
        end = offset + count
        return filtered[start:end]

    monkeypatch.setattr(main, "vk_wall_since", fake_wall_since)

    async def no_sleep(_):
        pass

    monkeypatch.setattr(vk_intake.asyncio, "sleep", no_sleep)

    caplog.set_level(logging.WARNING)
    stats = await vk_intake.crawl_once(db)

    assert stats["added"] == total_posts
    assert stats["safety_cap_hits"] == 1
    assert any("vk.crawl.inc.safety_cap" in rec.message for rec in caplog.records)

    async with db.raw_conn() as conn:
        cur = await conn.execute("SELECT post_id FROM vk_inbox ORDER BY post_id")
        inbox_rows = await cur.fetchall()
        cur = await conn.execute(
            "SELECT last_seen_ts, last_post_id FROM vk_crawl_cursor WHERE group_id=?",
            (1,),
        )
        cursor_row = await cur.fetchone()

    assert [pid for (pid,) in inbox_rows] == [100 + i for i in range(total_posts)]

    newest_ts = max(p["date"] for p in posts)
    expected_cursor_ts = max(last_seen_ts, newest_ts - vk_intake.VK_CRAWL_OVERLAP_SEC)
    if expected_cursor_ts < newest_ts:
        expected_cursor = (expected_cursor_ts, 0)
    else:
        newest_pid = max(p["post_id"] for p in posts if p["date"] == newest_ts)
        expected_cursor = (expected_cursor_ts, newest_pid)
    assert cursor_row == expected_cursor


@pytest.mark.asyncio
async def test_hard_cap_triggers_backfill(tmp_path, monkeypatch, caplog):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    async with db.raw_conn() as conn:
        await conn.execute(
            "INSERT INTO vk_source(group_id, screen_name, name, location, default_time, default_ticket_link) VALUES(?,?,?,?,?,?)",
            (1, "g", "Group", "", None, None),
        )
        last_seen_ts = int(time.time()) - 1000
        await conn.execute(
            "INSERT INTO vk_crawl_cursor(group_id, last_seen_ts, last_post_id) VALUES(?,?,?)",
            (1, last_seen_ts, 0),
        )
        await conn.commit()

    monkeypatch.setattr(vk_intake, "VK_CRAWL_PAGE_SIZE", 1)
    monkeypatch.setattr(vk_intake, "VK_CRAWL_MAX_PAGES_INC", 1)
    monkeypatch.setattr(vk_intake, "VK_CRAWL_OVERLAP_SEC", 5)
    monkeypatch.setattr(vk_intake, "VK_CRAWL_PAGE_SIZE_BACKFILL", 4)
    monkeypatch.setattr(vk_intake, "VK_CRAWL_MAX_PAGES_BACKFILL", 5)
    monkeypatch.setattr(vk_intake, "VK_CRAWL_BACKFILL_DAYS", 30)
    monkeypatch.setattr(vk_intake, "match_keywords", lambda text: (True, ["test"]))
    monkeypatch.setattr(vk_intake, "detect_date", lambda text: True)
    monkeypatch.setattr(
        vk_intake,
        "extract_event_ts_hint",
        lambda text, default_time=None, *, tz=None: int(time.time()) + 86400,
    )

    base_ts = int(time.time()) - 900
    total_posts = 12
    posts = [
        {
            "date": base_ts + i * 10,
            "post_id": 200 + i,
            "text": "концерт 01.01.2099",
            "photos": [],
        }
        for i in range(total_posts)
    ]

    call_log: list[tuple[int, int]] = []

    async def fake_wall_since(gid, since, count, offset=0):
        call_log.append((since, offset))
        filtered = [p for p in posts if p["date"] >= since]
        filtered.sort(key=lambda p: (p["date"], p["post_id"]), reverse=True)
        start = offset
        end = offset + count
        return filtered[start:end]

    monkeypatch.setattr(main, "vk_wall_since", fake_wall_since)

    async def no_sleep(_):
        pass

    monkeypatch.setattr(vk_intake.asyncio, "sleep", no_sleep)

    caplog.set_level(logging.WARNING)
    stats_first = await vk_intake.crawl_once(db)

    hard_cap_pages = max(1, vk_intake.VK_CRAWL_MAX_PAGES_INC) * 10
    expected_first_added = min(total_posts, hard_cap_pages * vk_intake.VK_CRAWL_PAGE_SIZE)

    assert stats_first["added"] == expected_first_added
    assert stats_first["deep_backfill_triggers"] == 1
    assert any("vk.crawl.inc.deep_backfill_trigger" in rec.message for rec in caplog.records)

    async with db.raw_conn() as conn:
        cur = await conn.execute(
            "SELECT last_seen_ts, last_post_id, updated_at FROM vk_crawl_cursor WHERE group_id=?",
            (1,),
        )
        cursor_row = await cur.fetchone()

    assert cursor_row[0] == last_seen_ts
    assert cursor_row[1] == 0
    assert cursor_row[2] <= time.time() - vk_intake.VK_CRAWL_BACKFILL_AFTER_IDLE_H * 3600

    caplog.clear()
    stats_second = await vk_intake.crawl_once(db)

    assert stats_second["added"] == total_posts - stats_first["added"]
    assert any(since == 0 for since, _ in call_log)

    async with db.raw_conn() as conn:
        cur = await conn.execute("SELECT post_id FROM vk_inbox ORDER BY post_id")
        inbox_rows = await cur.fetchall()

    assert [pid for (pid,) in inbox_rows] == [200 + i for i in range(total_posts)]
