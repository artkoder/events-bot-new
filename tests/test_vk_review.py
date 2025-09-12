import os, sys
import os, sys
import os, sys
import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import time as _time

import vk_review
from db import Database


@pytest.mark.asyncio
async def test_pick_next_and_skip(tmp_path):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    # insert two posts with different dates
    async with db.raw_conn() as conn:
        future_ts = int(_time.time()) + 10_000
        rows = [
            (1, 1, 100, "t1", "k", 1, future_ts, "pending"),
            (1, 2, 200, "t2", "k", 1, future_ts, "pending"),
        ]
        await conn.executemany(
            "INSERT INTO vk_inbox(group_id, post_id, date, text, matched_kw, has_date, event_ts_hint, status) VALUES(?,?,?,?,?,?,?,?)",
            rows,
        )
        await conn.commit()
    post = await vk_review.pick_next(db, 10, "batch1")
    assert post and post.post_id == 2  # newest by date
    # verify locked
    async with db.raw_conn() as conn:
        cur = await conn.execute("SELECT status, locked_by FROM vk_inbox WHERE id=?", (post.id,))
        st, lb = await cur.fetchone()
    assert st == "locked" and lb == 10

    await vk_review.mark_skipped(db, post.id)
    async with db.raw_conn() as conn:
        cur = await conn.execute("SELECT status, locked_by FROM vk_inbox WHERE id=?", (post.id,))
        st, lb = await cur.fetchone()
    assert st == "skipped" and lb is None

    post2 = await vk_review.pick_next(db, 10, "batch1")
    assert post2 and post2.id == post.id


@pytest.mark.asyncio
async def test_mark_imported_accumulates_month(tmp_path):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    async with db.raw_conn() as conn:
        await conn.execute(
            "INSERT INTO vk_review_batch(batch_id, operator_id, months_csv) VALUES(?,?,?)",
            ("batch1", 10, ""),
        )
        future_ts = int(_time.time()) + 10_000
        await conn.execute(
            "INSERT INTO vk_inbox(group_id, post_id, date, text, matched_kw, has_date, event_ts_hint, status) VALUES(?,?,?,?,?,?,?,?)",
            (1, 1, 100, "t1", "k", 1, future_ts, "pending"),
        )
        await conn.commit()
    post = await vk_review.pick_next(db, 10, "batch1")
    await vk_review.mark_imported(db, post.id, "batch1", 77, "2025-09-10")
    async with db.raw_conn() as conn:
        cur = await conn.execute("SELECT status, imported_event_id FROM vk_inbox WHERE id=?", (post.id,))
        st, eid = await cur.fetchone()
        cur = await conn.execute("SELECT months_csv FROM vk_review_batch WHERE batch_id=?", ("batch1",))
        months = (await cur.fetchone())[0]
    assert st == "imported" and eid == 77
    assert months == "2025-09"


@pytest.mark.asyncio
async def test_finish_batch_clears_months(tmp_path):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    async with db.raw_conn() as conn:
        await conn.execute(
            "INSERT INTO vk_review_batch(batch_id, operator_id, months_csv) VALUES(?,?,?)",
            ("batch1", 10, "2025-09,2025-10"),
        )
        await conn.commit()

    called = []

    async def fake_rebuild(_db, month):
        called.append(month)

    months = await vk_review.finish_batch(db, "batch1", fake_rebuild)
    assert called == ["2025-09", "2025-10"]
    assert months == ["2025-09", "2025-10"]
    async with db.raw_conn() as conn:
        cur = await conn.execute("SELECT months_csv, finished_at FROM vk_review_batch WHERE batch_id=?", ("batch1",))
        months_csv, finished_at = await cur.fetchone()
    assert months_csv == "" and finished_at is not None
