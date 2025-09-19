import os, sys
import os, sys
import os, sys
from datetime import datetime as real_datetime, timezone

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
            (1, 1, 100, "Событие 25.12.2099", "k", 1, future_ts, "pending"),
            (1, 2, 200, "Событие 26.12.2099", "k", 1, future_ts, "pending"),
        ]
        await conn.executemany(
            "INSERT INTO vk_inbox(group_id, post_id, date, text, matched_kw, has_date, event_ts_hint, status) VALUES(?,?,?,?,?,?,?,?)",
            rows,
        )
        await conn.commit()
    post = await vk_review.pick_next(db, 10, "batch1")
    assert post and post.post_id == 2  # newest by date

    # Skip the first post and ensure the other pending one is returned
    await vk_review.mark_skipped(db, post.id)
    post2 = await vk_review.pick_next(db, 10, "batch1")
    assert post2 and post2.post_id == 1

    # After resolving remaining pending posts the skipped one should reappear
    await vk_review.mark_rejected(db, post2.id)
    post3 = await vk_review.pick_next(db, 10, "batch1")
    assert post3 and post3.post_id == 2


@pytest.mark.asyncio
async def test_pick_next_rejects_outdated(tmp_path):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    async with db.raw_conn() as conn:
        now = int(_time.time())
        rows = [
            # event starting in ~100s should be rejected
            (1, 1, 100, "old", "k", 1, now + 100, "pending"),
            # far future event should be shown first
            (1, 2, 200, "Концерт 05.02.2099 в 19:00", "k", 1, now + 10_000, "pending"),
            # event without timestamp should now be rejected on recompute
            (1, 3, 300, "unknown", "k", 0, None, "pending"),
        ]
        await conn.executemany(
            "INSERT INTO vk_inbox(group_id, post_id, date, text, matched_kw, has_date, event_ts_hint, status) VALUES(?,?,?,?,?,?,?,?)",
            rows,
        )
        await conn.commit()

    post = await vk_review.pick_next(db, 10, "batch1")
    assert post and post.post_id == 2
    async with db.raw_conn() as conn:
        cur = await conn.execute("SELECT status FROM vk_inbox WHERE post_id=1")
        assert (await cur.fetchone())[0] == "rejected"

    await vk_review.mark_rejected(db, post.id)
    post2 = await vk_review.pick_next(db, 10, "batch1")
    assert post2 is None
    async with db.raw_conn() as conn:
        cur = await conn.execute("SELECT status FROM vk_inbox WHERE post_id=3")
        assert (await cur.fetchone())[0] == "rejected"


@pytest.mark.asyncio
async def test_pick_next_recomputes_hint_and_rejects_recent_past(tmp_path, monkeypatch):
    class FixedDatetime(real_datetime):
        @classmethod
        def now(cls, tz=None):
            tzinfo = tz or timezone.utc
            return real_datetime(2024, 10, 1, tzinfo=tzinfo)

    monkeypatch.setattr("vk_intake.datetime", FixedDatetime)
    fixed_epoch = int(real_datetime(2024, 10, 1, tzinfo=timezone.utc).timestamp())
    monkeypatch.setattr(vk_review._time, "time", lambda: fixed_epoch)

    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    async with db.raw_conn() as conn:
        rows = [
            # Stored hint is far in future but text describes a past event
            (1, 1, 100, "7 сентября прошла лекция", "k", 1, fixed_epoch + 1_000_000, "pending"),
            # Valid future event with even later hint so it becomes next candidate
            (1, 2, 200, "7 января состоится концерт", "k", 1, fixed_epoch + 2_000_000, "pending"),
        ]
        await conn.executemany(
            "INSERT INTO vk_inbox(group_id, post_id, date, text, matched_kw, has_date, event_ts_hint, status) VALUES(?,?,?,?,?,?,?,?)",
            rows,
        )
        await conn.commit()

    post = await vk_review.pick_next(db, 10, "batch1")
    assert post and post.post_id == 2

    async with db.raw_conn() as conn:
        cur = await conn.execute("SELECT status FROM vk_inbox WHERE post_id=1")
        assert (await cur.fetchone())[0] == "rejected"
        cur = await conn.execute("SELECT status FROM vk_inbox WHERE post_id=2")
        assert (await cur.fetchone())[0] == "locked"


@pytest.mark.asyncio
async def test_far_gap_override_triggers_after_k_non_far(tmp_path, monkeypatch):
    vk_review._FAR_BUCKET_HISTORY.clear()
    monkeypatch.setenv("VK_REVIEW_FAR_GAP_K", "3")
    monkeypatch.setenv("VK_REVIEW_W_SOON", "1")
    monkeypatch.setenv("VK_REVIEW_W_LONG", "1")
    monkeypatch.setenv("VK_REVIEW_W_FAR", "1")
    fixed_now = 1_700_000_000
    monkeypatch.setattr(vk_review._time, "time", lambda: fixed_now)
    monkeypatch.setattr(vk_review.random, "random", lambda: 0.0)

    def fake_extract(text):
        assert text.startswith("TS:")
        return int(text.split(":", 1)[1])

    monkeypatch.setattr(vk_review, "extract_event_ts_hint", fake_extract)

    urgent_cutoff = fixed_now + int(48 * 3600)
    long_cutoff = fixed_now + int(30 * 86400)

    soon_hints = [urgent_cutoff + 1000 + i for i in range(4)]
    far_hint = long_cutoff + 1000

    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    async with db.raw_conn() as conn:
        rows = [
            (
                1,
                idx + 1,
                100 + idx,
                f"TS:{hint}",
                "k",
                1,
                hint,
                "pending",
            )
            for idx, hint in enumerate(soon_hints)
        ]
        rows.append((1, 100, 500, f"TS:{far_hint}", "k", 1, far_hint, "pending"))
        await conn.executemany(
            "INSERT INTO vk_inbox(group_id, post_id, date, text, matched_kw, has_date, event_ts_hint, status) VALUES(?,?,?,?,?,?,?,?)",
            rows,
        )
        await conn.commit()

    operator_id = 77
    batch_id = "batch"

    for expected_history_length in (1, 2, 3):
        post = await vk_review.pick_next(db, operator_id, batch_id)
        assert post is not None
        assert post.post_id != 100
        await vk_review.mark_rejected(db, post.id)
        history = vk_review._FAR_BUCKET_HISTORY.get(operator_id)
        assert history is not None
        assert len(history) == expected_history_length
        assert all(bucket == "SOON" for bucket in history)

    post = await vk_review.pick_next(db, operator_id, batch_id)
    assert post is not None
    assert post.post_id == 100
    await vk_review.mark_rejected(db, post.id)
    history = vk_review._FAR_BUCKET_HISTORY.get(operator_id)
    assert history is not None
    assert list(history) == ["SOON", "SOON", "FAR"]


@pytest.mark.asyncio
async def test_bucket_boundaries_use_weighted_selection(tmp_path, monkeypatch):
    vk_review._FAR_BUCKET_HISTORY.clear()
    monkeypatch.setenv("VK_REVIEW_FAR_GAP_K", "2")
    monkeypatch.setenv("VK_REVIEW_W_SOON", "1")
    monkeypatch.setenv("VK_REVIEW_W_LONG", "1")
    monkeypatch.setenv("VK_REVIEW_W_FAR", "0")
    fixed_now = 1_750_000_000
    monkeypatch.setattr(vk_review._time, "time", lambda: fixed_now)
    monkeypatch.setattr(vk_review.random, "random", lambda: 0.0)

    urgent_cutoff = fixed_now + int(48 * 3600)
    soon_cutoff = fixed_now + int(14 * 86400)

    def fake_extract(text):
        return int(text.split(":", 1)[1])

    monkeypatch.setattr(vk_review, "extract_event_ts_hint", fake_extract)

    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    async with db.raw_conn() as conn:
        rows = [
            (1, 101, 100, f"TS:{urgent_cutoff}", "k", 1, urgent_cutoff, "pending"),
            (1, 202, 200, f"TS:{soon_cutoff}", "k", 1, soon_cutoff, "pending"),
        ]
        await conn.executemany(
            "INSERT INTO vk_inbox(group_id, post_id, date, text, matched_kw, has_date, event_ts_hint, status) VALUES(?,?,?,?,?,?,?,?)",
            rows,
        )
        await conn.commit()

    operator_id = 55
    batch_id = "boundaries"

    post = await vk_review.pick_next(db, operator_id, batch_id)
    assert post is not None
    assert post.post_id == 101
    history = vk_review._FAR_BUCKET_HISTORY.get(operator_id)
    assert history is not None
    assert list(history) == ["SOON"]

    await vk_review.mark_rejected(db, post.id)

    post2 = await vk_review.pick_next(db, operator_id, batch_id)
    assert post2 is not None
    assert post2.post_id == 202
    history = vk_review._FAR_BUCKET_HISTORY.get(operator_id)
    assert history is not None
    assert list(history) == ["SOON", "LONG"]


@pytest.mark.asyncio
async def test_pick_next_weighted_bucket_without_sqlite_math(tmp_path, monkeypatch):
    vk_review._FAR_BUCKET_HISTORY.clear()
    monkeypatch.setenv("VK_REVIEW_W_SOON", "0")
    monkeypatch.setenv("VK_REVIEW_W_LONG", "0")
    monkeypatch.setenv("VK_REVIEW_W_FAR", "5")
    monkeypatch.setenv("VK_REVIEW_FAR_GAP_K", "1")
    fixed_now = 1_720_000_000
    monkeypatch.setattr(vk_review._time, "time", lambda: fixed_now)
    monkeypatch.setattr(vk_review.random, "random", lambda: 0.25)

    far_hint = fixed_now + int(60 * 86400)

    def fake_extract(text: str) -> int:
        return int(text.split(":", 1)[1])

    monkeypatch.setattr(vk_review, "extract_event_ts_hint", fake_extract)

    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    async with db.raw_conn() as conn:
        rows = [
            (1, 101, 1_000, f"TS:{far_hint}", "k", 1, far_hint, "pending"),
            (1, 102, 900, f"TS:{far_hint + 10}", "k", 1, far_hint + 10, "pending"),
            (2, 201, 800, f"TS:{far_hint + 20}", "k", 1, far_hint + 20, "pending"),
        ]
        await conn.executemany(
            "INSERT INTO vk_inbox(group_id, post_id, date, text, matched_kw, has_date, event_ts_hint, status) VALUES(?,?,?,?,?,?,?,?)",
            rows,
        )
        await conn.commit()

    post = await vk_review.pick_next(db, 99, "weighted")
    assert post is not None
    history = vk_review._FAR_BUCKET_HISTORY.get(99)
    assert history is not None
    assert list(history) == ["FAR"]


@pytest.mark.asyncio
async def test_history_tracks_fallback_bucket(tmp_path, monkeypatch):
    vk_review._FAR_BUCKET_HISTORY.clear()
    monkeypatch.setenv("VK_REVIEW_FAR_GAP_K", "2")
    monkeypatch.setenv("VK_REVIEW_W_SOON", "0")
    monkeypatch.setenv("VK_REVIEW_W_LONG", "0")
    monkeypatch.setenv("VK_REVIEW_W_FAR", "0")
    fixed_now = 1_710_000_000
    monkeypatch.setattr(vk_review._time, "time", lambda: fixed_now)
    monkeypatch.setattr(vk_review.random, "random", lambda: 0.5)

    fallback_hint = fixed_now + int(7 * 86400)

    def fake_extract(_text):
        return fallback_hint

    monkeypatch.setattr(vk_review, "extract_event_ts_hint", fake_extract)

    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    async with db.raw_conn() as conn:
        await conn.execute(
            "INSERT INTO vk_inbox(group_id, post_id, date, text, matched_kw, has_date, event_ts_hint, status) VALUES(?,?,?,?,?,?,?,?)",
            (1, 500, 900, "fallback", "k", 0, None, "pending"),
        )
        await conn.commit()

    operator_id = 88
    post = await vk_review.pick_next(db, operator_id, "batch-fallback")
    assert post is not None
    history = vk_review._FAR_BUCKET_HISTORY.get(operator_id)
    assert history is not None
    assert list(history) == ["FALLBACK"]


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
            (1, 1, 100, "Фестиваль 10.10.2099", "k", 1, future_ts, "pending"),
        )
        await conn.commit()
    post = await vk_review.pick_next(db, 10, "batch1")
    await vk_review.mark_imported(db, post.id, "batch1", 10, 77, "2025-09-10")
    async with db.raw_conn() as conn:
        cur = await conn.execute("SELECT status, imported_event_id FROM vk_inbox WHERE id=?", (post.id,))
        st, eid = await cur.fetchone()
        cur = await conn.execute("SELECT months_csv FROM vk_review_batch WHERE batch_id=?", ("batch1",))
        months = (await cur.fetchone())[0]
    assert st == "imported" and eid == 77
    assert months == "2025-09"


@pytest.mark.asyncio
async def test_mark_imported_creates_batch_when_missing(tmp_path):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    async with db.raw_conn() as conn:
        future_ts = int(_time.time()) + 10_000
        await conn.execute(
            "INSERT INTO vk_inbox(group_id, post_id, date, text, matched_kw, has_date, event_ts_hint, status) VALUES(?,?,?,?,?,?,?,?)",
            (1, 5, 300, "Бал 11.11.2099", "k", 1, future_ts, "pending"),
        )
        await conn.commit()

    post = await vk_review.pick_next(db, 42, "batch-new")
    assert post is not None

    await vk_review.mark_imported(db, post.id, "batch-new", 42, 99, "2025-10-01")

    async with db.raw_conn() as conn:
        cur = await conn.execute(
            "SELECT status, review_batch, imported_event_id FROM vk_inbox WHERE id=?",
            (post.id,),
        )
        status, review_batch, imported_event_id = await cur.fetchone()
        cur = await conn.execute(
            "SELECT operator_id, months_csv, finished_at FROM vk_review_batch WHERE batch_id=?",
            ("batch-new",),
        )
        operator_id, months_csv, finished_at = await cur.fetchone()

    assert status == "imported"
    assert review_batch == "batch-new"
    assert imported_event_id == 99
    assert operator_id == 42
    assert months_csv == "2025-10"
    assert finished_at is None


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


@pytest.mark.asyncio
async def test_pick_next_resumes_locked_for_operator(tmp_path):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    async with db.raw_conn() as conn:
        future_ts = int(_time.time()) + 10_000
        await conn.execute(
            "INSERT INTO vk_review_batch(batch_id, operator_id, months_csv) VALUES(?,?,?)",
            ("oldbatch", 5, ""),
        )
        await conn.execute(
            """
            INSERT INTO vk_inbox(
                group_id, post_id, date, text, matched_kw, has_date,
                event_ts_hint, status, locked_by, locked_at, review_batch
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?)
            """,
            (1, 1, 123, "Праздник 12.12.2099", "k", 1, future_ts, "locked", 5, None, "oldbatch"),
        )
        await conn.commit()
    post = await vk_review.pick_next(db, 5, "newbatch")
    assert post and post.post_id == 1
    assert post.review_batch == "newbatch"
    async with db.raw_conn() as conn:
        cur = await conn.execute(
            "SELECT status, locked_by, review_batch FROM vk_inbox WHERE id=?",
            (post.id,),
        )
        status, locked_by, review_batch = await cur.fetchone()
    assert status == "locked" and locked_by == 5 and review_batch == "newbatch"


@pytest.mark.asyncio
async def test_pick_next_unlocks_stale_rows(tmp_path):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    async with db.raw_conn() as conn:
        future_ts = int(_time.time()) + 10_000
        await conn.execute(
            "INSERT INTO vk_review_batch(batch_id, operator_id, months_csv) VALUES(?,?,?)",
            ("batch", 10, ""),
        )
        await conn.execute(
            """
            INSERT INTO vk_inbox(
                group_id, post_id, date, text, matched_kw, has_date,
                event_ts_hint, status, locked_by, locked_at, review_batch
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                1,
                2,
                456,
                "Пикник 13.12.2099",
                "k",
                1,
                future_ts,
                "locked",
                99,
                "2000-01-01 00:00:00",
                "oldbatch",
            ),
        )
        await conn.commit()
    post = await vk_review.pick_next(db, 10, "batch")
    assert post and post.post_id == 2
    async with db.raw_conn() as conn:
        cur = await conn.execute(
            "SELECT status, locked_by, review_batch FROM vk_inbox WHERE id=?",
            (post.id,),
        )
        status, locked_by, review_batch = await cur.fetchone()
    assert status == "locked" and locked_by == 10 and review_batch == "batch"
