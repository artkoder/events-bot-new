from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest

import main_part2
from db import Database
from models import Event, EventPoster


@pytest.mark.asyncio
async def test_cleanup_old_events_does_not_queue_shared_supabase_media(tmp_path):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    now_utc = datetime(2026, 2, 26, tzinfo=timezone.utc)
    old_date = (now_utc.date() - timedelta(days=8)).isoformat()
    new_date = (now_utc.date() + timedelta(days=1)).isoformat()

    shared_path = "p/dh16/aa/" + ("a" * 64) + ".webp"

    async with db.get_session() as session:
        old = Event(
            title="Old",
            description="",
            date=old_date,
            time="18:00",
            location_name="P",
            source_text="",
        )
        new = Event(
            title="New",
            description="",
            date=new_date,
            time="18:00",
            location_name="P",
            source_text="",
        )
        session.add(old)
        session.add(new)
        await session.flush()

        session.add(
            EventPoster(
                event_id=int(old.id),
                catbox_url=None,
                supabase_url=None,
                supabase_path=shared_path,
                poster_hash="old-hash",
            )
        )
        session.add(
            EventPoster(
                event_id=int(new.id),
                catbox_url=None,
                supabase_url=None,
                supabase_path=shared_path,
                poster_hash="new-hash",
            )
        )
        await session.commit()

    deleted = await main_part2.cleanup_old_events(db, now_utc=now_utc)
    assert deleted == 1

    async with db.raw_conn() as conn:
        cur = await conn.execute("SELECT bucket, path FROM supabase_delete_queue ORDER BY id ASC")
        rows = await cur.fetchall()

    # Shared media path is still referenced by the remaining event => do not queue a delete.
    assert rows == []

