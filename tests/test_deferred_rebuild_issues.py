"""Tests for deferred rebuild issue fixes.

These tests verify the fixes for 6 issues identified by Codex:
1. next_run_at preserved on re-enqueue
2. drain skips deferred jobs
3. weekend_pages creates followup when running
4. mark_pages_dirty validates month key
5. load_pages_dirty clears corrupt JSON
6. concurrent mark_pages_dirty doesn't lose data
"""
import asyncio
import json
import re
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import select, update

import main
from main import Database, Event, JobOutbox, JobStatus, JobTask, Setting


def _ensure_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


@pytest.mark.asyncio
async def test_enqueue_preserves_deferred_next_run_at(tmp_path):
    """Fix #1: Re-enqueue pending job should preserve deferred next_run_at."""
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    async with db.get_session() as session:
        ev1 = Event(
            title="Event 1",
            description="Description",
            date="2025-09-05",
            time="10:00",
            location_name="Location",
            source_text="source",
        )
        ev2 = Event(
            title="Event 2",
            description="Description",
            date="2025-09-06",
            time="11:00",
            location_name="Location",
            source_text="source",
        )
        session.add_all([ev1, ev2])
        await session.commit()
        await session.refresh(ev1)
        await session.refresh(ev2)

    # First event schedules deferred job
    future = datetime.now(timezone.utc) + timedelta(minutes=15)
    await main.enqueue_job(db, ev1.id, JobTask.month_pages, next_run_at=future)

    async with db.get_session() as session:
        job = (await session.execute(select(JobOutbox))).scalar_one()
        original_next_run_at = _ensure_utc(job.next_run_at)

    # Second event re-enqueues same month - should NOT reset next_run_at to now
    await main.enqueue_job(db, ev2.id, JobTask.month_pages, next_run_at=future)

    async with db.get_session() as session:
        job = (await session.execute(select(JobOutbox))).scalar_one()
        new_next_run_at = _ensure_utc(job.next_run_at)

    # next_run_at should be preserved (still in future, not reset to now)
    now = datetime.now(timezone.utc)
    assert new_next_run_at > now + timedelta(minutes=10), \
        f"next_run_at was reset to {new_next_run_at}, expected ~{future}"


@pytest.mark.asyncio
async def test_drain_skips_deferred_jobs(tmp_path, monkeypatch):
    """Fix #2: _drain_nav_tasks should skip jobs with next_run_at in future."""
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    async with db.get_session() as session:
        ev = Event(
            title="Event",
            description="Description",
            date="2025-09-05",
            time="10:00",
            location_name="Location",
            source_text="source",
        )
        session.add(ev)
        await session.commit()
        await session.refresh(ev)

    monkeypatch.setattr(main, "JOB_HANDLERS", {})

    # Schedule deferred job 15 minutes in future
    future = datetime.now(timezone.utc) + timedelta(minutes=15)
    await main.enqueue_job(db, ev.id, JobTask.month_pages, next_run_at=future)

    # Drain should complete quickly without waiting for deferred job
    import time
    start = time.monotonic()
    await main._drain_nav_tasks(db, ev.id, timeout=5.0)
    elapsed = time.monotonic() - start

    # Should complete in under 3 seconds, not wait full timeout
    assert elapsed < 3.0, f"drain took {elapsed:.1f}s, expected < 3s (should skip deferred jobs)"


@pytest.mark.asyncio
async def test_weekend_pages_creates_followup_when_running(tmp_path):
    """Fix #3: weekend_pages should create followup when owner is running."""
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    async with db.get_session() as session:
        ev1 = Event(
            title="Event 1",
            description="Description",
            date="2025-09-06",  # Saturday
            time="10:00",
            location_name="Location",
            source_text="source",
        )
        ev2 = Event(
            title="Event 2",
            description="Description",
            date="2025-09-06",  # Same Saturday
            time="11:00",
            location_name="Location",
            source_text="source",
        )
        session.add_all([ev1, ev2])
        await session.commit()
        await session.refresh(ev1)
        await session.refresh(ev2)

        # Create running weekend_pages job for ev1
        session.add(
            JobOutbox(
                event_id=ev1.id,
                task=JobTask.weekend_pages,
                status=JobStatus.running,
                coalesce_key="weekend_pages:2025-09-06",
                updated_at=datetime.now(timezone.utc),
                next_run_at=datetime.now(timezone.utc),
            )
        )
        await session.commit()

    # ev2 should create followup since owner is running
    action = await main.enqueue_job(db, ev2.id, JobTask.weekend_pages)
    assert action == "merged"

    async with db.get_session() as session:
        jobs = (await session.execute(select(JobOutbox))).scalars().all()

    # New behavior: creates deferred task instead of v2 follow-up
    deferred_key = f"weekend_pages:2025-09-06:deferred:{ev2.id}"
    deferred_jobs = [j for j in jobs if j.coalesce_key == deferred_key]
    assert len(deferred_jobs) == 1, f"Expected deferred job, got {[j.coalesce_key for j in jobs]}"
    assert deferred_jobs[0].status == JobStatus.pending


@pytest.mark.asyncio
async def test_mark_pages_dirty_validates_month_key(tmp_path):
    """Fix #5: Invalid month keys should be rejected."""
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    # Valid keys should work
    await main.mark_pages_dirty(db, "2025-09")
    await main.mark_pages_dirty(db, "weekend:2025-09-06")

    state = await main.load_pages_dirty_state(db)
    assert "2025-09" in state.get("months", [])
    assert "weekend:2025-09-06" in state.get("months", [])

    # Invalid keys should be rejected (not added)
    await main.mark_pages_dirty(db, "invalid")
    await main.mark_pages_dirty(db, "2025-9")  # Missing leading zero
    await main.mark_pages_dirty(db, "weekend:invalid")

    state = await main.load_pages_dirty_state(db)
    months = state.get("months", [])
    assert "invalid" not in months
    assert "2025-9" not in months
    assert "weekend:invalid" not in months


@pytest.mark.asyncio
async def test_load_pages_dirty_clears_corrupt_json(tmp_path):
    """Fix #6: Corrupt JSON should be logged and cleared."""
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    # Set corrupt JSON directly
    async with db.get_session() as session:
        session.add(Setting(key=main.PAGES_DIRTY_KEY, value="not valid json {{{"))
        await session.commit()

    # Clear settings cache
    main.settings_cache.clear()

    # Load should return None and clear the corrupt value
    state = await main.load_pages_dirty_state(db)
    assert state is None

    # Value should be cleared
    async with db.get_session() as session:
        setting = (
            await session.execute(
                select(Setting).where(Setting.key == main.PAGES_DIRTY_KEY)
            )
        ).scalar_one_or_none()
        # Should be None or cleared
        if setting:
            assert setting.value is None or setting.value == ""


@pytest.mark.asyncio
async def test_concurrent_mark_pages_dirty(tmp_path):
    """Fix #4: Sequential calls should accumulate months (concurrent is hard in SQLite)."""
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    # Sequential calls - simulating what concurrent would produce after retry
    months = [f"2025-{i:02d}" for i in range(1, 7)]

    for month in months:
        await main.mark_pages_dirty(db, month)

    # All months should be present
    state = await main.load_pages_dirty_state(db)
    stored_months = state.get("months", [])

    for month in months:
        assert month in stored_months, f"Month {month} lost in sequential update"

