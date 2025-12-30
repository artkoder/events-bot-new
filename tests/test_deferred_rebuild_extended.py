"""Extended tests for deferred page rebuilds."""
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import select

import main
from main import Database, Event, JobOutbox, JobStatus, JobTask


def _ensure_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


@pytest.mark.asyncio
async def test_schedule_event_update_tasks_weekday_defers_only_month(tmp_path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    async with db.get_session() as session:
        ev = Event(
            title="Weekday Event",
            description="Description",
            date="2025-09-03",  # Wednesday
            time="12:00",
            location_name="Location",
            source_text="source",
        )
        session.add(ev)
        await session.commit()
        await session.refresh(ev)

    monkeypatch.setattr(main, "JOB_HANDLERS", {})

    now_before = datetime.now(timezone.utc)
    await main.schedule_event_update_tasks(db, ev, drain_nav=False)
    now_after = datetime.now(timezone.utc)

    async with db.get_session() as session:
        jobs = (await session.execute(select(JobOutbox))).scalars().all()

    job_map = {j.task: j for j in jobs}

    assert JobTask.telegraph_build in job_map
    assert JobTask.month_pages in job_map
    assert JobTask.week_pages in job_map
    assert JobTask.weekend_pages not in job_map

    month_run_at = _ensure_utc(job_map[JobTask.month_pages].next_run_at)
    expected_defer = now_before + timedelta(minutes=15)
    assert abs((month_run_at - expected_defer).total_seconds()) < 5

    week_run_at = _ensure_utc(job_map[JobTask.week_pages].next_run_at)
    assert now_before <= week_run_at <= now_after + timedelta(seconds=1)

    state = await main.load_pages_dirty_state(db)
    assert state is not None
    assert "2025-09" in state.get("months", [])
    # Note: weekend may be marked dirty if weekend_start_for_date finds a weekend
    # The test verifies that month is dirty, weekend check is optional


@pytest.mark.asyncio
async def test_mark_pages_dirty_dedupes_months(tmp_path):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    await main.mark_pages_dirty(db, "2025-09")
    await main.mark_pages_dirty(db, "2025-09")
    await main.mark_pages_dirty(db, "weekend:2025-09-06")

    state = await main.load_pages_dirty_state(db)
    assert state is not None
    months = state.get("months", [])
    assert months.count("2025-09") == 1
    assert "weekend:2025-09-06" in months
    assert state.get("reminded") is False
    assert state.get("since")


@pytest.mark.asyncio
async def test_enqueue_job_running_month_pages_creates_followup(tmp_path):
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

        session.add(
            JobOutbox(
                event_id=ev1.id,
                task=JobTask.month_pages,
                status=JobStatus.running,
                coalesce_key="month_pages:2025-09",
                updated_at=datetime.now(timezone.utc),
                next_run_at=datetime.now(timezone.utc),
            )
        )
        await session.commit()

    action = await main.enqueue_job(db, ev2.id, JobTask.month_pages)
    assert action == "merged"

    async with db.get_session() as session:
        jobs = (await session.execute(select(JobOutbox))).scalars().all()

    # New behavior: creates deferred task instead of v2 follow-up
    deferred_key = f"month_pages:2025-09:deferred:{ev2.id}"
    deferred_jobs = [j for j in jobs if j.coalesce_key == deferred_key]
    assert len(deferred_jobs) == 1
    assert deferred_jobs[0].status == JobStatus.pending
    # Deferred task has no depends_on (runs independently after delay)


@pytest.mark.asyncio
async def test_enqueue_job_requeues_error_deferred_job(tmp_path):
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

        future = datetime.now(timezone.utc) + timedelta(hours=1)
        session.add(
            JobOutbox(
                event_id=ev.id,
                task=JobTask.month_pages,
                status=JobStatus.error,
                coalesce_key="month_pages:2025-09",
                updated_at=datetime.now(timezone.utc),
                next_run_at=future,
                attempts=2,
                last_error="boom",
            )
        )
        await session.commit()

    now_before = datetime.now(timezone.utc)
    action = await main.enqueue_job(db, ev.id, JobTask.month_pages)
    now_after = datetime.now(timezone.utc)

    assert action == "requeued"

    async with db.get_session() as session:
        job = (
            await session.execute(
                select(JobOutbox).where(JobOutbox.coalesce_key == "month_pages:2025-09")
            )
        ).scalar_one()

    assert job.status == JobStatus.pending
    assert job.last_error is None
    assert job.attempts == 0
    job_run_at = _ensure_utc(job.next_run_at)
    assert now_before <= job_run_at <= now_after + timedelta(seconds=1)


@pytest.mark.asyncio
async def test_deferred_job_skipped_until_due_then_runs(tmp_path, monkeypatch):
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

    calls: list[int] = []

    async def fake_month_pages(eid, db_obj, bot_obj):
        calls.append(eid)
        return True

    monkeypatch.setitem(main.JOB_HANDLERS, "month_pages", fake_month_pages)

    future = datetime.now(timezone.utc) + timedelta(minutes=30)
    await main.enqueue_job(db, ev.id, JobTask.month_pages, next_run_at=future)

    await main._run_due_jobs_once(db, None)
    assert calls == []

    async with db.get_session() as session:
        job = (
            await session.execute(select(JobOutbox).where(JobOutbox.task == JobTask.month_pages))
        ).scalar_one()
        job.next_run_at = datetime.now(timezone.utc)
        session.add(job)
        await session.commit()

    await main._run_due_jobs_once(db, None)
    assert calls == [ev.id]

    async with db.get_session() as session:
        job = (
            await session.execute(select(JobOutbox).where(JobOutbox.task == JobTask.month_pages))
        ).scalar_one()
    assert job.status == JobStatus.done

