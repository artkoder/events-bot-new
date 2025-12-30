"""Tests for deferred page rebuilds functionality."""
import pytest
from datetime import datetime, timedelta, timezone
from sqlalchemy import select

import main
from main import Database, Event, JobOutbox, JobTask, JobStatus


def _ensure_utc(dt: datetime) -> datetime:
    """Ensure datetime is timezone-aware (UTC)."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


@pytest.mark.asyncio
async def test_enqueue_job_with_next_run_at(tmp_path):
    """Test that enqueue_job respects the next_run_at parameter."""
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    
    async with db.get_session() as session:
        ev = Event(
            title="Test Event",
            description="Description",
            date="2025-09-05",
            time="12:00",
            location_name="Test Location",
            source_text="source",
        )
        session.add(ev)
        await session.commit()
        await session.refresh(ev)
    
    # Schedule job with future next_run_at
    future = datetime.now(timezone.utc) + timedelta(minutes=15)
    action = await main.enqueue_job(
        db, ev.id, JobTask.month_pages, next_run_at=future
    )
    
    assert action == "new"
    
    async with db.get_session() as session:
        jobs = (await session.execute(select(JobOutbox))).scalars().all()
        assert len(jobs) == 1
        job = jobs[0]
        assert job.task == JobTask.month_pages
        assert job.status == JobStatus.pending
        # next_run_at should be approximately the future time
        job_run_at = _ensure_utc(job.next_run_at)
        delta = abs((job_run_at - future).total_seconds())
        assert delta < 5  # within 5 seconds


@pytest.mark.asyncio
async def test_enqueue_job_defaults_to_now(tmp_path):
    """Test that enqueue_job defaults to now when next_run_at is not provided."""
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    
    async with db.get_session() as session:
        ev = Event(
            title="Test Event",
            description="Description",
            date="2025-09-05",
            time="12:00",
            location_name="Test Location",
            source_text="source",
        )
        session.add(ev)
        await session.commit()
        await session.refresh(ev)
    
    now_before = datetime.now(timezone.utc)
    await main.enqueue_job(db, ev.id, JobTask.telegraph_build)
    now_after = datetime.now(timezone.utc)
    
    async with db.get_session() as session:
        jobs = (await session.execute(select(JobOutbox))).scalars().all()
        job = jobs[0]
        job_run_at = _ensure_utc(job.next_run_at)
        # next_run_at should be between now_before and now_after
        assert now_before <= job_run_at <= now_after + timedelta(seconds=1)


@pytest.mark.asyncio
async def test_schedule_event_update_tasks_defers_page_tasks(tmp_path, monkeypatch):
    """Test that schedule_event_update_tasks defers month_pages and weekend_pages."""
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    
    # Create event on a Saturday for weekend_pages
    async with db.get_session() as session:
        ev = Event(
            title="Weekend Event",
            description="Description",
            date="2025-09-06",  # Saturday
            time="14:00",
            location_name="Location",
            source_text="source",
        )
        session.add(ev)
        await session.commit()
        await session.refresh(ev)
    
    # Mock JOB_HANDLERS to avoid actual execution
    monkeypatch.setattr(main, "JOB_HANDLERS", {
        "ics_publish": lambda *a: True,
        "telegraph_build": lambda *a: True,
    })
    
    now_before = datetime.now(timezone.utc)
    await main.schedule_event_update_tasks(db, ev, drain_nav=False)
    now_after = datetime.now(timezone.utc)
    
    async with db.get_session() as session:
        jobs = (await session.execute(select(JobOutbox))).scalars().all()
    
    job_map = {j.task: j for j in jobs}
    
    # telegraph_build should run immediately
    assert JobTask.telegraph_build in job_map
    telegraph_job = job_map[JobTask.telegraph_build]
    telegraph_run_at = _ensure_utc(telegraph_job.next_run_at)
    assert now_before <= telegraph_run_at <= now_after + timedelta(seconds=1)
    
    # month_pages should be deferred ~15 minutes
    assert JobTask.month_pages in job_map
    month_job = job_map[JobTask.month_pages]
    month_run_at = _ensure_utc(month_job.next_run_at)
    expected_defer = now_before + timedelta(minutes=15)
    delta = abs((month_run_at - expected_defer).total_seconds())
    assert delta < 5  # within 5 seconds
    
    # weekend_pages should be deferred ~15 minutes
    assert JobTask.weekend_pages in job_map
    weekend_job = job_map[JobTask.weekend_pages]
    weekend_run_at = _ensure_utc(weekend_job.next_run_at)
    delta = abs((weekend_run_at - expected_defer).total_seconds())
    assert delta < 5  # within 5 seconds


@pytest.mark.asyncio
async def test_mark_pages_dirty_called_on_schedule(tmp_path, monkeypatch):
    """Test that mark_pages_dirty is called when scheduling deferred tasks."""
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    
    async with db.get_session() as session:
        ev = Event(
            title="Test Event",
            description="Description",
            date="2025-09-06",  # Saturday
            time="14:00",
            location_name="Location",
            source_text="source",
        )
        session.add(ev)
        await session.commit()
        await session.refresh(ev)
    
    monkeypatch.setattr(main, "JOB_HANDLERS", {})
    
    await main.schedule_event_update_tasks(db, ev, drain_nav=False)
    
    # Check dirty state was set
    state = await main.load_pages_dirty_state(db)
    assert state is not None
    assert "2025-09" in state.get("months", [])
    assert "weekend:2025-09-06" in state.get("months", [])
