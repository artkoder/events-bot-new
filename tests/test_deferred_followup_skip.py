"""Test that follow-up is skipped when deferred task exists."""
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import select

import main
from main import Database, Event, JobOutbox, JobStatus, JobTask


@pytest.mark.asyncio
async def test_drain_nav_skips_followup_when_deferred_exists(tmp_path, monkeypatch):
    """When a deferred task exists for event_id, don't create follow-up."""
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

        now = datetime.now(timezone.utc)
        deferred_time = now + timedelta(minutes=15)
        
        # Create a running job for ev1
        session.add(
            JobOutbox(
                event_id=ev1.id,
                task=JobTask.month_pages,
                status=JobStatus.running,
                coalesce_key="month_pages:2025-09",
                updated_at=now,
                next_run_at=now,
            )
        )
        # Create a pending deferred job for ev2 (the one we're adding)
        session.add(
            JobOutbox(
                event_id=ev2.id,
                task=JobTask.month_pages,
                status=JobStatus.pending,
                coalesce_key="month_pages:2025-09:deferred:2",
                updated_at=now,
                next_run_at=deferred_time,  # Deferred to future
            )
        )
        await session.commit()

    # Disable handlers
    monkeypatch.setattr(main, "JOB_HANDLERS", {})
    
    # Simulate enqueue for ev2 into running job
    action = await main.enqueue_job(db, ev2.id, JobTask.month_pages)
    assert action == "merged"

    async with db.get_session() as session:
        jobs = (await session.execute(select(JobOutbox))).scalars().all()

    # Should NOT have created a v2 follow-up because deferred task exists
    v2_jobs = [j for j in jobs if ":v2:" in (j.coalesce_key or "")]
    assert len(v2_jobs) == 0, "Follow-up should not be created when deferred task exists"

    # Deferred job should still exist
    deferred_jobs = [j for j in jobs if "deferred" in (j.coalesce_key or "")]
    assert len(deferred_jobs) >= 1, "Deferred task should still exist"
