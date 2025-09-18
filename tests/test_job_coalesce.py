import pytest
from datetime import datetime, timezone
from sqlalchemy import select

import main
from main import Database, Event, JobOutbox, JobTask, JobStatus


@pytest.mark.asyncio
async def test_coalesce_requeue(tmp_path):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    async with db.get_session() as session:
        ev1 = Event(title="a", description="d", date="2025-09-05", time="10:00", location_name="x", source_text="s")
        ev2 = Event(title="b", description="d", date="2025-09-06", time="10:00", location_name="x", source_text="s")
        session.add_all([ev1, ev2])
        await session.commit()
        await session.refresh(ev1)
        await session.refresh(ev2)
        session.add(
            JobOutbox(
                event_id=ev1.id,
                task=JobTask.month_pages,
                status=JobStatus.done,
                coalesce_key="month_pages:2025-09",
                updated_at=datetime.now(timezone.utc),
                next_run_at=datetime.now(timezone.utc),
            )
        )
        await session.commit()
    action = await main.enqueue_job(db, ev2.id, JobTask.month_pages)
    assert action == "requeued"
    async with db.get_session() as session:
        jobs = (await session.execute(select(JobOutbox))).scalars().all()
        assert len(jobs) == 1
        job = jobs[0]
        assert job.status == JobStatus.pending
        assert job.coalesce_key == "month_pages:2025-09"


@pytest.mark.asyncio
async def test_coalesce_pending_dedup(tmp_path):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    async with db.get_session() as session:
        ev1 = Event(title="a", description="d", date="2025-09-05", time="10:00", location_name="x", source_text="s")
        ev2 = Event(title="b", description="d", date="2025-09-06", time="10:00", location_name="x", source_text="s")
        session.add_all([ev1, ev2])
        await session.commit()
        await session.refresh(ev1)
        await session.refresh(ev2)
    await main.enqueue_job(db, ev1.id, JobTask.month_pages)
    await main.enqueue_job(db, ev2.id, JobTask.month_pages)
    async with db.get_session() as session:
        jobs = (await session.execute(select(JobOutbox))).scalars().all()
        assert len(jobs) == 1
        job = jobs[0]
        assert job.status == JobStatus.pending
        assert job.coalesce_key == "month_pages:2025-09"
