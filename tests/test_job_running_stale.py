import pytest
from datetime import datetime, timedelta
from sqlalchemy import select

import main
from main import Database, Event, JobOutbox, JobTask, JobStatus


@pytest.mark.asyncio
async def test_running_stale_marked_and_replaced(tmp_path):
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
                status=JobStatus.running,
                coalesce_key="month_pages:2025-09",
                updated_at=datetime.utcnow() - timedelta(minutes=15),
                next_run_at=datetime.utcnow() - timedelta(minutes=15),
            )
        )
        await session.commit()
    action = await main.enqueue_job(db, ev2.id, JobTask.month_pages)
    assert action == "new"
    async with db.get_session() as session:
        jobs = (await session.execute(select(JobOutbox))).scalars().all()
        assert len(jobs) == 2
        statuses = {job.status for job in jobs}
        assert JobStatus.error in statuses
        assert JobStatus.pending in statuses
        pend = next(j for j in jobs if j.status == JobStatus.pending)
        assert pend.event_id == ev2.id
