import pytest
from sqlmodel import select

import main
from main import Database, Event, JobOutbox, JobTask, JobStatus


@pytest.mark.asyncio
async def test_enqueue_ics_dedup(tmp_path):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    async with db.get_session() as session:
        ev = Event(title="a", description="d", date="2025-09-05", time="10:00", location_name="x", source_text="s")
        session.add(ev)
        await session.commit()
        await session.refresh(ev)

    await main.enqueue_job(db, ev.id, JobTask.ics_publish)
    await main.enqueue_job(db, ev.id, JobTask.ics_publish)

    async with db.get_session() as session:
        jobs = (
            await session.execute(select(JobOutbox).where(JobOutbox.task == JobTask.ics_publish))
        ).scalars().all()
    assert len(jobs) == 1
    assert jobs[0].status == JobStatus.pending
