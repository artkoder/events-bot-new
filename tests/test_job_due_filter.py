import pytest
from datetime import datetime, timedelta
from sqlmodel import select

import main
from main import Database, Event, JobOutbox, JobTask, JobStatus


@pytest.mark.asyncio
async def test_future_job_does_not_block_month_pages(tmp_path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    async with db.get_session() as session:
        ev = Event(
            title="t",
            description="d",
            date="2025-09-05",
            time="12:00",
            location_name="loc",
            source_text="src",
        )
        session.add(ev)
        await session.commit()
        await session.refresh(ev)
        future = datetime.utcnow() + timedelta(hours=1)
        session.add(
            JobOutbox(
                event_id=ev.id,
                task=JobTask.vk_sync,
                status=JobStatus.pending,
                next_run_at=future,
            )
        )
        session.add(
            JobOutbox(
                event_id=ev.id,
                task=JobTask.month_pages,
                status=JobStatus.pending,
                next_run_at=datetime.utcnow(),
            )
        )
        await session.commit()

    calls: list[int] = []

    async def fake_month_pages(eid, db_obj, bot_obj):
        calls.append(eid)
        return True

    monkeypatch.setitem(main.JOB_HANDLERS, "month_pages", fake_month_pages)

    await main._run_due_jobs_once(db, None)

    assert calls == [ev.id]

    async with db.get_session() as session:
        jobs = (await session.execute(select(JobOutbox))).scalars().all()
    statuses = {j.task: j.status for j in jobs}
    assert statuses[JobTask.month_pages] == JobStatus.done
    assert statuses[JobTask.vk_sync] == JobStatus.pending
