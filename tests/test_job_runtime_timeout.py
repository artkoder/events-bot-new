import asyncio
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import select

import main
from main import Database, Event, JobOutbox, JobStatus, JobTask


@pytest.mark.asyncio
async def test_job_handler_timeout_marks_error(tmp_path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    async with db.get_session() as session:
        ev = Event(
            title="t",
            description="d",
            date="2025-09-05",
            time="10:00",
            location_name="x",
            source_text="s",
        )
        session.add(ev)
        await session.commit()
        await session.refresh(ev)
        session.add(
            JobOutbox(
                event_id=ev.id,
                task=JobTask.telegraph_build,
                status=JobStatus.pending,
                updated_at=datetime.now(timezone.utc),
                next_run_at=datetime.now(timezone.utc) - timedelta(seconds=1),
            )
        )
        await session.commit()

    async def slow_handler(event_id, db_obj, bot_obj):
        await asyncio.sleep(5)
        return True

    monkeypatch.setitem(main.JOB_HANDLERS, "telegraph_build", slow_handler)
    monkeypatch.setitem(main.JOB_MAX_RUNTIME, JobTask.telegraph_build, 0.1)

    processed = await main._run_due_jobs_once(db, None)
    assert processed == 1

    async with db.get_session() as session:
        job = (await session.execute(select(JobOutbox))).scalar_one()
        assert job.status == JobStatus.error
        assert job.last_error and "timeout" in job.last_error

