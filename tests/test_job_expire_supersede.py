import pytest
from datetime import datetime, timedelta
from sqlmodel import select

import main
from main import Database, Event, JobOutbox, JobTask, JobStatus


@pytest.mark.asyncio
async def test_job_expired(tmp_path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    async with db.get_session() as session:
        ev = Event(title="t", description="d", date="2025-09-05", time="12:00", location_name="loc", source_text="src")
        session.add(ev)
        await session.commit()
        await session.refresh(ev)
        session.add(
            JobOutbox(
                event_id=ev.id,
                task=JobTask.month_pages,
                status=JobStatus.pending,
                updated_at=datetime.utcnow() - timedelta(minutes=15),
                next_run_at=datetime.utcnow() - timedelta(minutes=15),
                coalesce_key="month_pages:2025-09",
            )
        )
        await session.commit()
    calls = []
    async def fake_month_pages(eid, db_obj, bot_obj):
        calls.append(eid)
        return True
    monkeypatch.setitem(main.JOB_HANDLERS, "month_pages", fake_month_pages)
    processed = await main._run_due_jobs_once(db, None)
    assert processed == 0
    assert calls == []
    async with db.get_session() as session:
        job = (await session.execute(select(JobOutbox))).scalar_one()
        assert job.status == JobStatus.error
        assert job.last_error == "expired"


@pytest.mark.asyncio
async def test_job_superseded(tmp_path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    async with db.get_session() as session:
        ev1 = Event(title="a", description="d", date="2025-09-05", time="10:00", location_name="x", source_text="s")
        ev2 = Event(title="b", description="d", date="2025-09-06", time="10:00", location_name="x", source_text="s")
        session.add_all([ev1, ev2])
        await session.commit()
        await session.refresh(ev1)
        await session.refresh(ev2)
        now = datetime.utcnow()
        session.add_all([
            JobOutbox(
                event_id=ev1.id,
                task=JobTask.month_pages,
                status=JobStatus.pending,
                updated_at=now,
                next_run_at=now,
                coalesce_key="month_pages:2025-09",
            ),
            JobOutbox(
                event_id=ev2.id,
                task=JobTask.month_pages,
                status=JobStatus.pending,
                updated_at=now,
                next_run_at=now,
                coalesce_key="month_pages:2025-09",
            ),
        ])
        await session.commit()
    calls = []
    async def fake_month_pages(eid, db_obj, bot_obj):
        calls.append(eid)
        return True
    monkeypatch.setitem(main.JOB_HANDLERS, "month_pages", fake_month_pages)
    processed = await main._run_due_jobs_once(db, None)
    assert processed == 1
    assert calls == [ev2.id]
    async with db.get_session() as session:
        jobs = (await session.execute(select(JobOutbox).order_by(JobOutbox.id))).scalars().all()
        assert jobs[0].status == JobStatus.error
        assert jobs[0].last_error == "superseded"
        assert jobs[1].status == JobStatus.done
