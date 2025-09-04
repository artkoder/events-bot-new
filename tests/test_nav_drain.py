import pytest
from datetime import datetime
from sqlalchemy import select

import main
from main import Database, Event, JobOutbox, JobTask, JobStatus


@pytest.mark.asyncio
async def test_drain_coalesced_month_task(tmp_path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    async with db.get_session() as session:
        ev1 = Event(title="a", description="d", date="2025-09-04", time="10:00", location_name="x", source_text="s")
        ev2 = Event(title="b", description="d", date="2025-09-05", time="11:00", location_name="x", source_text="s")
        session.add_all([ev1, ev2])
        await session.commit()
        await session.refresh(ev1)
        await session.refresh(ev2)
        session.add(
            JobOutbox(
                event_id=ev1.id,
                task=JobTask.month_pages,
                status=JobStatus.pending,
                coalesce_key="month_pages:2025-09",
                updated_at=datetime.utcnow(),
                next_run_at=datetime.utcnow(),
            )
        )
        await session.commit()

    processed: list[int] = []

    async def fake_month_pages(event_id: int, db: Database, bot):
        processed.append(event_id)
        return True

    monkeypatch.setitem(main.JOB_HANDLERS, "month_pages", fake_month_pages)

    await main._drain_nav_tasks(db, ev2.id, timeout=1.0)

    assert processed == [ev1.id, ev2.id]
    async with db.get_session() as session:
        jobs = (
            await session.execute(select(JobOutbox).order_by(JobOutbox.id))
        ).scalars().all()
        assert {j.event_id: j.status for j in jobs} == {
            ev1.id: JobStatus.done,
            ev2.id: JobStatus.done,
        }


@pytest.mark.asyncio
async def test_drain_coalesced_weekend_task(tmp_path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    async with db.get_session() as session:
        ev1 = Event(title="a", description="d", date="2025-09-06", time="10:00", location_name="x", source_text="s")
        ev2 = Event(title="b", description="d", date="2025-09-07", time="11:00", location_name="x", source_text="s")
        session.add_all([ev1, ev2])
        await session.commit()
        await session.refresh(ev1)
        await session.refresh(ev2)
        session.add(
            JobOutbox(
                event_id=ev1.id,
                task=JobTask.weekend_pages,
                status=JobStatus.pending,
                coalesce_key="weekend_pages:2025-09-06",
                updated_at=datetime.utcnow(),
                next_run_at=datetime.utcnow(),
            )
        )
        await session.commit()

    processed: list[int] = []

    async def fake_weekend_pages(event_id: int, db: Database, bot):
        processed.append(event_id)
        return True

    monkeypatch.setitem(main.JOB_HANDLERS, "weekend_pages", fake_weekend_pages)

    await main._drain_nav_tasks(db, ev2.id, timeout=1.0)

    assert processed == [ev1.id, ev2.id]
    async with db.get_session() as session:
        jobs = (
            await session.execute(select(JobOutbox).order_by(JobOutbox.id))
        ).scalars().all()
        assert {j.event_id: j.status for j in jobs} == {
            ev1.id: JobStatus.done,
            ev2.id: JobStatus.done,
        }
