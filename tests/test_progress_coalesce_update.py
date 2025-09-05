import pytest
from types import SimpleNamespace

import main
from main import Database, Event, JobTask, JobStatus


@pytest.mark.asyncio
async def test_progress_updated_for_merged_jobs(tmp_path, monkeypatch):
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

    key = "month_pages:2025-09"
    updates = []

    async def follower_updater(task, eid, status, changed, link, err):
        updates.append((task, eid, status))

    main._EVENT_PROGRESS.clear()
    main._EVENT_PROGRESS_KEYS.clear()
    main._EVENT_PROGRESS[ev2.id] = SimpleNamespace(
        updater=follower_updater, ics_progress=None, fest_progress=None, keys=[key]
    )
    main._EVENT_PROGRESS_KEYS.setdefault(key, set()).add(ev2.id)

    async def fake_month_pages(eid, db_obj, bot_obj):
        return True

    monkeypatch.setitem(main.JOB_HANDLERS, "month_pages", fake_month_pages)

    await main._run_due_jobs_once(db, None)

    assert updates == [(JobTask.month_pages, ev2.id, JobStatus.done)]
