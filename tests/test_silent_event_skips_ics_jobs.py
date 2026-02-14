import pytest
from sqlmodel import select

from main import Database, Event, JobOutbox, JobTask, schedule_event_update_tasks


@pytest.mark.asyncio
async def test_silent_event_does_not_schedule_ics_publish_or_tg_ics_post(tmp_path):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    ev = Event(
        title="Silent",
        description="d",
        source_text="s",
        date="2026-02-15",
        time="19:00",
        location_name="loc",
        silent=True,
    )
    async with db.get_session() as session:
        session.add(ev)
        await session.commit()
        await session.refresh(ev)

    await schedule_event_update_tasks(db, ev, drain_nav=False)

    async with db.get_session() as session:
        rows = (await session.execute(select(JobOutbox.task).where(JobOutbox.event_id == ev.id))).all()
    tasks = {r[0] for r in rows}
    assert JobTask.ics_publish not in tasks
    assert JobTask.tg_ics_post not in tasks
    assert JobTask.telegraph_build in tasks
