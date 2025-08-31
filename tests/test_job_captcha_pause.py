import pytest
from sqlmodel import select

import main
from main import Database, Event, JobOutbox, JobTask, JobStatus


@pytest.mark.asyncio
async def test_vk_jobs_paused_and_resumed(tmp_path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    ev = Event(
        title="t",
        description="d",
        date="2025-01-04",
        time="12:00",
        location_name="loc",
        source_text="src",
    )
    async with db.get_session() as session:
        session.add(ev)
        await session.commit()
        await session.refresh(ev)
        session.add(JobOutbox(event_id=ev.id, task=JobTask.vk_sync))
        session.add(JobOutbox(event_id=ev.id, task=JobTask.week_pages))
        await session.commit()

    calls: list[str] = []

    async def fake_vk_job(event_id, db, bot):
        calls.append("call")
        if len(calls) == 1:
            raise main.VKAPIError(14, "Captcha needed")
        return True

    monkeypatch.setitem(main.JOB_HANDLERS, "vk_sync", fake_vk_job)
    monkeypatch.setitem(main.JOB_HANDLERS, "week_pages", fake_vk_job)

    await main._run_due_jobs_once(db, None)

    async with db.get_session() as session:
        jobs = (await session.execute(select(JobOutbox))).scalars().all()
    assert all(j.status == JobStatus.paused for j in jobs)
    assert all(j.attempts == 0 for j in jobs)
    assert len(calls) == 1

    resume = main._vk_captcha_resume
    assert resume is not None
    await resume()
    main._vk_captcha_resume = None

    async with db.get_session() as session:
        jobs = (await session.execute(select(JobOutbox))).scalars().all()
        assert all(j.status == JobStatus.pending for j in jobs)

    await main._run_due_jobs_once(db, None)

    async with db.get_session() as session:
        jobs = (await session.execute(select(JobOutbox))).scalars().all()
    assert all(j.status == JobStatus.done for j in jobs)
    assert len(calls) == 3
