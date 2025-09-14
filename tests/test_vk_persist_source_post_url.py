import pytest
import main
import vk_intake
from main import Database
from models import Event, JobTask


@pytest.mark.asyncio
async def test_persist_event_and_pages_sets_source_post_url_and_skips_vk_sync(tmp_path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    tasks = []

    async def fake_enqueue_job(db_, eid, task, depends_on=None, coalesce_key=None):
        tasks.append(task)
        return "job"

    monkeypatch.setattr(main, "enqueue_job", fake_enqueue_job)

    draft = vk_intake.EventDraft(title="T", date="2025-09-02", time="10:00", source_text="T")
    res = await vk_intake.persist_event_and_pages(
        draft, [], db, source_post_url="https://vk.com/wall-1_2"
    )

    async with db.get_session() as session:
        ev = await session.get(Event, res.event_id)
    assert ev.source_post_url == "https://vk.com/wall-1_2"
    assert JobTask.vk_sync not in tasks


@pytest.mark.asyncio
async def test_schedule_event_update_tasks_enqueues_and_runs_vk_sync(tmp_path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    tasks = []

    async def fake_enqueue_job(db_, eid, task, depends_on=None, coalesce_key=None):
        tasks.append(task)
        return "job"

    async def fake_sync_vk_source_post(ev, text, db_, bot, ics_url=None):
        return "https://vk.com/wall-1_1"

    monkeypatch.setattr(main, "enqueue_job", fake_enqueue_job)
    monkeypatch.setattr(main, "sync_vk_source_post", fake_sync_vk_source_post)

    ev = Event(
        title="T",
        description="",
        festival=None,
        date="2025-09-02",
        time="10:00",
        location_name="",
        source_text="T",
        source_post_url="http://example.com",
    )

    async with db.get_session() as session:
        saved, _ = await main.upsert_event(session, ev)

    await main.schedule_event_update_tasks(db, saved)

    assert JobTask.vk_sync in tasks

    await main.job_sync_vk_source_post(saved.id, db, None)

    async with db.get_session() as session:
        updated = await session.get(Event, saved.id)
    assert updated.source_vk_post_url == "https://vk.com/wall-1_1"
