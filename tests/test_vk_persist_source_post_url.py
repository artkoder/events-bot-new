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
