import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pytest
from types import SimpleNamespace

import main
import vk_intake
import vk_review
from main import Database
from models import Event, JobTask


class DummyBot:
    def __init__(self):
        self.messages = []

    async def send_message(self, chat_id, text, **kwargs):
        self.messages.append(SimpleNamespace(chat_id=chat_id, text=text))
        return SimpleNamespace(message_id=1)

    async def send_media_group(self, chat_id, media):
        pass


@pytest.mark.asyncio
async def test_vkrev_import_flow_persists_url_and_skips_vk_sync(tmp_path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    async with db.raw_conn() as conn:
        await conn.execute(
            "INSERT INTO vk_source(group_id, screen_name, name, location, default_time) VALUES(?,?,?,?,?)",
            (1, "club1", "Test Community", "", None),
        )
        await conn.execute(
            "INSERT INTO vk_inbox(id, group_id, post_id, date, text, matched_kw, has_date, event_ts_hint, status) VALUES(?,?,?,?,?,?,?,?,?)",
            (1, 1, 2, 0, "text", None, 1, 0, "pending"),
        )
        await conn.commit()

    async def fake_fetch(*args, **kwargs):
        return []

    draft = vk_intake.EventDraft(title="T", date="2025-09-02", time="10:00", source_text="T")

    async def fake_build(text, source_name=None, location_hint=None, default_time=None, operator_extra=None):
        return draft

    captured = {}

    async def fake_mark_imported(db_, inbox_id, batch_id, event_id, event_date):
        captured["event_id"] = event_id

    tasks = []

    async def fake_enqueue_job(db_, eid, task, depends_on=None, coalesce_key=None):
        tasks.append(task)
        return "job"

    monkeypatch.setattr(main, "_vkrev_fetch_photos", fake_fetch)
    monkeypatch.setattr(vk_intake, "build_event_payload_from_vk", fake_build)
    monkeypatch.setattr(vk_review, "mark_imported", fake_mark_imported)
    monkeypatch.setattr(main, "enqueue_job", fake_enqueue_job)

    bot = DummyBot()
    await main._vkrev_import_flow(1, 1, 1, "batch1", db, bot)

    async with db.get_session() as session:
        ev = await session.get(Event, captured["event_id"])
    assert ev.source_post_url == "https://vk.com/wall-1_2"
    assert JobTask.vk_sync not in tasks
