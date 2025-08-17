import asyncio
import contextlib
import pytest
from aiohttp import web
import main

@pytest.mark.asyncio
async def test_init_starts_job_outbox_worker(tmp_path, monkeypatch):
    async def fake_get_tz_offset(db):
        return "0"
    async def fake_get_catbox_enabled(db):
        return False
    async def fake_get_vk_photos_enabled(db):
        return False
    async def fake_worker(*args, **kwargs):
        pass
    class DummyBot:
        async def set_webhook(self, *args, **kwargs):
            pass
    monkeypatch.setattr(main, "get_tz_offset", fake_get_tz_offset)
    monkeypatch.setattr(main, "get_catbox_enabled", fake_get_catbox_enabled)
    monkeypatch.setattr(main, "get_vk_photos_enabled", fake_get_vk_photos_enabled)
    monkeypatch.setattr(main, "scheduler_startup", lambda db, bot: None)
    monkeypatch.setattr(main, "daily_scheduler", fake_worker)
    monkeypatch.setattr(main, "add_event_queue_worker", fake_worker)
    monkeypatch.setattr(main, "_watch_add_event_worker", fake_worker)
    monkeypatch.setattr(main, "job_outbox_worker", fake_worker)
    prev_catbox = main.CATBOX_ENABLED
    prev_vk = main.VK_PHOTOS_ENABLED
    app = web.Application()
    db = main.Database(str(tmp_path / "db.sqlite"))
    await main.init_db_and_scheduler(app, db, DummyBot(), "https://example.com")
    assert "job_outbox_worker" in app
    main.CATBOX_ENABLED = prev_catbox
    main.VK_PHOTOS_ENABLED = prev_vk
    for key in [
        "daily_scheduler",
        "add_event_worker",
        "add_event_watch",
        "job_outbox_worker",
    ]:
        task = app[key]
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
