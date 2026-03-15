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


@pytest.mark.asyncio
async def test_create_app_startup_waits_for_init(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:abc")
    monkeypatch.setenv("WEBHOOK_URL", "https://example.com/webhook")

    called = False

    async def fake_init(app, db, bot, webhook):
        nonlocal called
        called = True
        assert webhook == "https://example.com/webhook"

    monkeypatch.setattr(main, "init_db_and_scheduler", fake_init)
    monkeypatch.setattr(main, "_startup_handler_registered", False)

    prev_db = getattr(main, "db", None)
    app = main.create_app()
    try:
        assert app.on_startup
        await app.on_startup[-1](app)
    finally:
        main.db = prev_db

    assert called


@pytest.mark.asyncio
async def test_watch_add_event_worker_stall_restart_updates_global_timestamp(monkeypatch):
    prev_ts = getattr(main, "_ADD_EVENT_LAST_DEQUEUE_TS", None)

    class _Queue:
        def qsize(self):
            return 1

    class _CancelledWorker:
        def done(self):
            return False

        def cancelled(self):
            return False

        def cancel(self):
            return None

        def __await__(self):
            async def _wait():
                raise asyncio.CancelledError

            return _wait().__await__()

    class _RestartedWorker:
        def done(self):
            return False

        def cancelled(self):
            return False

    class _DummyBot:
        pass

    created: list[object] = []

    def fake_create_task(coro):
        created.append(coro)
        coro.close()
        return _RestartedWorker()

    sleep_calls = 0

    async def fake_sleep(_seconds):
        nonlocal sleep_calls
        sleep_calls += 1
        raise asyncio.CancelledError

    fake_now = 1000.0
    monkeypatch.setattr(main, "add_event_queue", _Queue())
    monkeypatch.setattr(main._time, "monotonic", lambda: fake_now)
    monkeypatch.setattr(main.asyncio, "create_task", fake_create_task)
    monkeypatch.setattr(main.asyncio, "sleep", fake_sleep)

    app = {"add_event_worker": _CancelledWorker()}
    main._ADD_EVENT_LAST_DEQUEUE_TS = fake_now - 1001

    with pytest.raises(asyncio.CancelledError):
        await main._watch_add_event_worker(app, None, _DummyBot())

    assert isinstance(app["add_event_worker"], _RestartedWorker)
    assert main._ADD_EVENT_LAST_DEQUEUE_TS == fake_now
    assert sleep_calls == 1
    assert created

    main._ADD_EVENT_LAST_DEQUEUE_TS = prev_ts
