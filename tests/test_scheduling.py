from __future__ import annotations

import json
import sys
import time
from contextlib import asynccontextmanager
from zoneinfo import ZoneInfo

import pytest

import scheduling
import vk_intake
from db import Database
from heavy_ops import HeavyOpMeta


def test_scheduler_and_extract_do_not_import_main(monkeypatch):
    original_main = sys.modules.pop("main", None)
    monkeypatch.delenv("ENABLE_NIGHTLY_PAGE_SYNC", raising=False)

    class DummyExecutor:
        pass

    class DummyJob:
        def __init__(self, job_id: str) -> None:
            self.id = job_id
            self.next_run_time = None

    class DummyScheduler:
        def __init__(self, executors=None, timezone=None):
            self.executors = executors
            self.timezone = timezone
            self.jobs: dict[str, DummyJob] = {}
            self.listeners = []
            self.started = False

        def configure(self, job_defaults=None):
            self.job_defaults = job_defaults

        def add_job(self, func, trigger, id, args=None, **kwargs):
            job = DummyJob(id)
            self.jobs[id] = job
            return job

        def get_job(self, job_id):
            return self.jobs.get(job_id)

        def add_listener(self, listener, mask):
            self.listeners.append((listener, mask))

        def start(self):
            self.started = True

        def shutdown(self, wait=False):
            self.started = False

    monkeypatch.setattr(scheduling, "AsyncIOExecutor", lambda: DummyExecutor())
    monkeypatch.setattr(scheduling, "AsyncIOScheduler", DummyScheduler)
    monkeypatch.setattr(scheduling, "_scheduler", None)

    try:
        scheduler = scheduling.startup(
            db=None,
            bot=None,
            vk_scheduler=lambda *a, **k: None,
            vk_poll_scheduler=lambda *a, **k: None,
            vk_crawl_cron=lambda *a, **k: None,
            cleanup_scheduler=lambda *a, **k: None,
            partner_notification_scheduler=lambda *a, **k: None,
            nightly_page_sync=lambda *a, **k: None,
            rebuild_fest_nav_if_changed=lambda *a, **k: None,
        )
        assert isinstance(scheduler, DummyScheduler)
        assert "main" not in sys.modules

        tz = ZoneInfo("UTC")
        ts_hint = vk_intake.extract_event_ts_hint("завтра", tz=tz)
        assert ts_hint is not None
        assert "main" not in sys.modules
    finally:
        scheduling.cleanup()
        if original_main is not None:
            sys.modules["main"] = original_main
        else:
            sys.modules.pop("main", None)


@pytest.mark.asyncio
async def test_job_wrapper_records_skipped_heavy_ops_run(tmp_path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    monkeypatch.setenv("SCHED_HEAVY_GUARD_MODE", "skip")
    monkeypatch.delenv("SCHED_SERIALIZE_HEAVY_JOBS", raising=False)

    @asynccontextmanager
    async def fake_heavy_operation(**_kwargs):
        yield False

    async def should_not_run(*_args, **_kwargs):
        raise AssertionError("scheduled job body must not run when heavy guard skips it")

    blocked_meta = HeavyOpMeta(
        kind="tg_monitoring",
        trigger="scheduled",
        started_monotonic=time.monotonic(),
        run_id="blocked-run",
        operator_id=0,
        chat_id=None,
    )

    monkeypatch.setattr(scheduling, "heavy_operation", fake_heavy_operation)
    monkeypatch.setattr(scheduling, "current_heavy_meta", lambda: blocked_meta)

    wrapped = scheduling._job_wrapper("vk_auto_import", should_not_run)
    await wrapped(db, None)

    async with db.raw_conn() as conn:
        cur = await conn.execute(
            "SELECT kind, trigger, status, details_json FROM ops_run ORDER BY id ASC"
        )
        row = await cur.fetchone()

    assert row is not None
    kind, trigger, status, details_raw = row
    details = json.loads(details_raw)
    assert kind == "vk_auto_import"
    assert trigger == "scheduled"
    assert status == "skipped"
    assert details["skip_reason"] == "heavy_busy"
    assert details["blocked_by_kind"] == "tg_monitoring"


@pytest.mark.asyncio
async def test_run_scheduled_guide_excursions_autopublishes_after_success(tmp_path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    class DummyBot:
        def __init__(self) -> None:
            self.messages: list[tuple[int, str]] = []

        async def send_message(self, chat_id, text, **kwargs):
            self.messages.append((int(chat_id), str(text)))

    bot = DummyBot()

    class Result:
        errors: list[str] = []

    calls: list[tuple[str, object]] = []

    async def fake_resolve_superadmin_chat_id(_db):
        return 42

    async def fake_run_guide_monitor(db_obj, bot_obj, *, chat_id, operator_id, trigger, mode, send_progress):
        calls.append(("run", {"chat_id": chat_id, "trigger": trigger, "mode": mode, "send_progress": send_progress}))
        return Result()

    async def fake_publish_guide_digest(db_obj, bot_obj, *, family, chat_id, target_chat=None):
        calls.append(("publish", {"family": family, "chat_id": chat_id, "target_chat": target_chat}))
        return {"published": True, "issue_id": 9, "target_chat": "@keniggpt"}

    monkeypatch.setenv("ENABLE_GUIDE_DIGEST_SCHEDULED", "1")
    monkeypatch.setattr(scheduling, "resolve_superadmin_chat_id", fake_resolve_superadmin_chat_id)

    import guide_excursions.service as guide_service

    monkeypatch.setattr(guide_service, "run_guide_monitor", fake_run_guide_monitor)
    monkeypatch.setattr(guide_service, "publish_guide_digest", fake_publish_guide_digest)

    await scheduling._run_scheduled_guide_excursions(db, bot, mode="full")

    assert calls[0][0] == "run"
    assert calls[1][0] == "publish"
    assert calls[1][1]["family"] == "new_occurrences"
    assert any("Scheduled guide digest published" in text for _, text in bot.messages)


@pytest.mark.asyncio
async def test_run_scheduled_guide_excursions_skips_autopublish_for_light_mode(tmp_path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    class DummyBot:
        async def send_message(self, chat_id, text, **kwargs):
            return None

    class Result:
        errors: list[str] = []

    calls: list[str] = []

    async def fake_resolve_superadmin_chat_id(_db):
        return 42

    async def fake_run_guide_monitor(db_obj, bot_obj, *, chat_id, operator_id, trigger, mode, send_progress):
        calls.append("run")
        return Result()

    async def fake_publish_guide_digest(db_obj, bot_obj, *, family, chat_id, target_chat=None):
        calls.append("publish")
        return {"published": True}

    monkeypatch.setenv("ENABLE_GUIDE_DIGEST_SCHEDULED", "1")
    monkeypatch.setattr(scheduling, "resolve_superadmin_chat_id", fake_resolve_superadmin_chat_id)

    import guide_excursions.service as guide_service

    monkeypatch.setattr(guide_service, "run_guide_monitor", fake_run_guide_monitor)
    monkeypatch.setattr(guide_service, "publish_guide_digest", fake_publish_guide_digest)

    await scheduling._run_scheduled_guide_excursions(db, DummyBot(), mode="light")

    assert calls == ["run"]
