from __future__ import annotations

import sys
from zoneinfo import ZoneInfo

import scheduling
import vk_intake


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

