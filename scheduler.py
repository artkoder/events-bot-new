from apscheduler.executors.asyncio import AsyncIOExecutor
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.events import (
    EVENT_JOB_ERROR,
    EVENT_JOB_EXECUTED,
    EVENT_JOB_MISSED,
    EVENT_JOB_SUBMITTED,
)
import asyncio
import logging
import time as _time
from functools import partial
from uuid import uuid4
import os

from db import optimize, wal_checkpoint_truncate, vacuum

_scheduler: AsyncIOScheduler | None = None
_run_meta: dict[str, tuple[str, float]] = {}


def _job_wrapper(job_id: str, func):
    async def _run(*args):
        run_id, start = _run_meta.get(job_id, (uuid4().hex, _time.perf_counter()))
        done = asyncio.Event()

        async def heartbeat():
            while not done.is_set():
                await asyncio.sleep(10)
                took_ms = (_time.perf_counter() - start) * 1000
                logging.info(
                    "job_heartbeat job_id=%s run_id=%s took_ms=%.0f",
                    job_id,
                    run_id,
                    took_ms,
                )

        hb_task = asyncio.create_task(heartbeat())
        try:
            return await func(*args, run_id=run_id)
        finally:
            done.set()
            hb_task.cancel()

    return _run


def _on_event(event):
    job_id = event.job_id
    name_map = {
        EVENT_JOB_SUBMITTED: "JOB_SUBMITTED",
        EVENT_JOB_EXECUTED: "JOB_EXECUTED",
        EVENT_JOB_ERROR: "JOB_ERROR",
        EVENT_JOB_MISSED: "JOB_MISSED",
    }
    event_name = name_map.get(event.code, str(event.code))
    run_id = None
    start = None
    if event.code == EVENT_JOB_SUBMITTED:
        run_id = uuid4().hex
        start = _time.perf_counter()
        _run_meta[job_id] = (run_id, start)
    else:
        run_id, start = _run_meta.get(job_id, (uuid4().hex, None))
    took_ms = None
    if event.code in (EVENT_JOB_EXECUTED, EVENT_JOB_ERROR) and start is not None:
        took_ms = (_time.perf_counter() - start) * 1000
        _run_meta.pop(job_id, None)
    if event.code == EVENT_JOB_MISSED:
        _run_meta.pop(job_id, None)
        run_id = uuid4().hex
    next_run = None
    if _scheduler:
        job = _scheduler.get_job(job_id)
        next_run = job.next_run_time if job else None
    tb_excerpt = None
    tb = getattr(event, "traceback", None)
    if tb:
        tb_excerpt = " | ".join(tb.strip().splitlines()[-3:])
    logging.info(
        "%s job_id=%s run_id=%s next_run=%s took_ms=%s traceback_excerpt=%s",
        event_name,
        job_id,
        run_id,
        next_run,
        f"{took_ms:.0f}" if took_ms is not None else "0",
        tb_excerpt,
    )


def startup(db, bot) -> AsyncIOScheduler:
    global _scheduler
    if _scheduler is None:
        # AsyncIOExecutor has no configuration options; use default settings
        executor = AsyncIOExecutor()
        _scheduler = AsyncIOScheduler(executors={"default": executor}, timezone="UTC")
        _scheduler.configure(
            job_defaults={
                "max_instances": 1,
                "coalesce": True,
                "misfire_grace_time": 30,
            }
        )

    from main import (
        vk_scheduler,
        vk_poll_scheduler,
        cleanup_scheduler,
        partner_notification_scheduler,
        nightly_page_sync,
    )

    _scheduler.add_job(
        _job_wrapper("vk_scheduler", vk_scheduler),
        "cron",
        id="vk_scheduler",
        minute="1,16,31,46",
        args=[db, bot],
        replace_existing=True,
        max_instances=1,
        coalesce=True,
        misfire_grace_time=30,
    )
    _scheduler.add_job(
        _job_wrapper("vk_poll_scheduler", vk_poll_scheduler),
        "cron",
        id="vk_poll_scheduler",
        minute="2,17,32,47",
        args=[db, bot],
        replace_existing=True,
        max_instances=1,
        coalesce=True,
        misfire_grace_time=30,
    )
    _scheduler.add_job(
        _job_wrapper("cleanup_scheduler", cleanup_scheduler),
        "cron",
        id="cleanup_scheduler",
        hour="2",
        minute="7",
        args=[db, bot],
        replace_existing=True,
        max_instances=1,
        coalesce=True,
        misfire_grace_time=30,
    )
    _scheduler.add_job(
        _job_wrapper("partner_notification_scheduler", partner_notification_scheduler),
        "cron",
        id="partner_notification_scheduler",
        minute="5",
        args=[db, bot],
        replace_existing=True,
        max_instances=1,
        coalesce=True,
        misfire_grace_time=30,
    )

    if os.getenv("ENABLE_NIGHTLY_PAGE_SYNC") == "1":
        _scheduler.add_job(
            _job_wrapper("nightly_page_sync", nightly_page_sync),
            "cron",
            id="nightly_page_sync",
            hour="2",
            minute="30",
            args=[db],
            replace_existing=True,
            max_instances=1,
            coalesce=True,
            misfire_grace_time=30,
        )

    async def _run_maintenance(job, name: str, timeout: float, run_id: str | None = None) -> None:
        start = _time.perf_counter()
        try:
            await asyncio.wait_for(job(), timeout=timeout)
            dur = (_time.perf_counter() - start) * 1000
            logging.info("db_maintenance %s done in %.0f ms", name, dur)
        except asyncio.TimeoutError:
            logging.warning(
                "db_maintenance %s timed out after %.1f s", name, timeout
            )
        except Exception:
            logging.warning("db_maintenance %s failed", name, exc_info=True)

    if db is not None:
        _scheduler.add_job(
            _job_wrapper("db_optimize", _run_maintenance),
            "interval",
            id="db_optimize",
            hours=1,
            args=[partial(optimize, db.engine), "PRAGMA optimize", 10.0],
            replace_existing=True,
            max_instances=1,
            coalesce=True,
            misfire_grace_time=30,
        )
        _scheduler.add_job(
            _job_wrapper("db_wal_checkpoint", _run_maintenance),
            "cron",
            id="db_wal_checkpoint",
            hour="3",
            minute="5",
            args=[
                partial(wal_checkpoint_truncate, db.engine),
                "PRAGMA wal_checkpoint(TRUNCATE)",
                10.0,
            ],
            replace_existing=True,
            max_instances=1,
            coalesce=True,
            misfire_grace_time=30,
        )
        _scheduler.add_job(
            _job_wrapper("db_vacuum", _run_maintenance),
            "cron",
            id="db_vacuum",
            day_of_week="sun",
            hour="4",
            minute="30",
            args=[partial(vacuum, db.engine), "VACUUM", 60.0],
            replace_existing=True,
            max_instances=1,
            coalesce=True,
            misfire_grace_time=30,
        )

    _scheduler.add_listener(
        _on_event,
        EVENT_JOB_SUBMITTED
        | EVENT_JOB_EXECUTED
        | EVENT_JOB_ERROR
        | EVENT_JOB_MISSED,
    )

    if not _scheduler.running:
        _scheduler.start()

    return _scheduler


def cleanup() -> None:
    global _scheduler
    if _scheduler:
        _scheduler.shutdown(wait=True)
        _scheduler = None
