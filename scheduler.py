from apscheduler.executors.asyncio import AsyncIOExecutor
from apscheduler.schedulers.asyncio import AsyncIOScheduler
import asyncio
import logging
import time as _time
from functools import partial

from db import optimize, wal_checkpoint_truncate, vacuum

_scheduler: AsyncIOScheduler | None = None


def startup(db, bot) -> AsyncIOScheduler:
    global _scheduler
    if _scheduler is None:
        # AsyncIOExecutor has no configuration options; use default settings
        executor = AsyncIOExecutor()
        _scheduler = AsyncIOScheduler(
            executors={"default": executor},
            job_defaults={
                "max_instances": 1,
                "coalesce": True,
                "misfire_grace_time": 30,
            },
        )

    from main import (
        vk_scheduler,
        vk_poll_scheduler,
        cleanup_scheduler,
        page_update_scheduler,
        partner_notification_scheduler,
    )

    _scheduler.add_job(
        vk_scheduler,
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
        vk_poll_scheduler,
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
        cleanup_scheduler,
        "cron",
        id="cleanup_scheduler",
        minute="3,18,33,48",
        args=[db, bot],
        replace_existing=True,
        max_instances=1,
        coalesce=True,
        misfire_grace_time=30,
    )
    _scheduler.add_job(
        page_update_scheduler,
        "cron",
        id="page_update_scheduler",
        minute="4,19,34,49",
        args=[db],
        replace_existing=True,
        max_instances=1,
        coalesce=True,
        misfire_grace_time=30,
    )
    _scheduler.add_job(
        partner_notification_scheduler,
        "cron",
        id="partner_notification_scheduler",
        minute="5,20,35,50",
        args=[db, bot],
        replace_existing=True,
        max_instances=1,
        coalesce=True,
        misfire_grace_time=30,
    )

    async def _run_maintenance(job, name: str, timeout: float) -> None:
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
            _run_maintenance,
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
            _run_maintenance,
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
            _run_maintenance,
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

    if not _scheduler.running:
        _scheduler.start()

    return _scheduler


def cleanup() -> None:
    global _scheduler
    if _scheduler:
        _scheduler.shutdown(wait=True)
        _scheduler = None
