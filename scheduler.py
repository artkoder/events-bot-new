from apscheduler.executors.asyncio import AsyncIOExecutor
from apscheduler.schedulers.asyncio import AsyncIOScheduler
import asyncio
import logging
import time as _time

_scheduler: AsyncIOScheduler | None = None


def startup(db, bot) -> AsyncIOScheduler:
    global _scheduler
    if _scheduler is None:
        executor = AsyncIOExecutor()
        executor._max_workers = 2
        _scheduler = AsyncIOScheduler(
            executors={"default": executor},
            job_defaults={
                "max_instances": 1,
                "coalesce": True,
                "misfire_grace_time": 60,
            },
        )

    if _scheduler.get_jobs():
        return _scheduler

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
    )
    _scheduler.add_job(
        vk_poll_scheduler,
        "cron",
        id="vk_poll_scheduler",
        minute="2,17,32,47",
        args=[db, bot],
        replace_existing=True,
    )
    _scheduler.add_job(
        cleanup_scheduler,
        "cron",
        id="cleanup_scheduler",
        minute="3,18,33,48",
        args=[db, bot],
        replace_existing=True,
    )
    _scheduler.add_job(
        page_update_scheduler,
        "cron",
        id="page_update_scheduler",
        minute="4,19,34,49",
        args=[db],
        replace_existing=True,
    )
    _scheduler.add_job(
        partner_notification_scheduler,
        "cron",
        id="partner_notification_scheduler",
        minute="5,20,35,50",
        args=[db, bot],
        replace_existing=True,
    )

    async def _maint(sql: str, op: str):
        delay = 1
        for attempt in range(3):
            start = _time.perf_counter()
            try:
                await db.exec_driver_sql(sql)
                dur = (_time.perf_counter() - start) * 1000
                logging.info("db_maintenance: %s done in %.0f ms", op, dur)
                break
            except Exception as e:
                if "locked" in str(e).lower() and attempt < 2:
                    await asyncio.sleep(delay)
                    delay *= 2
                    continue
                logging.error("db_maintenance %s failed: %s", op, e)
                break

    _scheduler.add_job(
        _maint,
        "cron",
        id="db_optimize",
        hour="3",
        args=["PRAGMA optimize;", "optimize"],
        replace_existing=True,
    )
    _scheduler.add_job(
        _maint,
        "cron",
        id="db_wal_checkpoint",
        hour="3",
        minute="5",
        args=["PRAGMA wal_checkpoint(TRUNCATE);", "wal_checkpoint"],
        replace_existing=True,
    )
    _scheduler.add_job(
        _maint,
        "cron",
        id="db_vacuum",
        day_of_week="sun",
        hour="4",
        minute="30",
        args=["VACUUM;", "vacuum"],
        replace_existing=True,
    )

    if not _scheduler.running:
        _scheduler.start()

    return _scheduler


def cleanup() -> None:
    global _scheduler
    if _scheduler:
        _scheduler.shutdown(wait=True)
        _scheduler = None
