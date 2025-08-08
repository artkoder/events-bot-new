from apscheduler.executors.asyncio import AsyncIOExecutor
from apscheduler.schedulers.asyncio import AsyncIOScheduler
import asyncio
import logging
import time as _time

_scheduler: AsyncIOScheduler | None = None


def startup(db, bot) -> AsyncIOScheduler:
    global _scheduler
    if _scheduler is None:
        from main import (
            vk_scheduler,
            vk_poll_scheduler,
            cleanup_scheduler,
            page_update_scheduler,
            partner_notification_scheduler,
        )

        executor = AsyncIOExecutor()
        executor._max_workers = 2
        _scheduler = AsyncIOScheduler(
            executors={"default": executor},
            job_defaults={"coalesce": True, "misfire_grace_time": 30},
        )
        _scheduler.add_job(
            vk_scheduler,
            "cron",
            minute="1,16,31,46",
            max_instances=1,
            args=[db, bot],
        )
        _scheduler.add_job(
            vk_poll_scheduler,
            "cron",
            minute="2,17,32,47",
            max_instances=1,
            args=[db, bot],
        )
        _scheduler.add_job(
            cleanup_scheduler,
            "cron",
            minute="3,18,33,48",
            max_instances=1,
            args=[db, bot],
        )
        _scheduler.add_job(
            page_update_scheduler,
            "cron",
            minute="4,19,34,49",
            max_instances=1,
            args=[db],
        )
        _scheduler.add_job(
            partner_notification_scheduler,
            "cron",
            minute="5,20,35,50",
            max_instances=1,
            args=[db, bot],
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
            hour="3",
            args=["PRAGMA optimize;", "optimize"],
        )
        _scheduler.add_job(
            _maint,
            "cron",
            hour="3",
            minute="5",
            args=["PRAGMA wal_checkpoint(TRUNCATE);", "wal_checkpoint"],
        )
        _scheduler.add_job(
            _maint,
            "cron",
            day_of_week="sun",
            hour="4",
            minute="30",
            args=["VACUUM;", "vacuum"],
        )
        _scheduler.start()
    return _scheduler


def cleanup() -> None:
    global _scheduler
    if _scheduler:
        _scheduler.shutdown(wait=False)
        _scheduler = None
