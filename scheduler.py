from apscheduler.executors.asyncio import AsyncIOExecutor
from apscheduler.schedulers.asyncio import AsyncIOScheduler

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
        _scheduler.start()
    return _scheduler


def cleanup() -> None:
    global _scheduler
    if _scheduler:
        _scheduler.shutdown(wait=False)
        _scheduler = None
