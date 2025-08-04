from apscheduler.executors.asyncio import AsyncIOExecutor
from apscheduler.schedulers.asyncio import AsyncIOScheduler


class LimitedAsyncIOExecutor(AsyncIOExecutor):
    def __init__(self, max_workers: int = 2):
        super().__init__()
        self._max_workers = max_workers


def setup_scheduler(db, bot):
    """Configure periodic jobs with staggered start times."""
    # Import here to avoid circular dependencies during module import.
    from main import (
        vk_scheduler,
        vk_poll_scheduler,
        cleanup_scheduler,
        page_update_scheduler,
        partner_notification_scheduler,
    )

    scheduler = AsyncIOScheduler(
        executors={"default": LimitedAsyncIOExecutor(2)},
        job_defaults={"coalesce": True, "misfire_grace_time": 30},
    )
    scheduler.add_job(
        vk_scheduler,
        "cron",
        minute="1,16,31,46",
        max_instances=1,
        args=[db, bot],
    )
    scheduler.add_job(
        vk_poll_scheduler,
        "cron",
        minute="2,17,32,47",
        max_instances=1,
        args=[db, bot],
    )
    scheduler.add_job(
        cleanup_scheduler,
        "cron",
        minute="3,18,33,48",
        max_instances=1,
        args=[db, bot],
    )
    scheduler.add_job(
        page_update_scheduler,
        "cron",
        minute="4,19,34,49",
        max_instances=1,
        args=[db],
    )
    scheduler.add_job(
        partner_notification_scheduler,
        "cron",
        minute="5,20,35,50",
        max_instances=1,
        args=[db, bot],
    )
    return scheduler

