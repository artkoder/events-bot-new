from apscheduler.schedulers.asyncio import AsyncIOScheduler


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

    scheduler = AsyncIOScheduler()
    scheduler.add_job(
        vk_scheduler,
        "cron",
        minute="1,16,31,46",
        misfire_grace_time=60,
        max_instances=1,
        args=[db, bot],
    )
    scheduler.add_job(
        vk_poll_scheduler,
        "cron",
        minute="2,17,32,47",
        misfire_grace_time=60,
        max_instances=1,
        args=[db, bot],
    )
    scheduler.add_job(
        cleanup_scheduler,
        "cron",
        minute="3,18,33,48",
        misfire_grace_time=60,
        max_instances=1,
        args=[db, bot],
    )
    scheduler.add_job(
        page_update_scheduler,
        "cron",
        minute="4,19,34,49",
        misfire_grace_time=60,
        max_instances=1,
        args=[db],
    )
    scheduler.add_job(
        partner_notification_scheduler,
        "cron",
        minute="5,20,35,50",
        misfire_grace_time=60,
        max_instances=1,
        args=[db, bot],
    )
    return scheduler

