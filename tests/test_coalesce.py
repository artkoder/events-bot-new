import pytest

from scheduling import CoalescingScheduler, schedule_event_batch


@pytest.mark.asyncio
async def test_coalesce_creates_single_jobs():
    scheduler = CoalescingScheduler()
    dates = ["2025-07-16"] * 5
    schedule_event_batch(scheduler, festival_id=1, dates=dates)

    keys = scheduler.jobs.keys()
    assert len([k for k in keys if k.startswith("festival_pages:")]) == 1
    assert len([k for k in keys if k.startswith("month_pages:")]) == 1
    assert len([k for k in keys if k.startswith("week_pages:")]) == 1
    assert len([k for k in keys if k.startswith("weekend_pages:")]) == 1
    assert len([k for k in keys if k.startswith("vk_week_post:")]) == 1
    assert len([k for k in keys if k.startswith("vk_weekend_post:")]) == 1
