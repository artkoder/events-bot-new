import pytest

from scheduling import CoalescingScheduler, schedule_event_batch


@pytest.mark.asyncio
async def test_coalescing_merges_jobs():
    scheduler = CoalescingScheduler()
    dates1 = ["2025-07-19"] * 3
    dates2 = ["2025-07-19"] * 3
    schedule_event_batch(scheduler, festival_id=1, dates=dates1)
    schedule_event_batch(scheduler, festival_id=1, dates=dates2)

    keys = scheduler.jobs.keys()
    assert len([k for k in keys if k.startswith("festival_pages:")]) == 1
    assert len([k for k in keys if k.startswith("month_pages:")]) == 1
    assert len([k for k in keys if k.startswith("week_pages:")]) == 1
    assert len([k for k in keys if k.startswith("weekend_pages:")]) == 1
    assert len([k for k in keys if k.startswith("vk_week_post:")]) == 1
    assert len([k for k in keys if k.startswith("vk_weekend_post:")]) == 1
    assert len(scheduler.jobs["weekend_pages:2025-07-19"].payload) == 6
    assert len(scheduler.jobs["vk_weekend_post:2025-07-19"].payload) == 6
