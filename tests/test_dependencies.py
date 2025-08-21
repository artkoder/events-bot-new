import pytest

from scheduling import BatchProgress, CoalescingScheduler, schedule_event_batch


@pytest.mark.asyncio
async def test_festival_runs_before_month_and_vk():
    dates = ["2025-07-19"]
    progress = BatchProgress(total_events=len(dates))
    scheduler = CoalescingScheduler(progress)
    schedule_event_batch(scheduler, festival_id=1, dates=dates)
    await scheduler.run()

    order = scheduler.order
    festival_key = "festival_pages:1"
    month_key = "month_pages:2025-07"
    week_key = "week_pages:2025-29"
    weekend_key = "weekend_pages:2025-07-19"
    vk_week_key = "vk_week_post:2025-29"
    vk_weekend_key = "vk_weekend_post:2025-07-19"

    assert order[0] == festival_key
    assert order.index(month_key) > order.index(festival_key)
    assert order.index(week_key) > order.index(festival_key)
    assert order.index(weekend_key) > order.index(festival_key)
    assert order.index(vk_week_key) > order.index(month_key)
    assert order.index(vk_week_key) > order.index(week_key)
    assert order.index(vk_week_key) > order.index(weekend_key)
    assert order.index(vk_weekend_key) > order.index(month_key)
    assert order.index(vk_weekend_key) > order.index(week_key)
    assert order.index(vk_weekend_key) > order.index(weekend_key)
