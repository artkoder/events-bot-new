import pytest

from scheduling import BatchProgress, CoalescingScheduler, schedule_event_batch


@pytest.mark.asyncio
async def test_progress_updates_to_final_state():
    dates = ["2025-07-16"] * 6
    progress = BatchProgress(total_events=len(dates))
    scheduler = CoalescingScheduler(progress)
    schedule_event_batch(scheduler, festival_id=1, dates=dates)
    await scheduler.run()

    assert progress.events_done == len(dates)
    # All tracked jobs should have a final state, none left as 'pending'
    assert progress.status
    assert all(status != "pending" for status in progress.status.values())
