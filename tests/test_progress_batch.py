import pytest

from scheduling import BatchProgress, CoalescingScheduler, schedule_event_batch


@pytest.mark.asyncio
async def test_progress_updates_to_final_state():
    dates = [
        "2025-07-31",
        "2025-08-01",
        "2025-08-02",
        "2025-08-02",
        "2025-08-03",
        "2025-08-03",
    ]
    progress = BatchProgress(total_events=len(dates))
    scheduler = CoalescingScheduler(progress)
    schedule_event_batch(scheduler, festival_id=1, dates=dates)
    await scheduler.run()

    assert progress.events_done == len(dates)
    assert len(progress.status) == 7
    final = progress.snapshot_text()
    assert "⏳" not in final
    assert final.count("Страница месяца") == 2
    assert final.count("Неделя:") == 1
    assert final.count("Выходные:") == 1
    assert final.count("Пост недели VK") == 1
    assert final.count("Пост выходных VK") == 1
