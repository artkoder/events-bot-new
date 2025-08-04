import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))
from scheduler import startup, cleanup


@pytest.mark.asyncio
async def test_scheduler_offsets_and_limits():
    scheduler = startup(None, None)
    jobs = scheduler.get_jobs()
    assert all(job.misfire_grace_time == 30 for job in jobs)
    assert all(job.max_instances == 1 for job in jobs)
    assert any("minute='1,16,31,46'" in str(job.trigger) for job in jobs)
    assert scheduler._executors["default"]._max_workers == 2
    cleanup()
