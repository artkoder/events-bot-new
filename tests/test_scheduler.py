import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from scheduler import setup_scheduler


def test_scheduler_offsets_and_limits():
    scheduler = setup_scheduler(None, None)
    jobs = scheduler.get_jobs()
    assert all(job.misfire_grace_time == 60 for job in jobs)
    assert all(job.max_instances == 1 for job in jobs)
    assert "minute='1,16,31,46'" in str(jobs[0].trigger)
