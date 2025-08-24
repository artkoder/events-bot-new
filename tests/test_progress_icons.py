import pytest
from scheduling import BatchProgress


def test_icons_single_and_batch_states():
    progress = BatchProgress(total_events=0)
    progress.status = {
        "job_pending": "pending",
        "job_running": "running",
        "job_deferred": "deferred",
        "job_captcha": "captcha",
        "job_done": "done",
        "job_error": "error",
        "job_skipped": "skipped_nochange",
    }
    text = progress.snapshot_text()
    assert "⏳" in text
    assert "🔄" in text
    assert "⏸" in text
    assert "🧩⏸" in text
    assert "✅" in text
    assert "❌" in text
    assert "⏭" in text
