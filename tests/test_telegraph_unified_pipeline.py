from __future__ import annotations

from models import JobTask
import main


def test_event_telegraph_job_handler_is_single_entrypoint() -> None:
    """All event pipelines must converge to one Telegraph builder handler."""
    handler = main.JOB_HANDLERS.get(JobTask.telegraph_build.value)
    assert callable(handler)
    assert getattr(handler, "__name__", "") == "update_telegraph_event_page"
    assert getattr(handler, "__module__", "") == "main"
