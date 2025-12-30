# Phase 3 — Added Tests

Created `tests/test_deferred_rebuild_extended.py` with new cases:

- Boundary: weekday event defers `month_pages` only, `week_pages` immediate, `weekend_pages` omitted; dirty state updated with month key only.
- Boundary: `mark_pages_dirty` dedupes months and preserves `reminded=False`.
- Race/concurrency: running `month_pages` job triggers follow-up job with `depends_on` when a new event coalesces into the same month.
- Cancel/restart: requeue an `error` nav job resets status/attempts and makes it runnable immediately.
- Deferred execution: job with future `next_run_at` is skipped until due, then runs and marks done.

