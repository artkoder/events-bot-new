Added phase reports and an extended deferred‑rebuild test suite to cover boundary behavior, race scenarios, and restart/requeue paths. Reports are in `.codex/reports/PHASE-1-review.md`, `.codex/reports/PHASE-2-eval.md`, `.codex/reports/PHASE-3.md`, and tests are in `tests/test_deferred_rebuild_extended.py`.

**Final Report**
- **Zones**: Preserve `next_run_at` when re‑enqueueing pending/coalesced jobs (`main.py`) so deferral isn’t cancelled on subsequent events; `_drain_nav_tasks` currently waits on deferred nav jobs and can block for ~90s (`main.py:10180+`); add follow‑up handling for `weekend_pages` when a coalesced job is already running; make dirty‑flag updates atomic to prevent lost months (`main.py:2411+`); validate month keys derived from `ev.date` before storing or coalescing.
- **Recommendations**: Skip or relax nav draining for deferred jobs (e.g., ignore pending `next_run_at` in the future); add a consumer for `pages_dirty_state` (reminders/auto `/pages_rebuild`) or remove the unused state; log and/or clear corrupt dirty‑state JSON; add a test asserting deferral survives re‑enqueue for coalesced jobs.
- **Readiness**: Not production‑ready for strict deferred semantics; correctness and latency risks remain (deferral can be bypassed, drain waits, weekend race). Once deferral preservation + drain behavior are fixed and dirty state is integrated, readiness improves.

Tests not run (not requested).

If you want, I can:
1) run the new tests and report results  
2) patch `enqueue_job`/`_drain_nav_tasks` to preserve deferral and avoid waiting on future jobs  
3) wire `pages_dirty_state` into a periodic rebuild/reminder flow