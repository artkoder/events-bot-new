# Phase 1 — Code Review (Deferred Rebuilds)

## Findings

1) **Deferred schedule is lost on re-enqueue (likely bug)**
- `enqueue_job` resets `next_run_at` to `now` for existing `pending` and `error/done` jobs, ignoring the passed `next_run_at`.
- Impact: deferred `month_pages`/`weekend_pages` coalesced jobs are re-armed to run immediately when a second event hits the same month/weekend. This undermines the intended debounce.
- Evidence: `main.py:9931-9944`, `main.py:10023-10033`.

2) **Deferred nav tasks still trigger drain loop (risk of long waits)**
- `schedule_event_update_tasks` defers nav tasks but still calls `_drain_nav_tasks` when `drain_nav=True`. `_drain_nav_tasks` waits until nav tasks are done; deferred tasks keep it pending until timeout.
- Impact: can add ~90s latency to workflows that call `schedule_event_update_tasks` with default `drain_nav=True`.
- Evidence: `main.py:10063-10127`, `main.py:10180-10340`.

3) **Race: weekend_pages lacks follow-up when owner running**
- `enqueue_job` has a follow-up creation path when a coalesced `month_pages` job is `running`, but no equivalent for `weekend_pages` or `week_pages`.
- Impact: if a `weekend_pages` job starts before a new event is committed, the running job may miss the event with no follow-up run to reconcile.
- Evidence: `main.py:9968-10012` (follow-up only for `JobTask.month_pages`).

4) **Dirty flag updates are not atomic (lost updates)**
- `mark_pages_dirty` uses read-modify-write with JSON in `Setting` without locking/transactions across concurrent calls.
- Impact: concurrent schedules can overwrite each other, dropping months/weekend keys.
- Evidence: `main.py:2411-2428`.

5) **Weak input validation for month key**
- `schedule_event_update_tasks` builds `month = ev.date.split("..", 1)[0][:7]` even if `ev.date` is malformed; `mark_pages_dirty` will store invalid keys.
- Impact: dirty state and coalesce keys may become invalid, causing rebuilds against malformed month identifiers.
- Evidence: `main.py:10090-10097`.

6) **Corrupt dirty state is silently ignored**
- `load_pages_dirty_state` returns `None` on JSON errors without logging or clearing the bad value.
- Impact: dirty state can become permanently ignored, leaving rebuild reminders/logic blind.
- Evidence: `main.py:2395-2408`.

## Notes
- `mark_pages_dirty` stores weekend keys in the `months` list; functional but naming is misleading and could cause confusion in later consumers.

