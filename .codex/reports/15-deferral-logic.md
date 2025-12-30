It runs once at T+20m (option 3). The second enqueue finds the existing pending `month_pages` job by `coalesce_key` and merges into it, updating `next_run_at` to the later time instead of creating a new job.

- `schedule_event_update_tasks` defers `month_pages` by 15 minutes via `next_run_at` (`main.py:10162` to `main.py:10169`).
- `enqueue_job` derives a month-based `coalesce_key` for `month_pages` (`main.py:9925` to `main.py:9928`) and selects the latest job with that key (`main.py:9960` to `main.py:9966`).
- If the existing job is `pending`, it updates in place and only extends `next_run_at` when the new time is later (`main.py:9995` to `main.py:10011`), returning `"merged-rearmed"`.

So Event A enqueues at T+15m; Event B at T+5m extends it to T+20m; only one job runs.