Summary  
- Keep `telegraph_build` and `ics_publish` immediate; defer only nav pages with a persistent dirty-flag and scheduled rebuild; disable VK sync for review flows and cap VK tasks that trigger captcha.  
- Filter/short-circuit `ics_publish` for past events to cut 50% expired churn; throttle VK/nav to avoid Telegraph flood and page rebuild races.

What stays HOT  
- `telegraph_build` (event page creation/update).  
- `ics_publish` (but skip past events to avoid expired errors).  
- `/start` VK sync for newly added events (no changes).

What is DEFERRED  
- `month_pages` and `weekend_pages` jobs: debounce to 10 minutes via dirty-flag, auto rebuild at +2h with APScheduler-triggered `/pages_rebuild` equivalent.

What is DISABLED  
- `vk_sync` when triggered from `/vk` and `/parse` flows; leave enabled only for `/start` user submissions.  
- VK post types that provoke captcha (week/weekend/festival VK posts) until type is identified; mark them paused with logging.

Implementation Steps (files/functions)  
- `main.py`  
  - In `schedule_event_update_tasks`: leave `telegraph_build`, `ics_publish`, `/start` `vk_sync` hot; for `month_pages/weekend_pages` set `next_run_at = now+2h`, skip `_drain_nav_tasks`, call new helper `mark_pages_dirty(db, month, now)` and set `remind_at = now+10m`.  
  - In `job_outbox_worker` or `enqueue_job`: respect pre-set `next_run_at` so deferred nav tasks stay delayed.  
  - In `JOB_HANDLERS` for `ics_publish`: early-return `done` when `event.ends_at < now()` or status past/archived to stop expired retries.  
- `main_part2.py`  
  - Add helpers `load_pages_dirty_state` / `mark_pages_dirty` / `clear_pages_dirty_state` using `Setting` (key `pages_dirty_state`).  
  - Extend `scheduling.startup` to add APScheduler job `maybe_rebuild_dirty_pages` every 3–5 minutes:  
    - If `since >= 10m` and not reminded → `notify_superadmin` with dirty months.  
    - If `since >= 2h` → call `_perform_pages_rebuild(db, months, force=True)`, then `clear_pages_dirty_state` and mark corresponding nav jobs `superseded/done`.  
  - In `_perform_pages_rebuild`: after success, `clear_pages_dirty_state` and reset pending nav jobs for the same months to avoid double runs.  
- VK captcha mitigation  
  - In VK handlers (`sync_vk_week_post`, `sync_vk_weekend_post`, `job_sync_vk_source_post` in `main_part2.py`/`main.py`): add guard to pause/skip execution when captcha detected; log `task_type` to identify offending post types; optionally tag jobs `error` with reason `captcha` instead of `paused 2035`.  
  - In `/vk` and `/parse` flows (`handle_vk_review_cb`, `_vkrev_import_flow` in `main_part2.py`): do not enqueue `vk_sync`; leave enqueue only in `/start` path.  
- Telegraph flood safety  
  - Add small jitter/backoff between `telegraph_call` invocations inside nav rebuild (e.g., sleep 0.5–1s after successful create/edit) to reduce flood risk for big batches.  
  - Add preflight size check for weekend pages (same 45k limit as months) before `telegraph_edit_page`; on overflow, compress/trim content or skip with explicit error to avoid repeated failures.

Risks and Mitigations  
- Telegraph flood / rate limits: serialize with added jitter; keep semaphore=1; retry 429/5xx in `telegraph_call`.  
- Dirty-flag loss on restart: store in `Setting`; APScheduler job re-reads and resumes.  
- Double nav updates: after auto rebuild, supersede pending `month_pages/weekend_pages` jobs for same months.  
- VK captcha backlog: mark offending tasks `error(captcha)` instead of `paused 2035`; logging per task type to decide which to disable.  
- ICS regression: skipping past events must not hide future ones—gate by `ends_at < now()` only.

Testing Plan  
- Unit/functional:  
  - Add test for `schedule_event_update_tasks` to assert deferred `next_run_at` for nav tasks and immediate for `telegraph_build/ics_publish`.  
  - Test `pages_dirty_state` helpers (set/load/clear, reminder thresholds).  
  - Test `maybe_rebuild_dirty_pages` logic with mocked time: reminder at 10m, rebuild at 2h, clears state and supersedes nav jobs.  
  - Test `ics_publish` early exit for past events.  
  - VK flow tests: `/vk` import does not enqueue `vk_sync`; `/start` still does.  
- Manual/QA:  
  - Add event → confirm Telegraph builds immediately, ICS runs, nav tasks scheduled +2h and dirty flag set.  
  - Advance clock or wait to trigger scheduler → ensure `/pages_rebuild` runs once, clears flag, nav jobs not double-run.  
  - Trigger weekend page with large content → preflight either trims or errors cleanly.  
  - Simulate VK captcha response → tasks marked error(captcha), not paused indefinitely.