# Cron Schedule

The bot uses APScheduler to run periodic maintenance tasks on a fixed schedule.

## Routing

When you need to change or inspect a schedule, use this route first instead of searching the repo from scratch:

- canonical ops doc and schedule policy: `docs/operations/cron.md`
- APScheduler job registration and default times: `scheduling.py`
- production schedule overrides for Fly: `fly.toml` (`[env]`)
- local/dev env template: `.env.example`

Rule of thumb:

- if you need to understand *what* runs and *why*, start here in `docs/operations/cron.md`;
- if you need to change fallback/default times in code, edit `scheduling.py`;
- if you need to change current production timings, edit `fly.toml`;
- if you need to keep local setup examples in sync, update `.env.example`.

Some jobs are lightweight (seconds), but **Kaggle/LLM/rendering** jobs can take **minutes or hours** (e.g. Telegram monitoring via Kaggle, VK auto-import via Smart Update, `/parse`, `/3di`).

To avoid parallel long-running operations (especially **manual** starts overlapping with **scheduled** ones), the scheduler uses a shared “heavy ops” gate:

- by default, scheduled heavy jobs **skip** if another heavy operation is already running and notify `ADMIN_CHAT_ID` about the skip;
- if you prefer waiting/serialization (run later instead of skipping), set `SCHED_HEAVY_GUARD_MODE=wait` (or legacy `SCHED_SERIALIZE_HEAVY_JOBS=1`).

VK crawling runs six times per day by default at `05:15`, `09:15`, `13:15`, `17:15`, `21:15` and `22:45` Europe/Kaliningrad time (`VK_CRAWL_TIMES_LOCAL` / `VK_CRAWL_TZ`).

## Observed runtimes (local runs)

Numbers below are from `ops_run` snapshots + local `/parse` logs (p50/p90/max). Use them to spread heavy jobs across the day.

- `tg_monitoring`: ~37m / ~2h53m / ~3h17m
- `/parse` (source parsing): ~9m / ~19m / (rare outliers up to ~6h+ when Kaggle stalls)
- `vk_auto_import`: ~45m / (few samples) / (rare outliers up to ~6h+ when unbounded)

## Recommended spacing (Europe/Kaliningrad)

Defaults were adjusted to reduce overlaps between the most common heavy jobs:

- nightly source parsing: `SOURCE_PARSING_TIME_LOCAL=04:30` (was `02:15`)
- `/3di` morning run: `THREEDI_TIMES_LOCAL=07:15,15:15,17:15` (was `05:30,15:15,17:15`; older default `03:15,15:15,17:15`)
- VK auto-import: `VK_AUTO_IMPORT_TIMES_LOCAL=06:15,10:15,12:00,18:30` with `VK_AUTO_IMPORT_LIMIT=15` by default, so queue draining relies on cadence instead of oversized single runs and stays away from the `08:00` daily announcement window and late-evening monitoring.

If you see skip notifications in admin chat often, spread the schedules further instead of switching to “wait”: skipping is a safety net, not a planning tool.

Skipped heavy-job attempts are now also written to `ops_run.status='skipped'` (with a reason), so `/general_stats` can show that the scheduler tried to start a job but skipped it before the job body ran.

For admin-facing scheduled reports, the bot now resolves the target chat from the superadmin row in SQLite first; `ADMIN_CHAT_ID` is only a bootstrap/legacy fallback.

## Jobs

- **partner reminders** – reminds inactive partners after 09:00 local time.
- **cleanup old events** – removes past events after 03:00 local time and notifies the superadmin.
- **general stats** – daily operational system report (`/general_stats`) for the previous 24 hours.
- **Telegram daily announcements** – posts `/daily` channel announcements after configured `daily_time`; scheduler has per-channel in-process dedup guard (inflight + sent-today cache) to prevent repeated sends while one run is still in progress.
- **VK daily posts and polls** – publishes daily announcements and festival polls when posting times are reached and a VK group is configured.
- **VK auto queue import** – imports queued VK posts (`vk_inbox`) via Smart Update on a fixed schedule when enabled.
- **Telegraph pages sync** – refreshes month and weekend Telegraph pages after 01:00 local time. Disabled by default; enable with `ENABLE_NIGHTLY_PAGE_SYNC=1`. Nightly runs update both page content and the month navigation block.
- **Telegraph cache sanitizer** – probes and warms Telegram web preview for Telegraph pages (via Kaggle/Telethon), tracks missing `cached_page` (Instant View) and warns on missing preview `photo`, and enqueues rebuilds for persistent “no cached_page” failures. Skips past pages (ended events / past weekends / past months). Manual `/telegraph_cache_sanitize` updates a single Kaggle status message while polling (like `/tg`), scheduled runs post a final summary to `ADMIN_CHAT_ID` when configured. Disabled by default; enable with `ENABLE_TELEGRAPH_CACHE_SANITIZER=1`.
- **festival navigation rebuild** – rebuilds festival navigation and landing page nightly.
- **festival queue processing** – processes the festival queue (VK/TG/site sources) on a fixed schedule when enabled.
- **ticket sites queue** – scans ticket-site URLs discovered in Telegram posts (pyramida.info / домискусств.рф / qtickets) via Kaggle and enriches events through Smart Update.
- **source parsing** – nightly + midday `/parse` runs when enabled (midday skips Kaggle if source pages did not change).
- **3D previews** – scheduled `/3di` run for “new” events:
  - events without `preview_3d_url` and with `photo_count >= 2`;
  - events whose 3D preview was invalidated because the illustration set changed (Smart Update clears `preview_3d_url` when `photo_urls` change).
- **Telegram monitoring** – scheduled daily import from Telegram sources (channels/groups) via Kaggle when enabled.
- **Guide excursions monitoring** – scheduled guide-only Kaggle scans when `ENABLE_GUIDE_EXCURSIONS_SCHEDULED=1`.
  - if `ENABLE_GUIDE_DIGEST_SCHEDULED=1`, the same successful `full` run immediately publishes `new_occurrences` after server-side import instead of using a separate cron slot.
- **Video announce `/v - Тест завтра`** – optional scheduled automatic test-render when `ENABLE_V_TEST_TOMORROW_SCHEDULED=1`.
  - uses the same `VideoAnnounceScenario.run_tomorrow_pipeline(... test_mode=True)` path as manual `/v`;
  - renders up to `12` scenes and sends the result to the configured `test` channel, or falls back to the operator/superadmin chat if no `test` channel is configured;
  - recommended default window: `21:20 Europe/Kaliningrad`, so it stays away from VK auto-import and leaves buffer before nightly Telegram monitoring.
- **kaggle recovery** – resumes in-flight Kaggle jobs after restarts, including `tg_monitoring` and `guide_monitoring`.

## Health Checks

- Fly probes `GET /healthz` every 15 seconds.
- `/healthz` no longer returns a blind static `ok`: it verifies that startup completed, the runtime heartbeat is fresh, required background tasks (`daily_scheduler`, `add_event_watch`, and `job_outbox_worker` when enabled) are alive, the bot session is open, and SQLite answers `SELECT 1`.
- `add_event_watch` is allowed to restart a stalled add-event worker in place; the watchdog now updates the shared dequeue timestamp correctly instead of tripping an `UnboundLocalError` during stall recovery and poisoning `/healthz`.
- If any of those checks fail, `/healthz` returns `503` with a JSON payload describing the failing component. This lets Fly recycle machines that are still serving HTTP but stopped processing Telegram webhooks or scheduler loops correctly.

## Environment variables

- `SCHED_HEAVY_GUARD_MODE` – scheduled heavy jobs gate mode: `skip` (default), `wait`, or `off`.
- `SCHED_HEAVY_TRY_TIMEOUT_SEC` – try-acquire timeout in seconds for `SCHED_HEAVY_GUARD_MODE=skip` (default: `0.2`).
- `SCHED_SERIALIZE_HEAVY_JOBS` – legacy flag: when enabled (`1|true|yes|on`) it implies `SCHED_HEAVY_GUARD_MODE=wait` + extra in-scheduler serialization.
- `VK_USER_TOKEN` – user token for VK posts (scopes: wall,groups,offline).
- `VK_TOKEN` – optional group token used as a fallback.
- `EVBOT_DEBUG` – enables extra logging and queue statistics.
- `ENABLE_SOURCE_PARSING` – enable nightly source parsing schedule.
- `SOURCE_PARSING_TIME_LOCAL` / `SOURCE_PARSING_TZ` – nightly parse time in local time zone.
- `ENABLE_SOURCE_PARSING_DAY` – enable midday source parsing schedule.
- `SOURCE_PARSING_DAY_TIME_LOCAL` / `SOURCE_PARSING_DAY_TZ` – midday parse time in local time zone.
- `ENABLE_3DI_SCHEDULED` – enable scheduled `/3di` runs.
- `THREEDI_TIMES_LOCAL` / `THREEDI_TZ` – `/3di` schedule times in local time zone.
- `ENABLE_GENERAL_STATS` – enable scheduled `/general_stats` report.
- `GENERAL_STATS_TIME_LOCAL` / `GENERAL_STATS_TZ` – `/general_stats` schedule time in local time zone.
- `ENABLE_TELEGRAPH_CACHE_SANITIZER` – enable scheduled Telegraph cache sanitizer.
- `TELEGRAPH_CACHE_TIME_LOCAL` / `TELEGRAPH_CACHE_TZ` – Telegraph cache sanitizer schedule time in local time zone.
- `TELEGRAPH_CACHE_DAYS_BACK` / `TELEGRAPH_CACHE_DAYS_FORWARD` – active window for collecting pages to probe.
- `TELEGRAPH_CACHE_LIMIT_EVENTS` / `TELEGRAPH_CACHE_LIMIT_FESTIVALS` – max number of event/festival pages to probe per run (defaults to safe values).
- `TELEGRAPH_CACHE_REGEN_AFTER_RUNS` – enqueue rebuilds after N consecutive failing sanitizer runs (default `2`).
- `ENABLE_TG_MONITORING` – enable daily Telegram monitoring job.
- `TG_MONITORING_TIME_LOCAL` / `TG_MONITORING_TZ` – Telegram monitoring schedule time in local time zone.
- `ENABLE_GUIDE_EXCURSIONS_SCHEDULED` – enable guide-only scheduled scans.
- `GUIDE_EXCURSIONS_LIGHT_TIMES_LOCAL` / `GUIDE_EXCURSIONS_FULL_TIME_LOCAL` / `GUIDE_EXCURSIONS_TZ` – guide monitoring light/full schedule in local time zone.
- `ENABLE_GUIDE_DIGEST_SCHEDULED` – after a successful scheduled `full` guide scan, automatically publish the `new_occurrences` digest in the same job instead of a separate cron slot.
- `ENABLE_V_TEST_TOMORROW_SCHEDULED` – enable scheduled automatic `/v - Тест завтра`.
- `V_TEST_TOMORROW_TIME_LOCAL` / `V_TEST_TOMORROW_TZ` – local schedule for automatic `/v - Тест завтра`.
- `V_TEST_TOMORROW_PROFILE` – video profile key for the scheduled `/v` run (default: `default`).
- `ENABLE_FESTIVAL_QUEUE` – enable festival queue schedule (disabled by default; next release keep off).
- `FESTIVAL_QUEUE_TIMES_LOCAL` / `FESTIVAL_QUEUE_TZ` – festival queue schedule times (default `03:30,16:30` local).
- `FESTIVAL_QUEUE_LIMIT` – optional limit of queue items per run.
- `ENABLE_TICKET_SITES_QUEUE` – enable scheduled ticket-sites queue processing.
- `TICKET_SITES_QUEUE_TIME_LOCAL` / `TICKET_SITES_QUEUE_TZ` – ticket-sites queue schedule time (default `11:20` local).
- `TICKET_SITES_QUEUE_LIMIT` – optional limit of queue items per scheduled run.
- `TICKET_SITES_QUEUE_INTERVAL_HOURS` – how often to rescan each URL after a successful run (default `24`).
- `ENABLE_VK_AUTO_IMPORT` – enable VK inbox auto import job.
- `VK_AUTO_IMPORT_TIMES_LOCAL` / `VK_AUTO_IMPORT_TZ` – VK auto-import schedule times in local time zone.
- `VK_AUTO_IMPORT_LIMIT` – max number of VK inbox rows to process per scheduled run (default `15`).
- `ENABLE_KAGGLE_RECOVERY` – enable background Kaggle recovery loop.
- `KAGGLE_RECOVERY_INTERVAL_MINUTES` – recovery interval in minutes (default: 5).
- `KAGGLE_JOBS_PATH` – path to Kaggle recovery registry JSON (default: `/data/kaggle_jobs.json`).
- `RUNTIME_HEALTH_HEARTBEAT_SEC` – how often the in-process runtime heartbeat updates (default: `15` seconds).
- `RUNTIME_HEALTH_STALE_SEC` – max allowed heartbeat age before `/healthz` turns unhealthy (default: `45` seconds, minimum `2x` heartbeat interval).
- `RUNTIME_HEALTH_STARTUP_GRACE_SEC` – startup grace window before “not ready yet” becomes a failing `/healthz` condition (default: `120` seconds).

To monitor real job durations, use the daily `/general_stats` report: it prints per-run `took=...` for `vk_auto_import` and `tg_monitoring` (and other ops-run instrumented jobs).
