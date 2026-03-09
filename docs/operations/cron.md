# Cron Schedule

The bot uses APScheduler to run periodic maintenance tasks on a fixed schedule.

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
- `/3di` morning run: `THREEDI_TIMES_LOCAL=05:30,15:15,17:15` (was `03:15,15:15,17:15`)

If you see skip notifications in admin chat often, spread the schedules further instead of switching to “wait”: skipping is a safety net, not a planning tool.

## Jobs

- **partner reminders** – reminds inactive partners after 09:00 local time.
- **cleanup old events** – removes past events after 03:00 local time and notifies the superadmin.
- **general stats** – daily operational system report (`/general_stats`) for the previous 24 hours.
- **Telegram daily announcements** – posts `/daily` channel announcements after configured `daily_time`; scheduler has per-channel in-process dedup guard (inflight + sent-today cache) to prevent repeated sends while one run is still in progress. If Telegram returns `forbidden/chat not found`, the day is marked as handled (no late backfill when rights are restored).
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
- **kaggle recovery** – resumes in-flight Kaggle jobs after restarts.

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
- `ENABLE_FESTIVAL_QUEUE` – enable festival queue schedule (disabled by default; next release keep off).
- `FESTIVAL_QUEUE_TIMES_LOCAL` / `FESTIVAL_QUEUE_TZ` – festival queue schedule times (default `03:30,16:30` local).
- `FESTIVAL_QUEUE_LIMIT` – optional limit of queue items per run.
- `ENABLE_TICKET_SITES_QUEUE` – enable scheduled ticket-sites queue processing.
- `TICKET_SITES_QUEUE_TIME_LOCAL` / `TICKET_SITES_QUEUE_TZ` – ticket-sites queue schedule time (default `11:20` local).
- `TICKET_SITES_QUEUE_LIMIT` – optional limit of queue items per scheduled run.
- `TICKET_SITES_QUEUE_INTERVAL_HOURS` – how often to rescan each URL after a successful run (default `24`).
- `ENABLE_VK_AUTO_IMPORT` – enable VK inbox auto import job.
- `VK_AUTO_IMPORT_TIMES_LOCAL` / `VK_AUTO_IMPORT_TZ` – VK auto-import schedule times in local time zone.
- `VK_AUTO_IMPORT_LIMIT` – max number of VK inbox rows to process per scheduled run.
- `ENABLE_KAGGLE_RECOVERY` – enable background Kaggle recovery loop.
- `KAGGLE_RECOVERY_INTERVAL_MINUTES` – recovery interval in minutes (default: 5).
- `KAGGLE_JOBS_PATH` – path to Kaggle recovery registry JSON (default: `/data/kaggle_jobs.json`).

To monitor real job durations, use the daily `/general_stats` report: it prints per-run `took=...` for `vk_auto_import` and `tg_monitoring` (and other ops-run instrumented jobs).
