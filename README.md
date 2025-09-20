# Events Bot

Telegram bot for publishing event announcements. Daily announcements can also be posted to a VK group.
Use `/regdailychannels` and `/daily` to manage both Telegram channels and the VK group including posting times.

Superadmins can use `/vk` to manage VK Intake: add or list sources, check or review events, and open the queue summary. The review UI highlights the bucket (URGENT/SOON/LONG/FAR) that produced the current card so operators understand the selection. Команда `/ocrtest` помогает сравнить распознавание афиш между `gpt-4o-mini` и `gpt-4o` на загруженных постерах.

## VK Intake & Review v1.1

Commands:

- `/vk` — add/list sources, check/review events, and open queue summary.
- `/vk_queue` — show VK inbox summary (pending/locked/skipped/imported/rejected) and a "🔎 Проверить события" button to start the review flow.
- `/vk_crawl_now` — run VK crawling now (admin only); reports "добавлено N, всего постов M" to the admin chat.

Background crawling collects posts from configured VK communities and filters
them by event keywords and date patterns. Matching posts land in the persistent
`vk_inbox` queue where an operator can accept, enrich with extra info, reject or
skip a candidate. Accepted items go through the standard import pipeline to
create a Telegraph page and calendar links. After each import the admin chat
receives links to the generated Telegraph and ICS pages, and the operator can
manually repost the source to the Afisha VK group via a dedicated button. Choosing
the ✂️ "Short post" action shows the draft with Publish/Edit buttons in the same
chat where the operator clicked. The
batch can be finished with "🧹 Завершить…" which sequentially rebuilds all
affected month pages. Operators can run `/vk_queue` to see current inbox counts
and get a button to start reviewing candidates.

### Bucket windows and priorities

Each queue item is assigned to a time-based bucket by comparing its `event_ts_hint` with the current moment:

- **URGENT** – events happening right now or within the next 48 hours (`VK_REVIEW_URGENT_MAX_H`). These are always served first if any exist.
- **SOON** – events between the urgent horizon and 14 days ahead (`VK_REVIEW_SOON_MAX_D`).
- **LONG** – events between the SOON limit and 30 days ahead (`VK_REVIEW_LONG_MAX_D`).
- **FAR** – events with hints beyond the LONG limit or without a parsed date at all.

Items with hints older than two hours (`VK_REVIEW_REJECT_H`) are automatically rejected and, if the queue is empty, previously skipped cards are re-opened. Within the SOON/LONG/FAR buckets the reviewer sees cards chosen by a weighted lottery that multiplies the bucket size by its weight (`VK_REVIEW_W_SOON=3`, `VK_REVIEW_W_LONG=2`, `VK_REVIEW_W_FAR=6`). After five non-FAR selections (`VK_REVIEW_FAR_GAP_K=5`) the system forces a FAR pick as a streak breaker so distant events do not get starved.

Within the winning bucket, cards are ordered by date with a tiny per-`group_id` jitter scaled by the square root of that group’s queue size. This spreads reviews across sources even when one community produces a large batch of similar events.

Reposts use images from the original post, link and doc attachments rely on their
previews, and only preview frames from videos are shown—video files are never
downloaded.

This is an MVP using **aiogram 3** and SQLite. It is designed for deployment on
Fly.io with a webhook.

Forwarded posts from moderators or admins are treated the same as the `/addevent` command.
Use `/setchannel` to pick one of the channels where the bot has admin rights and register it either as an announcement source or as the calendar asset channel. `/channels` lists all admin channels and lets you disable these roles. When the asset channel is set, forwarded posts from it get a **Добавить в календарь** button linking to the calendar file.

Bot messages display dates in the format `DD.MM.YYYY`. Public pages such as
Telegraph posts use the short form "D месяц" (e.g. `2 июля`).

Dates are shown as `DD.MM.YYYY` in bot messages. Telegraph pages and other
public posts use the format "D месяц" (for example, "2 июля").

See `docs/COMMANDS.md` for available bot commands, including `/events` to
browse upcoming announcements. The command accepts dates like `2025-07-10`,
`10.07.2025` or `2 августа`.

## Quick start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run locally (set `WEBHOOK_URL` to the public HTTPS address you plan to use):
   ```bash
   export TELEGRAM_BOT_TOKEN=xxx
   export WEBHOOK_URL=https://your-app.fly.dev
   export DB_PATH=/data/db.sqlite
   export FOUR_O_TOKEN=sk-...
  export FOUR_O_URL=https://api.openai.com/v1/chat/completions
  export SUPABASE_URL=https://<project>.supabase.co
  export SUPABASE_KEY=service_role_key
  # Optional: custom bucket name (defaults to events-ics)
  export SUPABASE_BUCKET=events-ics
  # Optional: provide Telegraph token. If omitted, the bot creates an account
  # automatically and saves the token to /data/telegraph_token.txt.
  export TELEGRAPH_TOKEN=your_telegraph_token
  # User access token for VK posts (scopes: wall,groups,offline)
  export VK_USER_TOKEN=vk_user_token
  # IDs of VK groups (without @)
  export VK_MAIN_GROUP_ID=123
  export VK_AFISHA_GROUP_ID=231828790
  # Group tokens
  export VK_TOKEN=vk_group_token
  export VK_TOKEN_AFISHA=vk_afisha_group_token
  # Optional: server-side token for read-only VK calls
  export VK_SERVICE_TOKEN=vk_service_token
  # Optional: disable service token reads (default true)
  export VK_READ_VIA_SERVICE=true
  # Optional: throttle VK requests (ms between calls, default 350)
  export VK_MIN_INTERVAL_MS=350
  # Optional: override VK API version (default 5.199)
  export VK_API_VERSION=5.199
  # Optional: max photos per VK post (default 10)
  export VK_MAX_ATTACHMENTS=10
  # Sending images to VK is disabled by default. Use /vkphotos to enable it.
  # Optional behaviour tuning
  export VK_ACTOR_MODE=auto  # group|user|auto

  # Weekly post edit scheduling
  export WEEK_EDIT_MODE=deferred  # immediate|deferred
  export WEEK_EDIT_CRON=02:30     # HH:MM local time
  # Optional: fine-grained weekly edit control
  export VK_WEEK_EDIT_ENABLED=false
  export VK_WEEK_EDIT_SCHEDULE=02:10
  export VK_WEEK_EDIT_TZ=Europe/Kaliningrad

  # Captcha handling parameters
  export CAPTCHA_WAIT_S=600
  export CAPTCHA_MAX_ATTEMPTS=2
  export CAPTCHA_NIGHT_RANGE=00:00-07:00
  export CAPTCHA_RETRY_AT=08:10
  export VK_CAPTCHA_TTL_MIN=60
  # optional quiet hours for captcha notifications (HH:MM-HH:MM)
  export VK_CAPTCHA_QUIET=
  # VK intake tuning (defaults shown)
  export VK_CRAWL_PAGE_SIZE=30
  export VK_CRAWL_MAX_PAGES_INC=1
  export VK_CRAWL_OVERLAP_SEC=300
  export VK_CRAWL_PAGE_SIZE_BACKFILL=50
  export VK_CRAWL_MAX_PAGES_BACKFILL=3
  export VK_CRAWL_BACKFILL_DAYS=14
  export VK_CRAWL_BACKFILL_AFTER_IDLE_H=24
  export VK_CRAWL_JITTER_SEC=600
  export VK_REVIEW_REJECT_H=2         # reject hints older than this many hours
  export VK_REVIEW_URGENT_MAX_H=48    # URGENT bucket spans up to this many hours ahead
  export VK_REVIEW_SOON_MAX_D=14      # SOON bucket reaches this many days ahead
  export VK_REVIEW_LONG_MAX_D=30      # LONG bucket extends to this many days ahead
  export VK_REVIEW_W_SOON=3           # weight for SOON in the bucket lottery
  export VK_REVIEW_W_LONG=2           # weight for LONG in the bucket lottery
  export VK_REVIEW_W_FAR=6            # weight for FAR in the bucket lottery
  export VK_REVIEW_FAR_GAP_K=5        # force FAR after this many non-FAR picks
  # keyword matching: regex by default; set to true to use pymorphy3 lemmas
  export VK_USE_PYMORPHY=false
  python main.py
  ```

By default the crawler uses regular-expression stems to detect event keywords.
Setting `VK_USE_PYMORPHY=true` (and installing `pymorphy3`) switches matching to
lemmatised forms for better coverage of Russian morphology.

## Service token

A VK service (server) token helps keep read-only API traffic away from the user token.

- **Why?** Crawling and preparing reposts rely on `wall.get*` and similar methods; using the service token reduces captcha prompts on these safe reads.
- **How do I enable it?** Set the `VK_SERVICE_TOKEN` secret and keep `VK_READ_VIA_SERVICE=true` (the default). Without the secret the bot behaves as before.
- **What does it cover?** Only public, read-only endpoints such as `utils.resolveScreenName`, `groups.getById`, `wall.get`, `wall.getById`, `photos.getById`, and `video.get*`. Publishing still uses user/group tokens.

## Deployment on Fly.io

1. Initialize app (once):
   ```bash
   fly launch
   fly volumes create data --size 1
   ```
2. Set secrets (the bot requires `WEBHOOK_URL` for webhook registration):
   ```bash
   fly secrets set TELEGRAM_BOT_TOKEN=xxx
   fly secrets set WEBHOOK_URL=https://<app>.fly.dev
   fly secrets set FOUR_O_TOKEN=xxxxx
    fly secrets set FOUR_O_URL=https://api.openai.com/v1/chat/completions
    fly secrets set DB_PATH=/data/db.sqlite
    # Optional: enable calendar files
    fly secrets set SUPABASE_URL=https://<project>.supabase.co
   fly secrets set SUPABASE_KEY=service_role_key
   # Optional: use your own Telegraph token. If not set, a new account will be
   # created on first run and the token saved to the data volume.
   fly secrets set TELEGRAPH_TOKEN=<token>
   # User access token for VK posts (scopes: wall,groups,offline)
   fly secrets set VK_USER_TOKEN=<token>
   # IDs of VK groups (without @)
   fly secrets set VK_MAIN_GROUP_ID=<id>
   fly secrets set VK_AFISHA_GROUP_ID=<id>
   # Group tokens
   fly secrets set VK_TOKEN=<token>
   fly secrets set VK_TOKEN_AFISHA=<token>
   # Optional: server-side token for read-only VK calls
   fly secrets set VK_SERVICE_TOKEN=<token>
   fly secrets set VK_READ_VIA_SERVICE=true
   fly secrets set VK_MIN_INTERVAL_MS=350
   # Optional: max photos per VK post
   fly secrets set VK_MAX_ATTACHMENTS=10
   # Sending images to VK is disabled by default. Toggle with /vkphotos.
   ```
3. Deploy:
   ```bash
   fly deploy
   ```
   > **Note:** If the app's average memory (RSS) approaches ~140&nbsp;MB, choose a
   > larger memory tier to avoid out-of-memory kills. On Apps V2 use `fly scale
   > memory 512`; on Machines:
   > `fly machines update -m 512 -c shared-cpu-1x <machine-id>`.

4. Optional tuning:
   - Enable swap if needed (for example `--swap 256`).
   - The app exposes a `/healthz` endpoint used by Fly health checks.
   - Avoid heavy requests at startup; warm caches lazily in background workers.

## Files
- `docs/COMMANDS.md` – full list of bot commands.
- `docs/USER_STORIES.md` – user stories.
- `docs/ARCHITECTURE.md` – system architecture.
- `docs/PROMPTS.md` – base prompt for model 4o (edit this for parsing rules).
- `docs/FOUR_O_REQUEST.md` – how requests to 4o are formed.
- `docs/LOCATIONS.md` – list of standard venues used when parsing events.
- `docs/RECURRING_EVENTS.md` – design notes for repeating events.
- `CHANGELOG.md` – project history.

Each added event stores the original announcement text in a Telegraph page. The link is shown when the event is added and in the `/events` listing. Events may also contain ticket prices and a purchase link. Use the edit button in `/events` to change any field.
Links from the announcement text are preserved on the Telegraph page whenever possible so readers can follow the original sources.
If the original message contains photos (under 5&nbsp;MB), they are uploaded to Catbox and displayed on the Telegraph page.
The project uses `telegraph>=2.2.0`. `create_page` returns the page `url` and `path`;
only `edit_page(path=...)` accepts a `path` argument when updating existing pages.
Editing an event lets you create or delete an ICS file for calendars. The file is uploaded to Supabase when `SUPABASE_URL` and `SUPABASE_KEY` are set. Files are named `Event-<id>-dd-mm-yyyy.ics` and include a link back to the event. Set `SUPABASE_BUCKET` if you use a bucket name other than `events-ics`.
When a calendar file exists the Telegraph page shows a link right under the title image: "📅 Добавить в календарь".
Events may note support for the Пушкинская карта, shown as a separate line in postings.
Run `/exhibitions` to see all ongoing exhibitions (events with a start and end date).

To verify Telegraph access manually run:
```bash
python main.py test_telegraph
```
The command prints the created page URL and confirms that editing works.

## Backup and restore

Use `/dumpdb` to download a SQL dump of the current database. When a Telegraph
token file exists the bot also sends `telegraph_token.txt`. The reply lists all
connected channels and detailed steps to set up the bot elsewhere. Copy the
token file to `/data/telegraph_token.txt` on the new host and send `/restore`
with the dump file attached to load the database.

## Telegraph caching

Telegram desktop may cache the first version of a Telegraph page and ignore
edits. Opening the link in a browser or the mobile client shows the latest
content. There is no reliable API to refresh the cached preview without creating
a new page.

## Telegraph page size

Telegraph rejects pages larger than about 64&nbsp;kB. When a month contains too
many events the bot automatically splits the announcement into two pages. The
first one ends with a prominent link "<месяц> продолжение" leading to the second
page.

## Production tips

- Periodic jobs are staggered to avoid CPU and I/O spikes. Each scheduler task
  runs every 15 minutes with an individual offset and is limited to a single
  concurrent instance.
- Database access reuses a single SQLite connection and disables per-request
  ping queries for faster read operations. A read-only context manager is
  available for pure `SELECT` statements.


## Batch publishing and coalescing

When multiple events from the same festival are added in quick succession the
bot groups related aggregation jobs. Each group is identified by a
`coalesce_key`:

- `festival_pages:{festival_id}`
- `month_pages:{yyyy-mm}`
- `week_pages:{yyyy-ww}`
- `weekend_pages:{yyyy-mm-dd}`
- `vk_week_post:{yyyy-ww}` and `vk_weekend_post:{yyyy-mm-dd}`

If another task with the same key is scheduled before the first one runs the
payloads are merged and only a single job executes. Aggregated jobs wait for a
short debounce window (5–10 seconds) before running and respect the dependency
order:

`festival_pages` → `month_pages`/`week_pages`/`weekend_pages` →
`vk_week_post`/`vk_weekend_post`.

This makes job execution idempotent and prevents duplicate rebuilds when many
events are added at once.

Example log extract for a batch of events:

```
INFO TASK_START task=festival_pages coalesce_key=festival_pages:42
INFO TASK_START task=month_pages coalesce_key=month_pages:2025-08 depends_on=festival_pages:42
INFO TASK_START task=vk_week_post coalesce_key=vk_week_post:2025-34 depends_on=month_pages:2025-08
INFO TASK_DONE task=vk_week_post status=done changed=True
```

## Batch progress

A batch of events shares a single progress message. Event processing and each
aggregated page update contribute to the same card:

```
Events (Telegraph): X/N
Festival: ✅/❌
Month: ✅/❌, Week: ✅/❌, Weekend: ✅/❌
VK week/weekend posts: ✅/❌/⏸
```

Every item eventually resolves to a final state (success, error or paused) so
the progress card never ends with a spinner.

## VK captcha handling

VK API errors with code `14` trigger a captcha flow. The bot pauses all pending
VK jobs, sends the captcha image to the super admin and waits for input. Jobs
resume automatically after the correct code is supplied or fail after a timeout
if the captcha is ignored.

When a post fails with `method is unavailable with group auth` (or error codes
15, 200 or 203) the bot automatically retries using the user token before
giving up. The progress card shows a temporary pause icon `⏸` while waiting for
captcha input and resolves to `✅` or `❌` afterwards.

## Festival links on month pages

When a festival already has a Telegraph page (`telegraph_url` is set) the month
page renders the festival name as a clickable link.

## Month navigation footer

Each month page ends with a navigation block that lists every available month.
When a new month is created the navigation is rebuilt on all pages so that each
footer links to all month pages and the current month is shown without a link.

## Link formatting

Telegraph pages and "source" pages use `linkify_for_telegraph` to convert
plain URLs and patterns like `Name (https://example.com)` into clickable
anchors. VK posts pass the text through `sanitize_for_vk`, which exposes the
original URLs in parentheses and strips unsupported HTML and Telegram emoji so
the full address remains visible.


## Календарь (ICS)
When an event is added or updated the bot builds an `.ics` calendar file. The file is
uploaded to Supabase and posted to the asset channel as a document. Both steps use a
separate semaphore so they do not block VK or Telegraph jobs. If `SUPABASE_DISABLED` is
set the upload is skipped. When the content hasn't changed the previous Supabase URL and
Telegram `file_id` are reused without extra network requests.
