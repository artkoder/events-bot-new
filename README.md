# Events Bot

Telegram bot for publishing event announcements. Daily announcements can also be posted to a VK group.
Use `/regdailychannels` and `/daily` to manage both Telegram channels and the VK group including posting times.

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
  # Optional: group token used as a fallback
  export VK_TOKEN=vk_group_token
  # Sending images to VK is disabled by default. Use /vkphotos to enable it.
  # Optional behaviour tuning
  export VK_ACTOR_MODE=auto  # group|user|auto
  export VK_FALLBACK_CODES="15,200,203"  # VK error codes for fallback
  export VK_FALLBACK_ON_CAPTCHA=false

  # Weekly post edit scheduling
  export WEEK_EDIT_MODE=deferred  # immediate|deferred
  export WEEK_EDIT_CRON=02:30     # HH:MM local time

  # Captcha handling parameters
  export CAPTCHA_WAIT_S=600
  export CAPTCHA_MAX_ATTEMPTS=2
  export CAPTCHA_NIGHT_RANGE=00:00-07:00
  export CAPTCHA_RETRY_AT=08:10
 python main.py
   ```

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
   # Optional: group token used as a fallback
   fly secrets set VK_TOKEN=<token>
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
When a calendar file exists the Telegraph page shows a link right under the title image: "📅 Добавить в календарь на телефоне (ICS)".
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

