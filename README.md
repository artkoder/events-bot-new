# Events Bot

Telegram bot for publishing event announcements.

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
   ```
3. Deploy:
   ```bash
   fly deploy
   ```

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

Use `/dumpdb` to download a SQL dump of the current database. The bot replies
with a list of connected channels and the steps required to restore the dump on
another server. Send `/restore` with the dump file attached to load it back.

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

