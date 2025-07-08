# Architecture

The bot is built with **aiogram 3** and runs on Fly.io using a webhook.

- **Web Server** – aiohttp application that receives updates on `/webhook`.
- **Bot Framework** – aiogram Dispatcher handles commands and callback queries.
- **Database** – SQLite accessed through SQLModel and `aiosqlite`. The default
  path is `/data/db.sqlite` mounted from a Fly volume.
- **Deployment** – Docker container on Fly.io with volume `data` attached to
  `/data`.

The MVP includes moderator registration, timezone setting and simple event
creation (`/addevent` and `/addevent_raw`). For each event the bot creates a
Telegraph page containing the original announcement text. When the event comes
from a registered announcement channel the title on that page links to the
source post. `/events` shows upcoming events by day (with links to these pages)
and allows deletion or editing through inline buttons. A
helper `python main.py test_telegraph` checks Telegraph access and creates a
Telegraph token automatically if needed.

Each event stores optional ticket information (`ticket_price_min`, `ticket_price_max`, `ticket_link`). If the event was forwarded from a channel, the link to that post is saved in `source_post_url`.
Free events are marked with `is_free`. Telegraph pages are stored with both URL and path so they can be updated when the event description changes. If a message includes images (under 5&nbsp;MB each), they are uploaded to Catbox and embedded at the start of the source page.
Events also keep `event_type` (one of six categories) and an `emoji` suggested by the LLM. Multi-day events store `end_date` and appear with "Открытие" or "Закрытие" on the respective days. `/exhibitions` lists active exhibitions.
`pushkin_card` marks events that accept the Пушкинская карта.
If a text describes several events at once the LLM returns an array of event objects and the bot creates separate entries and Telegraph pages for each of them.
Channels where the bot is admin are tracked in the `channel` table. Use `/setchannel` to choose an admin channel and mark it as an announcement source. The `/channels` command lists all admin channels and shows which ones are registered.
`docs/LOCATIONS.md` contains standard venue names; its contents are appended to the 4o prompt so events use consistent `location_name` values.
