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
Telegraph page with the raw announcement text. `/events` shows upcoming events by
day (with links to these pages) and allows deletion or editing through inline buttons. A
helper `python main.py test_telegraph` checks Telegraph access and creates a
Telegraph token automatically if needed.

Each event stores optional ticket information (`ticket_price_min`, `ticket_price_max`, `ticket_link`).
`docs/LOCATIONS.md` contains standard venue names; its contents are appended to
the 4o prompt so events use consistent `location_name` values.
