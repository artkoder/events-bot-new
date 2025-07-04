# Architecture

The bot is built with **aiogram 3** and runs on Fly.io using a webhook.

- **Web Server** – aiohttp application that receives updates on `/webhook`.
- **Bot Framework** – aiogram Dispatcher handles commands and callback queries.

- **Database** – SQLite accessed through SQLModel and `aiosqlite`. The default
  path is `/data/db.sqlite` mounted from a Fly volume.
- **Deployment** – Docker container on Fly.io with volume `data` attached to
  `/data`.


For PR‑1 the bot implements registration queue and timezone setting. Basic event
creation (`/addevent` and `/addevent_raw`) stores events in SQLite. A helper
`python main.py test_telegraph` checks Telegraph access.

