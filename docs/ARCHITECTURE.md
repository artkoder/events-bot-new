# Architecture

The bot is built with **aiogram 3** and runs on Fly.io using a webhook.

- **Web Server** – aiohttp application that receives updates on `/webhook`.
- **Bot Framework** – aiogram Dispatcher handles commands and callback queries.
- **Database** – SQLite accessed through SQLModel and `aiosqlite`.
- **Deployment** – Docker container on Fly.io with volume `data` for the database.

For PR-1 the bot implements registration queue and timezone setting. Future
sprints will extend it with event parsing and Telegraph pages.
