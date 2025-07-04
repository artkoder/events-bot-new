# Events Bot

Telegram bot for publishing event announcements.

This is an MVP using **aiogram 3** and SQLite. It is designed for deployment on
Fly.io with a webhook.


See `docs/COMMANDS.md` for available bot commands.

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
   fly secrets set DB_PATH=/data/db.sqlite
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
- `CHANGELOG.md` – project history.

To verify Telegraph access run:
```bash
python main.py test_telegraph
```


