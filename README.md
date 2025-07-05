# Events Bot

Telegram bot for publishing event announcements.

This is an MVP using **aiogram 3** and SQLite. It is designed for deployment on
Fly.io with a webhook.

See `docs/COMMANDS.md` for available bot commands, including `/events` to
browse upcoming announcements.

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
- `CHANGELOG.md` – project history.

Each added event stores the original announcement text in a Telegraph page. The link is shown when the event is added and in the `/events` listing.

To verify Telegraph access manually run:
```bash
python main.py test_telegraph
```
The command prints the created page URL and confirms that editing works.

