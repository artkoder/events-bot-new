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

Each event stores optional ticket information (`ticket_price_min`, `ticket_price_max`, `ticket_link`) together with the cached vk.cc short link (`vk_ticket_short_url`) and the associated stats key (`vk_ticket_short_key`). If the event was forwarded from a channel, the link to that post is saved in `source_post_url`.
Free events are marked with `is_free`. Telegraph pages are stored with both URL and path so they can be updated when the event description changes. If a message includes images (under 5&nbsp;MB each), they are uploaded to Catbox and embedded at the start of the source page. Under the cover the bot renders a «Быстрые факты» block summarizing key fields: the event date and time (or the closing date for ongoing exhibitions), the normalized location, and ticket information with a registration or ticket link when available. Each line is omitted when the underlying data is missing so moderators can see which facts are optional.
Month pages list upcoming events. When their content exceeds about 64&nbsp;kB the bot creates a second page and links to it from the first.
Events also keep `event_type` (one of eight categories: спектакль, выставка, концерт, ярмарка, лекция, встреча, мастер-класс, кинопоказ) and an `emoji` suggested by the LLM. Multi-day events store `end_date` and appear with "Открытие" or "Закрытие" on the respective days. `/exhibitions` lists active exhibitions.
`pushkin_card` marks events that accept the Пушкинская карта.
`ics_url` stores a link to a calendar file uploaded to Supabase. Moderators can generate or remove this file when editing an event. Calendar files are named `Event-<id>-dd-mm-yyyy.ics` and include a link back to the event page.
When present the link is inserted into the Telegraph source page below the title image so readers can quickly add the event to their phone calendar.
If a text describes several events at once the LLM returns an array of event objects and the bot creates separate entries and Telegraph pages for each of them.
Channels where the bot is admin are tracked in the `channel` table. Use `/setchannel` to choose an admin channel and mark it as an announcement source. The `/channels` command lists all admin channels and shows which ones are registered.
`docs/LOCATIONS.md` contains standard venue names; its contents are appended to the 4o prompt so events use consistent `location_name` values.

## Poster OCR pipeline

- Poster media is uploaded to Catbox once per unique image; the resulting bytes feed both the Telegraph page and the OCR stage.
- `poster_ocr.recognize_posters` caches results in `PosterOcrCache` by hash, detail level and model so retries reuse the stored text and token counts.
- Daily usage is tracked in the `OcrUsage` table and compared against the 10 000 000-token budget. Cached entries keep working, while new, uncached OCR requests are blocked until the quota resets.
- Recognized text is saved in `EventPoster` rows and injected into the downstream LLM pipeline so 4o sees both the operator draft and poster contents.

## Video Announce pipeline

The video announce feature generates promotional video clips with event highlights:

- **Session management** — `VideoAnnounceSession` and `VideoAnnounceItem` track selection state, rendering status and publication history.
- **Candidate selection** — `selection.py` ranks events by topic relevance, date proximity and manual boost (`🎬` counter). LLM generates the intro text.
- **Pattern preview** — `pattern_preview.py` renders client-side PNG previews of three intro patterns (`STICKER`, `RISING`, `COMPACT`) without Kaggle.
- **Payload generation** — `payload_as_json()` produces the JSON for the Kaggle kernel, including `cities`, `date`, `pattern`, and scene data.
- **Kaggle rendering** — the kernel (`kaggle/VideoAfisha/video_afisha.ipynb`) downloads assets, renders frames with MoviePy, and uploads the final video.
- **Publication** — once complete, the video is sent to the test or main channel; events are marked as published and their counters decremented.
