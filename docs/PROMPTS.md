# Prompt for model 4o

This repository uses an external LLM (model **4o**) for text parsing and
normalisation. The current instruction set for the model is stored here so that
it can be refined over time.

```
MASTER-PROMPT for Codex ― Telegram Event Bot
You receive long multi-line text describing one **or several** events.
Extract structured information and respond **only** with JSON.
If multiple events are found, return an array of objects. Each object uses these keys:
title             - name of the event
short_description - one-sentence summary
festival          - festival name or empty string
date              - single date or range (YYYY-MM-DD or YYYY-MM-DD..YYYY-MM-DD)
time              - start time or time range (HH:MM or HH:MM..HH:MM)
location_name     - venue name, use standard directory form if known
location_address  - street address if present
city              - city name
ticket_price_min  - minimum ticket price as integer or null
ticket_price_max  - maximum ticket price as integer or null
ticket_link       - URL for purchasing tickets if present
is_free           - true if explicitly stated the event is free
event_type       - one of: спектакль, выставка, концерт, ярмарка, лекция, встреча
emoji            - an optional emoji representing the event
end_date         - end date for multi-day events or null
When a range is provided, put the start date in `date` and the end date in `end_date`.
```

Do **not** include words like "Открытие" or "Закрытие" in exhibition titles.
The bot adds these markers automatically on the opening and closing dates.

Lines from `docs/LOCATIONS.md` are appended to the system prompt so the model
can normalise venue names. Please keep that file up to date.

The user message will start with the current date, e.g. "Today is
2025-07-05." Use this information to resolve missing years. **Ignore and do not
include any event whose date is earlier than today.**

Guidelines:
- If the event text does not specify a year, assume it happens in the current
  year.
- Omit any events dated before today.
- Respond with **plain JSON only** &mdash; do not wrap the output in code
  fences.

All fields must be present. No additional text.

Edit this file to tweak how requests are sent to 4o.
