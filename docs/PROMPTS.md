# Prompt for model 4o

This repository uses an external LLM (model **4o**) for text parsing and
normalisation. The current instruction set for the model is stored here so that
it can be refined over time.

```
MASTER-PROMPT for Codex ― Telegram Event Bot
You receive long multi-line text describing an event. Extract structured
information and respond **only** with JSON using these keys:
title             - name of the event
short_description - one-sentence summary
festival          - festival name or empty string
date              - single date or range (YYYY-MM-DD or YYYY-MM-DD..YYYY-MM-DD)
time              - start time or time range (HH:MM or HH:MM..HH:MM)
location_name     - venue name, use standard directory form if known
location_address  - street address if present
city              - city name
```

The user message will start with the current date, e.g. "Today is
2025-07-05." Use this information to resolve missing years so that parsed
events are never in the past.

Guidelines:
- If the event text does not specify a year, assume it happens in the current
  year.
- Respond with **plain JSON only** &mdash; do not wrap the output in code
  fences.

All fields must be present. No additional text.

Edit this file to tweak how requests are sent to 4o.
