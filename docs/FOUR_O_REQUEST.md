# 4o Request Guide

This document describes how the bot communicates with model **4o**.

Requests are sent as HTTP `POST` to the URL stored in the environment variable
`FOUR_O_URL` (defaults to `https://api.openai.com/v1/chat/completions`). The
header `Authorization: Bearer <FOUR_O_TOKEN>` is added. Set these values via Fly
secrets.

Payload:
```json
{
  "model": "gpt-4o",
  "messages": [
    {"role": "system", "content": "<contents of PROMPTS.md>"},
    {"role": "user", "content": "Today is YYYY-MM-DD. <original event text>"}
  ]
}
```

The response must be JSON with the fields listed in `docs/PROMPTS.md`.
The prefix "Today is YYYY-MM-DD." helps the model infer the correct year for
dates that omit it.
Edit this file or `docs/PROMPTS.md` to fine‑tune the request details.

The command `/ask4o <text>` sends an arbitrary user message to the same
endpoint and returns the assistant reply. It is intended for quick diagnostics
and available only to the superadmin.

When a new event might duplicate an existing one (same date/time/city but
slightly different title or venue), the bot sends both versions to 4o asking if
they describe the same event. The model replies with JSON
`{"duplicate": true|false, "title": "", "short_description": ""}`. If
`duplicate` is true the returned title and description replace the stored event
fields.
