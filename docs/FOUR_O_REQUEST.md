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

When a post is forwarded from a channel, its title is appended to the user
message on a new line. This helps the model infer the venue when it is omitted
in the text.

If `docs/LOCATIONS.md` exists, its lines are appended to the system prompt as a
list of known venues. This helps the model normalise `location_name` to a
standard form.

The response must be JSON with the fields listed in `docs/PROMPTS.md`. When the
text describes multiple events, return an array of such objects.
The prefix "Today is YYYY-MM-DD." helps the model infer the correct year for
dates that omit it and lets the model ignore any events scheduled before today.
When a post is forwarded from a Telegram channel, the channel title is added
before the announcement text as `Channel: <name>.` so the model can guess the
venue.
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

Festival pages also rely on 4o. When a festival is created, the bot sends the
full source text of the first event with the prompt “Сформируй описание
фестиваля <name> объёмом два‑три абзаца”. When more events are added, source
texts of several festival announcements (up to five) are combined and the model
is asked to write a new summary using only facts from these texts.
