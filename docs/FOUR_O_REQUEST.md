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

When a post is forwarded from a channel or imported from a VK group, its title
is appended to the user message on a new line. This helps the model infer the
venue when it is omitted in the text.

If `docs/LOCATIONS.md` exists, its lines are appended to the system prompt as a
list of known venues. This helps the model normalise `location_name` to a
standard form.

The response must be JSON with the fields listed in `docs/PROMPTS.md`. When the
text describes multiple events, return an array of such objects.
The prefix "Today is YYYY-MM-DD." helps the model infer the correct year for
dates that omit it and lets the model ignore any events scheduled before today.
When a post is forwarded from a Telegram channel or imported from a VK group,
the channel or group title is added before the announcement text as
`Channel: <name>.` so the model can guess the venue.
Edit this file or `docs/PROMPTS.md` to fine‑tune the request details.

The command `/ask4o <text>` sends an arbitrary user message to the same
endpoint and returns the assistant reply. It is intended for quick diagnostics
and available only to the superadmin.

## Digest intro example

To compose the introductory phrase for the lecture digest the bot sends a
single-message chat completion describing the number of lectures, the horizon
(``недели``/``двух недель``) and up to three topic hints. The response is plain
text with 1–2 short sentences:

```
POST $FOUR_O_URL
{ "model": "gpt-4o", "messages": [{"role": "user", "content": "..."}] }
```

Response:

```
{"choices": [{"message": {"content": "Подобрали для вас ..."}}]}
```

When a new event might duplicate an existing one (same date/time/city but
slightly different title or venue), the bot sends both versions to 4o asking if
they describe the same event. The model replies with JSON
`{"duplicate": true|false, "title": "", "short_description": ""}`. If
`duplicate` is true the returned title and description replace the stored event
fields.

Festival pages also rely on 4o. To craft a festival blurb the bot sends the
previous description (if any) together with the full text of up to five recent
announcements. The prompt instructs the model:
The model also returns `festival_full` alongside `festival` so the bot can store
the edition name separately.
"Стиль профессионального журналиста в сфере мероприятий и культуры. Не
используй типовые штампы, не выдумывай факты. Описание должно состоять из трёх
предложений, если сведений мало — из одного". Only information from the
provided texts may appear in the summary.
If the description contains a date range like "с 27 августа по 6 сентября 2025",
these dates define the festival period. When no range is present the period is
calculated from the events added to the festival.
