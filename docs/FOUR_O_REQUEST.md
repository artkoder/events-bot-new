# 4o Request Guide


This document describes how the bot communicates with model **4o**.

Requests are sent as HTTP `POST` to the URL stored in the environment variable
`FOUR_O_URL` (defaults to `https://api.example.com/parse`). The header
`Authorization: Bearer <FOUR_O_TOKEN>` is added.

Payload:
```json
{ "text": "<original event text>", "prompt": "<contents of PROMPTS.md>" }
```

The response must be JSON with the fields listed in `docs/PROMPTS.md`.
Edit this file or `docs/PROMPTS.md` to fine‑tune the request details.

The command `/ask4o <text>` sends arbitrary text to the same endpoint and
returns the field `response` from the JSON reply. This is meant for quick
diagnostics and is available only to the superadmin.

