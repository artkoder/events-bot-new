# Changelog

## v0.1.0 – Deploy + US-02 + /tz
- Initial Fly.io deployment config.
- Moderator registration queue with approve/reject.
- Global timezone setting via `/tz`.

## v0.1.1 – Logging and 4o request updates
- Added detailed logging for startup and 4o requests.
- Switched default 4o endpoint to OpenAI chat completions.
- Documentation now lists `FOUR_O_URL` secret.

## v0.2.0 – Event listing
- `/events` command lists events by day with inline delete buttons.

## v0.2.1 – Fix 4o date parsing
- Include the current date in LLM requests so events default to the correct year.

## v0.2.2 – Telegraph token helper
- Automatically create a Telegraph account if `TELEGRAPH_TOKEN` is not set and
  save the token to `/data/telegraph_token.txt`.
## v0.3.0 - Edit events and ticket info
- Added ticket price fields and purchase link
- Inline edit via /events
- Duplicate detection improved with 4o

## v0.3.1 - Forwarded posts
- Forwarded messages from moderators trigger event creation
- Events keep `source_post_url` linking to the original announcement

## v0.3.2 - Channel registration
- `/setchannel` registers a forwarded channel for source links
- `/channels` lists admin channels with removal buttons
- Bot tracks admin status via `my_chat_member` updates
