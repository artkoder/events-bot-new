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

## v0.3.3 - Free events and telegraph updates
- Added `is_free` field with inline toggle in the edit menu.
- 4o parsing detects free events; if unclear a button appears to mark the event as free.
- Telegraph pages keep original links and append new text when events are updated.

## v0.3.4 - Calendar files
- Events can upload an ICS file to Supabase during editing.
- Added `ics_url` column and buttons to create or delete the file.
- Use `SUPABASE_BUCKET` to configure the storage bucket (defaults to `events-ics`).
- Calendar files include a link back to the event and are saved as `Event-<id>-dd-mm-yyyy.ics`.
- Telegraph pages show a calendar link under the main image when an ICS file exists.
- Startup no longer fails when setting the webhook times out.

## v0.3.5 - Calendar asset channel
- `/setchannel` lets you mark a channel as the calendar asset source.
- `/channels` shows the asset channel with a disable button.
- Calendar files are posted to this channel and linked from month and weekend pages.
- Forwarded posts from the asset channel show a calendar button.

## v0.3.6 - Telegraph stats

- `/stats` shows view counts for the past month and weekend pages, plus all current and upcoming ones.

- `/stats events` lists stats for event source pages sorted by views.


## v0.3.7 - Large month pages

- Month pages are split in two when the content exceeds ~64&nbsp;kB. The first
  half ends with a link to the continuation page.

## v0.3.8 - Daily announcement tweak

- Daily announcements no longer append a "подробнее" link to the event's
  Telegraph page.

## v0.3.9 - VK daily announcements

- Daily announcements can be posted to a VK group. Set the group with `/vkgroup` and adjust
  times via `/vktime`. Use the `VK_TOKEN` secret for API access.

## v0.3.10 - Unified daily management

- `/regdailychannels` and `/daily` now show the VK group alongside Telegram channels.
  VK posting times can be changed there and test posts sent.
- Daily announcements include new hashtag lines for Telegram and VK posts.

## v0.3.11 - VK formatting tweaks

- VK daily posts show a calendar icon before "АНОНС" and include more spacing between events.
- Date, time and location are italicized if supported.
- Prices include `руб.` and ticket links move to the next line.
- The "подробнее" line now ends with a colon and calendar links appear on their own line as
  "📆 Добавить в календарь: <link>".

## v0.3.12 - VK announcement fixes

- Remove unsupported italic tags and calendar line from VK posts.
- Event titles appear in uppercase and the "подробнее" link follows the
  description.
- A visible separator line now divides events to improve readability.

## v0.3.13 - VK formatting updates

- VK posts use two blank separator lines built with the blank braille symbol.
- Ticket links show a ticket emoji before the URL.
- Date lines start with a calendar emoji and the location line with a location pin.

## v0.3.14 - VK link cleanup

- Removed the "Мероприятия на" prefix from month and weekend links in VK daily posts.

## v0.3.15 - Channel name context

- Forwarded messages include the Telegram channel title in 4o requests so the
  model can infer the venue.
- `parse_event_via_4o` also accepts the legacy `channel_title` argument for
  compatibility.

## v0.3.16 - Festival pages

- Added a `Festival` model and `/fest` command for listing festivals.
- Daily announcements now show festival links.
- Logged festival-related actions including page creation and edits.
- Festival pages automatically include an LLM-generated description and can be
  edited or deleted via `/fest`.

## v0.3.17 - Festival description update

- Festival blurbs use the full text of event announcements and are generated in
  two or three paragraphs via 4o.

## v0.3.18 - Festival contacts

- Festival entries store website, VK and Telegram links.
- `/fest` shows these links and accepts `site:`, `vk:` and `tg:` edits.
- **Edit** now opens a menu to update description or contact links individually.

## v0.3.19 - Festival range fix

- LLM instructions clarified: when festival dates span multiple days but only
  some performances are listed, only those performances become events. The bot
  no longer adds extra dates unless every day is described.

## v0.3.20 - Festival full name

- Festivals now store both short and full names. Telegraph pages and VK posts
  use the full name while events and lists keep the short version.
- `/fest` gained edit options for these fields. Existing records are updated
  automatically with the short name as the default full one.

## v0.3.21 - Partner activity reminder

- Partners receive a weekly reminder at 9 AM if they haven't added events in
  the past seven days.
- The superadmin gets a list of partners who were reminded.






