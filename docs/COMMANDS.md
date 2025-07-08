# Bot Commands

| Command | Arguments | Description |
|---------|-----------|-------------|
| `/start` | - | Register the first user as superadmin or display status. |
| `/register` | - | Request moderator access if slots (<10) are free. |
| `/requests` | - | Superadmin sees pending registrations with approve/reject buttons. |
| `/tz <±HH:MM>` | required offset | Set timezone offset (superadmin only). |
| `/addevent <text>` | event description | Parse text with model 4o and store one or several events. The original text is published to Telegraph (including the first photo or video if present) and the link is returned. Forwarded messages from moderators are processed the same way. |
| `/addevent_raw <title>|<date>|<time>|<location>` | manual fields | Add event without LLM. The bot also creates a Telegraph page with the provided text and optional attached photo. |
| `/ask4o <text>` | any text | Send query to model 4o and show plain response (superadmin only). |
| `/events [DATE]` | optional date `YYYY-MM-DD` or `DD.MM.YYYY` | List events for the day with delete and edit buttons. Dates are shown as `DD.MM.YYYY`. Choosing **Edit** lists all fields with inline buttons including a toggle for "Бесплатно". |
| `/setchannel` | - | Choose one of the admin channels to register as an announcement source. |
| `/channels` | - | List channels where the bot is admin and mark registered ones with a cancel button. |
| `/regdailychannels` | - | Choose admin channels for daily announcements (default 08:00). |

| `/daily` | - | Manage daily announcement channels: cancel, change time, test send. |

| `/exhibitions` | - | List active exhibitions similar to `/events`; each entry shows the period `c <start>` / `по <end>` and includes edit/delete buttons. |
| `/pages` | - | Show links to Telegraph month and weekend pages. |
| `python main.py test_telegraph` | - | Verify Telegraph API access. Automatically creates a token if needed and prints the page URL. |

Use `/addevent` to let model 4o extract fields. `/addevent_raw` lets you
input simple data separated by `|` pipes.
