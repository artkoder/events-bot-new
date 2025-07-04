# Bot Commands

| Command | Arguments | Description |
|---------|-----------|-------------|
| `/start` | - | Register the first user as superadmin or display status. |
| `/register` | - | Request moderator access if slots (<10) are free. |
| `/requests` | - | Superadmin sees pending registrations with approve/reject buttons. |
| `/tz <±HH:MM>` | required offset | Set timezone offset (superadmin only). |
| `/addevent <text>` | event description | Parse text with model 4o and store a new event. The bot replies with all parsed fields. |
| `/addevent_raw <title>|<date>|<time>|<location>` | manual fields | Add event without LLM. The bot echoes the stored fields. |
| `/ask4o <text>` | any text | Send query to model 4o and show plain response (superadmin only). |
| `/events [DATE]` | optional date `YYYY-MM-DD` or `DD.MM.YYYY` | List events for the day with delete buttons. |
| `python main.py test_telegraph` | - | Verify Telegraph API access. Automatically creates a token if needed and prints the page URL. |

Use `/addevent` to let model 4o extract fields. `/addevent_raw` lets you
input simple data separated by `|` pipes.
