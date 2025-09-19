# Bot Commands

| Command | Arguments | Description |
|---------|-----------|-------------|
| `/start` | - | Register the first user as superadmin or display status. |
| `/register` | - | Request moderator access if slots (<10) are free. |
| `/help` | - | Show commands available for your role. |
| `/requests` | - | Superadmin sees pending registrations with approve/reject buttons. |
| `/tz <±HH:MM>` | required offset | Set timezone offset (superadmin only). |
| `/addevent <text>` | event description | Parse text with model 4o and store one or several events. The original text is published to Telegraph. Images up to 5&nbsp;MB are uploaded to Catbox and shown on that page. Forwarded messages from moderators are processed the same way. |
| `/addevent_raw <title>|<date>|<time>|<location>` | manual fields | Add event without LLM. The bot also creates a Telegraph page with the provided text and optional attached photo. |
| `/images` | - | Toggle uploading photos to Catbox. |
| `/vkgroup <id|off>` | required id or `off` | Set or disable VK group for daily announcements. |
| `/vktime today|added <HH:MM>` | required type and time | Change VK posting times (default 08:00/20:00). |
| `/vkphotos` | - | Toggle sending images to VK posts. |
| `↪️ Репостнуть в Vk` | - | Safe repost via `wall.post` with photo IDs. |
| `✂️ Сокращённый рерайт` | - | LLM-сжатый текст без фото, предпросмотр и правка перед публикацией. |
| `/ask4o <text>` | any text | Send query to model 4o and show plain response (superadmin only). |
| `/ocrtest` | - | Сравнить распознавание афиш между gpt-4o-mini и gpt-4o (только супер-админ). |
| `/events [DATE]` | optional date `YYYY-MM-DD`, `DD.MM.YYYY` or `D месяц [YYYY]` | List events for the day with delete and edit buttons. Dates are shown as `DD.MM.YYYY`. Choosing **Edit** lists all fields with inline buttons including a toggle for "Бесплатно". |
| `/setchannel` | - | Choose an admin channel and register it as an announcement or calendar asset source. |
| `/channels` | - | List admin channels showing registered and asset ones with disable buttons. |
| `/regdailychannels` | - | Choose admin channels for daily announcements and set the VK group. |
| `/daily` | - | Manage daily announcement channels and VK posting times; send test posts. |
| `/exhibitions` | - | List active exhibitions similar to `/events`; each entry shows the period `c <start>` / `по <end>` and includes edit/delete buttons. |
| `/digest` | - | Build lecture digest with images, toggles and quick send buttons (superadmin only). |
| `/pages` | - | Show links to Telegraph month and weekend pages. |
| `/fest [archive] [page]` | optional `archive` flag and page number | List festivals with edit/delete options. Ten rows are shown per page with navigation buttons. Use `archive` to view finished festivals that no longer have upcoming events; omit it to see active ones. |



| `/stats [events]` | optional `events` | Superadmin only. Show Telegraph view counts starting from the past month and weekend pages up to all current and future ones. Includes the festivals landing page and stats for upcoming or recently ended (within a week) festivals. Use `events` to list event page stats. |
| `/dumpdb` | - | Superadmin only. Download a SQL dump and `telegraph_token.txt` plus restore instructions. |
| `/restore` | attach file | Superadmin only. Replace current database with the uploaded dump. |

| `python main.py test_telegraph` | - | Verify Telegraph API access. Automatically creates a token if needed and prints the page URL. |

Use `/addevent` to let model 4o extract fields. `/addevent_raw` lets you
input simple data separated by `|` pipes.

## Event topics

The classifier assigns up to three topic identifiers to each event. The same labels
appear in the `/events` listing so moderators instantly see the context. Current
labels:

- `STANDUP` — «Стендап и комедия»
- `QUIZ_GAMES` — «Квизы и игры»
- `OPEN_AIR` — «Фестивали и open-air»
- `PARTIES` — «Вечеринки»
- `CONCERTS` — «Концерты»
- `MOVIES` — «Кино»
- `EXHIBITIONS` — «Выставки и арт»
- `THEATRE` — «Театр»
- `LECTURES` — «Лекции и встречи»
- `MASTERCLASS` — «Мастер-классы»
- `SCIENCE_POP` — «Научпоп»
- `HANDMADE` — «Хендмейд/маркеты/ярмарки/МК» (сюда попадают ярмарки и pop-up маркеты)
- `NETWORKING` — «Нетворкинг и карьера»
- `ACTIVE` — «Активный отдых и спорт»
- `PERSONALITIES` — «Личности и встречи»
- `KIDS_SCHOOL` — «Дети и школа»
- `FAMILY` — «Семейные события»

Редактируя события, ориентируйтесь на эти подписи: их видят админы и читатели.
