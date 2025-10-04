# Bot Commands

| Command | Arguments | Description |
|---------|-----------|-------------|
| `/start` | - | Register the first user as superadmin or display status. |
| `/register` | - | Request moderator access if slots (<10) are free. |
| `/help` | - | Show commands available for your role. |
| `/requests` | - | Superadmin sees pending registrations with approve/reject buttons. |
| `/tz <±HH:MM>` | required offset | Set timezone offset (superadmin only). |
| `/addevent <text>` | event description | Parse text with model 4o and store one or several events. Poster images are uploaded to Catbox once, recognized via OCR, cached, and the extracted text is passed to 4o together with a token usage report for the operator. Forwarded messages from moderators are processed the same way. |
| `/addevent_raw <title>|<date>|<time>|<location>` | manual fields | Add event without LLM. The bot also creates a Telegraph page with the provided text and optional attached photo. |
| `/images` | - | Toggle uploading photos to Catbox. |
| `/vkgroup <id|off>` | required id or `off` | Set or disable VK group for daily announcements. |
| `/vktime today|added <HH:MM>` | required type and time | Change VK posting times (default 08:00/20:00). |
| `/vkphotos` | - | Toggle sending images to VK posts. |
| `↪️ Репостнуть в Vk` | - | Safe repost via `wall.post` with photo IDs. |
| `🎪 Сделать фестиваль` | - | Кнопка в меню редактирования события запускает пайплайн создания или привязки фестиваля; отображается только у событий без фестиваля. |
| `🧩 Склеить с…` | - | Кнопка в меню редактирования фестиваля открывает список дублей, переносит события, медиа, алиасы и ссылки в выбранную запись и удаляет источник. |
| `✂️ Сокращённый рерайт` | - | LLM-сжатый текст без фото, предпросмотр и правка перед публикацией. |
| `/ask4o <text>` | any text | Send query to model 4o and show plain response (superadmin only). |
| `/ocrtest` | - | Сравнить распознавание афиш между gpt-4o-mini и gpt-4o (только супер-админ). |
| `/events [DATE]` | optional date `YYYY-MM-DD`, `DD.MM.YYYY` or `D месяц [YYYY]` | List events for the day with delete, edit and VK rewrite buttons. The rewrite control launches the shortpost flow; it shows `✂️` when the event has no VK repost yet and `✅` once the saved `vk_repost_url` confirms publication. Ticket links appear as vk.cc short URLs, and each card includes a `Статистика VK: https://vk.com/cc?act=stats&key=…` line when a short key is available. Dates are shown as `DD.MM.YYYY`. Choosing **Edit** lists all fields with inline buttons including a toggle for "Бесплатно". |
| `/setchannel` | - | Choose an admin channel and register it as an announcement or calendar asset source. |
| `/channels` | - | List admin channels showing registered and asset ones with disable buttons. |
| `/regdailychannels` | - | Choose admin channels for daily announcements and set the VK group. |
| `/daily` | - | Manage daily announcement channels and VK posting times; send test posts. |
| `/exhibitions` | - | List active exhibitions similar to `/events`; each entry shows the period `c <start>` / `по <end>` and includes edit/delete buttons. |
| `/digest` | - | Build lecture digest with images, toggles and quick send buttons (superadmin only). |
| `/backfill_topics [days]` | optional integer horizon | Superadmin only. Re-run the topic classifier for events dated from today up to `days` ahead (default 90). Sends a summary `processed=... updated=... skipped=...`; manual topics are skipped. |
| `/pages` | - | Show links to Telegraph month and weekend pages. |
| `/fest [archive] [page]` | optional `archive` flag and page number | List festivals with edit/delete options. Ten rows are shown per page with navigation buttons. Use `archive` to view finished festivals that no longer have upcoming events; omit it to see active ones. |



| `/stats [events]` | optional `events` | Superadmin only. Show Telegraph view counts starting from the past month and weekend pages up to all current and future ones. Includes the festivals landing page and stats for upcoming or recently ended (within a week) festivals. Use `events` to list event page stats. |
| `/dumpdb` | - | Superadmin only. Download a SQL dump and `telegraph_token.txt` plus restore instructions. |
| `/restore` | attach file | Superadmin only. Replace current database with the uploaded dump. |
| `/tourist_export [period]` | optional `period=ГГГГ[-ММ[-ДД..ГГГГ-ММ-ДД]]` | Выгрузка событий в формате JSONL с полями `tourist_*`. Только для неблокированных модераторов и администраторов (включая суперадминов), уважается фильтр по диапазону дат. |

| `python main.py test_telegraph` | - | Verify Telegraph API access. Automatically creates a token if needed and prints the page URL. |

Use `/addevent` to let model 4o extract fields. `/addevent_raw` lets you
input simple data separated by `|` pipes.

Poster OCR reuses cached recognitions and shares a 10 000 000-token daily budget; once the limit is exhausted new posters wait
until the next reset at UTC midnight.

## Event topics

Автоклассификатор присваивает до трёх тем из фиксированного списка. Метки
видны администраторам в `/events` и в читательских карточках. Классификация
запускается когда:

- событие сохраняется через `/addevent` или VK-пайплайн (копии многодневных
  событий наследуют темы базовой записи);
- администратор меняет `title`, `description` или `source_text` и событие не
  находится в ручном режиме;
- супер-администратор вызывает `/backfill_topics`, чтобы пересчитать темы у
  будущих событий.

Актуальные идентификаторы и подписи:

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
- `HANDMADE` — «Хендмейд/маркеты/ярмарки/МК»
- `NETWORKING` — «Нетворкинг и карьера»
- `ACTIVE` — «Активный отдых и спорт»
- `PERSONALITIES` — «Личности и встречи»
- `KIDS_SCHOOL` — «Дети и школа»
- `FAMILY` — «Семейные события»
- `KRAEVEDENIE_KALININGRAD_OBLAST` — «Краеведение Калининградской области»

Чтобы закрепить ручные темы, установите `topics_manual` в меню редактирования
(кнопка **Edit** → поле `topics_manual` → введите `true`). Пока флаг включён,
автоклассификатор и `/backfill_topics` не переписывают метки. Вернуть автоматический
режим можно, отправив `false`. Детали пайплайна описаны в `docs/llm_topics.md`.

> **Региональные эвристики.** Даже если модель не вернула тему,
> постпроцессор добавит `KRAEVEDENIE_KALININGRAD_OBLAST`, когда город,
> адрес, хэштеги или ссылки указывают на Калининградскую область (например,
> `Калининград`, `Светлогорск`, `#калининград`, домен `klgd`). Метка не ставится,
> если событие переведено в ручной режим.
