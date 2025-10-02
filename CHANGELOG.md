# Changelog

## [x.y.z+2] – 2025-09-22
- `/exhibitions` теперь выводит будущие выставки без `end_date`, чтобы операторы видели их и могли удалить вручную при необходимости.
- Починили обработку фестивальных обложек: теперь `photo_urls = NULL` не приводит к ошибкам импорта.
- Исправлена обработка обложек фестивалей: отсутствие `photo_urls` больше не приводит к ошибкам.
- Добавили поддержку загрузки обложки лендинга фестивалей через `/weekendimg`: после загрузки страница пересобирается автоматически.
- `/addevent`, форварды и VK-очередь теперь распознают афиши (один проход Catbox+OCR), подмешивают тексты в LLM и показывают расход/остаток токенов.
- Результаты распознавания кешируются и уважают дневной лимит в 10 млн токенов.
- Added `/ocrtest` diagnostic command, чтобы сравнить распознавание афиш между `gpt-4o-mini` и `gpt-4o` с показом использования токенов.
- Clarified the 4o parsing prompt to warn about possible OCR mistakes in poster snippets.
- VK Intake помещает посты с одной фотографией и пустым текстом в очередь и отмечает их статусом «Ожидает OCR».
- В VK-очереди появились кнопки «Добавить (+ фестиваль)»/«📝🎉 …», а импорт теперь создаёт или обновляет фестиваль даже при отсутствии событий.
- На стартовом экране появилась кнопка «+ Добавить фестиваль»: оператор жмёт её, чтобы открыть ручное создание фестиваля, а при отсутствии распознанного фестиваля LLM-поток останавливает импорт с явным предупреждением.
- Уточнены правила очереди: URGENT с горизонтом 48 ч, окна SOON/LONG завершаются на 14 / 30 дней, FAR использует веса 3 / 2 / 6, джиттер задаётся по источнику, а стрик-брейкер FAR срабатывает после K=5 не-FAR выборов.
- Истории из VK перед публикацией прогоняются через редакторский промпт 4o: бот чинит опечатки, разбивает текст на абзацы и добавляет понятные подзаголовки.
- Month pages retry publishing without «Добавить в календарь» links when Telegraph rejects the split, preventing `/pages_rebuild` from failing on oversized months.
- На исторических страницах автоматически очищаем сторонние VK-ссылки, оставляя только ссылку на исходный пост.
- Импорт списка встреч из VK-очереди создаёт отдельные события для каждой встречи, а не только для первой.
- Список сообществ ВК показывает статусы `Pending | Skipped | Imported | Rejected` и поддерживает пагинацию.
- Ежедневный Telegram-анонс теперь ссылается на Telegraph-страницу для событий из VK-очереди (кроме партнёрских авторов).
- «✂️ Сокращённый рерайт» сохраняет разбивку на абзацы вместо склеивания всего текста в один блок.
- Запустили психологический дайджест: в `/digest` появилась отдельная кнопка, подбор идёт по тематике и автоматически создаётся интро.

- Introduced automatic topic classification with a closed topic list, editor display, and `/backfill_topics` command.
- Classifier/digest topic list now includes the `PSYCHOLOGY`, `THEATRE_CLASSIC`, and `THEATRE_MODERN` categories.
- Refreshed related documentation and tests so deploy notes match the current feature set.

- Fixed VK review queue issue where `vk_review.pick_next` recalculates `event_ts_hint` and auto-rejects posts whose event date
  disappeared or fell into the past (e.g., a 7 September announcement shown on 19 September).

## v0.3.10 – 2025-09-21
This release ships the updates that were previously listed under “Unreleased.”

- Компактные строки «Добавили в анонс» теперь начинаются с даты в формате `dd.mm`.
- `/events` теперь содержит кнопку быстрого VK-рерайта с индикаторами `✂️`/`✅`, чтобы операторы видели, опубликован ли шортпост.

## [x.y.z+1] - 2025-09-21
### Added
- Добавили JSON-колонку `aliases` у фестивалей и пробрасываем пары алиасов в промпт 4o, чтобы нормализовать повторяющиеся названия.
- В интерфейсе редактирования фестиваля появилась кнопка «🧩 Склеить с…», запускающая мастер объединения дублей с переносом событий и алиасов.

### Changed
- После объединения фестивалей описание пересобирается на основе актуальных событий и синхронизируется со страницами/постами.
- Промпт 4o для фестивальных описаний требует один абзац до 350 знаков без эмодзи, чтобы операторы придерживались нового стандарта.

## [x.y.z] - 2025-09-20
### Added
- Ввели ручной блок «🌍 Туристам» в Telegram и VK с кнопками «Интересно туристам» и «Не интересно туристам».
- Добавили меню причин и поддержку комментариев.
- Добавили экспорт `/tourist_export` в `.jsonl`.

### Changed
- Обновили справочник факторов: `🎯 Нацелен на туристов`, `🧭 Уникально для региона`, `🎪 Фестиваль / масштаб`, `🌊 Природа / море / лендмарк / замок`, `📸 Фотогенично / есть что постить`, `🍲 Местный колорит / кухня / крафт`, `🚆 Просто добраться`.
- Обновили документацию про TTL: 15 минут для причин и 10 минут для комментария.

### Security
- Доступ к кнопкам туристической метки и `/tourist_export` оставили только неблокированным модераторам и администраторам; авторазметка запрещена до окончания исследования.

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

## v0.3.10 - Festival stats filter and daily management updates

- `/stats` now lists festival statistics only for upcoming festivals or those
  that ended within the last week.
- `/regdailychannels` and `/daily` now show the VK group alongside Telegram channels.
  VK posting times can be changed there and test posts sent.
- Daily announcements include new hashtag lines for Telegram and VK posts.

## v0.3.11 - VK monitoring MVP and formatting tweaks

- Added `/vk` command for manual monitoring of VK communities: add/list/delete groups and review posts from the last three days.
- New `VK_API_VERSION` environment variable to override VK API version.
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

## v0.3.22 - Partner reminder frequency fix

- Partners who haven't added events no longer receive daily reminders; each
  partner is notified at most once a week.

## v0.3.23 - Weekend VK posts

- Creating a weekend Telegraph page now also publishes a simplified weekend
  post to VK and links existing weekend VK posts in chronological order.

## v0.3.24 - Weekend VK source filter

- Weekend VK posts include only events with existing VK source posts and no
  longer attempt to create source posts automatically.

## v0.3.25 - Daily VK title links

- Event titles in VK daily announcements link to their VK posts when available.

## v0.3.26 - Festival day creation

- Announcements describing a festival without individual events now create a
  festival page and offer a button to generate day-by-day events later.
- Existing databases automatically add location fields for festivals.

## v0.3.27 - Festival source text

- Festival descriptions are generated from the full original post text.
- Festival records store the original announcement in a new `source_text` field.

## v0.3.28 - VK user token

- VK posting now uses a user token. Set `VK_USER_TOKEN` with `wall,groups,offline` scopes.
- The group token `VK_TOKEN` is optional and used only as a fallback.

## v0.3.29 - Film screenings

- Added support for `кинопоказ` event type and automatic detection of film screenings.

## v0.3.30 - Festival ticket links

- Festival records support a `ticket_url` and VK/Telegraph festival posts show a ticket icon and link below the location.

## v0.3.31 - Unified publish progress

- Event publication statuses now appear in one updating message with inline status icons.

## v0.3.32 - Festival program links

- Festival records support a `program_url`. Telegraph festival pages now include a "ПРОГРАММА" section with program and site links when provided, and the admin menu allows editing the program link.

## v0.3.33 - Lecture digest improvements

- Caption length for lecture digests now uses visible HTML text to fit up to 9 lines.
- Removed URL shortener functionality and related configuration.
- 4o title normalization returns lecturer names in nominative form with `Имя Фамилия: Название` layout.

## v0.3.34 - VK Intake & Review v1.1

- Added database tables and helpers for VK crawling and review queue.
- Introduced `vk_intake` module with keyword and date detection utilities.

## v0.3.35 - VK repost link storage

- Event records now include an optional `vk_repost_url` to track reposts in the VK afisha.

## v0.3.36 - VK crawl utility

- Introduced `vk_intake.crawl_once` for cursor-based crawling and enqueueing of
  matching posts.
- Dropped the unused VK publish queue in favor of operator-triggered reposts;
  documentation updated.

## v0.3.37 - VK inbox review

- The review flow now reads candidates from the persistent `vk_inbox` table.
- Operators can choose to repost accepted events to the Afisha VK group.
- Removed remaining references to the deprecated publish queue from docs.

## v0.3.38 - VK queue summary

- `/vk_queue` displays current inbox counts and offers a button to start the
  review flow.

## v0.3.39 - VK review UI polish

- Review flow now presents media cards with action buttons and logs rebuilds
  per month.
- Accepted events immediately send Telegraph and ICS links to the admin chat.
- The "🧹 Завершить…" button rebuilds affected months sequentially.
- Operators can repost events to the Afisha VK group via a dedicated button
  storing the final post link.

## v0.3.40 - VK intake improvements

- Incremental crawling with pagination, overlap and optional 14‑day backfill.
- Randomised group order and schedule jitter to reduce API load.
- Keyword detector switched to regex stems with optional `pymorphy3` lemma
  matching via `VK_USE_PYMORPHY`.
- Date and time parser recognises more Russian variants and returns precise
  timestamps for scheduling.

## v0.3.41 - VK group context for 4o

- VK event imports now send the group title to 4o so venues can be inferred from
  `docs/LOCATIONS.md` when posts omit them.

## v0.3.42 - VK review media
- VK review: поддержаны фото из репостов (copy_history), link-preview, doc-preview; для video берём только превью-картинки, видео не загружаем

## v0.3.43 - Festival landing stats

- `/stats` now shows view counts for the festivals landing page.

## v0.3.44 - VK short posts

- VK review reposts now use safe `wall.post` with photo IDs.
- Added "✂️ Сокращённый рерайт" button that publishes LLM‑compressed text.

## v0.3.45 - VK shortpost preview

- "✂️ Сокращённый рерайт" отправляет черновик в админ-чат с кнопками
  публикации и правки.
- Посты больше не прикрепляют фотографии, только ссылку с превью.
