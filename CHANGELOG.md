# Changelog

## [1.11.0] - 2026-01-25
### Added
- **Telegram Monitor**: Full release of the Intelligent Monitoring System.
  - **Standard Pipeline**: Events from Kaggle are now processed via the standard `/addevent` pipeline (GPT-4o + deduplication).
  - **Secure Sessions**: Implemented Fernet-based session splitting (Key/Cipher) for Kaggle isolation.
  - **Inline UI**: New `/tg` command with interactive buttons.
  - **Docs**: Comprehensive walkthrough and setup guide.

## [1.10.6] - 2026-01-25
### Changed
- **Video Announce**: Improved poster overlay text cleaning by stripping emojis from `ocr_text` and `description` to prevent font rendering issues.

## [1.10.5] - 2026-01-25
### Added
- **Video Announce**: Implemented `cross_month` layout for "Compact" intro pattern, allowing distinct date placement when events span across month boundaries.

## [1.10.4] - 2026-01-25
### Changed
- **Video Announce**: Refactored `_filter_events_by_poster_ocr` in selection logic to improve code organization and testability.

## [1.10.3] - 2026-01-24
### Fixed
- **3D Preview**: Fixed logic in automatic generation to reliably detect and process events with missing previews ("gaps"), scanning the last 14 days.
- **3D Preview**: Added an extra scheduled run at 17:15 to ensure previews are ready for the 18:00 pinned button update.

## [1.10.2] - 2026-01-24
### Fixed
- Filtered out past events during parsing to prevent them from being announced.

## [1.10.1] - 2026-01-24
### Fixed
- Fixed `TypeError` in parsing results summary when using date objects (Philharmonia parser).

## [Unreleased]

## [1.10.0] - 2026-01-24
### Added
- **Source Parsing**: Added full support for **Kaliningrad Regional Philharmonia** (`filarmonia39.ru`).
  - Implemented Kaggle parser (`ParsePhilharmonia`) that scans proper 6-month window using direct URL navigation.
  - Integration in `/parse` command and scheduled jobs.
  - Supports automatic ticket status updates (`available` / `unavailable`) and price extraction.
  - Proper date normalization to avoid parsing errors.

### Fixed
- **Source Parsing**: Updated `requirements.txt` to include `beautifulsoup4` and `lxml` for local parsing utilities if needed.

## [1.9.13] - 2026-01-24
### Changed
- **CrumpleVideo**: Minor metadata updates in notebook.


## [1.9.12] - 2026-01-24
### Added
- **Video Announce**: Implemented "Poster Overlays" feature. Uses Google Gemma to check if posters are missing Title/Date/Time/Location. Adds an overlay badge to the video if critical info is missing.
- **Dependencies**: Added `google-generativeai`.

## [1.9.11] - 2026-01-24
### Fixed
- **Scheduler**: Fixed `_job_wrapper` to accept `**kwargs`, resolving `ValueError` when registering jobs with keyword arguments (like `3di_scheduler` with `chat_id`).

## [1.9.10] - 2026-01-24
### Fixed
- **Scheduler**: Added `_register_job` wrapper to prevent scheduler startup crashes if a single job fails to register.
- **Scheduler**: Added explicit "SCHED skipping" logs when optional jobs (source parsing, 3di) are disabled via env.

## [1.9.9] - 2026-01-24
### Fixed
- **CrumpleVideo**: Improved FFmpeg robustness with file existence checks, audio merge validation, and mpeg4 fallback.
- **Video Announce**: Enhanced polling reliability with retry logic (3 attempts) and recursive file search in output directory.

## [1.9.8] - 2026-01-23
### Changed
- **CrumpleVideo**: Updated test mode to use 5 scenes (was 1), samples=15, render_pct=84.
- **Video Announce**: Increased Kaggle timeout from 40 to 150 minutes to handle queue delays.
- **Kaggle Assets**: Fixed dataset slug format to `video-afisha-session-{id}` for compatibility.

## [1.9.7] - 2026-01-21
### Changed
- **CrumpleVideo**: Adjusted audio start timestamp to 1:17 (was 1:10) for better intro sync.
- **CrumpleVideo**: Increased `is_test_mode` render quality: samples raised to 18, percentage to 70% for clearer previews.


## [1.9.6] - 2026-01-21
### Fixed
- **Video Announce**: Improved random_order fallback and added notebook logging.
- **Tests**: Fixed import error in `test_video_announce_selection.py`.


## [1.9.5] - 2026-01-20
### Added
- **Preview 3D**: Автоматическая генерация 3D-превью (`/3di`) по расписанию (`ENABLE_3DI_SCHEDULED=1`, `THREEDI_TIMES_LOCAL`).
- **Source Parsing**: Поддержка дневного автозапуска (`ENABLE_SOURCE_PARSING_DAY`, `SOURCE_PARSING_DAY_TIME_LOCAL`).
- **Source Parsing**: Защита от холостого парсинга — если сигнатуры страниц театров не изменились, повторный парсинг пропускается.

### Changed
- **Config**: Часовые пояса для шедулеров теперь настраиваются явно (`SOURCE_PARSING_TZ`, `SOURCE_PARSING_DAY_TZ`, `THREEDI_TZ`).
### Added
- **Source Parsing**: Автозапуск парсинга по расписанию (по умолчанию 02:15 KGD). Настройка через `ENABLE_SOURCE_PARSING=1` и `SOURCE_PARSING_TIME_LOCAL`.
- **Source Parsing**: Таймауты для OCR (60 сек) и скачивания изображений.
- **Source Parsing**: Детальная диагностика событий (через `SOURCE_PARSING_DIAG_TITLE`).
- **Source Parsing**: Логи теперь сохраняются в Persistent Volume `/data/parse_debug`.

### Fixed
- **Source Parsing**: Улучшена обработка ошибок в боте и на сервере, предотвращены "молчаливые" падения.
- **Source Parsing**: OCR отключен по умолчанию для источника `tretyakov` (`SOURCE_PARSING_DISABLE_OCR_SOURCES`) для стабильности.
- **CrumpleVideo**: Test "Tomorrow" renders now use lower samples and resolution to speed up single-scene previews.
- **CrumpleVideo**: Test-mode intro previews now default to `STICKER_YELLOW` when no explicit pattern is provided.
- **Intro Visuals**: Restored the dark default palette and added a yellow theme via `_YELLOW` patterns.

## [1.9.3] - 2026-01-20
### Fixed
- **Source Parsing**: Исправлена нормализация локаций Третьяковки — теперь сохраняется информация о сцене (`Кинозал`/`Атриум`), что позволяет различать события в одном месте в одно время. Ранее события в разных залах ошибочно определялись как дубликаты.
- **Source Parsing**: Добавлен label `🎨 Третьяковка` в отчёты `/parse`.
- **Kaggle Assets**: Preserve existing Kaggle kernel dataset sources while appending new ones, and restore `generate_intro_image` in the CrumpleVideo notebook.
- **CrumpleVideo**: Move `_resolve_image_path` to module scope so the main pipeline can call it safely.
- **CrumpleVideo**: Define `is_last` before building the intro segment to avoid `UnboundLocalError` in production.

## [1.9.2] - 2026-01-20
### Fixed
- **Kaggle Assets**: Fixed `ModuleNotFoundError` by moving assets and `pattern_preview.py` to a dedicated Kaggle Dataset (`video-announce-assets`) and mounting it in the kernel.
### Fixed
- CrumpleVideo Kaggle kernel now loads `pattern_preview` via the `video-announce-assets` dataset instead of local files.

## [1.9.1] - 2026-01-20
### Fixed
- **Kaggle Kernel ID**: Fixed a bug where `kaggle_client.py` was forcing the legacy `video-afisha` kernel ID, preventing `CrumpleVideo` updates.
- **Intro Visuals**: Integrated verified `pattern_preview` logic into the `CrumpleVideo` kernel to ensure correct fonts and alignment in production.
- **Outro Animation**: Disabled physics simulation (crumpling) for the Outro scene, ensuring it remains static/readable.

## [1.9.0] - 2026-01-20

### Added
- **Video Announce**: Automated "Tomorrow" pipeline (`/v` -> `🚀 Запуск Завтра`).
- **Video Announce**: Test mode (`/v` -> `🧪 Тест Завтра`) for single-scene verification.
- **Video Announce**: Randomize event order selection (prioritizing OCR candidates).
- **Video Announce**: Visual improvements for City/Date intro layout.

## [1.8.2] - 2026-01-07

### Fixed
- **Channel Navigation Buttons**: Buttons ("Today", "Tomorrow" etc.) are now ONLY added if the post contains `#анонс`, `#анонсКалининград` or `#анонскалининград` hashtags. Fixes EVE-13 where buttons appeared in all channel posts.

## [1.8.1] - 2026-01-05

### Fixed
- **Channel Navigation Buttons**: Исправлено получение постов канала — добавлен `channel_post` в `allowed_updates` webhook.
- **Channel Navigation Buttons**: Исправлен доступ к `db` и `bot` в хэндлере — теперь берутся из модуля `main`.
- **Channel Navigation Buttons**: Исправлен фильтр команд — проверка `/` вынесена внутрь хэндлера.

## [1.8.0] - 2026-01-05

### Added
- **Channel Navigation Buttons**: Добавлены inline-кнопки навигации для постов в канале:
  - «📅 Сегодня» — ссылка на текущий месяц
  - «📅 Завтра» — ссылка на специальную страницу завтрашнего дня (33% шанс)
  - «📅 Выходные» — ссылка на ближайшие выходные (33% шанс)
  - «📅 [Месяц]» — ссылка на следующий месяц (33% шанс)
  - Алгоритм `random.choice` для равномерного распределения
  - Модель `TomorrowPage` для кэширования страниц «завтра»
  - Фильтрация рубрик и автоматических постов (не добавляем кнопки)
- **3D Preview on Split Pages**: На страницах месяца с большим количеством событий (>30) теперь отображаются 3D-превью, если они есть (обычные фото скрываются).

### Changed
- Навигация в подвале спец-страниц (`/special`, Tomorrow, Weekend): текущий месяц теперь кликабельный.

## [1.7.8] - 2026-01-04

### Added
- **🏛 Dom Iskusstv Parsing**: Полноценная интеграция парсера Дома искусств с Kaggle:
  - Кнопка "🏛 Дом искусств" в главном меню для ввода ссылки на спецпроект
  - Кнопка "🏛 Извлечь из Дом искусств" в VK review для автоматического парсинга
  - Kaggle notebook `ParseDomIskusstv` для скрейпинга событий с сайта
  - Автоматическое создание Telegraph страниц с билетами, фото и полным описанием
  - BDD E2E тесты для всех сценариев парсинга
- **E2E Testing**: Добавлен фреймворк для E2E BDD тестов (`tests/e2e/`):
  - `HumanUserClient` — обёртка Telethon с имитацией человеческого поведения
  - BDD сценарии на Gherkin с русским синтаксисом
  - Верификация контента Telegraph страниц (проверка наличия 🎟, Билеты, руб.)

### Fixed
- **Telegraph PAGE_ACCESS_DENIED Fallback**: При ошибке редактирования Telegraph страницы (PAGE_ACCESS_DENIED) теперь автоматически создаётся новая страница вместо сбоя.
- **Telegraph Rebuild on Event Update**: Вызов `update_event_ticket_status` теперь триггерит перестройку Telegraph страницы, гарантируя актуальность данных о билетах.
- **Dom Iskusstv Updated Events Links**: Ссылки на Telegraph отображаются для обновлённых событий (добавлено отслеживание `updated_event_ids`).
- **Events List Message Length**: Fallback на компактный формат при превышении лимита Telegram (4096 символов).
- **Фестивали**: 3D превью над заголовком события на странице фестиваля.

### Changed
- **Фестивали**: `/festivals_fix_nav` пропускает архивные фестивали без будущих событий.

## [1.7.7] - 2026-01-02

### Added
- **3D Preview**: Добавлена кнопка "🌐 All missing" в меню `/3di` для генерации превью всех будущих событий без preview_3d_url одним нажатием.
- **Фестивали**: Добавлена кнопка "🔄 Обновить события" в меню редактирования фестиваля (`/fest edit`) для обновления списка событий на Telegraph-странице фестиваля.

## [1.7.5] - 2026-01-02

### Changed
- Increased event limit for 3D previews on month pages from 10 to 30.

## [1.7.6] - 2026-01-02

### Fixed
- **3D Preview**:
  - **Notebook Cleanup**: Kaggle notebook now performs aggressive cleanup (`rm -rf`) of Blender binary and image directories before completion. This prevents the bot from downloading massive amount of data (hundreds of MBs) and ensures only the result JSON is retrieved, fixing "Result not applied" errors.

## [1.7.4] - 2026-01-02

### Added
- **Telegraph**: Для событий с длинным описанием (>500 символов) теперь отображается краткое описание (`search_digest`) над полным текстом, разделённое горизонтальной линией. Улучшает читаемость страниц событий.

### Fixed
- **Tretyakov Parser**: 
  - Исправлена навигация по календарю — теперь парсер корректно находит все даты через стрелку `.week-calendar-next`.
  - Исправлено извлечение времени — парсер теперь прокручивает календарь к нужной дате перед кликом, устраняя ошибки `00:00` для дат на других страницах календаря.
  - Добавлена полная поддержка min/max цен из всех секторов.
  - Добавлена дедупликация событий с объединением фото (исполнитель приоритет над фестивалем).

## [1.7.3] - 2026-01-02

### Added
- **3D Preview**: Added "Only New" button to `/3di` command. Allows generating missing previews for new events without reprocessing existing ones.
- **Pyramida**: Fixed price parsing from ticket widget. Now extracts specific prices (e.g. "500 ₽") and price ranges ("500 - 1000 ₽"), ensuring correct `ticket_status` ("available" instead of "unknown").

## [1.7.2] - 2026-01-02

### Changed
- **3D Preview Aesthetics**:
    - **Soft Shadows**: Increased light source angle to 10° for softer, more realistic shadows.
    - **Cinematic Rotation**: The first card in the stack is now slightly rotated (-3°) for a more dynamic look.

## [1.7.1] - 2026-01-02

### Fixed
- **3D Preview**: Fixed argument parsing in `/3di` command to support running from image captions and avoid errors when `message.text` is None (aiogram v3 compatibility).

## [1.7.0] - 2026-01-02

### Added
- **3D Preview**: Added `/3di multy` command mode. Generates previews only for events with 2 or more images, filtering out single-image events.
- **3D Preview**: Improved lighting with a new "Shadow Lift" fill light. This makes cards 2, 3, and 4 readable by softening the hard shadows while maintaining the dramatic texture.

## [1.6.11] - 2026-01-02

### Changed
- **Configuration**: Increased Kaggle polling timeout from 30 minutes to 4 hours to accommodate CPU fallback scenarios.

## [1.6.10] - 2026-01-01

### Fixed
- **Source Parsing**: Исправлено формирование `short_description` для событий из `/parse`. Усилен промпт LLM — добавлены подробные правила генерации `short_description` (REQUIRED поле, one-sentence summary с примерами). Убран fallback на `full_description` (многострочный текст), fallback на title используется только в крайнем случае с логированием warning.
- **Special Pages**: Added support for 3D generated previews (`preview_3d_url`) in special pages. If available, the 3D preview is used as the main event image, prioritizing it over regular photos.

## [1.6.9] - 2026-01-01

### Changed
- **3D Preview**: Changed the Blender background color from dark gray to pure Black (#000000) for better integration with both light and dark Telegraph themes.

## [1.6.8] - 2026-01-01

### Refinements
- **3D Preview**:
  - **Cover Logic**: 3D preview is now used as the Telegraph page cover ONLY if the event has 2 or more source photos. If there is only 1 photo, the original is preserved.
  - **Transparency**: Added a dark background to the Blender scene to fix transparency rendering issues in Telegraph.
  - **Composition**: Improved layout for single images (< 3 photos) to use a centered single plane instead of the carousel.

### Refined
- **3D Preview**: Use the preview image as the leading Telegraph photo, add a dark scene background, and simplify layout when fewer than three images are available.

## [1.6.7] - 2026-01-01

### Fixed
- **3D Preview**: Fixed critical bug where database session variable shadowed the user session dictionary, causing "AsyncSession object does not support item assignment" error.

## [1.6.6] - 2026-01-01

### Performance
- **3D Preview**:
  - Notebook now cleans up Blender binaries and input files before completion, leaving only `output.json`. This dramatically speeds up the result download (from minutes to seconds) and prevents timeouts.
  - Handler now actively cleans up temporary download directories in `/tmp` to save disk space.

## [1.6.5] - 2026-01-01

### Fixed
- **3D Preview**:
  - Increased output download retry limit to 10 attempts (50s total timeout).
  - Implemented automatic Month Page rebuild triggering after 3D preview application.
  - Added detailed final report in Telegram with links to the updated month page and events.

## [1.6.4] - 2026-01-01

### Fixed
- **3D Preview**: Added 3 retry attempts for downloading output.json from Kaggle (handles API race conditions).

## [1.6.3] - 2026-01-01

### Fixed
- **3D Preview**: Added 15s delay after dataset creation in handler (syncing pattern with video_announce) to ensure dataset availability before kernel start.

## [1.6.2] - 2026-01-01

### Fixed
- **3D Preview**: Added 60s retry loop for payload detection in Kaggle notebook to handle dataset mounting latency.

## [1.6.1] - 2026-01-01

### Fixed
- **3D Preview**:
  - Fixed payload path detection in Kaggle notebook (now uses `rglob`).
  - Added "fail fast" logic in notebook if payload is missing.
  - Implemented live status updates in Telegram message during polling.
  - Added `asyncio.Lock` to serialize concurrent generation requests.
  - Fixed output directory collisions by using per-session paths.

## [1.6.0] – 2026-01-01

### Added
- **3D Preview Feature**:
  - Added `preview_3d_url` to `Event` model.
  - Created `/3di` command for generating 3D previews using Kaggle.
  - Implemented Kaggle orchestration pipeline (dataset -> kernel -> polling -> db update).
  - Added support for GPU rendering on Kaggle.
  - Integrated 3D previews into Telegraph month pages (displayed as main image).

## [1.5.3] – 2026-01-01
- **Performance**: Оптимизация LLM-вызовов в `/parse` — унифицирована логика `find_existing_event` с `upsert_event`. Теперь существующие события распознаются до вызова LLM, что значительно снижает расход токенов и время обработки.

## [1.5.2] – 2025-12-31
- **Logging**: логируются выбор kernel, путь локального kernel и состав файлов при push в Kaggle.

## [1.5.1] – 2025-12-31
- **Fix**: В импортировании payload и перезапуске последней сессии добавлен шаг выбора kernel перед рендером, чтобы избежать 403.
## [1.5.0] – 2025-12-31
- **Fix**: Исправлена маска MoviePy в Kaggle-ноутбуке `video_afisha.ipynb` — маска остается 2D для корректного blit.
- **Feature**: Добавлена кнопка "📥 Импортировать payload" для запуска рендера видео-анонса из сохранённого `payload.json` без этапа подбора событий.

## [1.4.6] – 2025-12-31
- **Fix**: Исправлена ошибка фильтрации в `/special`: события больше не скрываются, если у него ошибочно указан `end_date` в прошлом (проверяется `max(date, end_date)`).
- **Refinement**: Очистка описаний Музтеатра и Кафедрального собора на продакшене.
- **Fix**: Исправлена дата и метаданные события в Веселовке.
- **Infrastructure**: Введено правило изоляции временных скриптов в папке `scripts/`.

## [1.4.5] – 2025-12-31

### Fixed
- **Muzteatr Parser**: Fixed empty descriptions by extracting text from `og:description` meta tags (site structure changed).

## [1.4.4] - 2025-12-31

### Fixed
- **Dramteatr Parser**: Fixed DOM traversal issue where date block was missed because it is a sibling of the link wrapper.

## [1.4.3] - 2025-12-31

### Fixed
- **Dramteatr Parser**: Fixed date extraction (incomplete dates like "31 ДЕКАБР") using CSS selectors.
- **Parsing**: Improved duplicate detection with fuzzy title matching (Codex).
- **Video Announce**: Filter out "sold_out" events from video digests by default.
- **UI**: Minor adjustment to ticket icon order in summaries.

## [1.4.2] - 2025-12-31

### Changed
- **Source Parsing**: Улучшен алгоритм сопоставления событий (parser.py) — добавлено извлечение стартового времени для более точного поиска дубликатов.
- **Source Parsing**: Добавлено детальное логирование (per-event logging) с метриками (LLM usage, duration).

## [1.4.1] - 2025-12-31

### Fixed
- **Source Parsing**: Раскомментированы блоки Драмтеатра и Музтеатра в ноутбуке `ParseTheatres`.

## [1.4.0] - 2025-12-31

### Added
- **Special Pages**: Новая команда `/special` для генерации праздничных Telegraph-страниц. Поддержка произвольного периода (1–14 дней), дедупликация событий с одинаковыми названиями (объединение в блок с несколькими временами), загрузка обложки, автоматическое сокращение периода при превышении лимита Telegraph.
- **Special Pages**: Нормализация названий локаций при генерации страницы (удаление дублей адресов).
- **Special Pages**: Улучшили навигацию — добавлена навигация по месяцам в футере страницы.
- **Source Parsing**: Улучшен Kaggle-ноутбук `ParsePyramida` для более надежного парсинга.

### Fixed
- **System**: Исправлен конфликт `sys.modules` при запуске бота, вызывавший ошибку доступа к базе данных (`get_db() -> None`) в динамически загружаемых модулях.
- **Month/Weekend Pages**: Исправлено отсутствие дат и времени на страницах месяцев и выходных в Telegraph. Теперь дата и время отображаются корректно в формате "_31 декабря 19:00, Место, Город_".

### Fixed

## [1.3.7] - 2025-12-31

### Added
- **Telegraph**: Телефонные номера на страницах событий теперь кликабельные (ссылки `tel:`). Поддерживаются форматы: +7, 8, локальные номера.
- **Performance**: Отложенные перестройки страниц (Deferred Rebuilds) — задачи `month_pages` и `weekend_pages` откладываются на 15 минут для оптимизации при массовом добавлении событий.
- **Conditional Images**: На месячных и выходных страницах Telegraph отображаются изображения событий, если на странице менее 10 событий.
- **EVENT_UPDATE_SYNC**: Добавлена поддержка синхронного режима для тестирования отложенных задач.

### Changed
- **/parse limit**: Лимит одновременно добавляемых событий снижен с 10 до 5 для стабильности.
- **/parse rebuild**: Убрана автоматическая пересборка Telegraph страниц после `/parse` — теперь используется стандартная очередь отложенных задач.

### Fixed
- **/parse month_pages**: При добавлении событий через `/parse` теперь гарантированно создаются задачи `month_pages` для всех затронутых месяцев for deferred rebuild.
- **Deferred Rebuilds**: Исправлен обход отложенности — `_drain_nav_tasks` больше не создаёт немедленные follow-up задачи если уже есть отложенная задача для event_id. Это предотвращает преждевременную пересборку страниц Telegraph.
- **VK Inbox**: Исправлено отсутствие ссылки на Telegraph страницу в отчёте оператору ("✅ Telegraph — "). Теперь бот ожидает создания страницы перед отправкой ответа (до 10 секунд).
- **Deferred Rebuilds**: Убран синхронный вызов `refresh_month_nav` при обнаружении нового месяца, вызывавший немедленную пересборку всех страниц. Теперь новые месяцы обрабатываются через отложенную очередь.
- **Deferred Rebuilds**: `schedule_event_update_tasks` по умолчанию теперь использует `drain_nav=False`, гарантируя соблюдение 15-минутной задержки перед сборкой.
- **Deferred Rebuilds TTL**: Исправлено преждевременное истечение (expiration) отложенных задач — TTL теперь считается от момента запланированного выполнения (`next_run_at`), а не от момента создания (`updated_at`). Ранее задачи с 15-минутной отложенностью истекали через 10 минут (TTL=600с).
- **Rebuild Notifications**: При автоматической пересборке страниц теперь суперадминам приходит уведомление с перечнем обновлённых месяцев.
- **Navigation Update**: При добавлении события на новый месяц (например, Апрель) теперь обновляются футеры навигации на всех существующих страницах (Январь, Февраль и т.д.).
- **Year Suffix**: Исправлено отображение года в навигации — "2026" добавляется только к Январю или при смене года, а не ко всем месяцам.
- **Spam Removal**: Удалены отладочные сообщения `NAV_WATCHDOG`, которые отправлялись в чат оператора при каждой отложенной задаче.
- **Retry Logic**: При ошибке `CONTENT_TOO_BIG` флаг `show_images` теперь корректно прокидывается в ретрай.
- **Test Stability**: `main_part2.py` теперь безопаснее импортировать напрямую (fallback для `LOCAL_TZ`, `format_day_pretty`).
- **Photo URL Validation**: Добавлена проверка схемы `http` для `photo_urls`.

## [1.3.5] - 2025-12-29

### Fixed
- **Pyramida**: Исправлен парсинг дат в формате `DD.MM.YYYY HH:MM` (например `21.03.2026 18:00`). Ранее такие даты не распознавались и события не добавлялись.

## [1.3.4] - 2025-12-29

### Fixed
- **Pyramida**: Исправлена ошибка ("missing FSInputFile"), из-за которой не отправлялся JSON файл с результатом парсинга.
- **Pyramida**: Включено OCR для событий, добавляемых через кнопку в VK Review (ранее работало только для `/parse`).

## [1.3.3] - 2025-12-29

### Fixed
- **Pyramida**: Добавлено отображение статуса работы Kaggle (Running/Poling) в чате. Теперь пользователь видит прогресс выполнения ноутбука.

## [1.3.2] - 2025-12-29

### Added
- **Source Parsing**: Добавлена поддержка OCR (распознавание текста) для событий из Pyramida и /parse. Теперь афиши скачиваются, распознаются и текст используется для улучшения описания события.

## [1.3.1] - 2025-12-29

### Fixed
- **Pyramida**: Исправлен парсинг описания событий (корректный селектор для Playwright/BS4)
- **Pyramida**: Добавлена отправка JSON файла с результатами парсинга в чат
- **Docs**: Уточнено, что OCR для Pyramida не выполняется

## [1.3.0] - 2025-12-29

### Added
- **Pyramida extraction**: Новая кнопка "🔮 Извлечь из Pyramida" в VK review flow для автоматического парсинга событий с pyramida.info. Извлекает ссылки из поста, запускает Kaggle notebook, добавляет события в базу. См. [docs/PYRAMIDA.md](docs/PYRAMIDA.md)
- **Pyramida manual input**: Кнопка "🔮 Pyramida" в меню /start (для супер-админов) для ручного ввода

## [1.2.17] - 2025-12-29



### Added
- **source_parsing**: Новый Kaggle-ноутбук `ParseTheatres` с полем `description`
- **docs**: Документация `/parse` в `docs/pipelines/source-parsing.md`

### Fixed
- **source_parsing**: События из `/parse` теперь корректно появляются в ежедневном анонсе — исправлен подсчёт новых vs обновлённых событий
- **source_parsing**: Отчёт теперь показывает 🔄 Обновлено для существующих событий (ранее не отображалось)
- **source_parsing**: Добавлено debug-логирование в `find_existing_event` для диагностики
- **source_parsing**: Прогресс теперь редактирует одно сообщение вместо множества
- **source_parsing**: Поле `description` корректно передаётся в БД из парсера

## [1.2.15] - 2025-12-28

### Fixed
- **source_parsing**: Исправлено добавление событий — теперь используется `persist_event_and_pages` вместо несуществующего `persist_event_draft`
- **source_parsing**: Добавлена отправка JSON файлов из Kaggle в ответ на `/parse`
- **source_parsing**: Улучшено логирование создания событий

## [1.2.14] - 2025-12-28

### Added
- Улучшено логирование модуля `source_parsing` для отладки команды `/parse`:
  - Логирование при получении команды и проверке прав
  - Логирование старта и завершения Kaggle-ноутбука
  - Логирование количества полученных событий

## [1.2.13] - 2025-12-28

### Fixed
- Улучшен промпт `about_fill_prompt` для видеоанонсов: теперь LLM явно включает title в about когда ocr_title пуст.
- Синхронизированы правила about в `selection_prompt` и `about_fill_prompt`.

## [1.2.1] - 2025-12-27

### Fixed
- Исправлено дублирование заголовков выходных дней ("суббота/воскресенье") на месячных Telegraph-страницах при инкрем ентальном обновлении

## [1.2.0] - 2025-12-27
### Fixed
- Fixed critical `TypeError` in video announce generation caused by mismatched arguments in `about` text normalization calls across `scenario.py`, `selection.py`, and `finalize.py`.

## [1.1.1] - 2025-12-27
### Fixed
- Fixed bug where `search_digest` was not saved to database during event creation via text import.
- Updated `about_fill_prompt` to preserve proper nouns (e.g. "ОДИН ДОМА") in about text.
- Removed anchor prepending logic in `about.py`, making LLM fully responsible for about text generation.
- Updated agent instructions to require explicit user command for production deployment.

<!-- Новые изменения добавляй сюда -->
- Исправлен сбор логов с Kaggle: теперь `poller.py` корректно скачивает логи из вложенных директорий и пакует их в zip-архив, если файлов больше 10.

---

## [1.1.0] – 2025-12-27

### Added

- **Развернуть всех кандидатов**: в UI выбора событий появилась кнопка «+ Все кандидаты», разворачивающая полный список событий в 5-колоночный формат для ручного добавления.
- **Экран сортировки**: кнопка «🔀 Сортировка» открывает экран с выбранными событиями и кнопками ⬆️/⬇️ для изменения порядка показа в видео.
- **Текущие выходные**: если сегодня суббота или воскресенье, в периодах появляются две кнопки — «Эти выходные (дата)» и «Выходные (следующая дата)».

---

## [1.0.0] – 2025-12-27

> **Первый мажорный релиз** — введено семантическое версионирование (SemVer).

### Added

- Исправили падение при запуске: добавили импорт `dataclass` для работы альбомов в видео-анонсах и других обработчиках.
- Видео-анонсы перестали искажать LLM-описания: строки `about` теперь лишь очищаются от лишних пробелов/эмодзи, а усечение и дедупликация слов остаются только для резервного текста.
- Перевели справочник сезонных праздников на локализованный формат дат `DD.MM` и текстовые диапазоны, сохранили столбец `tolerance_days` и обновили парсер импорта под новый формат.
- `/vk_misses` superadmins review fresh Supabase samples: the bot pulls post text, up to ten images, filter reasons, and matched keywords from `vk_misses_sample`, adds «Отклонено верно»/«На доработку» buttons, and records revision notes for the latter in `VK_MISS_REVIEW_FILE` (defaults to `/data/vk_miss_review.md`).
- Добавили `/ik_poster`, вынесли логику в новый модуль `imagekit_poster.py`, подключили зависимости ImageKit и пересылаем результаты в операторский чат.
- Фестивальные редакторы загружают кастомные обложки через кнопку «Добавить иллюстрацию»: фича опирается на Telegram-поток `festimgadd` в `main.py`, бот пересылает туда файлы, автоматически разворачивает обложку в альбомную ориентацию и сохраняет обновления.
- Исправили кросс-базовую совместимость `festival.activities_json`: SQLite снова работает и не падает при чтении поля, закрывая регрессию с крэшем.
- `GROUNDED_ANSWER_MODEL_ID` теперь фиксирован на `gemini-2.5-flash-lite`, чтобы grounded-ответы consistently шли через новую модель.
- Добавили пайплайн видео-анонсов: новые сущности `videoannounce_session` / `videoannounce_item` / `videoannounce_eventhit` хранят статусы, ошибки и историю включений, а watchdog переводит застрявшие рендеры в `FAILED`.
- В `/events` появилась кнопка `🎬` с циклическим счётчиком 0→5: доступна неблокированным модераторам/суперадминам (партнёры правят только свои события) и резервирует включения в ролик; после публикации основного ролика счётчик уменьшается.
- `/v` открывает суперадминское меню профилей с запуском новой сессии, показом пяти последних и перезапуском последней упавшей; пока Kaggle-рендер в статусе `RENDERING`, UI блокируется.
- Kaggle-интеграция для видео: сбор JSON-пейлоада, публикация датасета и kernel, трекинг статусов и `run_kernel_poller`, проверка учётки через `/kaggletest`.
- Готовый ролик и логи уходят в выбранный для профиля тестовый канал (если не задан, используется операторский чат); при наличии выбранного основного канала видео публикуется туда и фиксируется имя файла. События из ролика помечаются `PUBLISHED_MAIN` и тратят одну единицу включения.
- Исправили SQLite-инициализацию видео-анонсов: таблица `videoannounce_session` теперь создаётся с колонками `profile_key`, `test_chat_id` и `main_chat_id`, чтобы меню `/v` не падало на старых базах.
- Расширили SQLite-инициализацию видео-анонсов: добавляем `published_at`, `kaggle_dataset` и `kaggle_kernel_ref`, чтобы селекты `/v` не ломались на старых базах.
- Видео-анонсы показывают интро от LLM после подбора с кнопками для правки, просмотра JSON-файла и запуска Kaggle; пользовательские правки сохраняются в `selection_params`, а JSON отправляется как файл при предпросмотре и перед стартом рендеринга.
- Видео-анонс ранжирования собирает единый JSON-запрос с промптом, инструкциями, кандидатами, `response_format` и `meta` и отправляет его и результат в 4o и операторский чат даже без пользовательской инструкции.

### Video Announce Intro Patterns (2025-12-27)

- **Визуальные паттерны интро**: добавлены три паттерна для интро-экрана видео — `STICKER`, `RISING`, `COMPACT`. Пользователь выбирает паттерн в UI с кнопками и превью перед рендерингом.
- **Генератор превью паттернов**: новый модуль `video_announce/pattern_preview.py` генерирует PNG-превью паттернов на сервере без Kaggle.
- **Города и даты в интро**: `payload_as_json()` извлекает города и диапазон дат из событий и передаёт их в ноутбук для отображения на интро.
- **Шрифт Bebas Neue**: заменён шрифт Oswald на Bebas Neue Bold для Better Cyrillic rendering и соответствия референсному дизайну.
- **Fly-out анимация Sticker**: все текстовые стикеры вылетают с overshoot-эффектом в направлении наклона с задержкой между элементами.
- **Пропорциональные отступы**: вертикальные отступы между элементами интро теперь составляют 15% от высоты контента (как в превью), вместо фиксированных пикселей.
- **Исправлено отображение городов**: notebook теперь копирует `cities`, `date`, `pattern` из payload в `intro_data`, города отображаются в интро.
- **Дайджест в статусе импорта**: VK-импорт показывает поле `search_digest` события в сообщении об успехе.

## v0.3.17 – 2025-10-07

- VK crawler telemetry now exports group metadata, crawl snapshots, and sampled misses to Supabase (`vk_groups`, `vk_crawl_snapshots`, `vk_misses_sample`) with `SUPABASE_EXPORT_ENABLED`, `SUPABASE_RETENTION_DAYS` (default 60 days), and `VK_MISSES_SAMPLE_RATE` governing exports, sampling, and automatic cleanup.
- VK stories now ask whether to collect extra editor instructions and forward the answer plus any guidance to the 4o prompts.
- Добавлен справочник сезонных праздников (`docs/reference/holidays.md`), промпт 4o теперь перечисляет их с алиасами и описаниями, а импорт событий автоматически создаёт и переиспользует соответствующие фестивали.
- Log OpenAI token usage through Supabase inserts (guarded by `BOT_CODE`) and ship the `/usage_test` admin self-test so operators can verify the inserts and share usage snapshots during release comms.
- `/stats` подтягивает сводку токенов напрямую из Supabase (`token_usage_daily`/`token_usage`) и только при ошибке падает обратно на локальный снапшот, чтобы в релизных отчётах отображались свежие значения.

## v0.3.16 – 2025-10-05
- Telegraph event source pages now include a “Быстрые факты” block with date/time, location, and ticket/free status, hiding each line when the underlying data is missing so operators know it’s conditional.
- Системный промпт автоклассификации запрещает выбирать темы `FAMILY` и `KIDS_SCHOOL`, когда у события задан возрастной ценз; см. обновления в `main.py` (`EVENT_TOPIC_SYSTEM_PROMPT`) и документации `docs/llm_topics.md`.
- Уведомления в админ-чат для партнёров теперь включают первую фотографию события и ссылки на Telegraph и исходный VK-пост, чтобы операторы могли оперативно проверить публикацию.
- Fixed VK weekday-based date inference so it anchors on the post’s publish date and skips phone-number fragments like `474-30-04`, preventing false matches in review notes.
- Сокращённые VK-рерайты теперь дополняются тематическими хэштегами из ключевых тем события (например, `#стендап`, `#openair`, `#детям`, `#семье`).
- Тематический классификатор теперь доверяет 4o выбор темы `KRAEVEDENIE_KALININGRAD_OBLAST`, локальные эвристики отключены, а постобработка распределения тем удалена.
- Ограничение на количество тем увеличено до пяти, промпты 4o обновлены под новый лимит и требования к краеведению.
- Восстановили независимую тему `URBANISM`, чтобы классификатор снова различал городские трансформации и не смешивал их с инфраструктурой.
- Month-page splitter final fallback now removes both «Добавить в календарь» and «Подробнее» links, keeping oversized months deployable despite Telegraph size limits and closing the recent operator request.

- Запустили научпоп-дайджест: в `/digest` появилась отдельная кнопка, а кандидаты отбираются по тематике `SCIENCE_POP`, чтобы операторы могли быстро собрать подборку.
- В `/digest` добавили кнопку краеведения, чтобы операторы могли собирать подборки по теме `KRAEVEDENIE_KALININGRAD_OBLAST` без ручной фильтрации.

- Expanded THEATRE_CLASSIC and THEATRE_MODERN criteria to include canonical playwrights and contemporary production formats.
- `/digest` для встреч и клубов теперь подсказывает тону интро по `_MEETUPS_TONE_KEYWORDS`, использует запасной вариант «простота+любопытство» и отдельно просит сделать акцент на живом общении и нетворкинге, если в подборке нет клубов; описание обновлено в `docs/digests.md`.
- Заголовки митапов в `/digest` нормализуются постпроцессингом: если событие открывает выставку, к названию добавляется пояснение «— творческая встреча и открытие выставки», как задокументировано в `docs/digests.md`.

## v0.3.15 – 2025-10-04
- Clarified the 4o parsing prompt and docs for same-day theatre showtimes: posters with one date and multiple start times now yield separate theatre events instead of a single merged entry.
- Added admin digests for нетворкинг, развлечения, маркеты, классический/современный театр, встречи и клубы и кинопоказы; обновлён список синонимов тем и меню /digest.
- Library events without explicit prices now default to free, so operators can spot the change in billing behavior.

## v0.3.14 – 2025-09-23
- Нормализованы HTML-заголовки и абзацы историй перед публикацией.
- Равномерно распределяем inline-изображения в исторических текстах.
- Убрали символное усечение ответов `ask_4o`.
- Очищаем заголовки историй от VK-разметки.

## v0.3.13 – 2025-09-22
- `/exhibitions` теперь выводит будущие выставки без `end_date`, чтобы операторы видели их и могли удалить вручную при необходимости.
- Починили обработку фестивальных обложек: теперь `photo_urls = NULL` не приводит к ошибкам импорта.
- Исправлена обработка обложек фестивалей: отсутствие `photo_urls` больше не приводит к ошибкам.
- Добавили поддержку загрузки обложки лендинга фестивалей через `/weekendimg`: после загрузки страница пересобирается автоматически.
- `/addevent`, форварды и VK-очередь теперь распознают афиши (один проход Catbox+OCR), подмешивают тексты в LLM и показывают расход/остаток токенов.
- Результаты распознавания кешируются и уважают дневной лимит в 10 млн токенов.
- Added `/ocrtest` diagnostic command, чтобы сравнить распознавание афиш между `gpt-4o-mini` и `gpt-4o` с показом использования токенов.
- Clarified the 4o parsing prompt to warn about possible OCR mistakes in poster snippets.
- `/events` автоматически сокращает ссылки на билеты через vk.cc и добавляет строку `Статистика VK`, чтобы операторы могли открыть счётчик переходов.
- VK Intake помещает посты с одной фотографией и пустым текстом в очередь и отмечает их статусом «Ожидает OCR».
- В VK-очереди появились кнопки «Добавить (+ фестиваль)»/«📝🎉 …», а импорт теперь создаёт или обновляет фестиваль даже при отсутствии событий.
- На стартовом экране появилась кнопка «+ Добавить фестиваль»: оператор жмёт её, чтобы открыть ручное создание фестиваля, а при отсутствии распознанного фестиваля LLM-поток останавливает импорт с явным предупреждением.
- Уточнены правила очереди: URGENT с горизонтом 48 ч, окна SOON/LONG завершаются на 14 / 30 дней, FAR использует веса 3 / 2 / 6, джиттер задаётся по источнику, а стрик-брейкер FAR срабатывает после K=5 не-FAR выборов.
- Истории из VK перед публикацией прогоняются через редакторский промпт 4o: бот чинит опечатки, разбивает текст на абзацы и добавляет понятные подзаголовки.
- Month pages retry publishing without «Добавить в календарь» links when Telegraph rejects the split, preventing `/pages_rebuild` from failing on oversized months.
- На исторических страницах автоматически очищаем сторонние VK-ссылки, оставляя только ссылку на исходный пост.
- Ссылку «Добавить в календарь …» сокращаем через vk.cc; в VK используем формат без `https://`, а при ошибке API сохраняем исходный Supabase URL.
- Импорт списка встреч из VK-очереди создаёт отдельные события для каждой встречи, а не только для первой.
- Список сообществ ВК показывает статусы `Pending | Skipped | Imported | Rejected` и поддерживает пагинацию.
- Ежедневный Telegram-анонс теперь ссылается на Telegraph-страницу для событий из VK-очереди (кроме партнёрских авторов).
- «✂️ Сокращённый рерайт» сохраняет разбивку на абзацы вместо склеивания всего текста в один блок.
- VK source settings now store default ticket-link button text and prompt; ingestion applies the saved link only when a post lacks its own ticket or registration URL, keeping operator-provided links untouched.
- Запустили психологический дайджест: в `/digest` появилась отдельная кнопка, подбор идёт по тематике и автоматически создаётся интро.

- Introduced automatic topic classification with a closed topic list, editor display, and `/backfill_topics` command.
- Classifier/digest topic list now includes the `PSYCHOLOGY`, `THEATRE_CLASSIC`, and `THEATRE_MODERN` categories.
- Refreshed related documentation and tests so deploy notes match the current feature set.

- Fixed VK review queue issue where `vk_review.pick_next` recalculates `event_ts_hint` and auto-rejects posts whose event date
  disappeared or fell into the past (e.g., a 7 September announcement shown on 19 September).
- Карточки отзывов VK теперь показывают совпавшие события Telegraph для распознанной даты и времени.

## v0.3.10 – 2025-09-21
This release ships the updates that were previously listed under “Unreleased.”

- Компактные строки «Добавили в анонс» теперь начинаются с даты в формате `dd.mm`.
- `/events` теперь содержит кнопку быстрого VK-рерайта с индикаторами `✂️`/`✅`, чтобы операторы видели, опубликован ли шортпост.

## v0.3.12 – 2025-09-21
### Added
- Добавили JSON-колонку `aliases` у фестивалей и пробрасываем пары алиасов в промпт 4o, чтобы нормализовать повторяющиеся названия.
- В интерфейсе редактирования фестиваля появилась кнопка «🧩 Склеить с…», запускающая мастер объединения дублей с переносом событий и алиасов.

### Changed
- После объединения фестивалей описание пересобирается на основе актуальных событий и синхронизируется со страницами/постами.
- Промпт 4o для фестивальных описаний требует один абзац до 350 знаков без эмодзи, чтобы операторы придерживались нового стандарта.

## v0.3.11 – 2025-09-20
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
  `docs/reference/locations.md` when posts omit them.

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

## v0.3.46 - Video announce ranking context

- Ранжирование видеоподбора отправляет в LLM полный текст со страницы
  Telegraph и сохраняет в экспортируемом JSON полный промпт запроса.
