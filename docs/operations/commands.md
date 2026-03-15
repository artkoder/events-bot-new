# Bot Commands

| Command | Arguments | Description |
|---------|-----------|-------------|
| `/start` | - | Register the first user as superadmin or display status. |
| `/register` | - | Request moderator access if slots (<10) are free. |
| `/help` | - | Show commands available for your role. |
| `/assist (/a) <описание>` | required text | Суперадмин: описать действие простыми словами, Gemma подбирает подходящую команду, бот показывает план и просит подтверждение (✅/❌). |
| `/requests` | - | Superadmin sees pending registrations with approve/reject buttons. |
| `/tz <±HH:MM>` | required offset | Set timezone offset (superadmin only). |
| `/addevent <text>` | event description | Parse text with model 4o and store one or several events. Poster images are uploaded to Catbox once, recognized via OCR, cached, and the extracted text is passed to 4o together with a token usage report for the operator. Forwarded messages from moderators are processed the same way. |
| `/addevent_raw <title>|<date>|<time>|<location>` | manual fields | Add event without LLM. The bot also creates a Telegraph page with the provided text and optional attached photo. |
| `/images` | - | Toggle uploading photos to Catbox. |
| `/ik_poster` | - | Обработка афиш через ImageKit (Smart crop / GenFill). |
| `/vkgroup <id|off>` | required id or `off` | Set or disable VK group for daily announcements. |
| `/vktime today|added <HH:MM>` | required type and time | Change VK posting times (default 08:00/20:00). |
| `/vkphotos` | - | Toggle sending images to VK posts. |
| `/vk_auto_import [N|all] [--include-skipped]` | optional limit flags | Суперадмин: авторазбор очереди VK (таблица `vk_inbox`) через Smart Update. По умолчанию без аргументов обрабатывает всю активную очередь; `N` ограничивает количество постов. Показывает прогресс `X/Y`, унифицированный отчёт с фактами и количеством добавленных иллюстраций, ссылки на Telegraph и `/log`. |
| `/vk_auto_import_stop` | - | Суперадмин: запросить остановку текущего прогона `/vk_auto_import` (остановка после завершения текущего поста). |
| `/imp_groups_30d` | - | Суперадмин. Показать агрегированную статистику импорта за 30 дней по группам из Supabase-вьюха `vk_import_by_group`. Пример ответа:<br>`Импорт из VK по группам за последние 30 дн.:`<br>`1. club123: Импорт: 12, Отклонено: 4`. |
| `/imp_daily_14d` | - | Суперадмин. Сводка импорта по дням за последние 14 дней из `vk_import_daily`. Пример:<br>`Импорт из VK по дням за последние 14 дн.:`<br>`2024-05-17: Импорт: 6, Отклонено: 1`. |
| `/vk_misses [N]` | optional limit (default 10) | Суперадмин выгружает свежие пропуски из Supabase (`vk_misses_sample`), бот показывает карточки с текстом, ссылкой и причинами фильтрации, прикладывает до 10 изображений и добавляет кнопки «Отклонено верно»/«На доработку». Кнопка доработки записывает Markdown в `VK_MISS_REVIEW_FILE` (по умолчанию `/data/vk_miss_review.md`). |
| `↪️ Репостнуть в Vk` | - | Safe repost via `wall.post` with photo IDs. |
| `🎪 Сделать фестиваль` | - | Кнопка в меню редактирования события запускает пайплайн создания или привязки фестиваля; отображается только у событий без фестиваля. |
| `🧩 Склеить с…` | - | Кнопка в меню редактирования фестиваля открывает список дублей, переносит события, медиа, алиасы и ссылки в выбранную запись и удаляет источник. |
| `🎬 0…5` | - | Кнопка отбора в `/events`: увеличивает счётчик включений события в видео-анонсы по кругу 0→5 и обратно. Доступно неблокированным модераторам/суперадминам, партнёры могут трогать только свои события. После публикации ролика основной счётчик автоматически уменьшается на 1. |
| `Добавить иллюстрацию` | - | Запрашивает фото, изображение-документ или ссылку. Первая добавленная картинка становится обложкой, остальные попадают в альбом. |
| `✂️ Сокращённый рерайт` | - | LLM-сжатый текст без фото, предпросмотр и правка перед публикацией. |
| `/ask4o <text>` | any text | Send query to model 4o and show plain response (superadmin only). |
| `/ocrtest` | - | Сравнить распознавание афиш между gpt-4o-mini и gpt-4o (только супер-админ). |
| `/kaggletest` | - | Суперадмин: проверка авторизации Kaggle (возвращает заголовок тестовой записи или ошибку API). |
| `/tg` | - | Суперадмин: управление источниками Telegram Monitoring (добавить/удалить источник, ручной запуск). |
| `/parse [check]` | optional `check` | Суперадмин: запуск парсинга источников (театры/собор/Третьяковка) через Kaggle. `check` — диагностический режим без сохранения в БД. См. `docs/features/source-parsing/sources/theatres/README.md`. |
| `/events [DATE]` | optional date `YYYY-MM-DD`, `DD.MM.YYYY` or `D месяц [YYYY]` | List events for the day with delete, edit and VK rewrite buttons. The rewrite control launches the shortpost flow; it shows `✂️` when the event has no VK repost yet and `✅` once the saved `vk_repost_url` confirms publication. Ticket links appear as vk.cc short URLs, and each card includes a `Статистика VK: https://vk.com/cc?act=stats&key=…` line when a short key is available. Dates are shown as `DD.MM.YYYY`. Choosing **Edit** lists all fields with inline buttons including a toggle for "Бесплатно". |
| `/setchannel` | - | Choose an admin channel and register it as an announcement or calendar asset source. |
| `/channels` | - | List admin channels showing registered and asset ones with disable buttons. |
| `/regdailychannels` | - | Choose admin channels for daily announcements and set the VK group. |
| `/daily` | - | Manage daily announcement channels and VK posting times; send test posts. |
| `/v` | - | Суперадмин: меню видео-анонсов. После подбора событий и генерации intro-текста открывается экран выбора паттерна (`STICKER`, `RISING`, `COMPACT`) с PNG-превью. Кнопки: ◀ / ▶ для переключения, ✏️ редактирование текста, ✓ подтверждение. Далее — выбор kernel и запуск рендеринга на Kaggle. Результат уходит в тестовый или основной канал. |
| `/exhibitions` | - | List active exhibitions similar to `/events`; each entry shows the period `c <start>` / `по <end>` and includes edit/delete buttons. |
| `/digest` | - | Build digest with images, toggles and quick send buttons (superadmin only). The menu offers лекции, мастер-классы, психология, научпоп, краеведение Калининградской области и другие подборки. |
| `/backfill_topics [days]` | optional integer horizon | Superadmin only. Re-run the topic classifier for events dated from today up to `days` ahead (default 90). Sends a summary `processed=... updated=... skipped=...`; manual topics are skipped. |
| `/pages` | - | Show links to Telegraph month and weekend pages. |
| `/fest [archive] [page]` | optional `archive` flag and page number | List festivals with edit/delete options. Ten rows are shown per page with navigation buttons. Use `archive` to view finished festivals that no longer have upcoming events; omit it to see active ones. |
| `/fest_queue [--info|-i] [--limit=N] [--source=vk|tg|url]` | optional filters | Суперадмин: ручной запуск фестивальной очереди. `--info/-i` показывает состояние очереди без разбора (счётчики, pending список). Без `--info` — запускает обработку; показывает прогресс/статус, даёт ссылки на страницу фестиваля и общую страницу «Фестивали». |
| `/ticket_sites_queue [--info|-i] [--limit=N] [--source=pyramida|dom_iskusstv|qtickets] [--url=...]` | optional filters | Суперадмин: очередь мониторинга ticket-sites. Находится в Telegram Monitoring постах (pyramida.info / домискусств.рф / qtickets) и по расписанию (или вручную) запускает Kaggle-парсинг URL и Smart Update для обогащения событий (фото/цены/описание/статус билетов). `--info/-i` показывает состояние очереди. `--url=...` обрабатывает один конкретный URL. |



| `/stats [events]` | optional `events` | Superadmin only. Show Telegraph view counts starting from the past month and weekend pages up to all current and future ones. Includes the festivals landing page and stats for upcoming or recently ended (within a week) festivals. The footer now fetches daily OpenAI token totals from Supabase (`token_usage_daily`, falling back to live `token_usage` or the legacy snapshot on errors). Use `events` to list event page stats. |
| `/telegraph_cache_stats [kind]` | optional kind (`event|festival|month|weekend|festivals_index`) | Superadmin only. Show Telegram web preview health for Telegraph pages: whether the page has `cached_page` (Instant View) and `photo` (preview image). |
| `/telegraph_cache_sanitize [--limit=N] [--no-enqueue] ...` | optional flags | Superadmin only. Run Telegraph cache sanitizer (Kaggle/Telethon): warms + probes web preview for key pages and enqueues rebuild tasks for persistently failing pages. |
| `/general_stats` | - | Superadmin only. Daily operational system report for the previous 24 hours (scheduled at 07:30 Europe/Kaliningrad). |
| `/recent_imports [hours]` | optional integer horizon (default `24`) | Superadmin only. Rolling list of events created or updated from Telegram Monitoring, VK, and `/parse` over the last `N` hours. Rows are deduped by `event_id` and start with `id`, then compact status (`✅` created / `🔄` updated), then the Telegraph-linked title, event date/time, source labels, and last import timestamp. |
| `/popular_posts [N]` | optional integer limit (default 10) | Суперадмин: статистика “социальной популярности” постов, которые создали события (TG/VK). Показывает ТОП постов, где `views` или `likes` выше медианы внутри своего канала/сообщества: (1) окно 7 суток (предпочитается `age_day=6`, fallback на последний доступный `age_day<=6`), (2) окно 3 суток (предпочитается `age_day=2`, fallback на последний доступный `age_day<=2`), (3) окно 24 часа (`age_day=0`). В каждой строке: ссылка на исходник, список созданных событий (Telegraph + `id`), медианы канала и метрики поста, маркеры `⭐/👍`. Показываются только события на сегодня и позже; если у поста остались только завершившиеся события, он скрывается из блока. В конце блока — диагностика: размер выборки (посты/источники; метрики vs импорт), счётчики `skip(...)`, и сколько постов выше медианы по `views/likes/оба` (после фильтров); если 7-дневные снапшоты ещё не собираются, отчёт показывает явную подсказку по конфигу. |
| `/rebuild_event <event_id> [--regen-desc]` | required id, optional flag | Суперадмин: принудительно пересобрать пайплайн события (Telegraph + зависимые страницы). С `--regen-desc` дополнительно перегенерирует описание в режиме fact-first из сохранённых фактов перед пересборкой. |
| `/dumpdb` | - | Superadmin only. Download a SQL dump and `telegraph_token.txt` plus restore instructions. |
| `/restore` | attach file | Superadmin only. Replace current database with the uploaded dump. |
| `/tourist_export [period]` | optional `period=ГГГГ[-ММ[-ДД..ГГГГ-ММ-ДД]]` | Выгрузка событий в формате JSONL с полями `tourist_*`. Только для неблокированных модераторов и администраторов (включая суперадминов), уважается фильтр по диапазону дат. |

| `python main.py test_telegraph` | - | Verify Telegraph API access. Automatically creates a token if needed and prints the page URL. |

Use `/addevent` to let model 4o extract fields. `/addevent_raw` lets you
input simple data separated by `|` pipes.

Poster OCR reuses cached recognitions and shares a 10 000 000-token daily budget; once the limit is exhausted new posters wait
until the next reset at UTC midnight. If cached OCR is available when the limit is hit, `/addevent` includes a short cached OCR preview in its reply.

### VK review inline story creation

- **«Создать историю»** — кнопка в интерфейсе проверки VK-постов. После нажатия бот уточняет, нужны ли дополнительные указания
  редактора: «Да, нужны правки» открывает поле для текста, «Нет, всё понятно» пропускает шаг. В открывшемся поле отправьте
  короткое сообщение с тонами, фактами или табу; если ввод не нужен, нажмите «Пропустить», оставьте его пустым или отправьте `-`.
  Ответы сохраняются и подмешиваются в оба запроса 4o, которые строят план и финальную историю, поэтому модель следует заданным
  инструкциям.

## Event topics

Автоклассификатор присваивает до пяти тем из фиксированного списка. Метки
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
- `URBANISM` — «Урбанистика»
- `KRAEVEDENIE_KALININGRAD_OBLAST` — «Краеведение Калининградской области»

Чтобы закрепить ручные темы, установите `topics_manual` в меню редактирования
(кнопка **Edit** → поле `topics_manual` → введите `true`). Пока флаг включён,
автоклассификатор и `/backfill_topics` не переписывают метки. Вернуть автоматический
режим можно, отправив `false`. Детали пайплайна описаны в `../llm/topics.md`.

> **Региональная метка.** Теперь только LLM решает, когда выставлять
> `KRAEVEDENIE_KALININGRAD_OBLAST`. Если событие про Калининградскую область,
> постарайтесь упомянуть это в описании или хэштегах, чтобы модель увидела связь.
