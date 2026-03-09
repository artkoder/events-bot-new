# Telegram Monitoring

Ежедневный мониторинг публичных Telegram‑каналов/групп с автоматическим импортом событий в БД бота через Smart Event Update.

## Что делает

- По расписанию запускает Kaggle‑kernel `TelegramMonitor`.
- Kaggle читает сообщения источников, делает OCR и извлекает события; афиши по умолчанию грузятся в Supabase Storage (public media bucket), Catbox используется только в fallback/off режимах.
  - Ночной прогон теперь делает не только проход по новым сообщениям, но и bounded backfill по уже известным постам с событиями в последнем окне `TG_MONITORING_DAYS_BACK`, чтобы обновлять `views/likes` даже у каналов без новых публикаций и не терять снапшоты `age_day=1/2`.
  - Этот backfill работает в lightweight-режиме: producer не делает по старым постам OCR, загрузку/обработку картинок, видео и LLM extraction; он забирает только текст + `views/likes` по известным `message_id`.
  - Инвариант: extractor **не должен придумывать дату события** из даты публикации поста.
    - Дата/период должны быть явно в тексте/афише (или в виде относительных слов типа «сегодня/завтра», которые разрешено резолвить от даты поста).
    - Посты вида `open call` / «конкурсный отбор» / «приём заявок» считаются **не‑событиями** (это не «куда пойти») и должны отфильтровываться на стороне Kaggle (server-side guard существует как safety-net).
- Для афиш (постеров) по умолчанию использует **Supabase Storage** (public bucket) для стабильных URL:
  - `TG_MONITORING_POSTERS_SUPABASE_MODE=always` (default): Supabase‑upload всегда включён; Catbox используется только если Supabase недоступен;
  - `fallback`: приоритет Catbox, Supabase — только если Catbox‑загрузка не удалась;
  - `off`: только Catbox.
- Bucket для афиш берётся из `SUPABASE_MEDIA_BUCKET` (не из legacy `SUPABASE_BUCKET`), чтобы медиа не смешивались с ICS.
- При загрузке афиш в Supabase:
  - объект сохраняется **в WebP** (только WebP, без JPEG) для экономии объёма;
  - ключ объекта content‑addressed по перцептивному хешу (dHash16), чтобы одно и то же изображение (даже при разном разрешении/реэнкоде) не загружалось повторно:
    - `supabase_path`: `<prefix>/dh16/<first2>/<dhash>.webp` (prefix по умолчанию `p`, настраивается через `TG_MONITORING_POSTERS_PREFIX`);
    - качество WebP: `TG_MONITORING_POSTERS_WEBP_QUALITY` (default `82`).
- Сервер скачивает `telegram_results.json` и импортирует события через Smart Update:
  - создаёт новые события;
  - мерджит существующие;
  - добавляет источники в `event_source`.
  - обрабатывает Telegram-посты в хронологическом порядке (старые → новые), чтобы старые посты не перезатирали более свежие обновления того же события.
  - во время импорта в `/tg` показывает live-прогресс по каждому посту (`X/Y`, ссылка на пост, `Smart Update: ✅/🔄`, `event_ids`, иллюстрации, `took_sec`), чтобы оператор видел, что импорт не завис.
  - отправляет подробный блок `Smart Update (детали событий)` сразу после обработки конкретного поста (не дожидаясь завершения всего импорта).
  - в интерактивном режиме (`/tg`) финальный отчёт **не повторяет полный список созданных/обновлённых событий**, чтобы не дублировать ленту (подробности уже пришли per-post).
    - переопределение: `TG_MONITORING_FINAL_EVENT_LIST=1` (вернуть полный список в финале) или `=0`.
  - в `Smart Update (детали событий)` дополнительно показываются операторские блоки:
    - `🔥 Популярные посты` (⭐/👍) с метриками внутри канала; если пост уже привязан к событиям, рядом даётся ссылка на Telegraph страницы этих событий.
    - `📈 Метрики обновлены (без Smart Update)` для уже сканированных постов (обновление views/likes без повторного импорта).
    - `📌 Частично/пропущено` с причинами (past/invalid/rejected/nochange и т.п.), чтобы было понятно, почему «извлечено != создано».
  - перед отправкой per-post отчёта (best-effort) синхронно «дренит» JobOutbox задачи `ics_publish` + `telegraph_build` для затронутых `event_id`, чтобы ссылки Telegraph/ICS были актуальны сразу и DEV-снапшоты с чужим Telegraph token корректно пересоздавали страницы при `PAGE_ACCESS_DENIED` (см. `TG_MONITORING_DRAIN_EVENT_JOBS`).
    - Важно: inline‑drain ограничен по времени (`TG_MONITORING_INLINE_DRAIN_TIMEOUT_SEC`, default `10`) и не должен останавливать импорт `/tg` (если outbox занят/завис или задачи уже выполнены).
  - в per-post блоке `Источник:` Telegram-посты (`t.me/<channel>/<id>`) рендерятся с preview-friendly `href` (`?single`), чтобы ссылка лучше открывалась через web preview в клиентах Telegram; канонический `source_url` в БД при этом не меняется.
  - fallback на афиши/полный текст из публичной страницы `t.me/s/...` для single-event постов теперь выполняется независимо от наличия `bot`-объекта; в логи пишется явный результат (`tg_monitor.poster_fallback ... posters=N`) и ошибки fallback (debug), чтобы пропуски медиа диагностировались по логам.
  - если в fallback сломалась загрузка poster media в Catbox/Supabase, импорт не обнуляет иллюстрации: используется прямой CDN URL целевого Telegram media (`cdn*.telesco.pe`) как последний аварийный fallback.
  - `linked_source_urls` теперь обогащают медиа события: сервер пытается подтянуть афиши из linked Telegram постов (сначала из того же `telegram_results.json`, затем через `t.me/s/...` fallback) и добавляет их в candidate до Smart Update.
  - `linked_source_urls` также обогащают факты: для single-event постов сервер (best-effort) скачивает текст linked Telegram постов (payload-first, затем `t.me/s/...`) и прогоняет Smart Update по каждому linked источнику, чтобы в source log были факты по всем ссылкам.
  - если афиша явно содержит несколько дат/времён одного и того же события (например «12 июня 19:00» и «13 июня 15:00»), а extractor их схлопнул в одну дату, сервер (best-effort) расширяет карточку до нескольких событий по OCR афиши.
  - сохраняет `source_title`/`sources_meta[].title` в `telegram_source.title` (человекочитаемое название канала/группы).
  - сохраняет метаданные источника из `sources_meta[]`: `about`, `about_links_json`, `meta_hash`, `meta_fetched_at`.
  - сохраняет подсказки серии/сайта (`suggested_*`) в `telegram_source` и показывает их в UI `/tg` отдельной кнопкой принятия (без автоперезаписи ручного `festival_series`).
  - если Kaggle вернул неполную карточку события (например только `DD.MM | Title` без `location_name`/`ticket_link`),
    сервер делает best-effort восстановление из текста сообщения:
    - локация/адрес: по строкам вида `📍 ...` или `Площадка, улица/дом ...`;
    - контакт/ссылка для записи:
      - `@username` в контексте «запись/бронь/напиши» → `ticket_link=https://t.me/username`;
      - если в Kaggle‑payload пришли `messages[].links` (кнопки/hidden URL entities типа “More info”, “билеты”, “здесь”) и `ticket_link` пустой, сервер может best-effort выбрать один «сильный» registration/ticket URL.
    - заголовок: если extractor вернул мусор вроде `(4 места)`, заголовок берётся из первой содержательной строки поста.
- В Kaggle используются только модели Gemma (текст/vision); 4o там не участвует.

## Multi-event посты (несколько событий в одном сообщении)

Требование: если один Telegram‑пост содержит **несколько будущих событий**, мониторинг должен:

- создать/обновить **каждое** событие отдельной записью (по `title + date + time + location` якорям);
- **не создавать** события, которые уже в прошлом (по дате);
- перед созданием нового события сначала попытаться найти матч в БД, чтобы не плодить дубли;
- на странице Telegraph конкретного события не оставлять строки расписания/названия других событий из того же поста.

Проверки этого поведения зафиксированы в E2E сценариях: `tests/e2e/features/telegram_monitoring.feature`.

## Почему «извлечено много, а создано мало»

В отчёте мониторинга `Событий извлечено` — это количество кандидатов, которые Kaggle нашёл в сообщениях.
Дальше на сервере часть кандидатов **не импортируется** и попадает в `Пропущено`, например:

- событие уже в прошлом (по `date`, выставки/ярмарки — по `end_date` если он есть);
- не хватает якорей для Smart Update (чаще всего нет `location_name` даже после восстановления из текста);
- Smart Update нашёл матч, но изменений нет (`skipped_nochange`).

Чтобы отчёт был объясним, бот выводит разбиение `Пропущено` по основным причинам (прошедшие/невалидные/без изменений и т.д.).
Дополнительно бот отправляет оператору список **пропущенных/частично обработанных постов** (с ссылкой на пост, кратким фрагментом текста и breakdown причин), чтобы можно было вручную проверить “почему не импортировалось”.

Также мониторинг может сканировать сообщения «на несколько дней назад» (для обновления просмотров/лайков).
Такие сообщения **не прогоняются через Smart Update повторно** (идемпотентность по `message_id`), а учитываются как `Посты только для метрик` в отчёте.

## Метрики постов и популярность (⭐/👍)

Цель: собирать динамику `views/likes` у постов и подсвечивать «популярные» анонсы в отчётах Smart Update.

Каноника (общая для TG/VK): `docs/features/post-metrics/README.md` (таблицы, медианы, уровни `⭐/👍`, retention, ENV).

## Retention (очистка старых метрик)

Снапшоты метрик не хранятся вечно:

- по умолчанию оставляем только последние `90` дней (по publish timestamp);
- очистка выполняется scheduler job `post_metrics_cleanup` раз в сутки;
- настройка: `POST_METRICS_RETENTION_DAYS` (по умолчанию = `POST_POPULARITY_HORIZON_DAYS`).

## Ссылки на другие Telegram-посты (linked posts)

- Если в исходном посте найден URL вида `t.me/.../<message_id>`, мониторинг может добавить его в `linked_source_urls` конкретной карточки события.
- Цель: не потерять факты, которые могут быть разнесены между “коротким” и “полным” постами про одно и то же событие.
- На сервере linked посты обрабатываются так:
  - linked URL сохраняется в `event_source` рядом с основным источником;
  - для single-event постов импортёр (best-effort) подтягивает афиши и полный текст linked поста (payload-first, затем через публичный `t.me/s/...`);
  - затем выполняется дополнительный Smart Update-pass по linked URL: это нужно, чтобы в `🧾 Лог источников` были видны факты и их статусы по linked источнику (а не “без извлечённых фактов”).
- Ограничения (защита от рекурсии/лимитов):
  - только 1 уровень обхода ссылок (без цепочки);
  - обрабатываются только ссылки на посты (`t.me/<channel>/<id>`);
  - лимит linked-текста на карточку: `TG_MONITORING_LINKED_SOURCES_TEXT_LIMIT` (default `2`, max `5`);
  - отключение: `TG_MONITORING_LINKED_SOURCES_TEXT=0` (или `false/no/off`).

## Точки входа

- `/tg` — управление источниками и ручной запуск мониторинга (есть пагинация списка источников).
- `/tg` → `♻️ Импорт из JSON` — debug/import-only режим: позволяет выбрать один из последних локальных `telegram_results.json` (по умолчанию показываются 4, newest → older) и повторить server-import без нового запуска Kaggle.
  - После выбора файла показывается выбор режима:
    - `Импорт (обычно)` — обычный `run_telegram_import_from_results(...)`.
    - `DEV: Recreate + Reimport` — доступно только при `DEV_MODE=1`, сначала показывает preview (сколько событий/marks будет очищено), затем по подтверждению:
      - удаляет события детерминированно по `event_source(source_type='telegram', source_url IN links из JSON)`;
      - удаляет `joboutbox` по найденным `event_id` (без FK cascade);
      - очищает `telegram_scanned_message` по парам `(source_username, message_id)` из JSON;
      - запускает повторный импорт из того же файла.
  - В `DEV_MODE!=1` DEV-режим не показывается в UI и отклоняется на уровне callback/task, даже если callback вызван вручную.
- Планировщик (`scheduling.py`) — ежедневный запуск по ENV.

Канонический список источников (prod/test) и их настройки: `docs/features/telegram-monitoring/sources.yml` (см. также `docs/features/telegram-monitoring/sources.md`).

## Основные модули

- `source_parsing/telegram/commands.py` — UI/команды `/tg`.
- `source_parsing/telegram/service.py` — оркестрация Kaggle и загрузка результатов.
- `source_parsing/telegram/handlers.py` — разбор `telegram_results.json`.
- `smart_event_update.py` — Smart Event Update.

## Надёжность Kaggle polling

- Статус Kaggle kernel опрашивается с интервалом `TG_MONITORING_POLL_INTERVAL` (по умолчанию 30s) до динамического лимита ожидания (или фиксированного, если включён `fixed` mode).
- Транзиентные ошибки сети/SSL при опросе Kaggle API (например `UNEXPECTED_EOF_WHILE_READING`) **не валят прогон**: мониторинг продолжает опрос до получения `COMPLETE/FAILED` или таймаута, а в UI этап показывается как «временная ошибка сети».

## Надёжность импорта (SQLite lock)

- Если на этапе server-import возникает `sqlite3.OperationalError: database is locked`, мониторинг не падает сразу:
  - импорт `telegram_results.json` автоматически ретраится (`TG_MONITORING_IMPORT_RETRY_ATTEMPTS`, default `4`);
  - backoff между попытками: `TG_MONITORING_IMPORT_RETRY_BASE_DELAY_SEC` (default `2.0`, exponential).
- Во время ретрая оператор получает сообщение в `/tg`, что импорт повторяется.
- Для ORM-сессий SQLite на каждый новый connection применяются те же PRAGMA, что и для raw-connection (`journal_mode`, `busy_timeout`, `synchronous`, `foreign_keys`, `cache_size`), чтобы снизить вероятность lock-конфликтов под длительной нагрузкой.
- Telegraph job `telegraph_build` делает `commit()` перед сетевыми вызовами к Telegraph API (edit/create), чтобы не держать SQLite write-lock во время HTTP запросов (это снижает вероятность `database is locked` при параллельной работе импортов и воркеров).
- Тонкая настройка SQLite ожидания блокировок:
  - `DB_TIMEOUT_SEC` (default `30`);
  - `DB_BUSY_TIMEOUT_MS` (если задан, приоритетнее `DB_TIMEOUT_SEC`).

## Защита от параллельных запусков (global lock)

Проблема: если по ошибке запустить мониторинг/импорт из нескольких процессов бота одновременно (например, два polling-инстанса),
то операторский UI начнёт “дублировать” прогресс и отчёты, а SQLite чаще будет падать с `database is locked`.

Решение: сервер ставит cross-process lock на время `run_telegram_monitor` и `run_telegram_import_from_results` (включая DEV `Recreate + Reimport`).

- По умолчанию lock-файл создаётся в `tempdir` (обычно `/tmp`) и включает `BOT_CODE`, чтобы prod/test не блокировали друг друга.
- При попытке параллельного запуска второй процесс получает понятное сообщение в UI `/tg` и прогон пропускается.
- Можно переопределить путь через `TG_MONITORING_GLOBAL_LOCK_PATH`.

## Данные

- `telegram_source` — список источников (username, title, trust, defaults).
- `telegram_source.filters_json` — server-side фильтры на источник (см. `docs/features/telegram-monitoring/sources.yml`).
- `telegram_source.festival_source/festival_series` — признак фестивального канала и название серии.

## Иллюстрации (афиши)

Монитор (Kaggle) может прикреплять к постам список `posters[]` (URL + sha256 + OCR). На сервере эти афиши
переносятся в `event.photo_urls`/`event_poster` через Smart Update.

Важный нюанс:
- Для постов с одним событием мы переносим **все** фото из поста (dedupe по `sha256`). OCR используется только
  для приоритизации (первое изображение как обложка), а не для удаления фото.
- Для постов, где извлечено несколько событий (расписания/альбомы), мы стараемся **не** прикреплять “чужие”
  афиши ко всем событиям: используем event-level assignment от Kaggle или строгий OCR-матчинг.
- Нестандартный кейс: иногда канал публикует **текст** и сразу отдельным следующим сообщением пересылает афишу
  (forward из другого чата/канала). Если у текстового сообщения нет фото, а у следующего есть `posters[]`, сервер
  (best-effort) прикрепляет афишу к событию из предыдущего поста (poster-bridge) и не считает метрики второго сообщения
  “постом с событием” для популярности.
  - Safety: poster-bridge включается только при короткой подписи и малом временном дельта‑окне, и прикрепляет афиши
    только если OCR уверенно матчит `title/date/time` события (иначе лучше не прикреплять вовсе, чем прикрепить “чужую” афишу).
- Smart Update не “вымывает” уже прикреплённые афиши, если новая выборка `posters[]` оказалась пустой
  (защита от ложного prune).
- Если в payload мониторинга `posters[]` отсутствуют из-за upstream media сбоев, сервер может сделать best-effort
  fallback: вытащить фото из публичной HTML страницы `t.me/s/<username>/<message_id>`.
  Этот fallback извлекает **только** медиа‑изображения из самого поста (photo wrap + video thumbnail) и **не** должен подхватывать
  аватар канала или картинки из соседних постов. При сборке Telegraph страницы дополнительно есть safety‑net:
  слишком маленькие картинки (avatar‑like) удаляются, если в наборе есть полноценный постер.
- Если `message.text` в payload выглядит обрезанным (часто заканчивается на `…`/`...`), сервер может (best-effort)
  забрать полный текст поста из публичной HTML страницы `t.me/s/<username>/<message_id>` и использовать его как `source_text`
  для Smart Update (чтобы не терять строки про состав/поддержку/участников).
- `telegram_source.about/about_links_json/meta_hash/meta_fetched_at` — метаданные канала/группы, полученные в Kaggle через Telethon.
- `telegram_source.suggested_festival_series/suggested_website_url/suggestion_confidence/suggestion_rationale` — best-effort подсказки для оператора.
- `telegram_scanned_message` — идемпотентность сообщений.
- `telegram_post_metric` — снапшоты `views/likes` по дням после публикации (для аналитики и ⭐/👍).
- `event_source` — источники события (много на одно событие).
- `ticket_site_queue` — очередь обогащения событий по ticket‑ссылкам из постов (см. `docs/features/ticket-sites-queue/README.md`).
- `eventposter.phash` — опциональный перцептивный хеш.
- `eventposter.supabase_url/supabase_path` — fallback URL/путь в Supabase Storage для афиш (для надёжного preview и контролируемой очистки).

## Видео (Supabase)

- Kaggle (producer) может экспортировать `messages[].videos[]` с `sha256/size_bytes/mime_type/supabase_url/supabase_path` (часть полей опциональна).
- Режим загрузки видео: `TG_MONITORING_VIDEOS_SUPABASE_MODE=off|always` (default: `always`).
- Дедуп видео делается по `sha256` (content‑addressed ключ):
  - canonical `supabase_path`: `v/sha256/<first2>/<sha256>.<ext>`;
  - перед upload producer проверяет existence в Supabase Storage и при наличии не загружает видео повторно.
- Дополнительно есть fast‑path: если удаётся получить Telegram `document.id`, producer перед скачиванием проверяет
  legacy‑путь `v/tg/<document.id>.<ext>`; если объект уже есть — повторно не скачивает и использует его URL.
- Сервер прикрепляет такие видео в `event_media_asset`, когда сообщение мапится ровно на одно событие (включая `skipped_nochange`: событие уже существует, но прикрепление медиа всё равно полезно).
- Если из одного поста импортировано несколько событий, видео не мапится (статус для UI: `skipped:multi_event_message`), чтобы не прикрепить видео к неверному событию.
- В `/tg` per-post отчёте выводится явный статус: `🎬 видео (supabase)` или `🎬 видео (skipped: <reason>)`.
- Если видео прикреплено к событию, оно отображается на Telegraph-странице события:
  - как встроенное видео (preview) для прямых файлов (`.mp4/.webm/.mov`);
  - иначе как ссылка (🎬).
- После прикрепления новых видео сервер requeue-ит `telegraph_build`, чтобы Telegraph страница гарантированно отразила новые ссылки (защита от гонок с уже запущенной сборкой страницы).
- Очистка старых событий (`cleanup_old_events`) подхватывает `event_media_asset` и ставит объекты в `supabase_delete_queue` до удаления событий из БД.

## Seed источников (prod/test)

- Канонический список: `docs/features/telegram-monitoring/sources.yml`.
- Автосинхронизация отсутствующих источников выполняется при старте (SQLite seed).
- Ручная синхронизация: `/tg` → «🧩 Синхронизировать источники».

## OCR

- OCR выполняется **внутри Kaggle‑ноутбука** для сообщений с афишами, даже если в тексте поста уже есть описание.
- Результаты OCR сохраняются в `telegram_results.json`:
  - `messages[].posters[].ocr_text` и `messages[].posters[].ocr_title`;
  - агрегированный `messages[].ocr_text` (для удобства дебага).
- Дополнительно Kaggle (best-effort) сохраняет `messages[].links`:
  - URL, найденные в тексте;
  - URL из `MessageEntityTextUrl`/`MessageEntityUrl` (hidden links);
  - URL из кнопок (`reply_markup`) типа “More info”/“билеты”.
- В UI (`/events` → Edit) OCR виден в блоке **Poster OCR**.
- Проверка OCR в UI: см. `tests/e2e/features/telegram_monitoring.feature` (сценарий «Полный пользовательский поток мониторинга (UI)»).
- Для каналов с заданным `default_location` это значение считается **сильным prior** (защита от контекстных городов вроде «(г. Москва)» в описании участников), но это не «жёсткий игнор»:
  - если `default_location` задан строкой вида `Место, адрес, город`, при сборке кандидата она раскладывается по структурным полям `location_name/location_address/city`, а не копируется целиком в `location_name`;
  - если extractor извлёк город, противоречащий `default_location`, Smart Update делает короткую LLM‑проверку и может переключить `city/location_*` на извлечённые значения (после чего сработает регион‑фильтр и out‑of‑region пост будет корректно отвергнут);
  - если уверенности нет — остаётся `default_location`.
- Для постов-расписаний (несколько спектаклей в одном сообщении) применяется строгая фильтрация афиш по фактам события; если она неуверенна, но Kaggle уже выдал `event_data.posters` для конкретного события, используется event-level fallback (чтобы не терять релевантную афишу при отсутствующем времени в Telegram).

## Фильтры и санитаризация

- **Custom emoji** (Telegram `MessageEntityCustomEmoji` / `<tg-emoji>`) вычищаются из текста перед публикацией в Telegraph (обычные Unicode‑эмодзи остаются).
- **Розыгрыши билетов** (giveaway): Smart Update не делает детерминированного “вырезания” текста. LLM получает исходный текст и по инструкции игнорирует механику розыгрыша (условия участия, «подпишись/репост/коммент» и т.п.), сохраняя факты о событии. Если после разбора не остаётся признаков события — пост скипается.
- **Поздравления** (не‑ивент контент) не импортируются как события и не должны становиться источниками события (например посты «Поздравляем…» со списком ближайших спектаклей).
- **Акции/промо**: промо‑фрагменты (скидки/промокоды/«акция») должны игнорироваться/удаляться **внутри LLM** по инструкциям промпта (детерминированного regex‑стрипинга нет). Если в посте есть полноценный анонс события (дата/время/место), он импортируется/мерджится.
- OCR‑подсказка времени стала устойчивее: поддерживаются диапазоны (`10:00–18:00`), выбор времени при множественных упоминаниях на афише и защита от ложных совпадений типа `05.02` → `05:02` (дата на афише не должна становиться временем).
- Инференс года для дат без года ограничен границей года (декабрь → январь): посты февраля не должны превращать январские даты в `YYYY+1`.

## UI (/tg) — настройка источников без «параметров в сообщении»

Формат вида `@channel trust=low` поддерживается как расширенный, но операторский флоу — через кнопки:

- `/tg` → `📋 Список источников`
  - `Trust → ...` — циклически: low → medium → high
  - `📍 Локация → ...` — задать/очистить `default_location`
  - `🎟 Ticket → ...` — задать/очистить `default_ticket_link`
  - `🎪 Фестиваль → ...` — пометить источник как фестивальный и задать серию (очистка через `-`)
  - `✅ Принять подсказку` — появляется, если `festival_series` пустой и есть `suggested_festival_series`; копирует suggested в `festival_series` и включает `festival_source=1`.
  - `🌐 Suggested website` — ссылка на suggested `website_url` (без автосохранения в фестивальные сущности).
  - `♻️ Сбросить отметки ...` — очистить `telegram_scanned_message` и `last_scanned_message_id` для перескана
  - `🗑️ Удалить ...` — удалить источник

Если источники были удалены массово в тестовой БД, восстановление можно сделать без UI:
- `python scripts/restore_telegram_sources.py --db <DB_PATH> @username1 @username2 ...` (не удаляет/не трогает существующие настройки, только upsert + `enabled=1`).

- `/tg` → `♻️ Импорт из JSON`
  - После выбора файла: `Импорт (обычно)` или (только в `DEV_MODE=1`) `DEV: Recreate + Reimport`.
  - `DEV: Recreate + Reimport` использует 2-step confirm и очищает события/marks перед повторным импортом для детерминированного отладочного прогона Smart Update.

Канонический список источников (prod/test) и их настроек (trust/festival/defaults): `docs/features/telegram-monitoring/sources.md`.

## ENV

Минимум:

- `ENABLE_TG_MONITORING=1`
- `TG_MONITORING_TIME_LOCAL=23:40`
- `TG_MONITORING_TZ=Europe/Kaliningrad`
- `TELEGRAM_AUTH_BUNDLE_S22`, `TG_API_ID`, `TG_API_HASH`
- `GOOGLE_API_KEY`
- `KAGGLE_USERNAME`

Выбор auth bundle для мониторинга:

- по умолчанию используется `TELEGRAM_AUTH_BUNDLE_S22`;
- для ручной отладки можно явно переопределить источник через `TG_MONITORING_AUTH_BUNDLE_ENV=<ENV_KEY>` (например `TELEGRAM_AUTH_BUNDLE_E2E`).

Дополнительно:

- Supabase (нужно для poster fallback в Storage и для глобального rate-limit RPC, если включено):
  - `SUPABASE_URL`, `SUPABASE_KEY` (или `SUPABASE_SERVICE_KEY`), `SUPABASE_SCHEMA`, `SUPABASE_DISABLED`
  - bucket'и: legacy `SUPABASE_BUCKET` (default `events-ics`); плановое разделение: `SUPABASE_ICS_BUCKET`, `SUPABASE_MEDIA_BUCKET`
    (см. `docs/operations/supabase-storage.md`)
- Poster fallback настройка (Kaggle):
  - `TG_MONITORING_POSTERS_SUPABASE_MODE=off|fallback|always` (default `always`)
  - `TG_MONITORING_POSTERS_PREFIX` (default `p`)
  - `TG_MONITORING_POSTERS_WEBP_QUALITY` (default `82`)
- Видео в Supabase (Kaggle + import):
  - `TG_MONITORING_VIDEOS_SUPABASE_MODE=off|always` (default `always`)
  - `TG_MONITORING_VIDEO_MAX_MB` (default `10`)
  - `TG_MONITORING_VIDEO_BUCKET_SAFE_MB` (default `430`; для видео отдельный safe guard)
  - для фото/постеров остаётся safe-порог `490MB` (default общего helper-а bucket guard)
  - `SUPABASE_BUCKET_USAGE_GUARD_CACHE_SEC` (default `600`)
  - `SUPABASE_BUCKET_USAGE_GUARD_ON_ERROR=deny|allow` (default `deny`, рекомендуется `deny`)
- `TG_MONITORING_KERNEL_REF`
- `TG_MONITORING_KERNEL_PATH`
- `TG_MONITORING_CONFIG_CIPHER`
- `TG_MONITORING_CONFIG_KEY`
- `TG_MONITORING_POLL_INTERVAL`
- `TG_MONITORING_LOCAL_RESULTS_GLOB` — glob для поиска локальных результатов import-only кнопки (по умолчанию `tg-monitor-*/telegram_results.json` в системном temp-dir; в UI показываются 4 последних).

Таймаут ожидания Kaggle (на стороне бота, polling):

- `TG_MONITORING_TIMEOUT_MODE=dynamic|fixed` (default `dynamic`)
- `TG_MONITORING_TIMEOUT_MINUTES` — базовый/минимальный таймаут (default `90`)
- `TG_MONITORING_TIMEOUT_BASE_MINUTES` — базовая прибавка для dynamic (default `15`)
- `TG_MONITORING_TIMEOUT_PER_SOURCE_MINUTES` — baseline прибавка на источник (default `3.64`)
- `TG_MONITORING_TIMEOUT_SAFETY_MULTIPLIER` — safety multiplier для baseline (default `1.3`)
- `TG_MONITORING_TIMEOUT_MAX_MINUTES` — верхняя граница для dynamic (default `360`)

В режиме `dynamic` итоговый таймаут считается так:
`max(TG_MONITORING_TIMEOUT_MINUTES, TG_MONITORING_TIMEOUT_BASE_MINUTES + ceil(sources * TG_MONITORING_TIMEOUT_PER_SOURCE_MINUTES * TG_MONITORING_TIMEOUT_SAFETY_MULTIPLIER))`,
но не больше `TG_MONITORING_TIMEOUT_MAX_MINUTES`.

Скан лимиты (в Kaggle):

- `TG_MONITORING_LIMIT` — максимум сообщений **на источник** (по умолчанию 50).
- `TG_MONITORING_DAYS_BACK` — глубина по дням (по умолчанию 3).
- `TG_MONITORING_RECENT_RESCAN_ENABLED` — включить bounded backfill уже известных сообщений с событиями в окне `TG_MONITORING_DAYS_BACK` (по умолчанию 1).
- `TG_MONITORING_RECENT_RESCAN_LIMIT` — максимум уже известных event-постов на источник для этого backfill-прохода (по умолчанию 200, но не меньше `TG_MONITORING_LIMIT`).

Live E2E multi-source (VK+TG): `tests/e2e/features/multi_source_vk_tg.feature` (рекомендуемо запускать с `TG_MONITORING_LIMIT=10`).
- `TG_MONITORING_DAYS_BACK` — сколько дней сканировать назад. Для E2E держите дефолт `3`; для старых кейсов не расширяйте окно глобально, а добирайте конкретный `message_id` точечно.
- `TG_MONITORING_LIMIT` — лимит сообщений на источник за запуск.
- `TG_MONITORING_RECENT_RESCAN_LIMIT` — отдельный лимит на backfill уже известных event-постов; это не глобальный лимит по всем источникам, а лимит на один источник.
- `TG_MONITORING_MEDIA_MAX_PER_SOURCE` — лимит скачиваний медиа на источник (снижает шанс FloodWait).
- `TG_MONITORING_MEDIA_DELAY_MIN/MAX` — дополнительные задержки перед скачиванием медиа (снижает шанс FloodWait).
- `EVENT_TOPICS_LLM=gemma` — чтобы классификация тем не использовала 4o (Gemma-only).
- `EVENT_TOPICS_MODEL` — модель Gemma для классификации тем (по умолчанию `TG_MONITORING_TEXT_MODEL`).
- `TELEGRAPH_TOKEN_FILE` — путь к токену Telegraph. В dev среде автоматически фолбэкается на `artifacts/run/telegraph_token.txt`, если `/data` недоступен на запись.

## Контракт результата

Сервер принимает `telegram_results.json`:

- `schema_version=1` (legacy): только `messages[]` (без `sources_meta`);
- `schema_version=2`: `messages[]` + top-level `sources_meta[]` с метаданными источников и подсказками.

- Producer (Kaggle): `kaggle/TelegramMonitor/telegram_monitor.ipynb`
- Consumer (server): `source_parsing/telegram/handlers.py`

## FloodWait (Telegram rate limits)

Если в Kaggle логах появляется `FloodWaitError` или строки вида `Sleeping for Xs on GetHistoryRequest flood wait`, Telegram ограничил скорость запросов.

Типовые причины:

- Слишком большой объём сканирования: много источников и/или большой `TG_MONITORING_LIMIT`, `TG_MONITORING_DAYS_BACK` (особенно после очистки отметок мониторинга).
- Слишком большой bounded backfill по тихим/редко обновляемым каналам: завышенный `TG_MONITORING_RECENT_RESCAN_LIMIT` тоже увеличивает нагрузку, хотя и не расширяет само временное окно.
- Слишком агрессивные задержки (`TG_MONITORING_DELAY_*`, `TG_MONITORING_SOURCE_PAUSE_*`).
- Параллельные запуски мониторинга (ручной и scheduled) с одной и той же Telegram-сессией.

Митигации (ENV, пробрасываются в Kaggle):

- Увеличить “human-like” задержки: `TG_MONITORING_DELAY_MIN/MAX`, `TG_MONITORING_SOURCE_PAUSE_MIN/MAX`.
- Ограничить и замедлить скачивание медиа (частая причина FloodWait): `TG_MONITORING_MEDIA_MAX_PER_SOURCE`, `TG_MONITORING_MEDIA_DELAY_MIN/MAX`.
- Настроить поведение Telethon при FloodWait:
  - `TG_MONITORING_FLOOD_SLEEP_THRESHOLD` (по умолчанию 600) — авто-sleep при FloodWait до N секунд.
  - `TG_MONITORING_FLOOD_WAIT_MAX` (по умолчанию 1800) — максимум ожидания на один FloodWait.
  - `TG_MONITORING_FLOOD_MAX_RETRIES` (по умолчанию 4) — сколько раз подряд терпеть FloodWait на одном участке.
  - `TG_MONITORING_FLOOD_WAIT_JITTER_MIN/MAX` — небольшой джиттер к ожиданию.

Примечание: на сервере есть lock, который не даёт запустить два мониторинга одновременно в одном процессе (manual vs scheduler), но лучше всё равно избегать ручных запусков рядом с scheduled окном.

## E2E и старые посты

- Для регрессий по конкретному старому посту используйте point-fetch по `message_id` вместо расширения `TG_MONITORING_DAYS_BACK`.
- Базовый E2E профиль: `TG_MONITORING_DAYS_BACK=3`, умеренный `TG_MONITORING_LIMIT`.
- Причина: широкий перескан резко увеличивает время прогона, FloodWait-риск и количество лишних запросов в Gemma (лимиты ограничены).

## Очистка (DB + Supabase)

- Ежедневная очистка удаляет события, завершившиеся более 7 дней назад (по `end_date`, либо по `date` если `end_date` пуст).
- В рамках той же очистки (best-effort) удаляются связанные объекты из Supabase Storage:
  - ICS файлы события;
  - fallback афиши по `eventposter.supabase_path`.

## Acceptance (Gherkin)

Канонические сценарии (UI): `tests/e2e/features/telegram_monitoring.feature`.

Если нужно добавить/уточнить сценарий — правим `.feature` и шаги в `tests/e2e/features/steps/bot_steps.py`.

## Отложенное обновление страниц

Telegram Monitoring может обновлять/создавать много событий за один запуск, поэтому обновления month/weekend страниц делаются **отложенно и накопительно** (debounce 15 минут после последнего изменения). Каноническое описание механизма — в `docs/features/smart-event-update/README.md` («Отложенное обновление страниц (debounce)»).
