# Telegram Monitoring

Ежедневный мониторинг публичных Telegram‑каналов/групп с автоматическим импортом событий в БД бота через Smart Event Update.

## Что делает

- По расписанию запускает Kaggle‑kernel `TelegramMonitor`.
- Kaggle читает сообщения источников, грузит медиа в Catbox, делает OCR и извлекает события.
- Для афиш (постеров) поддерживает **fallback в Supabase Storage**, но по умолчанию **не делает dual‑upload**:
  - приоритет — Catbox;
  - Supabase используется только если Catbox‑загрузка не удалась (режим `TG_MONITORING_POSTERS_SUPABASE_MODE=fallback`) или явно включён режим `always`.
- При загрузке афиш в Supabase:
  - объект сохраняется **в WEBP** (best‑effort конвертация) для меньшего размера;
  - ключ объекта делается коротким (stable short id), а префикс по умолчанию — `p` (можно переопределить через `TG_MONITORING_POSTERS_PREFIX`), чтобы **минимизировать длину public URL**.
- Сервер скачивает `telegram_results.json` и импортирует события через Smart Update:
  - создаёт новые события;
  - мерджит существующие;
  - добавляет источники в `event_source`.
- В Kaggle используются только модели Gemma (текст/vision); 4o там не участвует.

## Multi-event посты (несколько событий в одном сообщении)

Требование: если один Telegram‑пост содержит **несколько будущих событий**, мониторинг должен:

- создать/обновить **каждое** событие отдельной записью (по `title + date + time + location` якорям);
- **не создавать** события, которые уже в прошлом (по дате);
- перед созданием нового события сначала попытаться найти матч в БД, чтобы не плодить дубли;
- на странице Telegraph конкретного события не оставлять строки расписания/названия других событий из того же поста.

Проверки этого поведения зафиксированы в E2E сценариях: `tests/e2e/features/telegram_monitoring.feature`.

## Ссылки на другие Telegram-посты (linked posts)

- Если в исходном посте найден URL вида `t.me/.../<message_id>`, Kaggle рассматривает его как связанный пост и сканирует дополнительно.
- Цель: не потерять факты, которые могут быть разнесены между “коротким” и “полным” постами про одно и то же событие.
- Для таких пар `primary_post + linked_post` добавляется отдельная LLM-проверка (Gemma) “это одно событие или разные”.
- LLM-проверка должна вернуть признак совпадения и нормализованные якоря события:
  - `same_event` (`true/false`);
  - `normalized_title`;
  - `normalized_location_name`;
  - `normalized_date`;
  - `normalized_time`;
  - `confidence` и краткое `reason`.
- Если `same_event=true`, в итоговый candidate берутся объединённые факты из двух постов, а в `event_source` сохраняются обе ссылки (`primary` и `linked`).
- Если `same_event=false`, связанные данные не сливаются в один candidate.
- Ограничения для защиты лимитов Gemma:
  - только 1 уровень обхода ссылок (без рекурсивной цепочки);
  - обрабатываются только ссылки на посты (`t.me/<channel>/<id>`);
  - для E2E и прод-прогонов используйте минимально достаточное число linked-post проверок.

## Точки входа

- `/tg` — управление источниками и ручной запуск мониторинга.
- Планировщик (`scheduling.py`) — ежедневный запуск по ENV.

## Основные модули

- `source_parsing/telegram/commands.py` — UI/команды `/tg`.
- `source_parsing/telegram/service.py` — оркестрация Kaggle и загрузка результатов.
- `source_parsing/telegram/handlers.py` — разбор `telegram_results.json`.
- `smart_event_update.py` — Smart Event Update.

## Данные

- `telegram_source` — список источников (username, trust, defaults).
- `telegram_scanned_message` — идемпотентность сообщений.
- `event_source` — источники события (много на одно событие).
- `eventposter.phash` — опциональный перцептивный хеш.
- `eventposter.supabase_url/supabase_path` — fallback URL/путь в Supabase Storage для афиш (для надёжного preview и контролируемой очистки).

## OCR

- OCR выполняется **внутри Kaggle‑ноутбука** для сообщений с афишами, даже если в тексте поста уже есть описание.
- Результаты OCR сохраняются в `telegram_results.json`:
  - `messages[].posters[].ocr_text` и `messages[].posters[].ocr_title`;
  - агрегированный `messages[].ocr_text` (для удобства дебага).
- В UI (`/events` → Edit) OCR виден в блоке **Poster OCR**.
- Проверка OCR в UI: см. `tests/e2e/features/telegram_monitoring.feature` (сценарий «Полный пользовательский поток мониторинга (UI)»).
- Для каналов с заданным `default_location` неверно распознанная локация игнорируется, приоритет у `default_location`.
- Для постов-расписаний (несколько спектаклей в одном сообщении) применяется строгая фильтрация афиш по фактам события; если она неуверенна, но Kaggle уже выдал `event_data.posters` для конкретного события, используется event-level fallback (чтобы не терять релевантную афишу при отсутствующем времени в Telegram).

## Фильтры и санитаризация

- **Custom emoji** (Telegram `MessageEntityCustomEmoji` / `<tg-emoji>`) вычищаются из текста перед публикацией в Telegraph (обычные Unicode‑эмодзи остаются).
- **Розыгрыши билетов** (giveaway): если пост содержит полноценный анонс события, импортируем/мерджим событие, но **вырезаем механику розыгрыша** (условия участия, «подпишись/репост/коммент» и т.п.). Если после очистки не остаётся признаков события — пост скипается.
- **Поздравления** (не‑ивент контент) не импортируются как события и не должны становиться источниками события (например посты «Поздравляем…» со списком ближайших спектаклей).
- **Акции/промо**: промо‑фрагменты (скидки/промокоды/«акция») **вырезаются из текста**, но если в посте есть полноценный анонс события (дата/время/место), он импортируется/мерджится.
- OCR‑подсказка времени стала устойчивее: поддерживаются диапазоны (`10:00–18:00`), выбор времени при множественных упоминаниях на афише и защита от ложных совпадений типа `05.02` → `05:02` (дата на афише не должна становиться временем).
- Инференс года для дат без года ограничен границей года (декабрь → январь): посты февраля не должны превращать январские даты в `YYYY+1`.

## UI (/tg) — настройка источников без «параметров в сообщении»

Формат вида `@channel trust=low` поддерживается как расширенный, но операторский флоу — через кнопки:

- `/tg` → `📋 Список источников`
  - `Trust → ...` — циклически: low → medium → high
  - `📍 Локация → ...` — задать/очистить `default_location`
  - `🎟 Ticket → ...` — задать/очистить `default_ticket_link`
  - `♻️ Сбросить отметки ...` — очистить `telegram_scanned_message` и `last_scanned_message_id` для перескана
  - `🗑️ Удалить ...` — удалить источник

## ENV

Минимум:

- `ENABLE_TG_MONITORING=1`
- `TG_MONITORING_TIME_LOCAL=23:40`
- `TG_MONITORING_TZ=Europe/Kaliningrad`
- `TELEGRAM_AUTH_BUNDLE_S22`, `TG_API_ID`, `TG_API_HASH`
- `GOOGLE_API_KEY`
- `KAGGLE_USERNAME`

Дополнительно:

- `TG_MONITORING_KERNEL_REF`
- `TG_MONITORING_KERNEL_PATH`
- `TG_MONITORING_CONFIG_CIPHER`
- `TG_MONITORING_CONFIG_KEY`
- `TG_MONITORING_TIMEOUT_MINUTES`
- `TG_MONITORING_POLL_INTERVAL`

Скан лимиты (в Kaggle):

- `TG_MONITORING_LIMIT` — максимум сообщений **на источник** (по умолчанию 50).
- `TG_MONITORING_DAYS_BACK` — глубина по дням (по умолчанию 3).

Live E2E multi-source (VK+TG): `tests/e2e/features/multi_source_vk_tg.feature` (рекомендуемо запускать с `TG_MONITORING_LIMIT=10`).
- `TG_MONITORING_DAYS_BACK` — сколько дней сканировать назад. Для E2E держите дефолт `3`; для старых кейсов не расширяйте окно глобально, а добирайте конкретный `message_id` точечно.
- `TG_MONITORING_LIMIT` — лимит сообщений на источник за запуск.
- `TG_MONITORING_MEDIA_MAX_PER_SOURCE` — лимит скачиваний медиа на источник (снижает шанс FloodWait).
- `TG_MONITORING_MEDIA_DELAY_MIN/MAX` — дополнительные задержки перед скачиванием медиа (снижает шанс FloodWait).
- `EVENT_TOPICS_LLM=gemma` — чтобы классификация тем не использовала 4o (Gemma-only).
- `EVENT_TOPICS_MODEL` — модель Gemma для классификации тем (по умолчанию `TG_MONITORING_TEXT_MODEL`).
- `TELEGRAPH_TOKEN_FILE` — путь к токену Telegraph. В dev среде автоматически фолбэкается на `artifacts/run/telegraph_token.txt`, если `/data` недоступен на запись.

## Контракт результата

Сервер ожидает файл `telegram_results.json` с `schema_version=1`.

- Producer (Kaggle): `kaggle/TelegramMonitor/telegram_monitor.ipynb`
- Consumer (server): `source_parsing/telegram/handlers.py`

## FloodWait (Telegram rate limits)

Если в Kaggle логах появляется `FloodWaitError` или строки вида `Sleeping for Xs on GetHistoryRequest flood wait`, Telegram ограничил скорость запросов.

Типовые причины:

- Слишком большой объём сканирования: много источников и/или большой `TG_MONITORING_LIMIT`, `TG_MONITORING_DAYS_BACK` (особенно после очистки отметок мониторинга).
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
