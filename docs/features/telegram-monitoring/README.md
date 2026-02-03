# Telegram Monitoring

Ежедневный мониторинг публичных Telegram‑каналов/групп с автоматическим импортом событий в БД бота через Smart Event Update.

## Что делает

- По расписанию запускает Kaggle‑kernel `TelegramMonitor`.
- Kaggle читает сообщения источников, грузит медиа в Catbox, делает OCR и извлекает события.
- Для афиш (постеров) также делает загрузку в Supabase Storage (fallback), чтобы Telegraph/Telegram preview не зависели от доступности Catbox.
- Сервер скачивает `telegram_results.json` и импортирует события через Smart Update:
  - создаёт новые события;
  - мерджит существующие;
  - добавляет источники в `event_source`.
- В Kaggle используются только модели Gemma (текст/vision); 4o там не участвует.

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

## Фильтры и санитаризация

- **Custom emoji** (Telegram `MessageEntityCustomEmoji` / `<tg-emoji>`) вычищаются из текста перед публикацией в Telegraph (обычные Unicode‑эмодзи остаются).
- **Розыгрыши билетов** (giveaway) не импортируются как события: сообщения с такими сигналами скипаются на этапе мониторинга и дополнительно защищены в Smart Update.
- **Акции/промо/поздравления** (не‑ивент контент) не импортируются как события и не должны становиться источниками события (например посты «Поздравляем…» со списком ближайших спектаклей).
- OCR‑подсказка времени стала устойчивее: поддерживаются диапазоны (`10:00–18:00`), выбор времени при множественных упоминаниях на афише и защита от ложных совпадений типа `05.02` → `05:02` (дата на афише не должна становиться временем).

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
- `TG_MONITORING_DAYS_BACK` — сколько дней сканировать назад (важно для E2E кейсов со старыми постами).
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
