# Telegram Monitoring

Ежедневный мониторинг публичных Telegram‑каналов/групп с автоматическим импортом событий в БД бота через Smart Event Update.

## Что делает

- По расписанию запускает Kaggle‑kernel `TelegramMonitor`.
- Kaggle читает сообщения источников, грузит медиа в Catbox, делает OCR и извлекает события.
- Сервер скачивает `telegram_results.json` и импортирует события через Smart Update:
  - создаёт новые события;
  - мерджит существующие;
  - добавляет источники в `event_source`.

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

## ENV

Минимум:

- `ENABLE_TG_MONITORING=1`
- `TG_MONITORING_TIME_LOCAL=23:40`
- `TG_MONITORING_TZ=Europe/Kaliningrad`
- `TG_SESSION`, `TG_API_ID`, `TG_API_HASH`
- `GOOGLE_API_KEY`
- `KAGGLE_USERNAME`

Дополнительно:

- `TG_MONITORING_KERNEL_REF`
- `TG_MONITORING_KERNEL_PATH`
- `TG_MONITORING_CONFIG_CIPHER`
- `TG_MONITORING_CONFIG_KEY`
- `TG_MONITORING_TIMEOUT_MINUTES`
- `TG_MONITORING_POLL_INTERVAL`

## Контракт результата

Сервер ожидает файл `telegram_results.json` с `schema_version=1` (см. `docs/backlog/features/telegram-monitoring/README.md`).
