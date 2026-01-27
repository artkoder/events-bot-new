# Smart Event Update (Интеллектуальный импорт)

Автоматический мердж события из разных источников без ручной модерации, с сохранением списка источников и защитой якорных полей.

## Что реализовано

- Матчинг кандидата с существующими событиями:
  - быстрые проверки по `ticket_link` и `poster_hash`;
  - LLM‑матчинг (JSON‑ответ с `match_event_id`, `confidence`).
- Мердж:
  - якорные поля (`date/time/location_name/location_address/end_date`) не меняются автоматически;
  - описание и необязательные поля обогащаются через LLM;
  - конфликты фиксируются в логах (`added_facts`, `skipped_conflicts`).
- Источники:
  - таблица `event_source` хранит все источники события;
  - idempotency по `telegram_scanned_message`.
- Trust‑логика:
  - хранится `event.ticket_trust_level` и применяется при обновлении ticket‑полей.

## Где используется

- Telegram Monitoring (`source_parsing/telegram/handlers.py` → `smart_event_update.py`).
- VK ingestion (`vk_intake.persist_event_and_pages`).
- Ручной импорт (`add_events_from_text`, `/addevent_raw`).

## Важные файлы

- `smart_event_update.py` — основная логика матчинга/мерджа.
- `docs/reference/location-flags.md` — `allow_parallel_events`.
- `models.py` / `db.py` — `event_source`, `eventposter.phash`, `telegram_*` таблицы.

## Примечания

- Smart Update защищает якорные поля и дополняет данные из разных источников.
