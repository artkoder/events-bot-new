# Smart Event Update (Интеллектуальный импорт)

Автоматический мердж события из разных источников без ручной модерации, с сохранением списка источников и защитой якорных полей.

## Что реализовано

- Матчинг кандидата с существующими событиями:
  - быстрые проверки по `ticket_link` и `poster_hash`;
  - LLM‑матчинг (JSON‑ответ с `match_event_id`, `confidence`).
- Мердж:
  - якорные поля (`date/time/location_name/location_address/end_date`) не меняются автоматически;
  - описание и необязательные поля обогащаются через LLM (для Telegram‑импорта — журналистский рерайт, не дословно);
  - конфликты фиксируются в логах (`added_facts`, `skipped_conflicts`).
- Санитаризация текста:
  - входящие хештеги удаляются из `title/description/source_text`, чтобы они не попадали на Telegraph страницы.
  - Telegram custom emoji (PUA / `<tg-emoji>`) вычищаются из текста перед публикацией.
- Фильтры:
  - розыгрыши билетов (ticket giveaway) не импортируются как события (`skipped_giveaway`).
  - акции/промо/поздравления (не‑ивент контент) не импортируются как события (`skipped_promo`).
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
- Каждый вызов Smart Update должен иметь заполненные `source_type` и `source_url` (они влияют на лог источников и счётчик в Telegraph).
- Для ручного добавления через бота (`/addevent`, `/addevent_raw`) источник фиксируется как **bot**.

## Инциденты / риски

### OCR может ошибаться в названии локации

Инцидент: OCR (Gemma) ошибочно распознал «Калининградская» как «Каминская», что может увести `location_name` при импорте из Telegram.

Митигируемо:
- Для Telegram‑источников с `default_location` используется **проверка на конфликт**: если распознанная локация не совпадает с `default_location`, берём `default_location` и сбрасываем `location_address`.
- В сценариях E2E есть проверка, что локация события из @kaliningradlibrary остаётся «Научная библиотека», даже если OCR даёт шум.

## Прозрачность источников (требование)

Чтобы оператор понимал, как сформирована карточка события:

1. **Telegraph footer** (в конец страницы события):
   - строка `Источников: N` (без перечисления самих источников);
   - строка `Последнее обновление: YYYY-MM-DD HH:MM (TZ)` — время последнего Smart Update (локальный TZ, Калининград).
2. **Лог фактов (added_facts) по источникам**:
   - доступен из `/events -> Edit` через отдельную кнопку;
   - формат — журнал с датой/временем, источником и фактом (added_facts), что именно было добавлено/смёрджено.

## Acceptance (Gherkin)

Канонические сценарии:

- `tests/e2e/features/smart_event_update.feature` (пограничные кейсы матчинга/мерджа).
- `tests/e2e/features/telegram_monitoring.feature` (обогащение событий из Telegram Monitoring).
