# Telegraph Cache Sanitizer

Проблема: часть Telegraph‑страниц (события/фестивали/месяцы/выходные) иногда открываются в Telegram с “чёрным экраном” или вместо Instant View открывается браузер. На уровне MTProto это обычно выглядит как отсутствие `webpage.cached_page` (Instant View) и/или `webpage.photo` (превью‑картинка) у web preview для ссылки.

Цель: **прогревать** Telegram web preview для ключевых Telegraph‑страниц, **собирать статистику** по “битым” страницам и **автоматически ставить их на пересборку** при повторяющихся фейлах.

## Почему нужен Telethon (а не bot API)

Проверка и получение полей `cached_page`/Instant View делается через MTProto web preview (`messages.getWebPagePreview` / `messages.getWebPage`) и доступна только **пользовательским** аккаунтам, не ботам. Поэтому прогрев/проверка выполняется Telethon‑сессией.

## Как работает

1) Сервер собирает список URL для проверки:
   - страницы событий (`event.telegraph_url/telegraph_path`);
   - фестивальные страницы (`festival.telegraph_url/telegraph_path`);
   - индекс фестивалей (landing);
   - страницы месяцев (`monthpagepart`/`monthpage`) и выходных (`weekendpage`) в окне дат.

   Чтобы не “застревать” в одном и том же `LIMIT`‑окне (например, ~200 первых ссылок), список событий **ротируется**:
   приоритет — страницы со streak‑фейлами, дальше — “давно не проверяли / никогда не проверяли”.

Важно: санитайзер **не проверяет и не пересобирает** “старые” страницы:

- события, которые уже закончились (`end_date < today`);
- страницы месяцев в прошлом (только текущий и будущие);
- страницы выходных, которые уже прошли (после окончания воскресенья).
2) Сервер запускает Kaggle kernel `TelegraphCacheProbe` (Telethon user session):
   - на каждую ссылку отправляет URL в “Saved Messages” (`me`);
   - ждёт, пока Telegram прикрепит `MessageMediaWebPage`;
   - ждёт `webpage.cached_page` (Instant View) в рамках тайм‑бюджета; если `cached_page` не появился в preview, делает best‑effort refresh через `messages.getWebPage`;
   - проверяет наличие `webpage.cached_page` (Instant View) и `webpage.photo` (превью‑картинка);
   - **OK** считается наличие `cached_page`; отсутствие `photo` — это **warning** (не повод автоматически пересобирать страницу само по себе);
   - повторяет попытку `N` раз с джиттером и таймаутом;
   - (опционально) удаляет отправленные сообщения.
3) Kaggle пишет `telegraph_cache_report.json`.
4) Сервер импортирует результаты в SQLite таблицу `telegraph_preview_probe` и считает streak’и.
5) Если страница падает **несколько прогонов подряд**, сервер ставит пересборку в JobOutbox:
   - `event` → `telegraph_build`
   - `festival` → `festival_pages`
   - `month` → `month_pages`
   - `weekend` → `weekend_pages`
   - `festivals_index` → принудительный rebuild индекса (best‑effort).

## Команды

- `/telegraph_cache_stats [kind]` — статистика по последним прогонам (`kind`: `event|festival|month|weekend|festivals_index`).
- `/telegraph_cache_sanitize [--limit=N] [--no-enqueue] [--back=N] [--forward=N] ...` — ручной запуск прогона (Kaggle/Telethon) + импорт + enqueue пересборок.

Во время `/telegraph_cache_sanitize` бот показывает прогресс как в `/tg`: одно сообщение со статусом Kaggle kernel обновляется (edit) во время опроса (`prepare → pushed → poll → complete/failed/timeout`).

Команды доступны только superadmin и добавлены в `/assist` (`/a`) для поиска естественным языком.

Канонический список команд: `docs/operations/commands.md`.

## Планировщик (cron)

По расписанию запускается APScheduler job `telegraph_cache_sanitize` (disabled by default):

- `ENABLE_TELEGRAPH_CACHE_SANITIZER=1`
- `TELEGRAPH_CACHE_TIME_LOCAL` (default `01:10`)
- `TELEGRAPH_CACHE_TZ` (default `Europe/Kaliningrad`)

Job помечен как “heavy”: не должен пересекаться с другими Kaggle/LLM/рендер‑операциями (см. общий guard в `docs/operations/cron.md`).

## ENV (основные)

Schedule:

- `ENABLE_TELEGRAPH_CACHE_SANITIZER`
- `TELEGRAPH_CACHE_TIME_LOCAL` / `TELEGRAPH_CACHE_TZ`

Targets:

- `TELEGRAPH_CACHE_DAYS_BACK` (default `7`)
- `TELEGRAPH_CACHE_DAYS_FORWARD` (default `120`)
- `TELEGRAPH_CACHE_LIMIT_EVENTS` (default `180`)
- `TELEGRAPH_CACHE_LIMIT_FESTIVALS` (default `80`)
- `TELEGRAPH_CACHE_INCLUDE_MONTHS=1`
- `TELEGRAPH_CACHE_INCLUDE_WEEKENDS=1`
- `TELEGRAPH_CACHE_INCLUDE_FESTIVALS=1`
- `TELEGRAPH_CACHE_INCLUDE_FESTIVALS_INDEX=1`

Probe tuning:

- `TELEGRAPH_CACHE_REPEATS` (default `2`)
- `TELEGRAPH_CACHE_ATTACH_WAIT_SEC` (default `20`)
- `TELEGRAPH_CACHE_PER_URL_TIMEOUT_SEC` (default `35`)
- `TELEGRAPH_CACHE_DELETE_MESSAGES` (default `1`)
- `TELEGRAPH_CACHE_DELAY_MIN_SEC` / `TELEGRAPH_CACHE_DELAY_MAX_SEC`
- `TELEGRAPH_CACHE_REPEAT_PAUSE_MIN_SEC` / `TELEGRAPH_CACHE_REPEAT_PAUSE_MAX_SEC`
- `TELEGRAPH_CACHE_KAGGLE_TIMEOUT_MIN` (default `35`)

Regeneration policy:

- `TELEGRAPH_CACHE_ENQUEUE_REGEN=1`
- `TELEGRAPH_CACHE_REGEN_AFTER_RUNS` (default `2`)

Secrets:

- Kaggle: `KAGGLE_USERNAME`, `KAGGLE_KEY`
- Telethon: `TG_API_ID` + `TG_API_HASH` и одна из:
  - `TELEGRAM_AUTH_BUNDLE_S22` / `TELEGRAM_AUTH_BUNDLE_E2E` (предпочтительнее)
  - `TG_SESSION` / `TELEGRAM_SESSION` (fallback)
- Выбор bundle‑ключа: `TELEGRAPH_CACHE_AUTH_BUNDLE_ENV` (если нужно форсировать, например `TELEGRAM_AUTH_BUNDLE_E2E`).

## Важное про “битые” медиа (Supabase 404 / Catbox)

Частая первопричина отсутствия `cached_page`/preview — Telegraph‑страница ссылается на картинки, которые:

- возвращают `404` (объект удалён/не долетел в Storage);
- требуют auth (signed URL истёк);
- недоступны для Telegram fetcher (TLS/redirect/anti‑bot).

Что помогает снизить воспроизводимость:

- держать `TELEGRAPH_VALIDATE_IMAGE_URLS=1` (по умолчанию включено): перед публикацией/пересборкой страницы URL’ы проверяются range‑GET и битые URL заменяются fallback’ами (Catbox↔Supabase) или выкидываются;
- избегать WEBP‑only cover’ов на Telegraph страницах: Telegram не всегда генерирует `cached_page`, когда **первое** изображение на странице — WEBP. Бот умеет best‑effort делать JPEG‑mirror для cover в Supabase (`SUPABASE_TELEGRAPH_COVER_PREFIX`, default `tgcover`).
- регулярно прогонять аудит Supabase Storage для ссылок из БД:
  - `python scripts/inspect/audit_media_dedup.py --db db_prod_snapshot.sqlite --hours 168 --check-storage`
  - если есть “missing objects”, имеет смысл либо восстановить объект (перезагрузка из источника), либо убрать ссылку из DB и пересобрать Telegraph страницы.

## Точки входа

- `telegraph_cache_sanitizer.py` — оркестрация Kaggle прогона, импорт статистики, enqueue пересборок.
- `kaggle/TelegraphCacheProbe/telegraph_cache_probe.py` — Telethon probe kernel.
- `handlers/telegraph_cache_cmd.py` — команды бота.
- `scheduling.py` — расписание.
- `db.py` — таблица `telegraph_preview_probe`.
