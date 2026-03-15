# Общий суточный отчёт (/general_stats)

> **Status:** spec / in progress  
> **Команда:** `/general_stats`  
> **Плановый запуск:** 07:30 `Europe/Kaliningrad`  
> **Назначение:** ежедневная “операторская” сводка по ключевым пайплайнам и техническим метрикам за **прошлые сутки**.

## 1) Поведение и маршрутизация

### 1.1. Период отчёта (“прошлые сутки”)

“Прошлые сутки” считаем **от отчёта к отчёту**: скользящее окно **24 часа**,
заканчивающееся в момент генерации отчёта.

Для планового утреннего отчёта это означает окно **с вчера ~07:30 до сегодня ~07:30**
по Калининградскому времени (фактические границы зависят от момента запуска job, но окно
всегда 24 часа).

Алгоритм:

- `tz = Europe/Kaliningrad` (или `GENERAL_STATS_TZ`)
- `end_local = now(tz)` (момент построения отчёта)
- `start_local = end_local - 24h`
- запросы к sqlite делаем по UTC-границам `start_utc..end_utc` (timestamps в sqlite пишутся как `CURRENT_TIMESTAMP`, т.е. UTC).

### 1.2. Куда отправляется отчёт

- **По расписанию**: отправляется в оба чата:
  - операторский (`OPERATOR_CHAT_ID`), если задан
  - чат superadmin из базы (`user.is_superadmin=1`); `ADMIN_CHAT_ID` используется только как legacy fallback, если superadmin ещё не зарегистрирован в БД
- **Вручную** (`/general_stats`): отправляется **в тот чат**, откуда вызвана команда.

### 1.3. Права доступа

Рекомендуемый доступ: только `superadmin` (операционный отчёт, включает тех.метрики и внутренние статусы).

## 2) Расписание (APScheduler)

Добавляется отдельный cron-job в `scheduling.py` по аналогии с `/parse`, `/3di`, `festival_queue`.

### ENV

- `ENABLE_GENERAL_STATS=1` — включает плановый отчёт
- `GENERAL_STATS_TIME_LOCAL=07:30`
- `GENERAL_STATS_TZ=Europe/Kaliningrad`

Поведение:
- `max_instances=1`, `coalesce=True`, `misfire_grace_time=30`
- если `ENABLE_GENERAL_STATS!=1` — тихо не запускаем
- если включено, но `OPERATOR_CHAT_ID` невалиден и chat superadmin не удалось получить ни из БД, ни из `ADMIN_CHAT_ID` fallback — логируем warning и пропускаем плановую отправку.

## 3) Что должно попасть в отчёт (метрики за `start_local..end_local`)

Ниже перечислены метрики, которые требуются в отчёте. Для каждой указаны **источник данных** и “точность”:
- **Exact** — можно посчитать детерминированно из текущих таблиц.
- **Needs run log** — требует отдельного логирования запусков (иначе невозможно восстановить “сколько и во сколько запускалось”).

### 3.1. VK intake / VK auto-import

1) **Сколько VK‑постов добавилось** (Exact)  
   Источник: `vk_inbox.created_at` (факт вставки в очередь).

2) **Сколько VK‑постов было разобрано за период** (Exact + fallback)  
   Основной источник: `ops_run.metrics_json.inbox_processed` для `kind='vk_auto_import'`
   в окне отчёта (`vk_queue_parsed_period` в `/general_stats`).  
   Fallback для старых периодов без корректного run-log: `COUNT(DISTINCT inbox_id)` из
   `vk_inbox_import_event.created_at` в окне отчёта.

3) **Сколько сейчас неразобранных в VK‑очереди** (Exact snapshot)  
   Источник: `vk_inbox.status`, считаем как
   `COUNT(*) WHERE status NOT IN ('imported', 'rejected')`
   (`vk_queue_unresolved_now` в `/general_stats`).

4) **Сколько VK‑постов было автоимпортировано в события** (Exact)  
   Источник: `vk_inbox_import_event.created_at`.  
   Примечание: один VK‑пост может дать несколько событий → считаем:
   - `vk_posts_auto_imported = COUNT(DISTINCT inbox_id)`  
   - `vk_events_from_auto_import = COUNT(*)` (по строкам `vk_inbox_import_event`)

5) **Сколько раз успешно была запущена авторазборка VK‑очереди (время + статус)** (Needs run log)  
   Без отдельного run-log нельзя восстановить:
   - время старта/финиша прогона
   - итоговый статус (success/failed/canceled)
   - лимит/режим (`--include-skipped`, `limit`)

### 3.2. Telegram monitoring

6) **Сколько Telegram‑каналов/источников было промониторено** (Exact)  
   Источник: `telegram_scanned_message.processed_at` → `COUNT(DISTINCT source_id)`.

7) **Сколько Telegram‑постов содержали события из какого количества каналов** (Exact)  
   Источник: `telegram_scanned_message`:
   - `messages_with_events = COUNT(*) WHERE events_extracted>0 OR events_imported>0`
   - `sources_with_events = COUNT(DISTINCT source_id) WHERE events_extracted>0 OR events_imported>0`

### 3.2.1. Guide excursions monitoring

Дополнительно `/general_stats` теперь показывает отдельный блок `Guide excursions`:

- `sources_scanned` — `COUNT(DISTINCT source_id)` из `guide_monitor_post.last_scanned_at` в окне отчёта;
- `posts_prefiltered` — сколько guide-постов прошло prefilter и были признаны потенциально экскурсионными;
- `occurrences_new` — сколько новых `guide_occurrence` впервые появилось в окне отчёта;
- `digest_published` — сколько выпусков `guide_digest_issue(status='published')` было отправлено;
- `guide_monitoring runs` — список прогонов `ops_run(kind='guide_monitoring')`.

### 3.3. /parse (source parsing)

8) **Сколько и во сколько было успешных запусков `/parse`** (Needs run log)  
9) **Сколько он обработал источников, сколько новых событий дал каждый источник** (Needs run log)  
   Важно: показываем **все успешные запуски**, даже если `0` новых событий/`0` источников — это “health monitor”.

Примечание: “best-effort” можно восстановить по `event_source.imported_at` для `source_type` театральных источников, но это:
- не даёт времена “запусков”
- плохо отделяет ручные действия и другие пайплайны
- не даёт корректный breakdown “на один запуск”.

### 3.4. /3di (3D previews)

10) **Сколько и во сколько было успешных запусков `/3di`** (Needs run log)  
11) **Сколько каждый обработал превью** (Needs run log)  
   Важно: показываем **все успешные запуски**, даже если обработано `0`.

Без run-log можно только “приблизительно” оценить по событиям, где `preview_3d_url` стал непустым в период — но у `event` нет `updated_at`, поэтому нужен отдельный журнал.

### 3.5. События (Smart Update итог)

12) **Сколько событий было создано** (Exact)  
    Источник: `event.added_at`.

13) **Сколько событий было обновлено** (Approx → лучше сделать Exact через run log Smart Update)  
    Предложение v1 (детерминированно, но приближённо): считать “обновлённым” событие, если:
    - событие создано до `start_utc` и
    - в период добавился хотя бы один источник (Smart Update): `event_source.imported_at in [start_utc,end_utc)`.

14) **Из обновлённых — сколько имеет источников 2, 3, …** (Exact при наличии определения “обновлено”)  
    Источник: `event_source` → `sources_per_event = COUNT(*) GROUP BY event_id` и распределение по множеству обновлённых.

15) **Сколько событий было по неизвестным ранее городам/НП и каким** (Needs schema support)  
    Текущее состояние:
    - есть кэш `geo_city_region_cache(city_norm, …, updated_at)`
    - есть allowlist `docs/reference/kaliningrad_oblast_places.md`

    Требование: “неизвестные” = **впервые встреченные в системе** и в итоге попавшие в кэш.
    Для этого нужно сделать “first-seen” детерминированным:

    - добавить в `geo_city_region_cache` поле `created_at` (не меняется при upsert) и считать новые `city_norm`,
      у которых `created_at in [start_utc,end_utc)`;
    - `updated_at` остаётся как “последнее обновление решения”.

### 3.6. Фестивали и фестивальная очередь

16) **Сколько событий содержало упоминаний о фестивале и добавлено в фестивальную очередь** (Exact)  
    Источник: `festival_queue.created_at` (новые элементы очереди) + фильтрация по `festival_context` при необходимости.

17) **Сколько и во сколько было успешных запусков разбора фестивальной очереди** (Needs run log)  
    Дополнительно (Exact, но не “по запускам”): можно показать, сколько элементов обработано и сколько успешно:
    - `festival_queue.updated_at` + `status in ('done','error')` в период.

18) **Сколько фестивалей было создано** (Exact)  
    Источник: `festival.created_at`.

19) **Сколько фестивалей было обновлено** (Needs schema support)  
    У `festival` нет `updated_at`. Чтобы считать “обновлено”, нужно:
    - добавить `festival.updated_at` и обновлять при изменениях, или
    - логировать изменения в отдельную таблицу фактов/источников (аналогично `event_source_fact`).

## 4) Технические параметры

20) **Сколько было запросов к Gemma за сутки от этого бота** (Depends on Supabase)  
    Требование: включать и Kaggle-run’ы (если они пишут в те же таблицы Supabase).

    Источник: Supabase таблица `google_ai_requests` (фильтр по `created_at` + `model LIKE 'gemma-%'`).  
    Fallback: `google_ai_usage_counters` (сумма `rpm_used` по `minute_bucket` в окне, `model LIKE 'gemma-%'`).
    Fallback (если таблицы Google AI не доступны): `token_usage` (фильтр по `at/created_at` + `endpoint LIKE 'google_ai%'` + `model LIKE 'gemma-%'`). Это даёт минимум по парсингу (и тем вызовам Gemma, которые логируются через `log_token_usage`).

21) **Насколько занят Supabase bucket (МБ)** (Depends on Supabase Storage)  
    Источник: Supabase Storage API (legacy bucket по `SUPABASE_BUCKET`, дефолт `events-ics`).
    Рекомендация: считать 1 раз в сутки и кэшировать результат (listing может быть дорогим).

    Примечание: ожидается миграция на 2 bucket’а (ICS отдельно, картинки отдельно); каноника по rollout/ENV:
    `docs/operations/supabase-storage.md`.

## 5) Требование: единый run log для “сколько/во сколько запускалось”

Для метрик класса **Needs run log** вводится минимальный журнал запусков.

### 5.1. Минимальная таблица (sqlite)

`ops_run` (предложение):
- `id INTEGER PK`
- `kind TEXT` (например: `vk_auto_import`, `parse`, `3di`, `festival_queue`, `tg_monitoring`, `general_stats`)
- `trigger TEXT` (`scheduled|manual`)
- `chat_id INTEGER NULL`
- `operator_id INTEGER NULL`
- `started_at TIMESTAMP NOT NULL`
- `finished_at TIMESTAMP NULL`
- `status TEXT` (`running|success|error|canceled|skipped`)
- `metrics_json JSON` (агрегаты прогона)
- `details_json JSON` (breakdown: по источникам, по шагам)
- индексы: `(kind, started_at)`, `(status, started_at)`

### 5.2. Что логировать по прогонам

- `vk_auto_import`: `limit`, `include_skipped`, `inbox_imported`, `inbox_rejected`, `events_created`, `events_updated`, `duration_sec`
- `/parse`: список источников + `new_added`/`updated`/`failed` по каждому
- `/3di`: `events_considered`, `previews_rendered`, `duration_sec`
- `festival_queue`: `limit`, `source_kind`, `processed/success/failed`, `duration_sec`
- `tg_monitoring`: `sources_scanned`, `messages_processed`, `messages_with_events`, `events_imported`

## 6) Формат сообщения (рекомендация)

Один текст (HTML или Markdown), структура:

1) Заголовок: дата + период (Kaliningrad), версия/бот-код  
2) VK (`vk_queue_added_period`, `vk_queue_parsed_period`, `vk_queue_unresolved_now` + автоимпорт метрики)  
3) Telegram  
4) `/parse` и `/3di` (список запусков с временем Kaliningrad)  
5) События (created/updated + breakdown по кол-ву источников)  
6) География (новые города)  
7) Фестивали/очередь  
8) Тех.метрики (Gemma, bucket size)

Для блоков `vk_auto_import runs`, `tg_monitoring runs`, `/parse runs`, `/3di runs`:

- каждая строка должна печатать `status=...` и `trigger=...` (`scheduled|manual|...`);
- если запуск завершился как `skipped`, в строке нужно показывать `reason=...`;
- для scheduler skip из-за общего heavy guard дополнительно показываем `blocked_by=...` (какая тяжёлая операция занимала слот).

## 7) /help

Нужно добавить строку в `HELP_COMMANDS`:

- `usage`: `/general_stats`
- `desc`: “Daily general system report for the previous 24 hours”
- `roles`: `{"superadmin"}`
