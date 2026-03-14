# Guide Excursions MVP

> **Status:** Implementation-ready design  
> **Goal:** получить полезный рабочий мониторинг и реальные digest-публикации уже на первом запуске, без public pages на собственном домене и без UI для управления источниками.

Каноника по домену и кейсам:

- high-level design: `docs/backlog/features/guide-excursions-monitoring/README.md`
- source taxonomy + case analysis: `docs/backlog/features/guide-excursions-monitoring/casebook.md`
- future static pages / own domain: `docs/backlog/features/static-event-pages/README.md`

## 1. Что должно дать ценность сразу

MVP считается успешным, если в тот же день после запуска даёт три вещи:

1. находит новые или изменившиеся экскурсии в seed-списке каналов;
2. публикует полезный digest в тестовый канал `@keniggpt`;
3. даёт оператору понятные admin-поверхности и блок в `/general_stats`, чтобы видеть, что мониторинг живой и что он нашёл.

## 2. Что входит в MVP

### In scope

- Telegram-only monitoring;
- seed источников через migration / seed script, без UI добавления;
- OCR картинок;
- regex/heuristic prefilter до LLM;
- подтверждение и нормализация через Gemma по отдельному ключу `GOOGLE_API_KEY2`;
- occurrence-level хранение в отдельных guide-таблицах;
- два публичных digest family:
  - `new_occurrences`
  - `last_call`
- admin-инструменты для ручного запуска, просмотра результатов и диагностики;
- отдельный блок в `/general_stats`;
- автозапуск по расписанию;
- публикация только в тестовый канал `@keniggpt`.

### Out of scope

- VK monitoring;
- public static pages / custom-domain site;
- guide profile pages;
- excursion template pages;
- stories / Bento images;
- UI для ручного добавления/редактирования guide sources;
- convergence в обычные `Event` текущего бота.

## 3. Seed pack источников для MVP

Источники seed’ятся миграцией или idempotent seed-командой в отдельную таблицу `guide_source`.

### Первичный пакет

| Username | Source kind | Trust | Notes |
|---|---|---:|---|
| `tanja_from_koenigsberg` | `guide_personal` | `high` | сильный первичный source, много коллабораций |
| `gid_zelenogradsk` | `guide_personal` | `high` | структурные анонсы, смешение с лекциями |
| `katimartihobby` | `guide_personal` | `high` | caption-heavy, авторская подача |
| `amber_fringilla` | `guide_personal` | `high` | природа/орнитология, mixed content |
| `alev701` | `guide_personal` | `medium` | редкие анонсы, шумная историческая лента |
| `twometerguide` | `guide_project` | `medium` | multi-region, сильный media/lifestyle шум |
| `valeravezet` | `guide_project` | `medium` | promo-heavy, бренд-автор |
| `ruin_keepers` | `organization_with_tours` | `medium` | heritage-organization, tours as one product line |
| `vkaliningrade` | `aggregator` | `medium` | fallback-aggregator, meeting-point updates |

### Обязательные source flags

- `allow_out_of_region=false` по умолчанию
- `allow_out_of_region=true` для `twometerguide`
- `aggregator=true` для `vkaliningrade`
- `organization=true` для `ruin_keepers`
- `caption_heavy=true` для `katimartihobby`
- `promo_noise=true` для `valeravezet`
- `collaboration_heavy=true` для `tanja_from_koenigsberg`, `amber_fringilla`

## 4. Стратегия хранения для MVP

### Решение

Использовать **основную SQLite базу**, но **отдельные guide-таблицы**, а не текущую `event`.

Это даёт три преимущества:

- нулевая утечка в `/daily`, month pages, weekend pages;
- отдельная domain-model для occurrence/status/template hints;
- простая интеграция с `/general_stats`, scheduler и админ-командами без второй БД.

### Почему не `event` table

Для MVP это создаёт больше проблем, чем пользы:

- текущая `event`-модель заточена под публичные event pages;
- occurrence/status-update логика экскурсий отличается от обычных культурных событий;
- придётся сразу решать слишком много public-surface guardrails.

### Минимальные таблицы MVP

#### `guide_source`

- `id`
- `platform='telegram'`
- `username`
- `title`
- `source_kind`
- `trust_level`
- `priority_weight`
- `enabled`
- `flags_json`
- `base_region`
- `added_via`
- `created_at`
- `updated_at`

#### `guide_monitor_post`

- `id`
- `source_id`
- `message_id`
- `grouped_id`
- `post_date`
- `views`
- `forwards`
- `reactions_total`
- `content_hash`
- `post_kind`
- `prefilter_passed`
- `llm_status`
- `last_scanned_at`

Назначение:

- idempotency;
- admin-diagnostics;
- база для future metrics windows.

#### `guide_occurrence`

- `id`
- `canonical_title`
- `guide_names_json`
- `organizer_names_json`
- `date`
- `time`
- `duration_text`
- `city`
- `meeting_point`
- `price_text`
- `booking_text`
- `booking_url`
- `channel_url`
- `cover_image_url`
- `image_phash`
- `occurrence_status`
  - `planned`
  - `few_seats`
  - `waitlist`
  - `sold_out`
  - `cancelled`
  - `done`
- `source_quality`
  - `original`
  - `organization`
  - `aggregator_fallback`
- `summary_one_liner`
- `raw_facts_json`
- `first_seen_at`
- `last_seen_at`
- `published_new_digest_at`
- `published_last_call_digest_at`

#### `guide_occurrence_source`

- `id`
- `occurrence_id`
- `source_id`
- `message_id`
- `role`
  - `primary`
  - `duplicate`
  - `status_update`
  - `template_signal`
- `source_url`
- `views`
- `reactions_total`
- `snapshot_at`

#### `guide_digest_issue`

- `id`
- `digest_family`
- `scope`
- `channel_username`
- `published_at`
- `items_count`
- `payload_hash`
- `status`

## 5. Notebook / pipeline MVP

### Новый guide-specific notebook

Отдельный Kaggle notebook, не копия текущего `TelegramMonitor`.

Рабочее название:

- `GuideExcursionsMonitor`

### Почему отдельный notebook

- current TG monitoring слишком дорогой для этой задачи;
- guide sources требуют другой prefilter и другой JSON contract;
- удобнее отдельно подключить `GOOGLE_API_KEY2`.

### Runtime stages

1. `Telethon fetch`
   - последние `N` сообщений на источник;
   - grouped albums collapse;
   - views / forwards / reactions / links / handles;
   - media references.
2. `OCR`
   - только для candidate posts с изображениями.
3. `Deterministic prefilter`
   - regex + heuristic scoring;
   - coarse post-kind guess.
4. `Gemma confirmation`
   - только по prefiltered items;
   - output: `ignore | announce_occurrence | status_update | template_signal`.
5. `Server import`
   - occurrence merge;
   - digest candidate extraction;
   - stats + admin report.

## 6. Scan modes

MVP должен быть **двухрежимным**.

### `full_scan`

Назначение:

- находить новые occurrences;
- обновлять полноту карточек;
- находить template hints;
- собирать cover-image и summary.

Поведение:

- `15` последних сообщений на источник;
- OCR по candidate media;
- полный LLM-pass по prefiltered items;
- occurrence merge/create/update;
- digest family `new_occurrences`.

### `light_scan`

Назначение:

- быстро ловить few seats / sold out / meeting point / reminder updates;
- не занимать много heavy-job окна.

Поведение:

- `5` последних сообщений на источник;
- recheck active occurrences на горизонте `7` дней;
- OCR только если без OCR нельзя принять решение;
- LLM только для:
  - `status_update`
  - очень сильных `announce_*` candidates
- digest family `last_call`.

### Почему не достаточно одного типа скана

Потому что `last_call`-сигналы живут быстро и коротко.

Если делать только один вечерний full scan:

- освобождение мест утром будет найдено слишком поздно;
- точка встречи “на завтра” может стать бесполезной;
- оператор потеряет главный operational benefit фичи.

## 7. Расписание MVP

Нужно избегать пересечений с текущими heavy jobs:

- `/parse`: `04:30`, `14:15`
- `/3di`: `05:30`, `15:15`, `17:15`
- `vk_auto_import`: `06:15`, `10:15`, `12:00`, `18:30`
- `/general_stats`: `07:30`
- `tg_monitoring`: `23:40`

### Рекомендуемое расписание

#### Light scans

- `09:05` Europe/Kaliningrad
- `13:20` Europe/Kaliningrad

Причины:

- уже после утренних `/daily`;
- не пересекается с `07:30` `/general_stats`;
- остаётся запас до `10:15` и до `14:15`.

#### Full scan

- `20:10` Europe/Kaliningrad

Причины:

- после `18:30` `vk_auto_import` с запасом;
- далеко до `23:40` `tg_monitoring`;
- хороший слот для публикации вечернего digest в тестовый канал.

### Scheduler policy

- `light_scan` должен считаться heavy job, но коротким;
- если heavy gate занят, scan можно skip’нуть, но это должно попасть в `ops_run` и `/general_stats`;
- full scan пропускать хуже, чем light, поэтому для него лучше policy `wait` либо отдельный retry slot.

## 8. Публичные дайджесты MVP

### 8.1. `new_occurrences`

Автопубликация после `20:10 full_scan`.

Содержит:

- только occurrences, которые ещё не были опубликованы в `new_occurrences`;
- максимум `8` карточек за выпуск;
- сортировка:
  1. близость даты
  2. completeness
  3. source priority
  4. relative popularity inside source

Формат карточки:

- гид / гиды
- дата и время
- маршрут
- одно предложение summary
- место встречи
- стоимость
- запись / ссылка
- канал гида

Медиа:

- до `1` картинки на occurrence;
- дедуп по `image_phash`;
- если картинок нет или все дубльные, digest остаётся текстовым.

### 8.2. `last_call`

Автопубликация после light scan **только по delta**, если есть новые сигналы:

- `few_seats`
- `waitlist`
- `sold_out`
- `meeting_point_update`
- `rescheduled`

Содержит:

- только ещё не публиковавшиеся status deltas;
- максимум `6` карточек;
- приоритет ближайшим датам и scarcity signals.

Формат компактнее, чем у `new_occurrences`.

### Что не публикуем в MVP автоматически

- `weekend_soon`
- `premieres_and_new_routes`
- `popular_inside_channel`

Их можно делать позже, когда накопятся occurrence data и metrics windows.

## 9. Test-channel publishing

MVP публикует digest’ы **только** в тестовый канал:

- `@keniggpt`

Рекомендуемые ENV:

- `ENABLE_GUIDE_MONITORING=1`
- `GUIDE_MONITORING_TZ=Europe/Kaliningrad`
- `GUIDE_MONITORING_LIGHT_TIMES_LOCAL=09:05,13:20`
- `GUIDE_MONITORING_FULL_TIMES_LOCAL=20:10`
- `GUIDE_MONITORING_TEST_CHANNEL=@keniggpt`
- `GUIDE_MONITORING_GOOGLE_KEY_ENV=GOOGLE_API_KEY2`

## 10. Admin surfaces MVP

### `/guide_excursions`

Главное меню:

- `Run light scan`
- `Run full scan`
- `Preview new_occurrences`
- `Preview last_call`
- `Publish latest to @keniggpt`
- `Active occurrences`
- `Sources`
- `Stats`

### `/guide_recent [hours]`

Показывает:

- created occurrences;
- updated occurrences;
- status updates;
- aggregator-only findings.

### `/guide_sources`

Показывает:

- source kind;
- trust;
- sample medians;
- recent hit-rate;
- source flags.

### `/guide_digest [family]`

Manual preview of:

- `new_occurrences`
- `last_call`

## 11. `/general_stats` integration

Добавить отдельный блок `guide_monitoring`:

- runs:
  - `light/full`
  - `success/partial/error/skipped`
- `sources_total`
- `sources_scanned`
- `posts_scanned`
- `posts_prefiltered`
- `llm_checked`
- `occurrences_created`
- `occurrences_updated`
- `status_updates`
- `aggregator_only_active`
- `new_digest_items`
- `last_call_digest_items`
- breakdown by `source_kind`

## 12. MVP output schema from notebook

Серверу нужен компактный JSON без лишнего raw мусора.

### Top-level

```json
{
  "scan_mode": "light",
  "started_at": "2026-03-14T09:05:00+00:00",
  "finished_at": "2026-03-14T09:12:10+00:00",
  "sources": [...]
}
```

### Per source

```json
{
  "username": "tanja_from_koenigsberg",
  "source_kind": "guide_personal",
  "posts_scanned": 5,
  "candidates": [...]
}
```

### Candidate

```json
{
  "message_id": 3895,
  "grouped_id": null,
  "post_kind": "announce_multi",
  "source_url": "https://t.me/tanja_from_koenigsberg/3895",
  "views": 1800,
  "reactions_total": 75,
  "text": "...",
  "ocr_text": "...",
  "images": ["https://..."],
  "llm_decision": "announce_occurrence",
  "occurrences": [
    {
      "title": "Город К. Женщины, которые вдохновляют",
      "date": "2026-03-07",
      "time": "11:00",
      "guide_names": ["Татьяна Удовенко", "Юлия Гришанова"],
      "meeting_point": "у Матери России",
      "price_text": "2000 руб",
      "booking_url": "https://t.me/Yulia_Grishanova",
      "status": "planned",
      "summary_one_liner": "..."
    }
  ]
}
```

## 13. Value on day one

После реализации MVP оператор должен получить уже в первый день:

1. seed monitoring девяти каналов;
2. первый вечерний `new_occurrences` digest в `@keniggpt`;
3. как минимум один `last_call` / status digest при появлении сигнала;
4. `/guide_recent` для ручной верификации;
5. блок в `/general_stats`, чтобы не искать состояние фичи по логам.

## 14. Что откладываем осознанно

- template clustering как first-class сущность;
- guide profile synthesis;
- public pages;
- stories;
- user-facing source management UI;
- convergence в общую event-модель.

Это сознательное упрощение ради того, чтобы получить полезный operational продукт быстро, а не строить сразу “идеальную систему”.
