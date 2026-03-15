# Guide Excursions MVP

> **Status:** Implementation-ready design  
> **Goal:** получить полезный рабочий мониторинг и реальные digest-публикации уже на первом запуске, без public pages на собственном домене и без UI для управления источниками.

Каноника по домену и кейсам:

- high-level design: `docs/backlog/features/guide-excursions-monitoring/README.md`
- source taxonomy + case analysis: `docs/backlog/features/guide-excursions-monitoring/casebook.md`
- facts-first architecture: `docs/backlog/features/guide-excursions-monitoring/architecture.md`
- LLM-first pack: `docs/backlog/features/guide-excursions-monitoring/llm-first.md`
- frozen eval set: `docs/backlog/features/guide-excursions-monitoring/eval-pack.md`
- digest/media delivery spec: `docs/backlog/features/guide-excursions-monitoring/digest-spec.md`
- planned live E2E scenarios: `docs/backlog/features/guide-excursions-monitoring/e2e.md`
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
- screening и `Tier 1` extraction через Gemma по отдельному ключу `GOOGLE_API_KEY2`, с server-side enrichment через existing LLM Gateway;
- facts-first накопление для `guide / template / occurrence` в отдельных guide-таблицах;
- короткая текстовая выдача для digest, а не long-form pages;
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

### Versioned component pack for MVP

- `Trail Scout v1` — Telegram intake, OCR, prefilter, `Tier 1` extraction
- `Route Weaver v1` — facts-first merge, status binding, enrichment for `guide/template/occurrence`
- `Lollipop Trails v1` — batched short digest text generation from facts
- `Trail Digest v1` — ranking, preview, publish
- `Media Bridge v1` — temporary media reuse via `forward -> file_id` bridge
- `Guide Atlas v1` — admin/read-model surfaces and stats
- `Guide E2E Pack v1` — manual/live E2E flow and Gherkin contract

Каноника по naming/versioning: `docs/backlog/features/guide-excursions-monitoring/digest-spec.md`

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
| `excursions_profitour` | `excursion_operator` | `medium` | agency/operator, много school-group и `on_request` программ |
| `ruin_keepers` | `organization_with_tours` | `medium` | heritage-organization, tours as one product line |
| `vkaliningrade` | `aggregator` | `medium` | fallback-aggregator, meeting-point updates |

### Обязательные source flags

- `allow_out_of_region=false` по умолчанию
- `allow_out_of_region=true` для `twometerguide`
- `operator=true` для `excursions_profitour`
- `aggregator=true` для `vkaliningrade`
- `organization=true` для `ruin_keepers`
- `caption_heavy=true` для `katimartihobby`
- `promo_noise=true` для `valeravezet`
- `collaboration_heavy=true` для `tanja_from_koenigsberg`, `amber_fringilla`
- `on_request_heavy=true` для `excursions_profitour`

## 4. Стратегия хранения для MVP

### Решение

Использовать **основную SQLite базу**, но **отдельные guide-таблицы**, а не текущую `event`.

Это даёт три преимущества:

- нулевая утечка в `/daily`, month pages, weekend pages;
- отдельная domain-model для guide/template/occurrence facts и status deltas;
- простая интеграция с `/general_stats`, scheduler и админ-командами без второй БД.

### Почему не `event` table

Для MVP это создаёт больше проблем, чем пользы:

- текущая `event`-модель заточена под публичные event pages;
- occurrence/status-update логика экскурсий отличается от обычных культурных событий;
- придётся сразу решать слишком много public-surface guardrails.

### Минимальные таблицы MVP

MVP не должен быть “только про digest row”.

Правильный компромисс:

- surface MVP остаётся кратким и occurrence-first;
- underlying storage уже копит факты по `GuideProfile`, `GuideExcursionTemplate` и `GuideExcursionOccurrence`;
- generated text остаётся минимальным и может пересобираться из фактов.

#### `guide_source`

- `id`
- `platform='telegram'`
- `username`
- `title`
- `primary_profile_id`
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
- `media_refs_json`
- `post_kind`
- `prefilter_passed`
- `llm_status`
- `last_scanned_at`

Назначение:

- idempotency;
- admin-diagnostics;
- база для future metrics windows.

#### `guide_profile`

- `id`
- `profile_kind`
  - `person`
  - `project`
  - `organization`
  - `operator`
- `display_name`
- `marketing_name`
- `source_links_json`
- `base_region`
- `audience_strengths_json`
- `summary_short`
- `facts_rollup_json`
- `first_seen_at`
- `last_seen_at`

#### `guide_template`

- `id`
- `canonical_title`
- `title_normalized`
- `aliases_json`
- `base_city`
- `availability_mode`
- `audience_fit_json`
- `participant_profiles_json`
- `summary_short`
- `facts_rollup_json`
- `first_seen_at`
- `last_seen_at`

#### `guide_occurrence`

- `id`
- `template_id`
- `canonical_title`
- `title_normalized`
- `participant_profiles_json`
- `guide_names_json`
- `organizer_names_json`
- `digest_eligible`
- `rescheduled_from_id`
- `date`
- `time`
- `duration_text`
- `city`
- `meeting_point`
- `audience_fit_json`
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
- `digest_blurb`
- `raw_facts_json`
- `first_seen_at`
- `last_seen_at`
- `published_new_digest_at`
- `published_last_call_digest_at`

Примечания:

- `booking_url` в MVP остаётся raw source of truth;
- tracking columns и redirect slug осознанно не materialize’ятся в MVP: future click analytics строится поверх `booking_url + occurrence_id` в отдельной итерации;
- уже прошедшие occurrences в MVP не держим: row либо не создаётся, либо удаляется cleanup path после перехода в прошлое.

#### `guide_fact_claim`

- `id`
- `entity_kind`
  - `guide`
  - `template`
  - `occurrence`
- `entity_id`
- `fact_key`
- `fact_value_json`
- `claim_role`
  - `anchor`
  - `support`
  - `status_delta`
  - `template_hint`
  - `guide_profile_hint`
  - `audience_fit_hint`
- `confidence`
- `source_id`
- `message_id`
- `observed_at`
- `last_confirmed_at`

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
- `media_refs_json`
- `snapshot_at`

#### `guide_digest_issue`

- `id`
- `digest_family`
- `scope`
- `channel_username`
- `published_at`
- `items_count`
- `media_manifest_json`
- `payload_hash`
- `status`

## 5. Notebook / pipeline MVP

### Новый guide-specific notebook

Мониторинг источников должен выполняться в **отдельном Kaggle notebook**.

Но это должен быть не greenfield и не “совсем другой pipeline”, а **focused fork/reuse** текущего `TelegramMonitor`.

Рабочее название:

- `GuideExcursionsMonitor`

### Почему отдельный notebook

- current TG monitoring слишком дорогой для этой задачи;
- guide sources требуют другой prefilter и другой JSON contract;
- удобнее отдельно подключить `GOOGLE_API_KEY2`.

### Kaggle-first execution boundary

Граница MVP должна быть жёсткой:

- Kaggle notebook делает raw Telegram scanning;
- серверный бот не делает Telethon fetch исходных каналов;
- серверный бот только импортирует notebook output, мерджит факты и публикует digest.

Это позволяет:

- переиспользовать уже отработанный Kaggle operational path;
- не плодить вторую Telethon session в runtime бота;
- держать тяжёлые OCR/LLM/fetch операции там же, где уже живёт текущий Telegram monitoring.

### Что именно должно переиспользоваться из текущего Telegram Monitoring

Канонический baseline:

- notebook baseline: `kaggle/TelegramMonitor/telegram_monitor.ipynb`
- Kaggle push/poll/download lifecycle: `source_parsing/telegram/service.py`
- import/reporting patterns: `source_parsing/telegram/handlers.py`
- encrypted split-secrets flow: `source_parsing/telegram/split_secrets.py`

#### Reuse as-is

- Kaggle launcher and output download;
- encrypted runtime secret delivery;
- Telethon session bootstrap;
- grouped album collapse;
- source metadata fetch;
- OCR/media fingerprinting;
- recent-results artifact discipline.

#### Reuse with guide-specific fork

- candidate JSON contract;
- message prefilter heuristics;
- LLM prompt/output contract;
- source-level metrics payload;
- linked-post and album heuristics where they help guide matching.

#### New guide-specific logic

- source taxonomy and source flags;
- `announce_single / announce_multi / status_update / reportage / template_signal / on_demand_offer`;
- extraction of `guide / template / occurrence` facts in `Tier 1`;
- `audience_fit`, `availability_mode`, `digest_eligible`;
- occurrence/template clustering and `aggregator_fallback` semantics.

### Runtime stages

1. `Telethon fetch`
   - последние `N` сообщений на источник;
   - grouped albums collapse;
   - views / forwards / reactions / links / handles;
   - media references for future digest reuse.
2. `OCR`
   - только для candidate posts с изображениями.
3. `Deterministic prefilter`
   - regex + heuristic scoring;
   - coarse post-kind guess.
4. `Gemma screening + Tier 1 extraction`
   - только по prefiltered items;
   - output: `ignore | announce | status_update | template_only` plus compact JSON payload.
5. `Server import`
   - transport validation + partial-run handling;
   - status bind + profile/template/occurrence match + merge;
   - server-side semantic enrichment;
   - digest candidate extraction;
   - media refs persistence for `Media Bridge v1`;
   - stats + admin report.

### Responsibilities split

#### Kaggle notebook responsibilities

- fetch recent source messages;
- collapse grouped albums and collect media refs;
- collect source meta and post metrics;
- run OCR;
- run deterministic prefilter;
- run Gemma screening + `Tier 1` extraction with `GOOGLE_API_KEY2`;
- export compact result artifact for server import.

#### Server responsibilities

- ingest notebook output;
- run `Route Weaver v1` status bind, merge, enrichment and materialization;
- update `Guide Atlas v1` read models and `/general_stats`;
- prepare `Trail Digest v1` preview/publish payloads with batched `Lollipop Trails v1` copy;
- resolve media through `Media Bridge v1` using bot-side `forward -> file_id`.

### Transport contract for notebook output

`GuideExcursionsMonitor` не выдумывает новый канал доставки.

Каноника MVP:

- reuse текущего Kaggle push/poll/download lifecycle из Telegram Monitoring;
- артефакт называется `guide_excursions_results.json`;
- top-level payload обязан содержать:
  - `run_id`
  - `scan_mode`
  - `started_at`
  - `finished_at`
  - `partial`
  - `sources`;
- per-source payload обязан содержать:
  - `source_status`
  - `posts_scanned`
  - `candidates`;
- server import обязан импортировать healthy sources даже при `partial=true`.

### Past occurrence policy

Пока что уже прошедшие экскурсии не храним как `guide_occurrence`.

Правило MVP:

- candidate с `date_local < today_local` не materialize’ится в `guide_occurrence`;
- report/template posts могут усиливать `guide_template` и `guide_profile`, но не создают past occurrence row;
- active occurrence удаляется cleanup path после перехода в прошлое, чтобы инвентарь оставался operational, а не историческим.

### Временная схема медиа в MVP

До постоянной infra на Yandex Cloud digest должен использовать `Media Bridge v1`, а не отдельный storage upload.

Что делаем:

- сохраняем media refs из исходных Telegram-постов;
- bot temporary-forward’ит выбранный source media в helper/admin chat и извлекает `file_id`;
- сохраняем `bot_file_id` для server-side publish;
- публикуем media group в `@keniggpt`;
- Kaggle staging держим только как fallback;
- если media-path ломается, digest уходит как text-only.

Каноника: `docs/backlog/features/guide-excursions-monitoring/digest-spec.md`

### Текстовая стратегия MVP

MVP не делает long-form guide pages и template pages, но и не ограничивается хранением “только title и одна строка”.

Что генерируется в MVP:

- `occurrence.canonical_title`
- `occurrence.summary_one_liner`
- optional `occurrence.digest_blurb`
- optional `occurrence.audience_line`

Что уже копится в MVP как факты:

- occurrence logistics and status;
- template hints и recurring anchors;
- `audience_fit` и `availability_mode`;
- guide profile claims, если они явно grounded в постах/OCR.

Текстовая политика MVP:

- card shell строится deterministic;
- смысловые короткие строки пишет `Lollipop Trails v1` batch-режимом из fact pack;
- fully deterministic digest-copy не является каноникой, но deterministic fallback допустим только как аварийный режим.

Иначе говоря, digest family оперируют базовыми информационными единицами, но underlying storage уже готов к следующей итерации.

## 6. Scan modes

MVP должен быть **двухрежимным**.

### `full_scan`

Назначение:

- находить новые occurrences;
- обновлять полноту карточек;
- находить template hints;
- находить `on_demand_offer` без автоматического создания digest item;
- собирать guide/profile signals;
- собирать cover-image и краткие тексты.

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
- только `digest_eligible=yes`;
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
- short `для кого`, если сигнал короткий и уверенный
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
- только `digest_eligible=yes`;
- максимум `6` карточек;
- приоритет ближайшим датам и scarcity signals.

Формат компактнее, чем у `new_occurrences`.

### Что не публикуем в MVP автоматически

- template-only `on_request` / `private-group-only` предложения;
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
- `Send test report`
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

Рекомендуемое имя артефакта:

- `guide_excursions_results.json`

### Top-level

```json
{
  "run_id": "guide-scan-2026-03-14T09-05-00Z",
  "scan_mode": "light",
  "started_at": "2026-03-14T09:05:00+00:00",
  "finished_at": "2026-03-14T09:12:10+00:00",
  "partial": false,
  "sources": [...]
}
```

### Per source

```json
{
  "username": "tanja_from_koenigsberg",
  "source_kind": "guide_personal",
  "source_status": "ok",
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
  "llm_decision": "announce",
  "announce_tier1": [
    {
      "title_raw": "Город К. Женщины, которые вдохновляют",
      "date_local": "2026-03-07",
      "time_local": "11:00",
      "guide_names": ["Татьяна Удовенко", "Юлия Гришанова"],
      "meeting_point": "у Матери России",
      "price_text": "2000 руб",
      "booking_target": "https://t.me/Yulia_Grishanova",
      "status_hint": "planned",
      "raw_text_snippet": "7 марта — Тематическая пешеходная прогулка..."
    }
  ],
  "status_claims": [],
  "template_candidates": []
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
