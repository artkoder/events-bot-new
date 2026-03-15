# Guide Excursions Architecture

> **Status:** Proposed canonical architecture  
> **Intent:** зафиксировать facts-first модель для сущностей `guide / excursion template / excursion occurrence`, чтобы MVP и следующие итерации не расходились по данным и текстовому контуру.

Связанные документы:

- feature overview: `docs/backlog/features/guide-excursions-monitoring/README.md`
- casebook: `docs/backlog/features/guide-excursions-monitoring/casebook.md`
- MVP: `docs/backlog/features/guide-excursions-monitoring/mvp.md`
- LLM-first pack: `docs/backlog/features/guide-excursions-monitoring/llm-first.md`
- eval pack: `docs/backlog/features/guide-excursions-monitoring/eval-pack.md`
- live `Opus` audit artifact: `artifacts/codex/reports/guide-excursions-monitoring-opus-audit-2026-03-14.md`
- каноника Smart Update: `docs/features/smart-event-update/README.md`
- lollipop funnel: `docs/llm/smart-update-lollipop-funnel.md`

## 1. Главный тезис

Экскурсионный трек нужно строить **facts-first с первого дня**, даже если на старте публичная выдача ограничена коротким digest.

Это значит:

- храним не только итоговый `title + one-liner`, а **извлечённые и провенансированные факты**;
- тексты считаем **производными артефактами**, которые можно пересобирать;
- один и тот же пост может одновременно обновлять:
  - `GuideProfile`
  - `ExcursionTemplate`
  - `ExcursionOccurrence`

Именно это отличает экскурсионный контур от упрощённой модели “нашли анонс и сразу опубликовали”.

## 2. Три канонические сущности

### `GuideProfile`

Кто проводит экскурсию.

Что накапливаем:

- имя и маркетинговое имя;
- тип профиля:
  - `person`
  - `project`
  - `organization`
  - `operator`
- позиционирование;
- специализацию и типичные темы;
- geography / base region;
- стиль подачи, сильные стороны, narrative claims;
- сильные audience сегменты:
  - школьные группы
  - семьи
  - взрослые туристы
  - местные жители
  - корпоративные группы
- контакты и ссылки;
- устойчивые коллаборации.

### `ExcursionTemplate`

Что это за **типовая экскурсия**.

Это не “один конкретный пост”, а устойчивый маршрут / продукт / recurring format, который может много раз повторяться и меняться в деталях.

Что накапливаем:

- каноническое название маршрута;
- альтернативные названия;
- маршрут / ключевые точки / темы;
- `availability_mode`
  - `scheduled_public`
  - `on_request_private`
  - `mixed`
- чем экскурсия отличается от других;
- `audience_fit`
  - для кого подходит;
  - возраст / класс / тип группы;
  - locals vs tourists;
  - темп и уровень сложности;
  - образовательный / развлекательный / профориентационный акцент;
  - требуется ли предварительный интерес/контекст;
- эмоциональные и narrative характеристики;
- recurring evidence из прошлых анонсов, отчётов, отзывов;
- typical media / visual anchors;
- связи с гидами и соорганизаторами.

### `ExcursionOccurrence`

Когда именно проходит конкретный выход.

Что накапливаем:

- дата, время, длительность;
- `digest_eligible`
  - `yes`
  - `no`
  - template-level default может быть переопределён на occurrence уровне;
- место встречи;
- стоимость / free / donation;
- audience-specific overrides:
  - возрастные рамки
  - group size
  - адаптация под школьный класс / корпоратив / семью
- статус мест:
  - `planned`
  - `few_seats`
  - `waitlist`
  - `sold_out`
  - `cancelled`
  - `done`
- `rescheduled_from_id` для связи нового выхода с исходным переносом;
- ссылка на запись;
- raw `booking_target_url` как source of truth;
- stable occurrence identity для будущего redirect/click-tracking слоя;
- active tracking columns осознанно откладываются до отдельной реализации tracking feature;
- актуальный source of truth;
- engagement snapshot и status deltas.

Важный MVP guardrail:

- пока что materialize’им только future/active occurrences;
- уже прошедшие выходы не храним как отдельные `ExcursionOccurrence`, хотя их посты могут усиливать `ExcursionTemplate` и `GuideProfile`.

### Matching and enum contracts

Чтобы matching не поплыл между MVP и следующими итерациями, нужны жёсткие договорённости уже сейчас.

`title_normalized` считается deterministic-полем и строится так:

- lower-case;
- stripping emoji and decorative prefixes;
- stripping кавычек и пунктуации;
- collapse whitespace;
- унификация длинного/короткого тире;
- без выдумывания новых слов.

Для fuzzy matching используем token-level similarity поверх `title_normalized`, а не сырое сравнение строк.

Начальный канонический enum для `audience_fit_tags`:

- `children`
- `school_groups`
- `families`
- `adults`
- `seniors`
- `tourists`
- `locals`
- `corporate_groups`
- `mixed_age`

Открывать enum можно позже, но в MVP drift по названиям недопустим.

## 3. Связи между сущностями

Базовая связь не должна моделироваться как жёсткое `GuideProfile 1:N Template 1:N Occurrence`.

Корректнее:

- `GuideProfile M:N ExcursionTemplate`
- `ExcursionTemplate 1:N ExcursionOccurrence`
- `GuideProfile M:N ExcursionOccurrence`

Почему так:

- бывают коллаборативные экскурсии;
- у одного маршрута может быть несколько участников с разными ролями;
- у организации или проекта может быть несколько лиц;
- конкретный выход может отличаться от обычного состава шаблона.

Поэтому у связей должны быть роли:

- `lead_guide`
- `co_guide`
- `organizer`
- `host`
- `aggregator_reference`

Для MVP допустим компромисс: роли пока можно хранить в `json` на materialized rows, а отдельные join-таблицы нормализовать позже.

Известное ограничение MVP:

- cross-guide dedup для коллабораций между разными origin channels остаётся best-effort;
- если один и тот же физический выход опубликован в двух разных каналах с сильно разными title/text, оператор может увидеть два occurrence-кандидата до ручной коррекции.

## 4. Вспомогательные сущности

### `GuideSource`

Мониторимый канал/источник.

Нужен для:

- Telethon fetch;
- source priority;
- trust;
- source-kind taxonomy;
- health/statistics.

### `GuideSourcePost`

Снимок конкретного поста с метриками и служебной классификацией.

Нужен для:

- idempotency;
- OCR linkage;
- повторных сканов;
- аудита решений;
- metric windows.

### `GuideFactClaim`

Ключевая сущность facts-first слоя.

Каждый claim должен иметь:

- `entity_kind`
  - `guide`
  - `template`
  - `occurrence`
- `fact_key`
- `fact_value`
- `confidence`
- `source_post`
- `provenance`
- `observed_at`
- `last_confirmed_at`
- `claim_role`
  - `anchor`
  - `support`
  - `status_delta`
  - `template_hint`
  - `guide_profile_hint`

Особо важные fact families:

- `audience_fit.*`
- `availability_mode`
- `group_constraints.*`
- `pace_and_access.*`
- `educational_focus.*`
- `vibe_and_delivery.*`

Именно `GuideFactClaim` позволяет:

- потом строить карточки гида;
- накапливать данные о типовой экскурсии;
- переиспользовать факты в digest, статических страницах и admin reports;
- не зависеть от одноразового текста, сгенерированного на старте.

### `GuideTextAsset`

Необязательный для MVP, но полезный как целевая модель.

Назначение:

- хранить производные тексты по surface/type:
  - `digest_one_liner`
  - `digest_blurb`
  - `template_short`
  - `guide_summary_short`
  - later: `template_long_page`, `guide_page`

Если отдельной таблицы пока нет, эти поля можно держать прямо в materialized rows.

## 5. Контур обработки

Канонический pipeline:

```text
GuideSource
-> source fetch
-> OCR / media grouping
-> candidate prefilter
-> scope mapping
-> fact extraction
-> entity match
-> fact merge
-> snapshot materialization
-> text generation on demand
-> digest / admin surfaces
```

Ключевой момент: текст не должен быть центральным объектом пайплайна.  
Центральный объект — это **matched and merged fact set**.

Отдельный принцип: `on-demand` предложение без публичного набора не обязано порождать `ExcursionOccurrence`.

В таких случаях pipeline может закончиться на:

- `GuideProfile` update;
- `ExcursionTemplate` update;
- `digest_eligible = no`.

### Kaggle-first execution boundary

Для этой фичи raw Telegram scanning должен жить не в runtime бота, а в отдельном Kaggle notebook.

Каноническая граница такая:

- `Trail Scout v1` работает в Kaggle и владеет единственной Telethon user-session;
- серверный бот не делает source fetch через Telethon и не сканирует исходные каналы сам;
- серверный бот получает только notebook output, импортирует его через `Route Weaver`, строит admin/read-model и публикует digest.

Это нужно зафиксировать явно, потому что у проекта уже есть рабочий operational контур для Telegram Monitoring, и экскурсионный трек должен его **переиспользовать**, а не изобретать второй ingestion stack.

Что относится к Kaggle стороне:

- Telethon auth/session bootstrap;
- fetch сообщений и grouped albums collapse;
- source metadata fetch;
- OCR и media fingerprinting;
- deterministic prefilter;
- Gemma confirmation по `GOOGLE_API_KEY2`;
- export компактного `guide_excursions_results.json`.

Что относится к серверной стороне:

- import notebook output;
- transport validation and partial-run handling;
- `guide/template/occurrence` matching и fact merge;
- `general_stats` и admin surfaces;
- digest preview/publish;
- `Media Bridge` через bot-side `forward -> file_id`.

Принципиальное ограничение:

- Telethon нужен только в Kaggle notebook;
- bot runtime должен работать без второй Telethon session.

### Kaggle -> server transport contract

`Trail Scout v1` не изобретает новый способ доставки результатов.

Каноника:

- reuse того же push/poll/download lifecycle, что и у текущего `TelegramMonitor`;
- артефакт называется `guide_excursions_results.json`;
- top-level contract обязан содержать:
  - `run_id`
  - `scan_mode`
  - `started_at`
  - `finished_at`
  - `partial`
  - `sources`;
- каждая source entry обязана содержать собственный `source_status`, чтобы сервер мог импортировать partial run без потери здоровых источников.

## 6. Именованные компоненты

### `Trail Scout`

Рабочее имя для ingestion/monitoring слоя.

Отвечает за:

- Telethon fetch;
- grouped albums;
- OCR;
- regex/heuristic prefilter;
- coarse post classification;
- candidate export.

`Trail Scout v1` должен быть реализован как **отдельный Kaggle notebook**, но не как greenfield rewrite.

Правильная форма: focused fork/reuse текущего `TelegramMonitor`:

- notebook baseline: `kaggle/TelegramMonitor/telegram_monitor.ipynb`
- launcher / polling / output download: `source_parsing/telegram/service.py`
- import/reporting patterns: `source_parsing/telegram/handlers.py`
- split secrets / Kaggle runtime config: `source_parsing/telegram/split_secrets.py`

То есть у `Trail Scout v1` своя доменная логика, но operational skeleton должен максимально наследовать уже работающий Telegram monitoring stack.

#### Reuse policy for `Trail Scout v1`

Переиспользовать как есть:

- Kaggle push/poll/download lifecycle;
- encrypted split-secrets flow;
- Telethon session bootstrap;
- grouped album reconstruction;
- source metadata extraction;
- OCR/media fingerprinting;
- recent-results artifact pattern.

Переиспользовать через guide-specific fork:

- message candidate schema;
- prefilter heuristics;
- LLM request/output contract;
- metrics payload for median-based ranking;
- linked-post and album handling rules, где это полезно для guide domain.

Реализовать как новую guide-specific логику:

- source taxonomy `guide_personal / guide_project / excursion_operator / organization_with_tours / aggregator`;
- classification в `announce_single / announce_multi / status_update / reportage / template_signal / on_demand_offer`;
- extraction для `GuideProfile / ExcursionTemplate / ExcursionOccurrence`;
- `audience_fit`, `availability_mode`, `digest_eligible`;
- occurrence/template clustering и `aggregator_fallback` semantics.

### `Route Weaver`

Рекомендуемое имя для экскурсионного аналога `Smart Update`.

Почему именно он:

- его задача не просто распарсить пост;
- он “сшивает” посты, факты и сущности в единый knowledge graph трека.

`Route Weaver` отвечает за:

- `scope_map`
  - что этот пост обновляет: `guide`, `template`, `occurrence`, или несколько сущностей сразу;
- `status_bind`
  - привязать raw status claim из Kaggle к active occurrence;
- `entity_match`
  - найти существующего гида / типовую экскурсию / конкретный выход;
- `fact_merge`
  - поднять, обновить или конфликтно отложить факты;
- `status_reconcile`
  - few seats / sold out / перенос / точка встречи;
- `materialize`
  - обновить канонические snapshot rows;
- `digest_delta_emit`
  - отдать новые/изменившиеся элементы в digest-контур.

Критичный частный случай для `scope_map`:

- `scheduled public occurrence`
  - создаём/патчим occurrence;
- `on-demand template offer`
  - occurrence не создаём автоматически;
  - усиливаем template/profile facts;
  - помечаем template default как `digest_eligible = no`, пока нет публичного набора.

Технически это и есть “Smart Update для экскурсий”, только multi-entity, а не event-only.

### `Lollipop Trails`

Рекомендуемое имя для экскурсионного text family поверх `lollipop` или его форка.

Назначение:

- генерировать тексты **из fact pack**, а не из сырого поста;
- отдельно обслуживать разные surfaces;
- не смешивать rendering задачи с entity merge.

Важно:

- `Lollipop Trails` не становится fully deterministic шаблонизатором;
- deterministic shell отвечает за layout и omission policy;
- смысловые короткие тексты пишет batch-LLM слой поверх fact pack.

На старте достаточно небольшого family:

- `digest_one_liner`
- `digest_blurb`
- later:
  - `template_short`
  - `template_long`
  - `guide_summary`
  - `guide_profile_page`

### `Trail Digest`

Digest builder / ranking / publishing.

Отвечает за:

- ranking внутри family;
- grouping;
- image dedup;
- output formatting;
- posting to target channel.

### `Media Bridge`

Временный media-delivery слой до постоянной infra.

Отвечает за:

- выбор исходных фото/видео из source posts;
- дедуп медиа внутри digest;
- temporary `forward -> file_id` bridge через helper/admin chat;
- fallback staging path, если direct forward не сработал;
- relay cleanup sweep and bridge logging;
- отправку media group в целевой канал без отдельного permanent hosting.

### `Guide Atlas`

Имя для internal read-model и admin surfaces.

Туда относятся:

- `/guide_recent`
- `/guide_sources`
- `/guide_excursions`
- `/guide_digest`
- блок в `/general_stats`

Это не публичный сайт, а операционная knowledge surface.

## 7. Как должен выглядеть экскурсионный Smart Update

`Route Weaver` должен наследовать принципы обычного `Smart Update`:

- LLM-first для смысла;
- deterministic layer только для нормализации, guardrails и дешёвых shortlist;
- facts before text;
- provenance и source priority;
- generated text как downstream artifact.

Но у него есть дополнительная сложность: **один пост может обновлять несколько уровней модели**.

Пример:

- анонс нового выхода даёт `occurrence` facts;
- тот же анонс даёт `template` facts про маршрут;
- размышления гида о том, как он ведёт экскурсию, дают `guide` facts;
- отчёт о прошедшей экскурсии не создаёт новый digest item, но усиливает `template`.

Поэтому `Route Weaver` нельзя делать как простой `event merge` clone.

## 8. Lollipop-контур для экскурсий

Для экскурсионного трека целесообразно делать не “прямой copy-paste” текущего event lollipop, а **forked family поверх тех же принципов**.

Минимальный рекомендуемый funnel:

```text
scope.map
-> facts.extract_tier1
-> status.bind
-> facts.enrich
-> facts.dedup
-> facts.merge
-> facts.prioritize
-> writer.digest_batch
```

Следующие stages можно добавлять позже:

```text
-> writer.template_short
-> writer.template_long
-> writer.guide_summary
```

То есть в экскурсиях `lollipop` нужен не только “для красивого текста”, а как дисциплина разделения:

- что является фактом;
- что является uncertain hint;
- что вообще должно попасть в конкретную surface.

## 9. Что считается текстом в MVP

MVP не должен ограничиваться моделью “храним только название экскурсии и одно предложение”.

Правильный компромисс для MVP:

- **копим facts уже сейчас**;
- **генерируем мало текстов сейчас**.

Минимальный набор текстов MVP:

- `occurrence.canonical_title`
- `occurrence.summary_one_liner`
- optional `occurrence.digest_blurb`

Минимальный набор фактов MVP:

- occurrence logistics and status;
- template hints and recurring anchors;
- `audience_fit` и `availability_mode`;
- guide profile claims, если они явно grounded.

То есть digest действительно оперирует базовыми информационными единицами, но underlying store при этом уже готов к росту.

## 10. `Audience Fit` как golden fund

Для экскурсионного трека `для кого это` должно стать одним из центральных фактов, а не второстепенной редакторской пометкой.

Это нужно сразу для трёх вещей:

- будущих выборок и рекомендаций;
- понятных страниц гида / агентства / типовой экскурсии;
- лучшего digest ranking и summarization.

Практически это означает, что система должна уметь копить такие grounded признаки:

- дети / взрослые / mixed;
- школьные классы / семьи / корпоративные группы / туристы / местные;
- `6+`, `7-11 классы`, `организованные группы 20-30 человек`;
- спокойная прогулка vs активный интерактив;
- познавательная / гастрономическая / природная / профориентационная / иммерсивная подача;
- нужен ли интерес к истории / искусству / природе заранее;
- можно ли адаптировать под возраст и уровень группы.

## 11. Практический вывод для MVP

Если делать MVP “только как digest finder”, придётся потом почти заново проектировать:

- карточки гидов;
- типовые экскурсии;
- related materials;
- template pages;
- richer digest families.

Поэтому правильнее с первого дня заложить:

- `GuideProfile`
- `ExcursionTemplate`
- `ExcursionOccurrence`
- `GuideFactClaim`

Но при этом ограничить surface MVP:

- публикация только digest’ов;
- без public pages;
- без длинных profile/template текстов;
- с короткими admin read-model surfaces.

Это даёт сразу и текущую ценность, и нормальный upgrade path.
