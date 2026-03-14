# Guide Excursions Architecture

> **Status:** Proposed canonical architecture  
> **Intent:** зафиксировать facts-first модель для сущностей `guide / excursion template / excursion occurrence`, чтобы MVP и следующие итерации не расходились по данным и текстовому контуру.

Связанные документы:

- feature overview: `docs/backlog/features/guide-excursions-monitoring/README.md`
- casebook: `docs/backlog/features/guide-excursions-monitoring/casebook.md`
- MVP: `docs/backlog/features/guide-excursions-monitoring/mvp.md`
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
- позиционирование;
- специализацию и типичные темы;
- geography / base region;
- стиль подачи, сильные стороны, narrative claims;
- контакты и ссылки;
- устойчивые коллаборации.

### `ExcursionTemplate`

Что это за **типовая экскурсия**.

Это не “один конкретный пост”, а устойчивый маршрут / продукт / recurring format, который может много раз повторяться и меняться в деталях.

Что накапливаем:

- каноническое название маршрута;
- альтернативные названия;
- маршрут / ключевые точки / темы;
- чем экскурсия отличается от других;
- эмоциональные и narrative характеристики;
- recurring evidence из прошлых анонсов, отчётов, отзывов;
- typical media / visual anchors;
- связи с гидами и соорганизаторами.

### `ExcursionOccurrence`

Когда именно проходит конкретный выход.

Что накапливаем:

- дата, время, длительность;
- место встречи;
- стоимость / free / donation;
- статус мест:
  - `planned`
  - `few_seats`
  - `waitlist`
  - `sold_out`
  - `cancelled`
  - `done`
- ссылка на запись;
- актуальный source of truth;
- engagement snapshot и status deltas.

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

Это аналог “monitoring notebook”, но со своей продуктовой идентичностью.

### `Route Weaver`

Рекомендуемое имя для экскурсионного аналога `Smart Update`.

Почему именно он:

- его задача не просто распарсить пост;
- он “сшивает” посты, факты и сущности в единый knowledge graph трека.

`Route Weaver` отвечает за:

- `scope_map`
  - что этот пост обновляет: `guide`, `template`, `occurrence`, или несколько сущностей сразу;
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

Технически это и есть “Smart Update для экскурсий”, только multi-entity, а не event-only.

### `Lollipop Trails`

Рекомендуемое имя для экскурсионного text family поверх `lollipop` или его форка.

Назначение:

- генерировать тексты **из fact pack**, а не из сырого поста;
- отдельно обслуживать разные surfaces;
- не смешивать rendering задачи с entity merge.

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
-> facts.extract
-> facts.dedup
-> facts.merge
-> facts.prioritize
-> writer.digest_short
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
- guide profile claims, если они явно grounded.

То есть digest действительно оперирует базовыми информационными единицами, но underlying store при этом уже готов к росту.

## 10. Практический вывод для MVP

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
