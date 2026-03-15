# Smart Update Lollipop Funnel

Статус: `dry-run`

`lollipop` — исследовательская shadow-ветка для `Smart Update`. Она не меняет baseline runtime и нужна для сборки более качественного fact-first pipeline.

## Каноническая схема

```text
source.scope
-> facts.extract
-> facts.dedup
-> facts.merge
-> facts.prioritize
-> editorial.layout
-> writer_pack.compose
-> writer_pack.select
-> writer.final_4o
```

## Принципы

- `Smart Update` baseline остаётся эталоном.
- Все grounded facts должны доживать до `facts.merge.emit` или явно классифицироваться как background/uncertain.
- Стадии маленькие и одноцелевые.
- Для `Gemma` используются компактные self-contained prompt'ы с явной JSON-схемой и короткими примерами.
- Для full-family lab reruns с одним Gemma key нужно соблюдать TPM-aware execution discipline:
  - upstream families должны поддерживать `EVENT_IDS` subset reruns и safe run-label/input overrides;
  - shared Gemma caller должен иметь proactive pacing (`LOLLIPOP_GEMMA_CALL_GAP_S`), а не только reactive `429/tpm` retries.
  - Gemma-heavy canary reruns допустимо переносить в Kaggle через `kaggle/execute_lollipop_canary.py`, а не продолжать локальный монолитный batch, если local provider path режется по TPM или location policy.
- Финальный `4o` допускается один раз и только в самом конце.

## Активные families

### `source.scope`

Назначение: отделить нужный event scope от шума, multi-event и mixed-phase contamination.

Базовые стадии:

- `scope.extract`
- `scope.select`

Отдельный риск-класс:

- mixed-phase series post
  Документ: [smart-update-lollipop-casebook.md](/workspaces/events-bot-new/docs/llm/smart-update-lollipop-casebook.md)
  Prompt pack: [smart-update-lollipop-mixed-phase-prompts.md](/workspaces/events-bot-new/docs/llm/smart-update-lollipop-mixed-phase-prompts.md)

### `facts.extract`

Текущий рабочий bank:

- `baseline_fact_extractor`
- `facts.extract.subject`
- `facts.extract.card`
- `facts.extract.agenda`
- `facts.extract.support`
- `facts.extract.performer`
- `facts.extract.participation`
- `facts.extract.stage.tightened`
- `facts.extract.theme.challenger`

Текущая каноника:

- после консультационного owner-audit `2026-03-11` следующий richness-owner подтверждён как `facts.extract`, а не `writer.final_4o` / `editorial.layout`
- для `выставка` extract prompt теперь жёстче сохраняет curatorial/history/collection-detail facts как first-class evidence:
  - `facts.extract.card` может поднимать название экспозиции, размер коллекции, эпоху и институциональную связку
  - `facts.extract.profiles` может сохранять maker/designer/item-level detail даже без named people
  - `facts.extract.theme` / `facts.extract.concept` должны предпочитать исторический контекст и кураторскую рамку, а не только общий `выставка посвящена ...`
  - `facts.extract.performer` не должен вытаскивать bare subject-name из названия выставки; performer stage для выставок остаётся пустым без явной role/status/credibility evidence
- `facts.extract.support` больше не должен выводить широкую аудиторию / возраст / accessibility из friendly title или marketing tone; такие visitor facts допустимы только при явном source evidence

### `facts.dedup`

Назначение: различать `covered / reframe / enrichment` без потери meaningful facts.

### `facts.merge`

Назначение: собрать canonical fact pack:

- `event_core`
- `program_list`
- `people_and_roles`
- `forward_looking`
- `logistics_infoblock`
- `support_context`
- `uncertain`
- `provenance`

Текущая каноника:

- `facts.merge iter5` может гидрировать старый `bucket.v2` trace только если состав `record_id` совпадает с текущим `merge_records`
- если upstream `facts.extract` / `facts.dedup` дал новый record-set, `bucket.v2` должен пересчитываться заново на актуальном payload, а не падать на stale hydrated decisions

### `facts.prioritize`

Назначение: расставить salience для последующих editorial steps.

Текущая каноника:

- full `12`-event family-lab `iter3` уже прогнан на `facts.merge iter5`
- weight stage по-прежнему сохраняет полный grounded pack, но теперь может добавлять узкий deterministic rescue для exhibition/history context из `raw_facts`, если upstream evidence уже существует
- `lead` cleaner теперь знает про title opacity:
  - для bare/opaque `presentation` lead должен вытаскивать event-action fact вроде `на презентации расскажут ...`, а не открываться только описанием проекта;
  - для bare/opaque `кинопоказ` secondary lead fallback теперь может уходить из `people_and_roles` в более событийный / film-defining fact из `support_context`, если чистого screening anchor upstream не хватает
- сам `lead` prompt теперь тоже жёстче фиксирует этот contract:
  - в input/prompt явно передаются `title_is_bare` и `title_needs_format_anchor`;
  - prompt содержит positive/negative examples для `screening`, `presentation`, `lecture`;
  - biography/cast/project-definition openings считаются wrong lead, если есть более событийный event-facing fact
- после weighting применяется deterministic `narrative_policy = include|suppress`
- `suppress` используется для:
  - cross-promo schedules и других `other events` spillovers
  - low-specificity support fillers, когда событие уже закрыто более сильными `high/medium` facts
  - hospitality/service-detail lines (`печенье`, `чай`, подобные visitor-comfort notes), если они не несут narrative value
  - generic audience-pitch lines вроде `мероприятие будет интересно ...`, если pack уже закрыт более содержательными facts
- post-iter4 cleanup добавил ещё один narrow deterministic carry без нового stage split:
  - для `кинопоказ`, где upstream не дал `event_core/forward_looking`, но в `support_context` есть synopsis / adaptation / plot facts, до `editorial.layout` они поднимаются из `low` в `medium`, чтобы screening copy не схлопывался в cast-only reference note
- downstream `editorial.layout` должен потреблять только `include` facts; suppressed items остаются audit-only
- audit layer дополнительно считает `lead_format_anchor_present`, чтобы opaque-title cases можно было мерить не только вручную

### `editorial.layout`

Текущая каноника:

- один `Gemma` stage `editorial.layout.plan.v1`
- deterministic `precompute`
- deterministic `validate`
- full `12`-event family-lab `iter2` уже прогнан на `facts.prioritize iter3`
- post-run `Gemini` verdict: `GO` для перехода к deterministic `writer_pack.compose`
- carry из post-run review уже вшит в prompt contract:
  - `title_is_bare` подаётся прямо в prompt input
  - `all_fact_ids` подаётся прямо в prompt input как явный checklist
- follow-up clarity retune после `iter2 vs baseline` добавил ещё два deterministic carries:
  - `title_needs_format_anchor` считается до `Gemma`
  - `non_logistics_total` и `heading_guardrail_recommended` теперь тоже передаются в prompt, чтобы dense cases не схлопывались в один blob
  - semantic headings теперь можно сохранять не только при `rich`, но и на opaque-title `presentation` / `кинопоказ`, а также вообще при `non_logistics_total >= 4`, если event реально разваливается на смысловые блоки; сами heading labels снова выбирает `Gemma`, а не deterministic cleaner
  - dense cases с `non_logistics_total >= 5` без headings не переписываются детерминированно, но получают явный audit flag `missing_headings_for_dense_case`
- `iter6` carry after the 2026-03-11 rerun:
  - precompute now explicitly carries `body_cluster_count`, `body_block_floor`, and `multi_body_split_recommended`, so rich post-lead material can ask for two narrative sections without deterministic heading labels
  - deterministic cleaner may split one oversized body block at a bucket-cluster boundary as a safety floor, but still leaves second-block heading selection to the model / downstream prose rather than inventing labels in Python
- current `iter2` aggregate:
  - `events_with_flags = 0`
  - `missing_fact_total = 0`
  - `duplicate_fact_total = 0`
  - `auto_fixed_total = 1`

Prompt pack:
- [smart-update-lollipop-editorial-layout-prompts.md](/workspaces/events-bot-new/docs/llm/smart-update-lollipop-editorial-layout-prompts.md)

### `writer_pack.compose`

Текущая каноника:

- deterministic `writer_pack.compose.v1`
- deterministic `writer_pack.select.v1` как identity/no-op
- full `12`-event family-lab `iter2` уже прогнан поверх `editorial.layout iter2`
- post-run `Gemini` verdict: `GO` для перехода к `writer.final_4o`
- current `iter2` aggregate:
  - `events_with_flags = 0`
  - `missing_fact_total = 0`
  - `duplicate_fact_total = 0`
  - `events_with_literal_items = 3`
  - `absorbed_by_list_total = 1`
- literal program items now survive through explicit `literal_items` + `coverage_plan`
- post-baseline retune carry, подтверждённый в `iter2` run:
  - suppressed facts не должны попадать в `sections` или `must_cover_fact_ids`
  - sections с `literal_items` теперь могут нести `literal_list_is_partial = true`
- `iter4` carry after the 2026-03-11 rerun:
  - selected pack now explicitly carries `event_type` into `writer.final_4o`, so final prose can restore format clarity even when lead facts sound like film/project reference notes

Prompt/contract pack:
- [smart-update-lollipop-writer-pack-prompts.md](/workspaces/events-bot-new/docs/llm/smart-update-lollipop-writer-pack-prompts.md)

### `writer.final_4o`

Текущая каноника:

- один final `writer.final_4o.v1` call на `gpt-4o`
- deterministic validator после call
- Python-side apply rule: `title_strategy = keep` всегда принудительно возвращает `original_title`
- full `12`-event family-lab `iter2` уже прогнан поверх `writer_pack.select iter2`
- current `iter2` aggregate:
  - `attempt_total = 13`
  - `retry_event_total = 1`
  - `events_with_errors = 0`
  - `events_with_warnings = 0`
  - `infoblock_leak_total = 0`
  - `literal_missing_total = 0`
  - `literal_mutation_total = 0`
- final post-run `Gemini 3.1 Pro Preview` verdict: `GO`
- post-baseline retune carry, подтверждённый в `iter2` run:
  - partial literal lists должны подаваться как примеры, а не как полный перечень
  - validator блокирует partial list без явного non-exhaustive intro-marker
  - `2498` больше не тащит cross-promo
  - `2657` снова несёт сильный исторический контекст
  - `2734` вводит список через non-exhaustive framing
- текущий safety retune после quality consultation усилил только prompt contract, без нового downstream split:
  - prompt получает explicit structure plan по `sections` и exact headings;
  - для bare/opaque titles есть отдельный `title_needs_format_clarity` signal;
  - prompt держит rough length band, чтобы rich cases не схлопывались в короткую справку;
  - прямо запрещены openings вида `Режиссёр фильма — ...` / `Проект представляет собой ...`, если они не объясняют формат события
- `iter4` rerun on 2026-03-11 подтвердил ещё один рабочий carry:
  - final writer теперь получает `event_type` и вычисляет `lead_needs_format_bridge`, чтобы для screening/presentation cases явно назвать показ/презентацию в первом предложении, если lead facts сами не дают format anchor
  - practical result: headings вернулись во всех `12/12` текстах, а `2673/2659/2747` перестали открываться как чистая справка о проекте/фильме
- `iter6` carry after the 2026-03-11 full rerun:
  - prompt now treats every `section` boundary as a paragraph boundary, so extra body sections from `editorial.layout` survive into public prose even when the later block has `heading = null`
  - practical result: `writer.final_4o iter6` stayed clean (`0 errors`, `0 warnings`, `0 retries`) while average description length recovered from `449.6` to `471.2`

Prompt/contract pack:
- [smart-update-lollipop-writer-final-prompts.md](/workspaces/events-bot-new/docs/llm/smart-update-lollipop-writer-final-prompts.md)

## Mixed-phase class

Для источников вида `past recap + future anchor` используется узкий interceptor:

```text
scope.extract.phase_map.v1
-> scope.select.target_phase.v1
-> facts.extract.phase_scoped.v1
```

Смысл:

- прошедшая фаза уходит в `background_context`;
- будущая фаза становится target;
- прошлые venue/time facts не должны протекать в будущую карточку;
- при слабом future anchor pipeline должен работать по принципу `fail closed`.
