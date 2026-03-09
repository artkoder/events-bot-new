# Smart Update Gemma Event Copy V2.16.2 Lollipop Funnel Design Brief

Дата: 2026-03-09

Основание:

- [v2.15.2 design brief](/workspaces/events-bot-new/docs/reports/smart-update-gemma-event-copy-v2-15-2-design-brief-2026-03-08.md)
- [v2.15.3 design brief](/workspaces/events-bot-new/docs/reports/smart-update-gemma-event-copy-v2-15-3-design-brief-2026-03-08.md)
- [v2.16.1 design brief](/workspaces/events-bot-new/docs/reports/smart-update-gemma-event-copy-v2-16-1-design-brief-2026-03-09.md)
- [ice-cream iter3 duel](/workspaces/events-bot-new/docs/reports/smart-update-gemma-event-copy-v2-16-1-ice-cream-duel-iter3-2026-03-09.md)
- [iter3 consultation synthesis](/workspaces/events-bot-new/docs/reports/smart-update-event-copy-v2-16-1-ice-cream-duel-iter3-consultation-synthesis-2026-03-09.md)
- [master retrospective](/workspaces/events-bot-new/docs/reports/smart-update-event-copy-opus-gemini-recommendations-master-retrospective-2026-03-08.md)
- [external consultation retrospective](/workspaces/events-bot-new/docs/reports/smart-update-event-copy-external-consultation-retrospective-2026-03-08.md)
- [Gemma deep research impact](/workspaces/events-bot-new/docs/reports/smart-update-gemma-deep-research-impact-on-v2-15-2-2026-03-08.md)
- [Smart Update feature docs](/workspaces/events-bot-new/docs/features/smart-event-update/README.md)

## 0. Суть новой ветки

`v2.16.2 lollipop` — это уже не продолжение `ice-cream`.

Это новая исследовательская ветка со следующей логикой:

- baseline `Smart Update` остаётся production-эталоном;
- baseline fact-first accumulation остаётся;
- baseline infoblock logic остаётся;
- baseline list handling остаётся;
- вся подготовка фактов и editorial planning идёт через длинную funnel-цепочку маленьких Gemma-вызовов;
- один и тот же атомарный шаг может иметь несколько prompt-версий;
- результаты этих версий проходят отдельный шаг отбора/дедупликации/слияния;
- только один финальный вызов `4o` разрешён и используется только для финальной генерации публичного текста;
- после финального `4o` нет обязательного LLM-repair pass.

Короткая иерархия:

- `Smart Update`
- `fact-first accumulation`
- `Gemma funnel preparation`
- `single final 4o text generation`

## 1. Что именно считается правильным пониманием задачи

### 1.1. Не giant prompt, а funnel

Цель больше не в том, чтобы найти один большой “правильный” prompt.

Цель:

- собрать банк маленьких prompt-стадий;
- версионировать их;
- гонять их по кейсбуку;
- фиксировать статистику;
- находить несколько локально сильных prompt-версий на один и тот же атомарный шаг;
- затем строить из них funnel:
  - несколько candidate-веток одного шага
  - затем `select_best / merge_best / dedupe_best`
  - затем следующий логический шаг

### 1.2. Writer не должен чинить плохую подготовку

`write -> audit -> repair` больше не считается нормальной главной линией.

Если финальный writer приходится чинить, это означает:

- fact buckets были грязными;
- support/poster/service шум дошёл до narrative-pack;
- pattern/hook/layout были выбраны некачественно;
- narrative-ready pack ещё не готов к финальной генерации.

Правильная цель:

- финальный `4o` должен получать уже clean, filtered, narrative-ready package;
- после него допустимы только safe checks и fallback decision;
- не должно быть зависимости от широкого LLM-repair.

### 1.3. 4o используется строго один раз

Из-за лимита `4o`:

- нельзя строить ветку, где `4o` участвует в extraction / planning / repair / audit;
- весь upstream funnel работает только на `Gemma`;
- `4o` получает только итоговый pack и генерирует публичный текст один раз.

### 1.4. `lollipop v1` должен быть candidate-first, а не route-first

Для этой ветки нужно исходить из жёсткой предпосылки:

- на десятках и тем более тысячах реальных source-постов не будет достаточно надёжной routing-map;
- ранний `route-first` будет системно терять факты и mis-handle unusual shapes;
- устойчивость должна достигаться не качеством routing, а качеством `select / merge / priority` после группы candidate-прогонов.

Следствие:

- в `v1` ключевые stage-families запускаются широко;
- несколько версий одного атомарного шага на одном и том же событии считаются нормой;
- `selection` идёт после candidate-bank, а не вместо него;
- routing допустим только как поздняя optimization-layer после доказанного quality win над baseline.

## 2. Что обязательно сохраняем из baseline

### 2.1. Infoblock

Готовая baseline-логика инфоблока считается замороженной сильной частью системы.

Следствие:

- `lollipop` не переизобретает infoblock;
- infoblock рендерится baseline-способом;
- в dry-run / duel отчётах infoblock показывается ПЕРВЫМ, затем уже narrative text.

### 2.2. List discipline

Готовая baseline-логика списков тоже считается обязательной к переиспользованию.

Нельзя терять:

- short program lists;
- репертуар;
- setlist;
- numbered/bulleted fragments;
- порядок пунктов;
- буквальное сохранение названий.

### 2.3. Baseline fact-first spine

Нельзя ломать:

- `event_source_fact`
- source-by-source accumulation
- event-level rebuild
- multi-source enrichment
- multi-event source scoping

`lollipop` встраивается сверху как preparation/generation branch, а не заменяет этот spine.

## 2.4. Что обязательно вытаскиваем из более поздних веток

Отдельный обязательный слой для `lollipop`:

- нельзя ограничиваться только carry из `2.15.2 / 2.15.3`;
- нужно явно вытащить поздние локальные win из `2.15.5+` и `ice-cream`;
- но вытаскивать не ветки целиком, а только те stage-version / prompt contracts, которые совместимы с funnel-архитектурой.

Главное правило:

- если поздняя наработка полезна для extraction / typing / selection / layout pack — переносим;
- если она полезна только как финальный prose-writer или repair-loop внутри `ice-cream` — не переносим как core flow.

### 2.4.1. Что переносим из `2.15.5`

`presentation_project` дал важный proof, что split normalize лучше одного перегруженного normalize-step.

В stage-bank надо положить как seed-кандидаты:

- `normalize_subject`
- `normalize_program`
- `expand_agenda`

Они не обязаны сохранить старые названия буквально, но их логика переносится в `facts.extract_*` funnel.

### 2.4.2. Что переносим из `2.15.6`

`lecture_person` дал лучший shape-specific разрез:

- `normalize_cluster`
- `normalize_theme`
- `normalize_profiles`

Это прямые seed-кандидаты для:

- `facts.extract_shapeaware`
- `facts.type`
- `hook/person` и `pattern/person-led`

### 2.4.3. Что переносим из `2.15.7`

`program_rich` дал самый ценный carry для fact-floor preparation.

Нужно явно перенести в seed-bank:

- `normalize_concept`
- `normalize_setlist`
- `normalize_performer`
- `normalize_stage`
- `setlist_v1_grouped`

Именно это подтверждает, что `setlist` и program-items нельзя надеяться “дописать потом” writer-ом.

### 2.4.4. Что переносим из `2.15.8`

Из `atomic shape batch` переносим только extraction-side micro-contracts, а не их prose-stage.

Seed-кандидаты для `v1`:

- `normalize_card_v1`
- `normalize_support_v1`
- `normalize_identity_v2_strict`
- `normalize_participation_v1`
- `normalize_program_v1`

Отдельный salvage outside `v1`:

- `extract_plot_v1`

`extract_plot_v1` не должен входить в `seed-bank v1` как normal path.
Его можно держать только как later experiment после отдельной валидации screening/theater guards.

Непереносимое из `2.15.8`:

- `plan_lead`
- `generate_lead`
- `generate_body`
- `repair`

Их локальные результаты тогда не дали переносимого win.

### 2.4.5. Что переносим из `2.15.9`

Из `2.15.9` переносим не downstream assembly-path целиком, а идею:

- strict separation of:
  - list block
  - logistics block
  - narrative prose

Это должно стать частью:

- `layout.plan`
- `pack.compose`

Но не writer-core.

Сам deterministic routing не считается mainline carry для `lollipop v1`.
Если он когда-либо вернётся, то только как поздняя optimization после того, как broad-run candidate-first режим уже докажет устойчивое превосходство над baseline.

### 2.4.6. Что переносим из `2.15.10`

Из screening-grounding round переносим:

- screening-specific metadata framing;
- осторожность к world-knowledge bleed;
- идею, что некоторые shape требуют особого prep-contract до writer-а.

Но не переносим screening prose-path как universal template.

### 2.4.7. Что переносим из `ice-cream`

`ice-cream` дал важные вещи не как итоговая writer-ветка, а как instrumentation и fact-pack discipline.

Нужно перенести:

- `normalize_fact_floor` как stage family;
- shape-aware buckets:
  - `narrative_facts`
  - `list_facts`
  - `logistics_facts`
- stage-level prompt/result traces;
- per-stage profiling;
- `primary_failure_stage` thinking;
- строгое недопущение свободного переписывания list/logistics/support в prose;
- shape-specific anti-pattern knowledge для:
  - `screening_card`
  - `party_theme_program`
  - `theater_history`

Нужно отдельно зафиксировать, что `ice-cream` полезен и как negative knowledge base:

- какие prompt families дают modality drift;
- какие shape rules не удерживают Gemma;
- какие support/poster facts загрязняют narrative.

### 2.4.8. Что из `ice-cream` НЕ переносим как core

Не переносим:

- `generate_narrative_core` как основной writer-path;
- `audit_narrative_core -> repair_narrative_core` как mainline;
- идею, что tightening bans сам по себе стабилизирует prose;
- финальный Gemma writer в роли главного public-text generator.

## 3. Что обязательно переносим из `2.15.2 / 2.15.3`

Ниже — non-negotiables.

### 3.1. Text quality bar

Итоговый текст должен оставаться:

- естественным;
- профессиональным;
- нешаблонным;
- grounded;
- полным по фактам;
- ясным;
- вариативным;
- с сильным лидом;
- с осмысленными подзаголовками;
- со списками там, где они реально нужны;
- без AI clichés;
- без сервисного мусора;
- без бюрократического prose drift.

### 3.2. Pattern-driven generation

Pattern idea не отменяется.

Но:

- patterning живёт в generation/presentation layer;
- patterning не должен рано искажать extraction;
- execution patterns для Gemma остаются компактными и sharp.

### 3.3. Hook selection

Выбор hooks остаётся нужен.

Это включает:

- direct lead;
- quote / epigraph only when justified;
- person-led / program-led / project-led / scene-led entry;
- выбор opening move не по vibes, а по фактам и shape.

### 3.4. Formatting discipline

Остаются в силе:

- meaningful `###` headings;
- optional `blockquote`, но только при evidence;
- functional markdown lists;
- разделение `narrative` vs `infoblock`;
- отсутствие service leakage в public prose.

## 4. Что не делаем в `lollipop`

### 4.1. Не делаем deterministic semantic post-check regex-first

Пользовательское ограничение для этой ветки:

- нельзя полагаться на regex-heavy deterministic semantic post-check как на основной quality gate;
- регулярки допустимы только как safe structural helpers:
  - markdown shape counts
  - trivial duplicate cleanup
  - list rendering hygiene
  - reporting counters

Semantic validation, selection, routing, prioritization и cleanup должны жить в:

- Gemma stage passes;
- LLM-side comparison;
- фактологических merge/select шагах;
- финальной decision logic по артефактам.

Для `lollipop v1` канонический порядок такой:

- сначала несколько versioned candidate-runs;
- потом `select / merge / priority`;
- и только потом optional soft signals.

### 4.2. Не делаем final-repair loop by default

`lollipop` не строится как:

- final write
- big audit
- big repair
- big re-audit

Вместо этого:

- чистим upstream;
- writer получает только curated pack;
- если final result не проходит bar, candidate просто отклоняется.

## 5. Базовая идея `lollipop funnel`

Один логический шаг теперь может выглядеть так:

- `stage_x.v1`
- `stage_x.v2`
- `stage_x.v3`
- `stage_x_select.v1`

Где:

- `v1/v2/v3` — разные prompt-версии, решающие один и тот же атомарный подшаг;
- `select` — отдельный шаг, который:
  - выбирает лучшее,
  - объединяет лучшее,
  - убирает дубли,
  - не теряет факты.

Следующий шаг работает уже не с сырым source, а с output предыдущей funnel-stage.

Итоговая цепочка может быть длинной:

- `scope.extract.v2`
- `scope.extract.v4`
- `scope.select.v1`
- `facts.narrative_candidates.v3`
- `facts.narrative_candidates.v5`
- `facts.merge.tier1.v1`
- `facts.merge.tier2.v1`
- `facts.priority.v1`
- `hook.seed.v1`
- `hook.seed.v4`
- `hook.select.v2`
- `pattern.signal.v2`
- `layout.plan.v2`
- `pack.compose.v1`
- `pack.select.v1`
- `writer.final_4o.spec.v1`
- `writer.final_4o.v1`

Это нормально, если:

- каждый шаг дешёвый по TPM;
- шаг действительно атомарный;
- артефакты сохраняются;
- статистика по stage version ведётся отдельно.
- несколько версий одного stage-family реально запускаются на одном и том же событии;
- `select/merge` stage считается обязательной частью качества, а не лишним overhead.

## 6. Новый целевой pipeline

## 6.1. Stage Group A — `scope`

Задача:

- извлечь только данный event из source;
- не потянуть sibling event lines;
- отделить multi-event digest noise;
- собрать только event-relevant source slice.

Пример стадий:

- `scope.extract.v1`
- `scope.extract.v2`
- `scope.select.v1`

Output:

- `scoped_source_excerpt`
- `scoped_ocr_excerpt`
- `scope_confidence`
- `scope_notes`

Важно:

- `scope` здесь ограничивает sibling-noise и multi-event bleed;
- `scope` не должен становиться hard router для остального funnel.

## 6.2. Stage Group B — `fact candidates`

Задача:

- из scoped source получить несколько candidate-наборов фактов;
- разные prompt-версии могут лучше работать на:
  - names
  - program items
  - forward-looking promises
  - support/visitor conditions
  - stage/venue context

Это broad-run family:

- несколько extraction-версий должны запускаться на одном событии по умолчанию;
- устойчивость появляется потом на `merge / select / priority`, а не на upstream routing.

Пример стадий:

- `facts.extract_literal.v1`
- `facts.extract_shapeaware.v1`
- `facts.extract_support.v1`
- `facts.extract_forward.v1`
- `facts.merge.tier1.v1`
- `facts.merge.tier2.v1`
- `facts.priority.v1`

Output:

- canonical claim candidates
- provenance
- duplicates map

## 6.3. Stage Group C — `fact typing`

Задача:

- разложить canonical claims по buckets.

Обязательные buckets:

- `identity_core`
- `narrative_core`
- `program_list`
- `visitor_conditions`
- `infoblock_ready`
- `support_context`
- `poster_context`
- `uncertain_or_conflict`

Критически важно:

- poster/support не должны автоматически попадать в narrative_core;
- фраза вроде `На афише размещены цитаты...` должна почти всегда уходить в `poster_context`, а не в event prose.

Пример стадий:

- `facts.type.v1`
- `facts.type.v2`
- `facts.type_select.v1`

Это тоже candidate-first family:

- допускается несколько typing-версий на одном событии;
- итоговый bucket-map должен выбираться `type_select`, а не single-pass typing.

## 6.4. Stage Group D — `hook seed`

Задача:

- не писать готовые лиды, а собрать несколько структурных `hook seeds`;
- каждый `hook seed` должен содержать:
  - angle / hook_type
  - anchor_fact
  - source grounding
  - short reason why this angle is usable
- потом выбрать лучший seed без потери фактов и без decorative drift.

Пример:

- `hook.seed.direct.v1`
- `hook.seed.direct.v2`
- `hook.seed.quote.v1`
- `hook.seed.person.v1`
- `hook.seed.person.v2`
- `hook.seed.program.v1`
- `hook.seed.program.v2`
- `hook.select.v1`

Это не single-version family:

- несколько hook-seed версий должны конкурировать на одном событии;
- `hook.select` выбирает лучший структурный angle уже после broad-run, а не после routing.

Output:

- `hook_type`
- `anchor_fact`
- `hook_reason`
- `hook_supporting_fact_ids`

## 6.5. Stage Group E — `pattern signal + layout`

Задача:

- не делать full creative pattern choice на Gemma в `v1`;
- instead вычислить `pattern signal / pattern hint` из shape и fact-density;
- выбрать section plan;
- решить, нужен ли epigraph;
- решить, нужен ли list block;
- построить exact final layout contract.

Важно:

- patterning живёт здесь, а не в extraction;
- infoblock не смешивается с narrative;
- list logic должна быть baseline-aligned.

Пример:

- `pattern.signal.v1`
- `pattern.signal.v2`
- `layout.plan.v1`

Output:

- `pattern_hint`
- `use_epigraph`
- `heading_titles[]`
- `list_block_plan`
- `infoblock_policy=baseline`

## 6.6. Stage Group F — `narrative-ready pack`

Задача:

- из прошлых стадий собрать финальный writer-input pack.

В pack входят:

- event identity
- selected hook
- selected pattern
- selected narrative claims
- selected list block items
- visitor conditions
- infoblock-ready fields
- banned noise buckets

Это последний Gemma-prep step перед `4o`.

Пример:

- `pack.compose.v1`
- `pack.compose.v2`
- `pack.select.v1`

Это тоже broad-run family:

- допускается несколько pack-вариантов на одном и том же event fact floor;
- `pack.select` обязателен, потому что здесь качество часто выигрывает best-of merge нескольких pack views.

## 6.7. Stage Group G — `writer.final_4o`

Перед первым реальным writer-pass должна существовать отдельная stage-spec:

- `writer.final_4o.spec.v1`

В ней фиксируется:

- exact input payload
- field semantics
- anti-pattern payload
- layout contract
- grounding contract
- what 4o may not invent

Единственный вызов `4o`.

Задача:

- сгенерировать итоговый public text;
- сохранить все факты;
- применить hook/pattern/layout;
- не трогать infoblock policy;
- не сворачивать списки;
- не тянуть support/poster в lead/body без явного разрешения.

Input:

- только curated narrative-ready pack;
- без сырых source texts;
- без лишнего historical noise.

Output:

- final event text

## 7. Что должно храниться как исследовательский банк

Для каждой stage-version нужно хранить:

- stage family
- exact stage id
- version
- prompt text
- schema / expected output
- input artifact ref
- output artifact ref
- case ids
- metrics per case
- known strengths
- known failure modes

Пример naming:

- `scope.extract.v2`
- `facts.extract_literal.v3`
- `facts.extract_forward.v1`
- `facts.type.v2`
- `hook.seed.program.v4`
- `hook.select.v2`
- `pattern.signal.v2`
- `facts.merge.tier2.v1`
- `facts.priority.v1`
- `pack.select.v1`
- `writer.final_4o.spec.v1`
- `writer.final_4o.v1`

## 7.1. Seed-bank для первого раунда

Перед началом новых prompt-экспериментов stage-bank должен стартовать не с пустого места, а с initial seed set.

Минимальный seed set:

- `facts.extract_subject.v1`
  - carry from `2.15.5 normalize_subject`
- `facts.extract_agenda.v1`
  - carry from `2.15.5 expand_agenda`
- `facts.extract_program.v1`
  - carry from `2.15.5 normalize_program`
- `facts.extract_cluster.v1`
  - carry from `2.15.6 normalize_cluster`
- `facts.extract_theme.v1`
  - carry from `2.15.6 normalize_theme`
- `facts.extract_profiles.v1`
  - carry from `2.15.6 normalize_profiles`
- `facts.extract_concept.v1`
  - carry from `2.15.7 normalize_concept`
- `facts.extract_setlist.v1`
  - carry from `2.15.7 normalize_setlist / setlist_v1_grouped`
- `facts.extract_performer.v1`
  - carry from `2.15.7 normalize_performer`
- `facts.extract_stage.v1`
  - carry from `2.15.7 normalize_stage`
- `facts.extract_card.v1`
  - carry from `2.15.8 normalize_card_v1`
- `facts.extract_support.v1`
  - carry from `2.15.8 normalize_support_v1`
- `facts.extract_identity.v1`
  - carry from `2.15.8 normalize_identity_v2_strict`
- `facts.extract_participation.v1`
  - carry from `2.15.8 normalize_participation_v1`
- `facts.type_buckets.v1`
  - carry from `ice-cream normalize_fact_floor + fact bucketing`
- `facts.merge.tier1.v1`
  - deterministic/simple dedup before LLM conflict resolution
- `facts.merge.tier2.v1`
  - semantic conflict merge only for hard cases
- `facts.priority.v1`
  - selection within buckets before final pack
- `hook.seed.v1`
  - structured hook seed family, not prose hook drafts
- `hook.select.v1`
  - select best structured hook seed
- `pattern.signal.v1`
  - structural pattern hint, not full stylistic choice
- `layout.plan.v1`
  - carry from baseline list/infoblock split plus `2.15.9` separation logic
- `pack.compose.v1`
  - first narrative-ready pack assembler
- `pack.select.v1`
  - best-of selection between pack variants
- `writer.final_4o.spec.v1`
  - explicit payload contract for the only 4o call

Этот seed set не финален.

И он не означает `по одной версии на stage family`.

Для `lollipop v1` нормой считается, что:

- один stage family может содержать несколько сильных version-candidates;
- эти версии могут запускаться на одном и том же событии;
- финальное качество достигается через `select / merge / priority`, а не через раннее схлопывание в single routed version.

Он нужен, чтобы:

- не выдумывать stage-bank с нуля;
- явно зафиксировать, какие поздние удачи уже должны войти в funnel;
- дальше сравнивать новые версии не с пустотой, а с лучшим известным кандидатом каждой stage family.

## 7.2. Salvage matrix для stage families

Для каждой новой stage family в brief и harness должна быть отдельная salvage-matrix:

- `source`
- `old stage / prompt id`
- `why it was locally strong`
- `what exactly is transferred`
- `what is NOT transferred`
- `new lollipop stage id`

Без этого легко потерять реальные поздние win и заново изобретать уже найденные рабочие куски.

## 8. Что должно входить в stage metrics

Метрики теперь ведутся не только по финальному тексту, но и по каждой stage version.

Примеры:

- `fact retention`
- `secondary-source gain`
- `duplicate removal quality`
- `poster contamination rate`
- `support-to-narrative leakage`
- `list preservation`
- `hook acceptability`
- `pattern signal usefulness`
- `heading usefulness`
- `fact loss after select`

Для final text stage:

- full fact retention
- unsupported claims
- list loss
- infoblock leak
- hook quality
- pattern naturalness
- section usefulness
- overall prose quality

## 9. Как делаем консультации

Консультации должны быть точечными.

Нельзя каждый раз таскать в `Opus/Gemini` весь проектный контекст.

Правильный режим:

- узкий stage brief;
- несколько кейсов;
- конкретная prompt-версия;
- конкретный expected behavior;
- конкретный mismatch.

То есть консультации идут не “про весь pipeline”, а, например:

- review только `facts.type.v2`
- review только `hook.select.v1`
- review только `pattern.signal.v2`

## 10. Что должно измениться в dry-run reports

Новый формат отчёта:

1. infoblock
2. final event text
3. stage chain
4. used stage versions
5. stage metrics
6. final metrics

Для каждого кейса нужно видеть:

- какой exact funnel использовался;
- какие version ids участвовали;
- что именно было выбрано на select-стадиях;
- что было отброшено;
- были ли потеряны факты.

## 11. Что переносим из `ice-cream`

Хотя ветка меняется, из `ice-cream` сохраняем полезное:

- multi-source fact-floor awareness;
- shape-aware micro-contract thinking;
- stage-level profiling;
- raw prompt/result trace discipline;
- failure localization по шагам;
- понимание, что final prose не должен смешивать logistics/list/support произвольно.

Что НЕ переносим как core flow:

- write -> repair as mainline;
- идея, что final prose можно стабилизировать только tightening bans;
- попытка лечить semantic problems финальным Gemma rewrite.

## 12. Первая исследовательская программа для `lollipop`

Порядок первых рабочих раундов:

1. Зафиксировать кейсбук.
2. Вытащить baseline-aligned infoblock/list invariants в harness.
3. Построить stage bank для первых трёх групп:
   - `scope`
   - `fact candidates`
   - `fact typing`
4. Прогнать несколько prompt-версий на каждый из этих шагов.
5. Построить `select/merge` стадии.
6. Только после этого переходить к:
   - `hook`
   - `pattern signal`
   - `layout`
   - `pack`
7. Параллельно спроектировать `writer.final_4o.spec.v1`.
8. И только затем подключать `writer.final_4o.v1`.

## 13. Decision rule для `lollipop`

Ветка считается перспективной только если одновременно выполняются все условия:

- не теряются факты;
- poster/support contamination в narrative падает;
- списки и infoblock baseline-сильны;
- pattern/hook quality возвращается на уровень `2.15.2/2.15.3` целей;
- final `4o` реально улучшает prose, а не просто косметически сглаживает плохой pack.

Если этого нет:

- проблема считается upstream;
- следующий раунд идёт в funnel stages, а не в final writer.

## 14. Короткий вердикт

`lollipop` — это:

- не writer-pipeline;
- не repair-pipeline;
- не regex-check pipeline;

а:

- versioned prompt-bank,
- funnel of Gemma micro-stages,
- best-of / merge-of stage outputs,
- baseline-aligned infoblock/list reuse,
- один финальный `4o` вызов для public text.
