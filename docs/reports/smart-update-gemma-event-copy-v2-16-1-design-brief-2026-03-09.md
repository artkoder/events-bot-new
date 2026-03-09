# Smart Update Gemma Event Copy V2.16.1 Design Brief

Дата: 2026-03-09

Supersedes:

- [v2.15.11 design brief](/workspaces/events-bot-new/docs/reports/smart-update-gemma-event-copy-v2-15-11-design-brief-2026-03-09.md)

Основание:

- [baseline -> v2.14 retrospective](/workspaces/events-bot-new/docs/reports/smart-update-gemma-event-copy-retrospective-baseline-v2-14-2026-03-08.md)
- [v2.6 review](/workspaces/events-bot-new/docs/reports/smart-update-opus-gemma-event-copy-pattern-dry-run-v2-6-review-2026-03-07.md)
- [v2.15.2 design brief](/workspaces/events-bot-new/docs/reports/smart-update-gemma-event-copy-v2-15-2-design-brief-2026-03-08.md)
- [v2.15.3 design brief](/workspaces/events-bot-new/docs/reports/smart-update-gemma-event-copy-v2-15-3-design-brief-2026-03-08.md)
- [v2.15.8 atomic shape batch](/workspaces/events-bot-new/docs/reports/smart-update-opus-gemma-event-copy-v2-15-8-atomic-shape-batch-2026-03-08.md)
- [v2.15.9 downstream assembly retune](/workspaces/events-bot-new/docs/reports/smart-update-opus-gemma-event-copy-v2-15-9-downstream-assembly-retune-2026-03-08.md)
- [v2.15.10 screening grounding retune](/workspaces/events-bot-new/docs/reports/smart-update-opus-gemma-event-copy-v2-15-10-screening-grounding-retune-2026-03-08.md)
- [external consultation retrospective](/workspaces/events-bot-new/docs/reports/smart-update-event-copy-external-consultation-retrospective-2026-03-08.md)
- [Opus + Gemini master retrospective](/workspaces/events-bot-new/docs/reports/smart-update-event-copy-opus-gemini-recommendations-master-retrospective-2026-03-08.md)
- [Gemma deep research impact on v2.15.2](/workspaces/events-bot-new/docs/reports/smart-update-gemma-deep-research-impact-on-v2-15-2-2026-03-08.md)
- [using-gemma-deep-research-report.md](/workspaces/events-bot-new/docs/researches/using-gemma-deep-research-report.md)
- [using-gemma-deep-research-report-gemini.md](/workspaces/events-bot-new/docs/researches/using-gemma-deep-research-report-gemini.md)
- [Smart Event Update feature README](/workspaces/events-bot-new/docs/features/smart-event-update/README.md)
- [Fact-first Smart Update](/workspaces/events-bot-new/docs/features/smart-event-update/fact-first.md)

## 0. Почему теперь `2.16.1`, а не `2.15.11`

`2.15.11` правильно вернул missing dimension:

- multi-source accumulation на уровне события;
- narrative vs infoblock split;
- block-level layout вместо одинакового two-paragraph prose.

Но после тупика `2.15.10` и смены постановки это уже не просто "следующий подпункт внутри `2.15.x`".

Это новая runtime-integration ветка:

- не offline replacement текущего пайплайна;
- не single-source experimental rewriter;
- а встраивание найденных atomic wins обратно в реальный Smart Update.

Поэтому канонический следующий brief = `2.16.1`.

`2.15.11` сохраняется как recovery-note.
`2.16.1` — как основной integration brief.

## 1. Каноническая иерархия `2.16.1`

Порядок больше не должен плавать.

### 1.1. Smart Update

Внешний runtime-контур уже задан текущим кодом:

- приходит один новый `candidate` из одного source/post;
- Smart Update либо создаёт событие, либо матчится в уже существующее;
- у события уже может быть 1, 2, 5 или 20 прошлых source;
- после каждого нового source событие должно быть пересобрано заново, но уже из накопленного event-level fact floor.

Это верхний слой.

### 1.2. Fact-first

Канонический источник публичного текста — не raw source text и не предыдущий `description`, а накопленные факты события:

- `event_source_fact` по всем source текущего события;
- legacy baseline fact seed, если он действительно нужен;
- event-level anchors и trust-aware structured fields;
- OCR-derived facts;
- source-specific enrichment из вторичных постов.

Это второй слой и он non-negotiable.

### 1.3. LLM-first

Семантические решения остаются за LLM:

- scoping сложных source;
- shape-aware extraction;
- semantic normalization;
- layout planning;
- lead/body wording;
- targeted repair.

Deterministic код допускается только как support:

- anchor rules;
- trust/merge policy;
- bucketing;
- validation;
- format guardrails;
- leakage/hallucination checks.

### 1.4. Pattern-driven

Variability текста и human-like writing behavior должны формироваться через pattern layer.

Но pattern layer живёт только в generation/layout:

- не создаёт смысл слишком рано;
- не подменяет fact floor;
- не превращается в giant routing tree с десятком расплывчатых карт.

### 1.5. Atomic pipeline / micro-contracts

Исполнительный уровень должен быть дробным и shape-aware:

- small prompts;
- step-local roles;
- ясные JSON/Markdown contracts;
- optional branching только там, где это реально даёт выигрыш.

Но atomicity не означает "всегда 8 вызовов".

Правильный принцип:

- micro-contracts детализируются там, где shape этого требует;
- happy path должен оставаться достаточно компактным для runtime Smart Update.

## 2. Ретроспектива: baseline -> `v2.6` -> `2.15.11`

## 2.1. Что baseline и текущий runtime уже делают правильно

Текущий Smart Update уже имеет правильный production skeleton:

- source-by-source processing;
- create/merge around one incoming source;
- хранение `event_source`;
- накопление `event_source_fact`;
- `added / duplicate / conflict / note` статусы;
- `source_texts` synchronization;
- legacy snapshot backfill для старых событий;
- trust-aware merge якорей и ticket fields;
- fact-first path для `description`;
- support-layer cleanup для headings, quotes, lists, Telegraph readability.

Это не нужно выкидывать.

Главный урок baseline:

- production success шёл от fact discipline и от того, что событие живёт как сущность, обогащающаяся во времени.

Главные baseline слабости:

- prose часто однотипен;
- leads местами слишком служебные;
- headings не всегда живые;
- layout variability не всегда раскрыта;
- часть rich cases в позднем baseline ощущается как safe-but-flat.

То есть baseline — это правильный каркас, но не потолок качества текста.

## 2.2. Что `v2.6` нельзя потерять

`v2.6` не был runtime candidate, но был важным quality marker.

Из него доказанно полезны:

- сильный compact handling для sparse case `2660`;
- более живые headings и подача;
- strongest recovery на `2734`;
- локальные prose wins против baseline на части кейсов.

При этом `v2.6` также ясно показал, что:

- lecture/presentation cases (`2687`, `2673`) нельзя "лечить" теми же общими moves;
- label-style facts и `посвящ*` легко возвращаются;
- coverage improvement без editorial discipline не даёт настоящей победы.

Вывод:

- `v2.6` — это не архитектура для копирования;
- но это обязательный quality bar по живости, headings, compactness и person/program-led phrasing.

## 2.3. Что `2.15.2` и `2.15.3` правильно зафиксировали

Именно `2.15.2` и `2.15.3` вернули важные architectural invariants:

- не giant prompt;
- атомарные шаги;
- planner structural-only;
- no prose-outline;
- no same-model full editorial pass;
- dynamic prompt assembly;
- facts-backed repair;
- pattern library остаётся, но execution patterns для Gemma компактнее;
- quote/epigraph/list gating должно быть evidence-based.

Это не временные идеи, а долгоживущие carries.

## 2.4. Что `2.15.5 - 2.15.10` реально доказали

По atomic win-cycle уже есть твёрдые выводы:

- step-level fine-tuning реально работает;
- shape-aware atomic extraction переносится между разными типами событий;
- `presentation`, `lecture_person`, `program_rich` дали локальные win;
- screening требует отдельного groundedness audit;
- exhibition не надо насильно загонять в один и тот же downstream, если он там хуже.

Главный negative lesson:

- downstream assembly ветка начала жить как single-source rewrite;
- event-level accumulation выпал из центра системы;
- rich formatting схлопнулся в одинаковый two-paragraph mold.

## 2.5. Что `2.15.11` исправил и чего ещё не хватало

`2.15.11` правильно вернул:

- event-level canonical fact floor;
- multi-source viewpoint;
- narrative vs infoblock separation;
- block-level layout planning;
- `###`, lists, quote blocks как first-class structure.

Но `2.15.11` был ещё design recovery, а не полный runtime-brief.

Чего не хватало:

- явной оценки текущего кода Smart Update;
- прямого встраивания в source-by-source merge flow;
- формального связывания atomic extraction wins с `event_source_fact` accumulation;
- нового номера ветки после фактического branch reset.

`2.16.1` закрывает именно этот пробел.

## 3. Оценка текущего кода Smart Update

## 3.1. Что в коде нужно сохранить буквально

В текущем runtime есть несколько вещей, которые уже попали в правильную архитектурную точку:

- `event_source` как source-level provenance;
- `_record_source_facts(...)` как идемпотентный лог фактов по source;
- `_sync_source_texts(...)` как event-level source corpus;
- legacy description snapshot + backfill;
- trust-aware anchor merge;
- ticket merge отдельно от prose;
- существующий sanitizer/cleanup stack;
- derived fields (`short_description`, `search_digest`) как отдельные outputs;
- факт, что Smart Update работает от одного incoming source, а не от "batch of six posts".

Это и есть production spine.

## 3.2. Главный integration gap в текущем коде

Сейчас `_llm_merge_event(...)` делает слишком много несовместимых задач одновременно:

1. оценивает fact delta;
2. решает duplicate/conflict;
3. использует raw candidate text и event description как материал для итогового описания;
4. одновременно предлагает новый `title`, `description`, `search_digest` и часть merge-решений.

Это плохое место по отношению к новой иерархии.

Почему:

- merge и authoring смешаны;
- raw source text снова становится semantic core;
- event-level fact floor теряет статус единственного источника истины для public text;
- single-source bias возвращается через merge prompt.

## 3.3. Второй integration gap

`_llm_fact_first_description_md(...)` уже идёт в правильную сторону:

- берёт только `facts_text_clean`;
- делает coverage/revise loop;
- не смотрит в raw source.

Но в текущем виде это всё ещё слишком плоский слой:

- на входе только flat list facts;
- нет typed canonical floor;
- нет event-level provenance priorities;
- нет shape-aware layout planner;
- нет block-level generation contracts;
- screening grounding и digest bleed там не выражены как отдельные gates.

Итог:

- это хороший промежуточный proof, но не финальная runtime architecture.

## 3.4. Третий integration gap

`_facts_text_clean_from_facts(...)` полезен как deterministic guardrail:

- отделяет `text_clean` от `infoblock` и `drop`;
- режет anchors;
- не даёт логистике утекать в narrative.

Но этого мало для `2.16.1`, потому что нужен не только flat cleaned list, а canonical event floor с richer semantics:

- group/list facts;
- quote candidates;
- sectionable clusters;
- source provenance;
- secondary-source enrichment tracking;
- must-include markers для фактов, пришедших позже и часто теряющихся в prose.

## 3.5. Архитектурный вывод по коду

Встраивание должно быть additive, а не destructive:

- Smart Update runtime skeleton сохраняется;
- fact log и source accumulation сохраняются;
- merge-решения по anchors/tickets сохраняются;
- новый pipeline встраивается в rebuild public text layer.

То есть менять нужно не "как живёт событие", а "как после каждого source строится новый текст события".

## 4. Все требования к качеству текста, которые нужно перенести из `2.15.2` / `2.15.3`

Ниже не wishlist, а extracted non-negotiables.

## 4.1. Общий quality bar

Итоговый текст должен быть:

- естественным;
- профессиональным;
- не шаблонным;
- не рекламным;
- без нейросетевых клише;
- grounded;
- с полным сохранением публикуемых фактов.

## 4.2. Правило полноты

Твоё базовое требование остаётся жёстким:

- все publishable facts должны оказаться в итоговом тексте;
- если факт list-like, должны сохраниться все элементы;
- если факт относится к infoblock, он не должен утекать в narrative ради формального coverage.

## 4.3. Lead rules

Нужно восстановить и удерживать:

- strong direct lead;
- прямой вход в subject/person/program/theme;
- без служебных заходов типа `мероприятие расскажет`, `в рамках проекта состоится`, `это не ..., а ...`;
- без question-heading tone и без bureaucratic transfer language.

## 4.4. Layout rules

Описание не должно быть forced into two paragraphs.

Нужно снова считать first-class layout vocabulary:

- лид;
- 0-3 осмысленных `###` подзаголовков;
- списки для program / line-up / setlist / object clusters / agenda;
- blockquote для реальной цитаты;
- sparse cases без headings, если так лучше;
- richer cases с block variation, если материал это оправдывает.

## 4.5. Heading rules

Подзаголовки:

- короткие;
- информативные;
- не общие;
- не полные предложения;
- не декоративные;
- не размножают микро-секции по одной фразе.

## 4.6. Quote and epigraph rules

Цитата допустима только если она:

- event-local;
- дословная;
- явно поддержана source fragment;
- не вытянута из соседнего digest item;
- не придумана моделью.

Если цитата сомнительна:

- quote-led path не активируется;
- epigraph не используется;
- repair удаляет dubious quote, а не "чинит" его творчески.

## 4.7. Pattern rules

Нужно сохранить различие:

- editorial pattern vocabulary может быть richer;
- execution pattern set у Gemma должен быть меньше и structurally distinct.

То есть:

- variability сохраняется;
- но runtime не заставляет Gemma выбирать между десятком размытых mood-cards.

## 4.8. Prompt discipline

Нужно сохранить:

- no giant prompt;
- no prose-outline;
- no blind full rewrite repair;
- targeted repair only;
- repair gets facts;
- dynamic prompt assembly;
- 2-3 core generation/planning calls в happy path, а extra micro-steps только когда case действительно rich/ambiguous.

## 4.9. Narrative vs infoblock split

Нужно вернуть baseline discipline как жёсткий инвариант:

- narrative описывает, что это за событие и чем оно наполнено;
- infoblock отдельно держит дату/время/место/цену/вход/ссылки/телефоны/возраст/служебные поля.

## 5. Все устойчивые рекомендации из ретроспектив `Opus` и `Gemini`

## 5.1. Что точно переносится

По всей consultation history подтвердились следующие carries:

- semantic core must stay `LLM-first`;
- full-floor normalization лучше dirty merge;
- pattern layer полезен только в generation;
- anti-duplication — central quality issue;
- positive transformations работают лучше giant ban walls;
- syntax-level prose rules лучше abstract persona;
- stop-phrase / anti-bureaucracy knowledge layer полезен;
- planner должен быть tiny/structural, not prose planner;
- repair must receive facts;
- preserve baseline strengths, а не replace everything.

## 5.2. Что особенно важно для `2.16.1`

Ниже те советы, которые прямо влияют на runtime integration:

- не строить новый semantic core на regex-heavy logic;
- не пытаться "вылечить всё" ещё одним intermediary full-review pass;
- не делать giant routing tree;
- не делать giant inline ban-list внутри main generation prompt;
- не over-abstract pattern names;
- не тащить headings/lists/epigraphs как always-on rules в один монолитный prompt;
- не отдавать всю variability только deterministic сборке.

## 5.3. Что нужно явно держать как anti-regression

- sparse cases не нужно насильно дробить headings;
- body должен быть self-contained;
- list blocks должны быть meaningful, not decorative;
- anti-quote control и anti-bureaucracy control должны быть системными, а не ситуативными;
- sentence-level lexical rules живут дольше, чем абстрактные "пиши как журналист".

## 6. Что обязательно перенести из двух Gemma research

## 6.1. Self-contained prompts

Каждый Gemma step должен быть self-contained:

- нельзя полагаться на hidden system role;
- критические правила должны жить внутри самого step prompt;
- предыдущий вызов не считается "магическим контекстом" для следующего.

## 6.2. Секционная структура prompt'ов

Промпты лучше организовывать через короткие смысловые блоки:

- `ROLE`
- `RULES`
- `OUTPUT`
- `FACTS`

Это особенно важно для:

- source scoping;
- normalize steps;
- planner;
- targeted repair.

## 6.3. Short-chain default

Gemma устойчивее на коротких цепочках вызовов, чем на одном giant prompt.

Для `2.16.1` это подтверждает:

- atomic micro-contracts — верный default;
- validate/repair — нормальный production path;
- но каждый шаг должен иметь одну главную обязанность.

## 6.4. Step-specific decoding profiles

Нужно держать разный decoding режим:

- source-scope / normalize / planner / repair / audits — максимально детерминированно;
- main prose blocks — умеренно креативнее;
- одинаковый decoding profile для всех шагов — плохая идея.

## 6.5. Structure-first outputs

Для Gemma важнее:

- JSON with clear schema;
- concise structural planner outputs;
- grounded block contracts.

И хуже работает:

- freeform explanations;
- visible chain-of-thought;
- giant analytical instructions внутри main prose step.

## 6.6. Schema semantics matter

Нужна не только форма схемы, но и смысл полей.

Особенно это относится к:

- `shape`
- `pattern`
- `use_headings`
- `use_quote_block`
- `blocks.kind`
- `must_include`
- `bucket`
- `group_kind`

## 6.7. Research-specific warnings

Нужно прямо учитывать два риска:

- Gemma может тянуть world knowledge, если screening/path плохо grounded;
- OCR/poster extraction нельзя считать безошибочной и нужно проверять числовые/списочные/известные-title facts отдельными gates.

## 6.8. Fine-tuning remains a legit next step

Если после стабилизации `2.16.x` окажется, что:

- prose quality plateau persists;
- repair loop rate остаётся высоким;
- schema compliance всё ещё постоянно чинится prompt-ами;

то fine-tuning Gemma остаётся легитимным следующим шагом.

Но не до того, как будет стабилизирован runtime architecture.

## 7. Целевая runtime-архитектура `2.16.1`

Это центральная часть brief.

## 7.1. Главный принцип

Smart Update по-прежнему обрабатывает один source за раз.

Новая схема не отменяет это, а правильно использует:

1. пришёл новый source;
2. он дал новые event-specific facts;
3. факты сохранились в `event_source_fact`;
4. из union всех фактов события rebuilt весь public text.

Именно это должно одинаково работать:

- для первого source;
- для шестого source;
- для single-event post;
- для multi-event digest post.

## 7.2. Step A. `source_scope_extract`

Это новый обязательный source-level слой.

Его задача:

- взять один incoming source;
- сузить его до event-specific fragment;
- не пропустить факты текущего события;
- не протащить sibling facts из соседних items digest-поста.

Input:

- raw source text;
- OCR text;
- candidate title/raw_excerpt/anchors;
- если это merge-path — existing event anchors/title/facts as scope hints.

Output contract должен уметь вернуть:

- `scoped_text`
- `scope_confidence`
- `quote_locality`
- `digest_risk`
- optional `scope_notes`

Этот шаг критичен для кейсов вроде `2498` и для всех false-quote failures.

## 7.3. Step B. `source_atomic_extract`

После scoping запускается shape-aware extraction, но всё ещё на одном source fragment.

Принцип:

- extraction остаётся атомарным;
- shape families могут иметь собственные `normalize_*` micro-contracts;
- не все микрошаги запускаются всегда.

Уже подтверждённые families:

- `presentation_like`: `normalize_subject -> expand_agenda -> normalize_program`
- `lecture_person`: `normalize_cluster -> normalize_theme -> normalize_profiles`
- `program_rich`: `normalize_concept -> normalize_setlist -> normalize_performer -> normalize_stage`
- `screening_card`: `normalize_card -> normalize_plot -> normalize_support -> grounding_audit`
- `party_theme_program`: `normalize_theme -> normalize_program -> normalize_support`
- `exhibition_context_collection`: `normalize_context -> normalize_collection -> normalize_object_groups`

Execution rule:

- universal runtime не должен пытаться решить всё одним generic extractor;
- но и не должен жёстко требовать одинакового call stack для каждого event.

Output каждого source extraction должен быть canonical enough for storage:

- atomic fact text;
- `bucket_hint`;
- `group_kind`;
- `shape_slot`;
- `quote_candidate`;
- `list_candidate`;
- `must_include_hint`;
- `source_id/source_url`.

## 7.4. Step C. `persist_source_facts`

После extraction факты записываются в текущий fact spine:

- `event_source_fact`
- source-local provenance
- statuses `added / duplicate / conflict / note`

Это existing Smart Update behavior и его нужно сохранить.

Новый принцип здесь такой:

- accumulation spine уже правильный;
- менять нужно качество входящих source facts и их event-level rebuilding, а не отказываться от этой таблицы.

## 7.5. Step D. `canonical_event_fact_floor`

Это главный missing layer, который теперь должен стать explicit.

Он строится не из одного source, а из всех relevant facts события:

- `added`
- `duplicate`
- при необходимости legacy baseline facts
- OCR-supported enrichment
- event structured fields

Цель не просто "почистить список", а построить typed event floor:

```json
{
  "narrative_facts": [],
  "list_groups": [],
  "quote_candidates": [],
  "section_clusters": [],
  "infoblock_facts": [],
  "drop_facts": [],
  "must_include_fact_ids": [],
  "secondary_source_fact_ids": []
}
```

Именно на этом шаге нужно:

- дедуплицировать event-level facts;
- сохранить source-specific enrichment;
- пометить факты, которые пришли из secondary source и historically теряются;
- собрать list/set/program groups;
- отделить narrative от infoblock.

`_facts_text_clean_from_facts(...)` может остаться как часть этого слоя, но уже не как его полная замена.

## 7.6. Step E. `shape_aware_layout_plan`

Planner должен работать только по canonical event floor, а не по raw source.

Его задача:

- определить lead pattern;
- решить, нужны ли headings;
- решить, нужен ли list block;
- решить, нужен ли quote block;
- разложить facts по lead/sections/lists/quote/infoblock;
- зафиксировать coverage targets.

Planner output — structural only.

Пример:

```json
{
  "lead_pattern": "person_led | concept_led | program_led | screening_led | sparse_led",
  "use_headings": true,
  "use_quote_block": false,
  "sections": [
    {"kind": "lead", "fact_ids": ["f1", "f2"]},
    {"kind": "section", "heading_kind": "program", "fact_ids": ["f3", "f4"]},
    {"kind": "list", "list_kind": "setlist", "fact_ids": ["f5", "f6", "f7"]}
  ],
  "excluded_infoblock_fact_ids": ["i1", "i2"]
}
```

Именно здесь pattern-driven и layout variation должны жить в `2.16.1`.

## 7.7. Step F. `generate_blocks`

Новый public-text path не должен быть одним blob prompt.

Нужны block-level steps:

- `generate_lead`
- `generate_section`
- `assemble_or_generate_list_block`
- `integrate_quote_block`

Ключевое правило:

- каждый блок получает только свои fact ids;
- не тянет новые факты;
- не повторяет смысл соседнего блока;
- не подменяет infoblock narrative-логистикой.

Здесь как раз должны жить:

- strong leads;
- headings;
- list preservation;
- quote attribution;
- pattern-driven variation.

## 7.8. Step G. `validate_and_targeted_repair`

Validator и repair остаются.

Но теперь они должны проверять не только `missing/extra`, а ещё:

- `digest_bleed`
- `infoblock_leak`
- `quote_grounding`
- `list_item_loss`
- `micro_section`
- `heading_genericity`
- `world_knowledge_drift`
- `duplicate_fact_expression`

Repair rule:

- narrow and facts-backed;
- only when validator says so;
- no full blind rewrite.

## 7.9. Step H. `derive_short_and_search_digest`

`short_description` и `search_digest` должны оставаться отдельными derived outputs.

Но их лучше строить уже после финального event-level rebuild и при необходимости сверять с canonical floor.

То есть они не должны снова тянуться из raw source мимо rebuilt event text.

## 8. Как это должно встроиться в текущий Smart Update код

## 8.1. Что остаётся на месте

Не меняются как spine:

- create/match/merge orchestration;
- anchor correction rules;
- ticket merge;
- source persistence;
- source facts persistence;
- source_texts sync;
- legacy backfill;
- existing sanitizers и safe deterministic cleanup.

## 8.2. Что нужно сузить по ответственности

`_llm_merge_event(...)` должен перестать быть главным authoring prompt.

Его правильная новая роль:

- fact delta classification;
- duplicate/conflict classification;
- optional title/ticket suggestions;
- optional source-scope notes;
- no final `description` as primary output.

Иначе single-source/raw-source bias будет возвращаться снова.

## 8.3. Что нужно заменить новым build layer

Текущий `_llm_fact_first_description_md(...)` должен быть разделён на family of runtime steps:

- `build_canonical_event_fact_floor`
- `shape_aware_layout_plan`
- `generate_blocks`
- `validate_and_targeted_repair`
- `derive_short_and_search_digest`

Это не обязательно означает точные имена функций, но означает точные роли.

## 8.4. Create path in `2.16.1`

Для нового события flow должен быть таким:

1. source arrives;
2. `source_scope_extract`;
3. `source_atomic_extract`;
4. записываем source facts;
5. строим initial canonical event fact floor;
6. собираем layout plan;
7. генерируем full event text;
8. валидируем и при необходимости repair;
9. deriving short/search;
10. persist.

## 8.5. Merge path in `2.16.1`

Для уже существующего события flow должен быть таким:

1. source arrives;
2. current Smart Update match/anchor/ticket logic decides event identity;
3. `source_scope_extract` работает с опорой на existing event context;
4. `source_atomic_extract` выделяет только event-specific facts из нового source;
5. факты сохраняются в `event_source_fact` со status;
6. canonical event fact floor пересобирается из union всех source facts события;
7. полный public text пересобирается заново из event-level floor;
8. derived fields refresh as needed.

Именно это сохраняет твой исходный fact-first принцип:

- не rewrite одного поста;
- а rebuild описания события из накопленного фактового набора.

## 8.6. Multi-event source handling

Это теперь explicit requirement.

Если один source/post содержит несколько событий:

- каждый candidate должен получить свой `source_scope_extract`;
- sibling facts не должны попадать в чужой event floor;
- quote/list/program fragments должны быть event-local.

Именно тут лежит починка digest-like failures.

## 9. Как использовать атомизацию из `2.15.5 - 2.15.10`, не ломая runtime

Главное правило:

- атомизация живёт на source extraction и block generation уровне;
- event accumulation spine остаётся общим.

То есть не надо выбирать между:

- "один giant prompt"
- и "каждый event всегда проходит 9 шагов".

Правильная схема:

- общий runtime layer один;
- common fact-first rebuild один;
- внутри source extraction и block generation существуют conditional shape-aware micro-contracts.

Практически это значит:

- `2673`, `2687`, `2734` сохраняют свои локально выигравшие `normalize_*` разрезы;
- новые shape families добавляются постепенно;
- но все они в итоге складываются в один canonical event fact floor и один rebuild pipeline.

## 10. Метрики `2.16.1` по шагам

Ниже фиксируются обязательные step-level metrics для следующих прогонов.

## 10.1. `source_scope_extract`

- `scope_fact_recall`
- `digest_bleed_rate`
- `quote_scope_precision`
- `scope_overcut_rate`

## 10.2. `source_atomic_extract`

- `slot_coverage`
- `fact_validity_rate`
- `list_item_retention`
- `quote_candidate_precision`
- `shape_route_accuracy`

## 10.3. `canonical_event_fact_floor`

- `cross_source_fact_gain`
- `retained_secondary_fact_rate`
- `dedupe_precision`
- `must_include_recall`
- `infoblock_bucket_precision`

## 10.4. `shape_aware_layout_plan`

- `lead_pattern_fit`
- `section_plan_recall`
- `layout_diversity_rate`
- `infoblock_exclusion_precision`

## 10.5. `generate_blocks`

- `lead_ok`
- `heading_quality`
- `list_integrity`
- `quote_grounding`
- `all_fact_coverage`
- `duplication_rate`

## 10.6. `validate_and_targeted_repair`

- `repair_trigger_precision`
- `repair_success_rate`
- `hallucination_escape_rate`
- `post_repair_missing_rate`
- `post_repair_extra_rate`

## 11. Обязательный evaluation suite для `2.16.1`

Следующий цикл нельзя считать валидным на 3 событиях.

Нужен обязательный suite из уже известных stress-shapes:

### 11.1. Quality carry / historical bar

- `2660` — sparse/compact
- `2673` — presentation/project
- `2687` — lecture_person / quote risk
- `2734` — program_rich / setlist / performer

### 11.2. Новые shape-batch cases

- `2659`
- `2747`
- `2701`
- `2732`
- `2759`
- `2657`

### 11.3. Реальные multi-source / enrichment cases

- `2731` — secondary source with playlist/support enrichment
- `2647` — multi-source lecture enrichment
- `2212` — secondary source adds lineup

### 11.4. Negative digest / bleed case

- `2498` — sibling schedule noise

Suite должен проверяться не только по одному "последнему лучшему тексту", а по step traces и fact-retention.

## 12. Что `2.16.1` прямо запрещает

- single-source final generation как основной semantic path;
- rebuild текста напрямую из `event_before.description + candidate.text`;
- forced two-paragraph default for all events;
- giant prompt with all rules always-on;
- prose-outline planner;
- blind full rewrite repair;
- patterning inside extraction/semantic core;
- loss of secondary-source facts;
- leakage of infoblock into narrative;
- acceptance of screening outputs without groundedness audit.

## 13. Definition of done for этой ветки

`2.16.1` считается successful только если одновременно выполняется всё ниже:

- Smart Update по-прежнему работает source-by-source;
- событие может быть как новым, так и уже существующим;
- новый source добавляет факты в общий event spine;
- итоговый текст пересобирается из canonical event fact floor;
- narrative содержит все publishable facts;
- infoblock остаётся отдельным слоем;
- layout снова вариативен и shape-aware;
- atomic micro-contracts доказуемо улучшают step metrics;
- multi-source enrichment и multi-event source scoping проходят обязательный suite.

## 14. Bottom line

`2.16.1` — это не отказ ни от Smart Update, ни от fact-first, ни от `LLM-first`, ни от pattern-driven ideas.

Это их правильная иерархия:

1. Smart Update как runtime orchestration.
2. Fact-first как единственный источник истины для public text.
3. `LLM-first` как semantic engine.
4. Pattern-driven как слой вариативной структуры и подачи.
5. Atomic micro-contracts как способ заставить Gemma работать устойчиво на разных shape.

Следующий практический шаг после этого brief:

- строить runtime-aware `2.16.1` harness;
- прогонять suite на single-source, multi-source и digest-cases;
- fine-tune каждый отдельный шаг по его метрикам, а не только по финальному тексту.
