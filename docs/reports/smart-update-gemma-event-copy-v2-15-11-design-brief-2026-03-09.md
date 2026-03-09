# Smart Update Gemma Event Copy V2.15.11 Design Brief

Дата: 2026-03-09

Основание:

- [v2.15.2 design brief](/workspaces/events-bot-new/docs/reports/smart-update-gemma-event-copy-v2-15-2-design-brief-2026-03-08.md)
- [v2.15.3 design brief](/workspaces/events-bot-new/docs/reports/smart-update-gemma-event-copy-v2-15-3-design-brief-2026-03-08.md)
- [v2.15.8 atomic shape batch](/workspaces/events-bot-new/docs/reports/smart-update-opus-gemma-event-copy-v2-15-8-atomic-shape-batch-2026-03-08.md)
- [v2.15.9 downstream assembly retune](/workspaces/events-bot-new/docs/reports/smart-update-opus-gemma-event-copy-v2-15-9-downstream-assembly-retune-2026-03-08.md)
- [v2.15.10 screening grounding retune](/workspaces/events-bot-new/docs/reports/smart-update-opus-gemma-event-copy-v2-15-10-screening-grounding-retune-2026-03-08.md)
- [Smart Event Update feature README](/workspaces/events-bot-new/docs/features/smart-event-update/README.md)
- [Fact-first Smart Update](/workspaces/events-bot-new/docs/features/smart-event-update/fact-first.md)

## 0. Как читать этот brief

`2.15.11` - это не новый architectural reset и не откат к baseline.

Это recovery-brief следующего раунда:

- сохранить выигрыш `2.15.5 - 2.15.10` по atomic extraction и shape-aware micro-contracts;
- вернуть и формально закрепить сильные стороны текущего runtime Smart Update;
- убрать главный пропуск последних прогонов: generation не должен смотреть только на один источник и не должен схлопываться в одинаковый two-paragraph prose.

Главная идея `2.15.11`:

- один общий event-level pipeline остаётся;
- extraction по-прежнему может быть shape-aware и дробным;
- но generation и layout обязаны работать уже по накопленному canonical fact floor всего события, а не по последнему source.

## 1. Что уже доказано и что оказалось сломано

### 1.1. Что реально доказано в `2.15.5 - 2.15.10`

Уже есть локально подтверждённые переносимые win:

- atomic normalization переносится между разными shape;
- для части кейсов именно дробление `normalize_*` шагов спасает factual coverage;
- screening требует отдельного groundedness gate;
- party / lecture / program-rich лучше работают через shape-aware micro-contracts, чем через один giant prompt.

Это нужно сохранить.

### 1.2. Что стало явным regressions cluster

Последние experimental downstream-кандидаты показали новый системный провал:

- generation начал жить как single-source prose rewrite;
- тексты стали слишком однотипными по форме;
- rich cases потеряли подзаголовки, списки и block-level variation;
- часть фактов из secondary sources не получает нормального пути в итоговый текст;
- часть логистики начала протекать из infoblock в narrative;
- screening и digest-sources показали, что без source scoping легко приходит sibling leakage.

То есть проблема уже не "как лучше написать один абзац", а "как собрать event-level public text из уже накопленного набора фактов без потери baseline behavior".

## 2. Непотеряемые инварианты `2.15.11`

### 2.1. Fact accumulation обязателен

Новое описание события должно строиться не из одного source text, а из event-level fact floor:

- все связанные `event_source_fact` по событию;
- legacy fact seed, если он реально нужен;
- event-level anchors и уже подтверждённые поля события;
- OCR-derived facts;
- source-specific enrichment, пришедший позже из второго, третьего или пятого источника.

Это non-negotiable.

### 2.2. Финальный текст должен содержать все публикуемые факты

Твоё правило остаётся жёстким:

- если факт относится к событию и пригоден для публичного narrative, он должен быть отражён в финальном тексте;
- если факт относится к infoblock, он не должен протекать в narrative только ради coverage score;
- если факт является list-like payload, нельзя "смыслово пересказать" его, потеряв элементы.

### 2.3. Description и infoblock должны снова быть разными слоями

Нужно сохранить baseline separation:

- `facts_infoblock` - дата, время, площадка, адрес, цена, билеты, регистрация, телефон, возраст, "Пушкинская карта", служебные media facts;
- `facts_text_clean` - содержательные narrative facts;
- `facts_drop` - CTA, хэштеги, промо-шум, sibling noise.

`2.15.11` не должен размывать это разделение.

### 2.4. Layout variation must come back

Описание не должно по умолчанию сводиться к двум абзацам.

Нужно вернуть baseline-capable layout vocabulary:

- лид;
- осмысленные `###` подзаголовки там, где они реально помогают;
- списки для line-up / setlist / program / object clusters;
- blockquote для реальной цитаты;
- редкое, функциональное выделение, а не uniform prose slab.

### 2.5. LLM-first сохраняется

Semantic decisions, wording, lead, hooks, section language и narrative form остаются в `LLM-first` path.

Deterministic слой допустим только как:

- scoping;
- bucketing;
- validation;
- format enforcement;
- guardrails against leakage/hallucination.

## 3. Live-data cases, которые новый brief обязан объяснить

### 3.1. Event `2731`, 2026-03-07

У события уже есть минимум два telegram source:

- основной анонс даёт формат вечеринки, организатора, длительность, условия участия;
- второй post даёт playlist-like enrichment и additional support fact (`50+` участников, Я.Музыка playlist).

Это canonical multi-source case:

- extraction по source может быть раздельным;
- generation должна брать union этих фактов на уровне события.

### 3.2. Event `2647`, 2026-03-05

У лекции несколько source с разными акцентами:

- один даёт базовую тему;
- другой расширяет профиль спикера;
- ещё один добавляет semantic scope исследования.

Здесь нужен не single-post rewrite, а accumulation + dedupe + stronger section planning.

### 3.3. Event `2212`, 2026-03-07

У концерта secondary source добавляет состав участников.

Это значит:

- `lineup/list facts` могут приходить не из первого источника;
- если generation не умеет работать по accumulated floor, будут теряться отдельные имена.

### 3.4. Event `2498`, 2026-03-10

В событие попал digest/schedule-like source с соседними спектаклями.

Это negative case:

- source scoping должен срезать sibling facts до event-specific slice;
- generation не должна брать в narrative общие schedule fragments только потому, что они лежат в одном посте.

## 4. Macro-architecture `2.15.11`

### Step A. `source_scope_extract`

Это source-level, а не event-level шаг.

Его задача:

- взять один новый source + OCR + event-specific slice этого source;
- извлечь атомарные факты только для текущего события;
- сохранить source-local provenance;
- не смешивать в одном шаге несколько событий из digest-поста.

Ключевой принцип:

- extraction может быть shape-aware и сильно дробным;
- но он всегда работает на одном source fragment.

Output contract минимум должен позволять хранить:

- `fact_text`
- `bucket_hint`
- `group_hint`
- `quote_candidate`
- `list_candidate`
- `shape_slot`
- `source_id`

### Step B. `canonical_event_fact_floor`

Это обязательный event-level слой, который нельзя пропускать.

Он собирается из всех связанных source-слоёв:

- `added`
- `duplicate`
- при необходимости legacy baseline facts

и строит canonical floor события с provenance-aware buckets:

- `title_floor`
- `lead_floor`
- `section_floor`
- `list_floor`
- `quote_floor`
- `infoblock_floor`
- `drop_floor`

Это и есть главный missing layer последних experimental batch-ов.

### Step C. `shape_aware_layout_plan`

Planner должен работать не по raw source, а по canonical event floor.

Он решает:

- какой lead pattern уместен;
- нужны ли `###` sections;
- нужен ли list block;
- нужна ли quote block;
- какие fact groups должны пойти в lead, section bodies, lists и infoblock;
- какие secondary-source facts являются обязательными coverage targets.

Planner output должен быть structural-only.

Примерно так:

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
  "excluded_infoblock_fact_ids": ["i1", "i2", "i3"]
}
```

### Step D. `generate_blocks`

Новый prose path не должен быть одним blob prompt.

Нужны block-level LLM steps:

- `generate_lead`
- `generate_section`
- `generate_list_intro` или strict `assemble_list_block`
- `integrate_quote_block`, если quote реально verified

Каждый шаг получает только свои fact ids и не имеет права тянуть новые сведения.

Ключевое отличие от `v2.15.9`:

- assembly остаётся более structured;
- но она не должна насильно flatten rich events до linear prose.

### Step E. `validate_and_targeted_repair`

Validation должна проверять не только `missing/extra`, но и layout contract:

- infoblock leakage;
- list integrity;
- quote grounding;
- heading compliance;
- sibling leakage markers;
- unsupported claims;
- shape-specific groundedness.

Repair остаётся узким:

- block-level;
- issue-specific;
- facts-backed;
- без full rewrite "на всякий случай".

## 5. Shape-aware micro-contracts остаются, но меняют место в архитектуре

`2.15.11` не отменяет локальные shape wins.

Они должны жить в двух местах:

- в `source_scope_extract`;
- в `shape_aware_layout_plan / generate_blocks`.

Текущий обязательный минимум shape library:

- `presentation_project`: subject -> agenda -> program
- `lecture_person`: cluster -> theme -> profiles
- `program_rich`: concept -> setlist -> performer -> stage
- `screening_card`: identity -> plot -> support
- `party_theme_program`: identity -> participation -> program
- `exhibition_context_collection`: theme -> objects -> signature/context
- `sparse_case`: identity -> core_reason -> support

Важно:

- shape-aware не значит single-template;
- universal part - это общий macro-cycle;
- variation идёт через small contracts и event-level planning.

## 6. Multi-source rules are first-class, not optional

### 6.1. Source-level extraction, event-level generation

Это главное правило `2.15.11`:

- каждый source обрабатывается отдельно;
- но итоговое описание строится только после накопления canonical event floor по всем source.

### 6.2. Secondary source facts must have an explicit path to surface

Нужно измерять и поддерживать:

- сколько новых narrative facts внёс secondary source;
- сколько из них дошли до canonical floor;
- сколько из них реально отражены в final narrative или осознанно ушли в infoblock.

### 6.3. Multi-event posts need event scoping before prose generation

Для digest/schedule posts pipeline обязан:

- сначала сделать event-specific slice;
- только потом извлекать facts;
- не разрешать quote/list/title candidates из соседних items попадать в текущий event floor.

### 6.4. Older stored facts cannot be erased by a new narrow source

Если ранний source дал setlist, lineup или object cluster, а поздний source его не повторил:

- эти facts не должны исчезать из canonical floor;
- новый generation pass обязан по-прежнему видеть их как coverage targets.

## 7. Baseline-aligned layout policy

### 7.1. Что должно вернуться как first-class output

- meaningful lead;
- осмысленные `###` подзаголовки для non-sparse cases;
- списки для program/list-heavy payload;
- blockquote только для verified quote;
- более богатая структура, чем uniform two-paragraph output.

### 7.2. Что нельзя делать

- не превращать любой rich case в 2 абзаца;
- не прятать line-up / setlist / object cluster в размытый prose summary;
- не таскать infoblock facts в body ради формального coverage;
- не добавлять unsupported world knowledge;
- не придумывать citation-like hook без source proof.

### 7.3. Title и description снова должны быть разведены

Новый brief не требует вставлять event title в `description` как heading.

Наоборот:

- title остаётся отдельным grounded field;
- layout richness внутри `description` обеспечивается подзаголовками, списками и block-level planning;
- title-update logic должна оставаться source-grounded и жить отдельно от copy generation.

## 8. Step-level metrics for the next run

### 8.1. `source_scope_extract`

- `source_slot_coverage`
- `source_slice_precision` для multi-event posts
- `list_item_preservation`
- `quote_candidate_precision`
- `unsupported_source_claims`

### 8.2. `canonical_event_fact_floor`

- `cross_source_fact_gain`
- `retained_secondary_fact_rate`
- `duplicate_precision`
- `conflict_precision`
- `bucket_accuracy`

### 8.3. `shape_aware_layout_plan`

- `plan_fact_coverage`
- `list_trigger_recall`
- `quote_gate_precision`
- `infoblock_exclusion_precision`
- `format_plan_compliance`

### 8.4. `generate_blocks`

- `lead_ok`
- `section_coverage`
- `list_integrity`
- `quote_fidelity`
- `unsupported_claims`
- `infoblock_leak_count`
- `layout_diversity_flag`

### 8.5. `validate_and_targeted_repair`

- `repair_needed_rate`
- `repair_success_rate`
- `repair_drift_count`
- `post_repair_missing`
- `post_repair_extra`

### 8.6. Event-level final score

`2.15.11` не считается win, если одновременно не выполнено:

- `public_missing = 0` или explainable near-zero only on explicitly deferred facts;
- `unsupported_claims = 0`;
- `infoblock_leak = 0`;
- `digest_bleed_rate = 0`;
- `list_item_loss = 0` для list-bearing cases;
- rich cases не схлопнулись в uniform two-paragraph prose.

## 9. Evaluation suite for the next experimental round

Следующий обязательный suite должен включать не только single-source transfer cases, но и multi-source accumulation cases.

Минимальный набор:

- `2673` - presentation/project
- `2687` - lecture/person
- `2734` - program-rich
- `2659` - screening
- `2747` - screening
- `2701` - party/theme
- `2732` - party/theme
- `2759` - exhibition/context
- `2731` - multi-source playlist enrichment
- `2647` - multi-source lecture enrichment
- `2212` - multi-source line-up enrichment
- `2498` - digest leakage negative case

То есть следующий round обязан проверять и:

- переносимость shape-aware steps;
- accumulation из нескольких sources;
- защиту от sibling leakage.

## 10. Что нельзя делать в `2.15.11`

- не возвращаться к giant prompt;
- не строить final description только по последнему source;
- не подменять event-level fact floor прямым rewrite из source text;
- не выкидывать baseline buckets `text_clean / infoblock / drop`;
- не делать deterministic editorial rewrites как core logic;
- не считать успехом красивый текст, если он потерял факты;
- не считать успехом полный coverage, если логистика утекла в body;
- не считать solved screening без grounding audit;
- не терять lists и block structure ради "универсального" assembly.

## 11. Bottom line

`2.15.11` - это brief не про новый стиль ради стиля.

Это brief про правильную сборку системы:

- source-level extraction;
- event-level canonical accumulation;
- shape-aware layout planning;
- block-level generation;
- baseline-aligned separation narrative vs infoblock;
- per-step metrics;
- factual completeness across one source и many sources.

Если коротко, следующий кандидат должен быть "нечто среднее" между текущим runtime bot и уже найденными atomic wins:

- baseline дает нам правильную event model и факт-накопление;
- atomic pipeline даёт нам более сильные micro-contracts;
- `2.15.11` должен соединить эти две линии в один universal event-copy flow.
