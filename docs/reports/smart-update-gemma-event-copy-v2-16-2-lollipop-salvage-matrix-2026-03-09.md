# Smart Update Gemma Event Copy V2.16.2 Lollipop Salvage Matrix

Дата: 2026-03-09

Основание:

- [v2.16.2 lollipop funnel design brief](/workspaces/events-bot-new/docs/reports/smart-update-gemma-event-copy-v2-16-2-lollipop-funnel-design-brief-2026-03-09.md)
- [v2.15.5 atomic tuning 2673](/workspaces/events-bot-new/docs/reports/smart-update-opus-gemma-event-copy-v2-15-5-atomic-step-tuning-event-2673-2026-03-08.md)
- [v2.15.6 atomic tuning 2687](/workspaces/events-bot-new/docs/reports/smart-update-opus-gemma-event-copy-v2-15-6-atomic-step-tuning-event-2687-2026-03-08.md)
- [v2.15.7 atomic tuning 2734](/workspaces/events-bot-new/docs/reports/smart-update-opus-gemma-event-copy-v2-15-7-atomic-step-tuning-event-2734-2026-03-08.md)
- [v2.15.8 atomic shape batch](/workspaces/events-bot-new/docs/reports/smart-update-opus-gemma-event-copy-v2-15-8-atomic-shape-batch-2026-03-08.md)
- [v2.15.9 downstream assembly retune](/workspaces/events-bot-new/docs/reports/smart-update-opus-gemma-event-copy-v2-15-9-downstream-assembly-retune-2026-03-08.md)
- [v2.15.10 screening grounding retune](/workspaces/events-bot-new/docs/reports/smart-update-opus-gemma-event-copy-v2-15-10-screening-grounding-retune-2026-03-08.md)
- [ice-cream iter2](/workspaces/events-bot-new/docs/reports/smart-update-gemma-event-copy-v2-16-1-ice-cream-duel-iter2-2026-03-09.md)
- [ice-cream iter3](/workspaces/events-bot-new/docs/reports/smart-update-gemma-event-copy-v2-16-1-ice-cream-duel-iter3-2026-03-09.md)
- [Smart Update feature docs](/workspaces/events-bot-new/docs/features/smart-event-update/README.md)

## 1. Назначение матрицы

Эта матрица фиксирует, что именно должно быть перенесено в `lollipop stage-bank` из:

- baseline runtime;
- поздних atomic rounds;
- `ice-cream`.

Важно:

- матрица не переносит старые ветки целиком;
- переносится только то, что совместимо с funnel-архитектурой;
- writer-loop и repair-loop сюда не включаются как core;
- матрица одновременно служит:
  - inventory сильных carry;
  - seed-bank blueprint;
  - checklist для консультаций.

## 2. Правила salvage

- Переносим только то, что помогает `scope / fact extraction / fact typing / selection / layout pack`.
- Не переносим старые writer-path как финальный public-text core.
- Не переносим старые repair-loop как обязательный runtime-mainline.
- Если стадия давала локальный factual win, но prose win не давала, это всё равно может быть хороший salvage для upstream funnel.
- Если стадия давала красивый текст, но через hallucination / modality drift / world knowledge bleed, она фиксируется только как negative knowledge.
- Для `lollipop v1` не предполагается hard routing как core-mechanism:
  - устойчивость должна достигаться через broad candidate generation;
  - затем через `select / merge / priority`;
  - routing допустим только как поздняя optimization, если broad-run уже доказанно сильнее baseline.

## 3. Матрица salvage

| Source round | Old stage / version | Local evidence of strength | What exactly is salvaged | New lollipop stage family | New seed id | Quality bar impact | Carry status |
|---|---|---|---|---|---|---|---|
| baseline runtime | infoblock rendering | production-stable, no user complaints | готовая логика инфоблока, separation from description | `pack / render` | `infoblock.baseline.v1` | ясность, без сервисного мусора, functional formatting | mandatory |
| baseline runtime | list preservation rules | production-stable, tested | literal list retention, order preservation, compact program lists | `facts.extract` + `pack.compose` | `lists.baseline.v1` | полное покрытие фактов, списки там где нужны | mandatory |
| baseline runtime | markdown heading / blockquote safety | production-stable | Telegraph-safe heading/list/blockquote handling | `layout.plan` + `render` | `format.baseline.v1` | подзаголовки, readability, no broken layout | mandatory |
| v2.15.5 | `subject_v1_strict` | dominant subject recovered on 2673 | dominant-subject-first extraction without weak frame | `facts.extract` | `facts.extract_subject.v1` | ясный lead substrate, groundedness | seed |
| v2.15.5 | `agenda_v2_prose_ready` | better reason/structure/opportunities coverage on 2673 | prose-ready agenda facts, not weak noun phrases | `facts.extract` | `facts.extract_agenda.v1` | полнота фактов, ясность, less bureaucracy | seed |
| v2.15.5 | `program_v1_compact` | secondary program preserved on 2673 | compact secondary-program extraction | `facts.extract` | `facts.extract_program.v1` | fact completeness, list readiness | seed |
| v2.15.5 | `plan_v1_basic` | lead/body split locally useful | idea of structural split only, not prose planning | `layout.plan` | `layout.split_basic.v1` | strong lead support, section logic | carry as concept |
| v2.15.6 | `cluster_v2_named_group` | all six names preserved on 2687 | grouped person-cluster extraction without fragmentation | `facts.extract` | `facts.extract_cluster.v1` | grounded, full facts, non-template person-led entry | seed |
| v2.15.6 | `theme_v1_compact` | contribution + british roots compactly preserved | theme extraction separate from profiles | `facts.extract` | `facts.extract_theme.v1` | ясность, strong thematic lead support | seed |
| v2.15.6 | `profiles_v1_literal` | all profile facts preserved | literal profile micro-facts | `facts.extract` | `facts.extract_profiles.v1` | fullness, groundedness, anti-hallucination | seed |
| v2.15.7 | `concept_v1_compact` | love-story + program-core preserved | compact concept extraction | `facts.extract` | `facts.extract_concept.v1` | strong lead substrate, concept clarity | seed |
| v2.15.7 | `setlist_v1_grouped` | setlist preserved literally | grouped setlist extraction | `facts.extract` | `facts.extract_setlist.v1` | attendee-useful facts, list quality | seed |
| v2.15.7 | `performer_v1_awards` | performer + laureate facts restored | identity performer card extraction | `facts.extract` | `facts.extract_performer.v1` | strong lead, professional tone, grounded identity | seed |
| v2.15.7 | `stage_v2_compact` | Muse/dancer detail recovered without embellishment | compact stage-image extraction | `facts.extract` | `facts.extract_stage.v1` | vivid but grounded details | seed |
| v2.15.8 | `normalize_card_v1` | screening metadata extraction win | screening metadata card extraction | `facts.extract` | `facts.extract_card.v1` | groundedness on screenings | seed |
| v2.15.8 | `extract_plot_v1` | useful extraction but later prose-risk | keep only as deferred experiment outside `v1` | `facts.extract` | `facts.extract_plot_support.v1` | possible hook support, but too risky for `v1` | deferred |
| v2.15.8 | `normalize_support_v1` | support facts recovered | separate support/visitor-condition extraction | `facts.extract` | `facts.extract_support.v1` | completeness, infoblock support, no service loss | seed |
| v2.15.8 | `normalize_identity_v2_strict` | party identity extracted cleanly | strict identity extraction for themed events | `facts.extract` | `facts.extract_identity.v1` | ясность, grounded opening | seed |
| v2.15.8 | `normalize_participation_v1` | participation rules extracted | participation/visitor framing extraction | `facts.extract` | `facts.extract_participation.v1` | clarity, useful visitor info, anti-filler | seed |
| v2.15.8 | `normalize_program_v1` | program fragments preserved | program extraction for party/event shapes | `facts.extract` | `facts.extract_program_shape.v1` | lists where needed, full facts | seed |
| v2.15.9 | deterministic routing idea | helped separate downstream assembly concerns | keep only block-separation lesson; do not carry route-first logic into `v1` mainline | `layout.plan` + `pack.compose` | `layout.separate_blocks.v1` | structure, anti-leakage, clarity | carry as concept only |
| v2.15.9 | strict assemble lead/body idea | useful as separation discipline | preserve separation idea, not prose generator itself | `pack.compose` | `pack.separate_blocks.v1` | strong lead support, section discipline | carry as concept |
| v2.15.10 | screening grounding lesson | proved world-knowledge risk | screening metadata must stay narrow and grounded | `facts.type` + `pack.compose` | `screening.metadata_guard.v1` | grounded, anti-hallucination | mandatory for screenings |
| ice-cream iter2 | `normalize_fact_floor` family | improved aggregate fact prep | normalized fact-floor family itself | `facts.merge / normalize` | `facts.normalize_floor.v1` | groundedness, full facts, cleaner pack | seed |
| ice-cream iter2 | shape-aware buckets | list/logistics separation worked | `narrative_facts / list_facts / logistics_facts` bucket logic | `facts.type` | `facts.type_buckets.v1` | no service leakage, lists preserved | seed |
| ice-cream iter2 | stage traces / profiling | useful for pinpoint tuning | prompt/result trace discipline | `research infra` | `trace.stage_profile.v1` | iterative tuning discipline | mandatory |
| ice-cream iter3 | anti-pattern knowledge for `party_theme_program` | explicit unsupported claims isolated | negative knowledge about modality drift and experience-promise failure | `hook / pattern / writer guard` | `neg.party_modality.v1` | grounded, anti-cliche, anti-hallucination | mandatory negative carry |
| ice-cream iter3 | anti-pattern knowledge for `theater_history` | interpretive drift isolated | negative knowledge about poster pollution and immersive reframe | `facts.type` + `hook` | `neg.theater_interpretive.v1` | grounded, non-template, no semantic drift | mandatory negative carry |
| ice-cream iter3 | anti-pattern knowledge for `screening_card` | screening must be metadata-first | screening narrative must be tightly constrained | `pack.compose` | `neg.screening_world_knowledge.v1` | grounded, professional, no world knowledge bleed | mandatory negative carry |

## 4. Что НЕ переносится как core

| Source round | Old component | Why not carried as core |
|---|---|---|
| v2.15.8 | `generate_lead` / `generate_body` | prose quality was not a transferable win |
| v2.15.8 | `repair_v1` | no stable local win |
| ice-cream | `generate_narrative_core` | still produces modality / scope drift on sensitive shapes |
| ice-cream | `audit_narrative_core -> repair_narrative_core` mainline | structural contract gap, not acceptable as core |
| v2.15.8 / v2.15.10 | screening prose branch | repeatedly unsafe or overcreative for screenings |

## 5. Quality Bar -> Required Seed-Bank Coverage

Ниже уже не история, а инженерное соответствие: какие stage families должны существовать в `seed-bank`, если мы реально хотим обеспечить quality bar из `lollipop brief`.

| Quality requirement | Mandatory seed-bank stage families | Why these stages are required |
|---|---|---|
| естественный | `hook.seed.*`, `hook.select.*`, `pattern.signal.*`, `layout.plan.*`, `pack.compose.*`, `writer.final_4o.*` | естественность не появляется из raw facts, нужен controlled presentation layer |
| профессиональный | `facts.extract_identity.*`, `facts.extract_theme.*`, `facts.type_buckets.*`, `pack.compose.*` | профессиональность требует чистых identity/theme packs и separation of prose vs logistics |
| нешаблонный | `hook.seed.*`, `hook.select.*`, `pattern.signal.*`, `pack.select.*` | вариативность невозможна без reusable hook/pattern library and structured presentation choice |
| grounded | `scope.*`, `facts.extract_*`, `facts.type_buckets.*`, `screening.metadata_guard.*`, negative carries from `ice-cream` | groundedness зависит от качества upstream filtering, не только от writer-а |
| полный по фактам | `facts.extract_*`, `facts.merge.tier1.*`, `facts.merge.tier2.*`, `facts.type_buckets.*`, `facts.priority.*`, `lists.baseline.v1`, `pack.compose.*` | факт-потери происходят до writer-а; списки и setlist требуют отдельных extraction contracts |
| ясный | `facts.extract_agenda.*`, `facts.extract_theme.*`, `layout.plan.*`, `pack.compose.*` | ясность требует prose-ready claims, а не raw noun fragments |
| вариативный | `hook.seed.*`, `hook.select.*`, `pattern.signal.*`, `pack.select.*` | без этого текст станет монотонным baseline-mold |
| с сильным лидом | `hook.seed.direct.*`, `hook.seed.person.*`, `hook.seed.program.*`, `hook.select.*`, `layout.split_basic.*`, `writer.final_4o.spec.*` | strong lead needs a dedicated seed-and-select layer and an explicit writer contract |
| с осмысленными подзаголовками | `layout.plan.*`, `format.baseline.v1` | headings should be chosen structurally, then rendered safely |
| со списками там, где они реально нужны | `facts.extract_setlist.*`, `facts.extract_program.*`, `lists.baseline.v1`, `layout.plan.*` | list worthiness must be preserved upstream, not invented late |
| без AI clichés | `hook.select.*`, `pattern.signal.*`, `pack.select.*`, negative knowledge bank, `writer.final_4o.spec.*` | clichés are mostly presentation-layer failures, but they must be filtered by seed-bank design |
| без сервисного мусора | `facts.type_buckets.*`, `infoblock.baseline.v1`, `pack.compose.*` | service leakage is a typing / packing failure |
| без бюрократического prose drift | `facts.extract_agenda.*`, `facts.extract_theme.*`, `hook.seed.*`, `hook.select.*`, `pattern.signal.*`, `pack.select.*`, `writer.final_4o.*` | bureaucracy grows from weak claims plus flat writer framing |

## 6. Что это значит practically для seed-bank v1

`seed-bank v1` нельзя ограничить только extraction-family.

И `seed-bank v1` не означает `по одной версии на family`.

Для `lollipop v1` нормой считается:

- несколько сильных version-candidates внутри одной stage-family;
- их broad-run на одном и том же событии;
- затем `select / merge / priority`;
- и только потом переход к следующей стадии.

Минимальный обязательный состав:

- `scope`
- `facts.extract`
- `facts.merge.tier1`
- `facts.merge.tier2`
- `facts.type`
- `facts.priority`
- `hook.seed`
- `hook.select`
- `pattern.signal`
- `layout.plan`
- `pack.compose`
- `pack.select`
- `writer.final_4o.spec`
- `writer.final_4o`
- `trace.stage_profile`

Если из первой версии убрать:

- `hook` слой,
- `layout.plan`,
- `pack.select`,
- или `writer.final_4o.spec`,

то quality bar из brief будет заранее недостижим даже при хороших facts.

Если же слишком рано заменить broad-run на hard routing, то `v1` снова станет хрупким и будет системно проигрывать на длинном production tail разнообразных source-постов.

## 7. Open risks already visible from the matrix

- `extract_plot_v1` не должен входить в `v1`; его нужно держать только как later experiment.
- `ice-cream` anti-pattern knowledge надо хранить как first-class bank, а не как заметки в отчётах.
- `hook.seed`, `pattern.signal`, `pack.select` и `writer.final_4o.spec` пока не имеют поздних локально выигравших версий такой же силы, как extraction family.
  Значит эти stage families придётся исследовать почти с нуля, но уже на хорошем upstream pack.
- `writer.final_4o` не должен компенсировать пробелы matrix; если pack плохой, writer only hides the problem.

## 8. Next step from this matrix

Следующий инженерный шаг:

1. На базе этой матрицы собрать `stage registry`.
2. Для каждого `seed id` завести:
   - canonical prompt file
   - version id
   - known source round
   - case evidence
3. Отдельно собрать узкий consultation round:
   - хватает ли stage families в `seed-bank` для quality bar;
   - какие stage families missing;
   - какие seed ids слишком слабые и требуют immediate replacement.
4. После консультации пересобрать `seed-bank v1` с:
   - `facts.merge.tier1/tier2`
   - `facts.priority`
   - `hook.seed`
   - `pattern.signal`
   - `pack.select`
   - `writer.final_4o.spec`
