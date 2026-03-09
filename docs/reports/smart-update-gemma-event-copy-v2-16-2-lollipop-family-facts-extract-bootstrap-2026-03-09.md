# Smart Update Gemma Event Copy V2.16.2 Lollipop Family Bootstrap - `facts.extract`

Дата: 2026-03-09

Основание:

- [lollipop funnel design brief](/workspaces/events-bot-new/docs/reports/smart-update-gemma-event-copy-v2-16-2-lollipop-funnel-design-brief-2026-03-09.md)
- [lollipop salvage matrix](/workspaces/events-bot-new/docs/reports/smart-update-gemma-event-copy-v2-16-2-lollipop-salvage-matrix-2026-03-09.md)
- [stage-bank bootstrap](/workspaces/events-bot-new/docs/reports/smart-update-gemma-event-copy-v2-16-2-lollipop-stage-bank-bootstrap-2026-03-09.md)
- [machine-readable stage bank bootstrap](/workspaces/events-bot-new/artifacts/codex/stage_bank/smart_update_lollipop_stage_bank_v2_16_2_2026-03-09.json)

## 1. Зачем эта family первая

`facts.extract` идёт первой, потому что:

- именно здесь уже есть самый сильный salvage из `2.15.5+`;
- именно здесь у нас больше всего локально выигравших atomic versions;
- без зрелой extraction-family весь downstream `facts.type -> hook -> pack` будет строиться на грязном материале;
- если `lollipop` не сможет стабильно вытаскивать facts лучше, чем предыдущие ветки, дальше идти бессмысленно.

## 2. Стартовый режим для family

Для `facts.extract` на старте действует жёсткое правило:

- не делаем hard routing;
- не схлопываем family до одной версии;
- сначала запускаем broad candidate bank;
- все outputs сохраняем до какого-либо merge/select;
- только после этого делаем family review.

То есть family сейчас проектируется как multi-version bank, а не как “выберем один extractor на shape”.

Стартовый рабочий кейсбук для этого family-lab:

Core `6`:

- `2673`
- `2687`
- `2734`
- `2659`
- `2731`
- `2498`

Extension `6`:

- `2747`
- `2701`
- `2732`
- `2759`
- `2657`
- `2447`

## 3. Стартовый candidate bank

### 3.1. Active candidates

| Stage id | Old source stage | Source round | Local evidence | Hypothesized fit |
|---|---|---|---|---|
| `facts.extract_subject.v1` | `subject_v1_strict` | `2.15.5` | local win on `2673` | single dominant person / performer / speaker |
| `facts.extract_agenda.v1` | `agenda_v2_prose_ready` | `2.15.5` | local win on `2673` | forward-looking agenda / what-will-happen posts |
| `facts.extract_program.v1` | `program_v1_compact` | `2.15.5` | local win on `2673` | compact program / sequence-bearing posts |
| `facts.extract_cluster.v1` | `cluster_v2_named_group` | `2.15.6` | local win on `2687` | grouped person-cluster lecture/history posts |
| `facts.extract_theme.v1` | `theme_v1_compact` | `2.15.6` | local win on `2687` | theme/context-forward cultural posts |
| `facts.extract_profiles.v1` | `profiles_v1_literal` | `2.15.6` | local win on `2687` | literal multi-person micro-facts |
| `facts.extract_concept.v1` | `concept_v1_compact` | `2.15.7` | local win on `2734` | concept-led program-rich posts |
| `facts.extract_setlist.v1` | `setlist_v1_grouped` | `2.15.7` | local win on `2734` | setlist / titles that must survive literally |
| `facts.extract_performer.v1` | `performer_v1_awards` | `2.15.7` | local win on `2734` | performer-led sources with identity/award facts |
| `facts.extract_stage.v1` | `stage_v2_compact` | `2.15.7` | local win on `2734` | grounded stage-image or support-performance details |
| `facts.extract_card.v1` | `normalize_card_v1` | `2.15.8` | local win on screening shape | metadata-first screening posts |
| `facts.extract_support.v1` | `normalize_support_v1` | `2.15.8` | local win on screening/support cases | support details, visitor conditions, secondary enrichment |
| `facts.extract_identity.v1` | `normalize_identity_v2_strict` | `2.15.8` | local win on party shape | identity-led theme/program posts |
| `facts.extract_participation.v1` | `normalize_participation_v1` | `2.15.8` | local win on party/exhibition shape | visitor participation framing |
| `facts.extract_program_shape.v1` | `normalize_program_v1` | `2.15.8` | local win on party/exhibition shape | dispersed program fragments and list-worthiness |

### 3.2. Deferred outside `v1`

| Stage id | Old source stage | Why deferred |
|---|---|---|
| `facts.extract_plot_support.v1` | `extract_plot_v1` | too risky for `v1`, later experiment only |

## 4. Что важно про fit

Управляющая логика здесь не “роль stage внутри абстрактной taxonomy”, а реальные типы исходных постов и событий.

То есть для family важны:

- тип source-поста;
- плотность программы;
- есть ли grouped persons;
- dominant subject vs distributed concept;
- forward-looking promises vs literal metadata;
- есть ли support/logistics enrichment;
- multi-source enrichment vs single-source summary.

Именно эти признаки со временем должны показать:

- какие версии выигрывают регулярно;
- какие версии выигрывают только на отдельных типах входа;
- какие версии не стоят цены broad-run.

На старте это ещё hypotheses, а не rules.

## 5. Что нужно получить в первом реальном family-lab раунде

По `facts.extract` нужно собрать не итоговый текст, а raw evidence:

1. Все candidate outputs по каждому stage id.
2. Все outputs до merge/select.
3. Сопоставление output с raw facts каждого кейса.
4. Видимые losses:
   - missing facts
   - over-fragmentation
   - fact duplication
   - weak wording
   - support leakage into wrong extractor
5. Сравнение внутри family:
   - какие версии дают локальный win;
   - какие версии комплементарны;
   - какие версии слабые и должны быть сняты;
   - где family надо расширить ещё одной версией prompt-а.

## 6. Что будет считаться хорошим итогом family review

После первого настоящего `facts.extract family lab` должны появиться:

- shortlist versions для регулярного broad-run;
- first rejected versions;
- список missing prompt variants;
- понимание, какие event/source traits реально коррелируют с win;
- только после этого проект `facts.merge.tier1/tier2`.

То есть success criterion на этом этапе не “написан хороший текст”, а:

- extraction-family стала зрелее;
- стало понятно, что именно потом надо мержить;
- стало видно, какие версии реально достойны жить в регулярном режиме.

## 7. Что ещё не сделано в этом bootstrap-документе

Здесь пока нет:

- канонических prompt files по каждому extractor;
- raw output matrix по кейсбуку;
- консультации по самой family;
- shortlist verdict after rerun.

Этот документ фиксирует именно стартовую форму family и то, как она будет исследоваться дальше.

## 8. Следующий шаг

Следующий рабочий раунд по `facts.extract` должен сделать уже операционную часть:

1. Вытащить raw prompt inventory для active candidates.
2. Прогнать family broad-run по стартовому casebook.
3. Сохранить outputs до merge/select.
4. Сделать отдельный family review report.
5. Только потом идти в узкий consultation round по этой family.
