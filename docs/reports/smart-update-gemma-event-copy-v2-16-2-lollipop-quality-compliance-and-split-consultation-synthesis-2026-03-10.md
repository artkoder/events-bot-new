# Smart Update Gemma Event Copy V2.16.2 Lollipop Quality Compliance And Split Consultation Synthesis

Дата: 2026-03-10

## 1. Inputs

- author brief: `artifacts/codex/tasks/smart-update-lollipop-v2-16-2-quality-compliance-and-split-consultation-brief-2026-03-10.md`
- compact `Opus` brief: `artifacts/codex/tasks/smart-update-lollipop-v2-16-2-quality-compliance-and-split-opus-brief-2026-03-10.md`
- fresh `Opus` result: `artifacts/codex/reports/smart-update-lollipop-v2-16-2-quality-compliance-and-split-consultation-opus-2026-03-10.raw.json`
- fresh `Gemini 3.1 Pro Preview` result: `artifacts/codex/reports/smart-update-lollipop-v2-16-2-quality-compliance-and-split-consultation-gemini-3.1-pro-preview-2026-03-10.raw.json`
- comparison corpus: `artifacts/codex/reports/smart-update-lollipop-v2-16-2-writer-final-iter2-vs-baseline-2026-03-10.md`
- prior canonical retune synthesis: `docs/reports/smart-update-gemma-event-copy-v2-16-2-lollipop-writer-retune-iter3-consultation-synthesis-2026-03-10.md`
- pipeline timing context: `docs/reports/smart-update-gemma-event-copy-v2-16-2-lollipop-pipeline-profiling-2026-03-10.md`

## 2. Shared Verdict

Оба свежих консультанта дают один и тот же high-level verdict:

- `overall_quality_verdict = mixed_but_salvageable`

Общий консенсус:

- `lollipop iter2` уже выигрывает по `no_logistics_leakage`, `list_handling`, groundedness и anti-clutter;
- главные незакрытые проблемы живут в `lead_strength`, `format_clarity`, `heading_structure`, `reader_scanability` и части `naturalness`;
- текущий regression class действительно reader-facing, а не только “внутренне архитектурный”.

Особенно устойчивый проблемный кластер:

- `2673` `Собакусъел`
- `2659` `Посторонний`
- `2747` `Киноклуб: «Последнее метро»`

Там оба консультанта видят одну и ту же поломку:

- title не даёт достаточного format anchor;
- lead открывается background/cast/content fact'ом;
- текст начинает звучать как справка о предмете, а не как анонс события.

## 3. Criteria Assessment

### Meets

- `groundedness`
- `list_handling`
- `no_logistics_leakage`
- `anti_cliche`

### Partial

- `format_clarity`
  - `Gemini` считает её ближе к `fails`;
  - `Opus` — к `partial`, потому что на self-explanatory titles она чаще держится.
- `coverage_completeness`
  - формального factual collapse нет;
  - но есть over-compression и потеря части useful detail balance.
- `naturalness`
  - не сломана полностью;
  - но тексты часто читаются как fact concatenation вместо editorial prose.

### Fails

- `lead_strength`
- `heading_structure`
- `reader_scanability`

Канонический practical read:

- baseline всё ещё часто ближе к target quality bar по headings, structure и reader navigation;
- `lollipop` уже сильнее по product-fit hygiene, но ещё не догнал по editorial readability.

## 4. Root Cause By Family

### `facts.extract`

Verdict: `none / minimal`

- ни `Opus`, ни `Gemini` не считают extraction главным источником текущих regressions;
- reopen имеет смысл только если следующий rerun покажет missing raw event-format anchors.

### `facts.dedup`

Verdict: `none`

- не выглядит text-shaping owner;
- split или redesign здесь не обещают reader-facing gains.

### `facts.merge`

Verdict: `minimal`

- `Opus` допускает, что generic cluster labels слегка ухудшают downstream framing;
- но не считает merge regression driver.

### `facts.prioritize`

Verdict: `primary`

- именно здесь ломается выбор первого anchor;
- stage слишком часто поднимает cast/background/content-detail fact вместо event-announcement fact;
- это главный owner `lead_strength` и значительной части `format_clarity`.

### `editorial.layout`

Verdict: `primary`

- heading loss и flattening rooted здесь;
- prompt/threshold слишком сжимают multi-theme cases;
- semantic shifts часто не получают отдельную structure.

### `writer_pack.compose`

Verdict: `secondary`

- не primary semantic bottleneck;
- но `Opus` отдельно указывает, что current structure handoff слишком advisory:
  - headings легко игнорируются;
  - нет explicit length floor;
  - prose register почти не калибруется.

### `writer.final_4o`

Verdict: `secondary`

- stage не должен быть главным owner semantic rescue;
- но он реально влияет на flat prose, dropped headings and over-compression;
- полезен как narrow carry for format disambiguation and connective prose.

## 5. Main Disagreement: Split Now Or Later

### `Gemini`

Более агрессивен к split-routing уже сейчас:

- `facts.prioritize.lead`: routed prompt variant for opaque-title cases;
- `editorial.layout.plan`: routed variants `compact_sparse` vs `semantic_split`.

### `Opus`

Более консервативен:

- считает, что текущая проблема всё ещё prompt-quality, а не architecture-first;
- рекомендует сначала unified retune + better examples + deterministic structural guardrail;
- split поднимать только если iter3 всё ещё провалит lead/heading metrics.

## 6. Canonical Decision

На этот раунд беру staged reading, а не “или Gemini, или Opus”.

### Immediate round: follow `Opus`-first safety line

Сейчас делать:

- `facts.prioritize`: strong prompt retune, без нового stage;
- `editorial.layout`: strong prompt retune, без split family;
- `writer_pack.compose`: усилить structure handoff и register guidance;
- `writer.final_4o`: оставить один stage, добавить narrow carry;
- `editorial.layout` safety guardrail на heading policy.

### Escalation rule: follow `Gemini` if unified retune is still insufficient

Если после rerun:

- `lead_strength` всё ещё failing `>3/12` events;
- или opaque-title cases всё ещё открываются wrong anchor;
- или heading recovery остаётся слишком слабым;

тогда уже justified:

- routed variant для `facts.prioritize.lead`;
- затем, при необходимости, routed variant для `editorial.layout.plan`.

То есть:

- **split не отвергнут**;
- но **не является первым безопасным ходом**, пока не исчерпан более дешёвый и менее рискованный unified retune.

## 7. Concrete Plan Of Action

### Phase 1. Safe retune without architectural split

1. Retune `facts.prioritize` prompt.
   - Добавить explicit rule:
     - lead must answer “что это за событие?”;
     - для opaque titles lead обязан назвать event format.
   - Добавить positive/negative examples:
     - screening;
     - lecture;
     - presentation.
   - Прямо запретить lead типа:
     - `Режиссёр фильма — ...`
     - `В главных ролях ...`
     - `Проект представляет собой ...`
     как opening for opaque-title cases.

2. Retune `editorial.layout` prompt.
   - Опустить `no-heading` threshold с `<=3` до `<=2` facts.
   - Передавать `event_type` прямо в layout input.
   - Добавить examples хороших headings:
     - screening: film/context/cast split;
     - lecture: topic/speaker split.

3. Add one deterministic structural guardrail in layout.
   - Если layout returned `heading_policy = none`, но fact count `>=5`, повышать policy до `add_minimal` и логировать warning.
   - Важно: guardrail не придумывает heading labels, только не даёт схлопнуть структуру до нуля.

4. Strengthen `writer_pack.compose`.
   - Делать structure imperative, а не advisory:
     - `follow this order`;
     - `keep ### headings exactly as shown`.
   - Добавить target length floor:
     - practical band `500-900 chars`;
     - не давать модели схлопывать rich cases в `~200-350 chars`.
   - Положить short register example в prompt pack.
   - Держать literal items внутри их target block, а не как хвост append-only.

5. Retune `writer.final_4o` via prompt pack only.
   - First sentence must remove format ambiguity when title opaque.
   - Avoid choppy sentence-per-fact style.
   - Weave related facts into connective editorial prose.
   - Не открывать second writer pass.

### Phase 2. Rerun and measure

6. Прогнать новый `iter3` на тех же `12` событиях.

7. Снять metrics:
   - `lead_strength` failures
   - `heading_structure` recovery count
   - `opaque-title lead correctness`
   - average length
   - `no_logistics_leakage`
   - list integrity
   - regressions on `2498`, `2657`, `2734`

### Phase 3. Conditional split only on evidence

8. Если после Phase 2:
   - `lead_strength` failing `>3/12`
   - или opaque-title cases всё ещё unstable
   тогда:
   - вводить routed variant в `facts.prioritize.lead`.

9. Если после этого headings всё ещё weak:
   - вводить routed variant в `editorial.layout.plan`.

10. Не split’ить:
   - `facts.extract`
   - `facts.dedup`
   - `facts.merge`
   - `writer.final_4o` draft/edit

## 8. Do Not Do

- не добавлять deterministic heading labels в Python;
- не вводить numeric `% of baseline` target;
- не делать second writer pass до исчерпания prompt retune;
- не раздувать schema contract в `writer_pack.compose` на этом шаге;
- не split’ить upstream families без hard evidence.

## 9. Final Practical Verdict

Задача с `Opus` теперь закрыта.

Его позиция полезно ужесточила план:

- quality gaps реальны;
- split everywhere не нужен;
- safest next move — сначала сильный retune `facts.prioritize + editorial.layout + writer_pack.compose + writer.final_4o`, с измеримыми stop/go metrics;
- split поднимать только как phase-2 escalation, если unified retune не закроет проблему.

Итого:

- **сейчас**: no new family split, but strong prompt/guardrail round;
- **после rerun**: data-driven decision on routed split for `facts.prioritize.lead`, then maybe `editorial.layout.plan`.
