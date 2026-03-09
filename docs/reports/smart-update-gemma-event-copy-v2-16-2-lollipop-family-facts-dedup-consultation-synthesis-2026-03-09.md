# Smart Update Gemma Event Copy V2.16.2 Lollipop Facts.Dedup Consultation Synthesis

Дата: 2026-03-09

## 1. Scope

Этот synthesis закрывает два последовательных вопроса по новой family `facts.dedup`:

1. какой должна быть сама архитектура `facts.dedup`;
2. что показал первый реальный pilot-run и можно ли уже идти в `facts.merge`.

Материалы раунда:

- design brief: `/workspaces/events-bot-new/artifacts/codex/tasks/smart-update-lollipop-v2-16-2-facts-dedup-family-design-consultation-brief-2026-03-09.md`
- design `Opus`: `/workspaces/events-bot-new/artifacts/codex/reports/smart-update-lollipop-v2-16-2-facts-dedup-family-design-consultation-opus-2026-03-09.md`
- design `Gemini` single-launch stderr: `/workspaces/events-bot-new/artifacts/codex/reports/smart-update-lollipop-v2-16-2-facts-dedup-family-design-consultation-gemini-3.1-pro-preview-2026-03-09.stderr.txt`
- dedup harness: `/workspaces/events-bot-new/artifacts/codex/smart_update_lollipop_facts_dedup_family_v2_16_2_2026_03_09.py`
- dedup lab report: `/workspaces/events-bot-new/docs/reports/smart-update-gemma-event-copy-v2-16-2-lollipop-family-facts-dedup-lab-2026-03-09.md`
- post-run brief: `/workspaces/events-bot-new/artifacts/codex/tasks/smart-update-lollipop-v2-16-2-facts-dedup-family-postrun-consultation-brief-2026-03-09.md`
- post-run `Opus` retry1: `/workspaces/events-bot-new/artifacts/codex/reports/smart-update-lollipop-v2-16-2-facts-dedup-family-postrun-consultation-opus-2026-03-09-retry1.md`
- post-run `Gemini` single-launch stderr: `/workspaces/events-bot-new/artifacts/codex/reports/smart-update-lollipop-v2-16-2-facts-dedup-family-postrun-consultation-gemini-3.1-pro-preview-2026-03-09.stderr.txt`

## 2. What Changed After Design Consultation

Главный design-turn произошёл ещё до первого run.

Изначальная внутренняя гипотеза была:

- `facts.dedup.normalize`
- `facts.dedup.cluster`
- `facts.dedup.review`

`Opus` design-review возразил против этой схемы и предложил:

- `facts.dedup.baseline_diff`
- `facts.dedup.cross_enrich`
- `facts.dedup.audit`

Сильный довод `Opus`:

- global clustering слишком тяжёл и неустойчив для `Gemma`;
- normalizing claims до dedup создаёт риск rewrite/provenance loss;
- baseline должен быть anchor, а не равный кандидат;
- dedup должен отвечать не на вопрос “как красиво объединить facts”, а на вопрос “что уже покрыто baseline, что реально новое, а что только reframe”.

Я принял этот structural turn и перестроил harness до первого run.

## 3. Design Verdict

По design-stage usable verdict есть только от `Opus`.

`Gemini` был запущен ровно один раз, но usable report не вернул: provider ответил `429 MODEL_CAPACITY_EXHAUSTED`.

Что считаю подтверждённым после design-stage:

- `baseline_diff -> cross_enrich -> audit` сильнее, чем `normalize -> cluster -> review`;
- `baseline_diff` должен быть коротким per-stage diff prompt;
- `cross_enrich` должен быть узким и pairwise, без global clustering;
- `audit` должен быть deterministic structural check, а не LLM self-review;
- `theme` не нужен в первом pilot-run и может идти отдельным challenger round позже.

## 4. Real Pilot Outcome

Pilot-run был запущен уже на новой архитектуре:

- events: `2673, 2687, 2734, 2447`
- shortlist inputs:
  - `subject`
  - `card`
  - `agenda`
  - `support`
  - `performer`
  - `participation`
  - `stage`
- `theme` intentionally excluded

Агрегаты:

```json
{
  "events": 4,
  "avg_total_extraction_facts": 30.0,
  "avg_enrichment_count": 3.25,
  "avg_reframe_count": 0.0,
  "avg_cross_duplicates": 0.0,
  "avg_unique_enrichment_final": 3.25,
  "events_with_flags": 1
}
```

Сильные стороны pilot:

- `100%` fact accounting на всех четырёх событиях;
- pipeline укладывается в Gemma TPM;
- реальные enrichment facts выживают:
  - setlist/program details,
  - performer credibility details,
  - workshop artist details,
  - ticket/logistics specifics.

Слабые стороны pilot:

- `reframe_count = 0` на всех событиях;
- `cross_enrich` почти пуст;
- есть `missing_baseline_match` на `2673`;
- enrichment count выглядит слишком ровно между очень разными event shapes.

## 5. Post-Run Verdict

Post-run usable verdict снова есть от `Opus`.

Первый post-run `Opus` response оказался unusable: вместо diagnosis модель попыталась эмулировать tool-calls. Это был явный failure state, поэтому был сделан один допустимый retry в ещё более жёстком `no-tools` framing. Retry дал usable report.

`Gemini` post-run снова был запущен ровно один раз и снова не вернул usable report из-за `429 MODEL_CAPACITY_EXHAUSTED`.

Главный post-run verdict `Opus`:

- архитектура `facts.dedup` правильная;
- менять family назад на clustering не нужно;
- current pilot для перехода в `facts.merge` пока **NO-GO**;
- bottleneck сидит не в architecture, а в `baseline_diff` prompt precision.

## 6. What Is Confirmed Now

Подтверждено:

- `facts.dedup` как отдельная family нужна и жизнеспособна;
- `baseline_diff-first` лучше clustering-first для `Gemma`;
- `cross_enrich` должен быть страховочным, а не главным dedup engine;
- `facts.merge` пока рано запускать на текущем output.

Неподтверждено:

- что текущий `baseline_diff.v1` корректно различает `COVERED` и `REFRAME`;
- что `baseline_diff.v1` не перекидывает часть enrichments в `COVERED`;
- что текущий `cross_enrich` действительно “лишний”, а не просто starving из-за over-covering upstream.

## 7. Exact Next Step

Следующий ход теперь должен быть не `facts.merge`, а узкий tuning round именно на `facts.dedup.baseline_diff.v1`.

Нужно исправить:

- явное различение `COVERED` vs `REFRAME`;
- hard requirement на `baseline_match` для `COVERED`/`REFRAME`;
- anti-quota wording, чтобы enrichment count не коллапсировал к `3-4`;
- bias rule:
  - unsure between `COVERED` and `REFRAME` -> choose `REFRAME`
  - unsure between `REFRAME` and `ENRICHMENT` -> choose `ENRICHMENT`

После этого:

1. rerun тот же `4`-event pilot;
2. проверить, что:
   - `reframe_count > 0`
   - `missing_baseline_match = 0`
   - enrichment variance выросла
   - `cross_enrich` получил хотя бы немного реальной работы
3. только потом решать `GO / NO-GO` для `facts.merge`.

## 8. Final Decision

Текущий статус:

- `facts.dedup architecture`: `GO`
- `facts.dedup current prompt pack`: `NO-GO`
- `facts.merge`: `NO-GO until dedup prompt retune`

То есть family как идея уже доказана, но её first prompt contract ещё не дотянут до merge-ready качества.
