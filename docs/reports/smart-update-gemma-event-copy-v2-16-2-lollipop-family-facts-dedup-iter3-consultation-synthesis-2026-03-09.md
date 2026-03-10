# Smart Update Gemma Event Copy V2.16.2 Lollipop Facts.Dedup Iter3 Consultation Synthesis

Дата: 2026-03-09

## 1. Scope

Этот synthesis закрывает `facts.dedup iter3` после:

- короткой prompt-level консультации по `baseline_diff.v3(id-anchor)`;
- полного `12`-event rerun на уже сохранённых `facts.extract` outputs;
- post-run consultation через `Opus`;
- single-launch `Gemini`, который не вернул usable answer из-за provider-side capacity failure.

Материалы:

- prompt-fix brief: `/workspaces/events-bot-new/artifacts/codex/tasks/smart-update-lollipop-v2-16-2-facts-dedup-prompt-fix-consultation-brief-2026-03-09.md`
- prompt-fix `Opus` JSON: `/workspaces/events-bot-new/artifacts/codex/reports/smart-update-lollipop-v2-16-2-facts-dedup-prompt-fix-consultation-opus-2026-03-09.json`
- prompt-fix `Gemini`: `/workspaces/events-bot-new/artifacts/codex/reports/smart-update-lollipop-v2-16-2-facts-dedup-prompt-fix-consultation-gemini-3.1-pro-preview-2026-03-09.md`
- iter3 harness: `/workspaces/events-bot-new/artifacts/codex/smart_update_lollipop_facts_dedup_family_v2_16_2_iter3_2026_03_09.py`
- iter3 raw JSON: `/workspaces/events-bot-new/artifacts/codex/smart_update_lollipop_facts_dedup_family_v2_16_2_iter3_2026-03-09.json`
- iter3 lab report: `/workspaces/events-bot-new/docs/reports/smart-update-gemma-event-copy-v2-16-2-lollipop-family-facts-dedup-lab-iter3-2026-03-09.md`
- iter3 post-run brief: `/workspaces/events-bot-new/artifacts/codex/tasks/smart-update-lollipop-v2-16-2-facts-dedup-family-postrun-consultation-iter3-brief-2026-03-09.md`
- iter3 `Opus`: `/workspaces/events-bot-new/artifacts/codex/reports/smart-update-lollipop-v2-16-2-facts-dedup-family-postrun-consultation-iter3-opus-2026-03-09.md`
- iter3 `Gemini` failure note: `/workspaces/events-bot-new/artifacts/codex/reports/smart-update-lollipop-v2-16-2-facts-dedup-family-postrun-consultation-iter3-gemini-3.1-pro-preview-2026-03-09.md`
- iter3 `Gemini` stderr: `/workspaces/events-bot-new/artifacts/codex/reports/smart-update-lollipop-v2-16-2-facts-dedup-family-postrun-consultation-iter3-gemini-3.1-pro-preview-2026-03-09.stderr.txt`

## 2. What Iter3 Actually Changed

`iter3` заменил brittle exact-text `baseline_match` на deterministic `baseline_match_id`.

Это не новый `facts.extract` run. `facts.dedup iter3` работает только поверх уже сохранённых extraction outputs.

Главные aggregate shifts относительно `iter2`:

```json
{
  "avg_enrichment_count": [3.75, 2.083],
  "avg_reframe_count": [1.333, 1.667],
  "avg_unique_enrichment_final": [5.083, 3.75],
  "avg_auto_reclassified_missing_match_count": [1.0, 0.0],
  "events_with_reframe": [10, 9],
  "enrichment_stddev": [2.976, 1.498],
  "events_with_flags": [7, 3]
}
```

Ключевой практический итог:

- старый blocker `auto_reclassified_missing_match` исчез полностью;
- `REFRAME` остался жив;
- signal стал чище, но enrichment counts заметно сжались.

## 3. External Review Outcome

### 3.1. Opus

`Opus` дал usable strict review и подтвердил:

- `facts.dedup iter3`: `Conditional GO`
- `facts.merge`: можно разблокировать после маленьких harness-only guardrails

Главная логика `Opus`:

- старая проблема `iter2` действительно решена;
- enrichment compression выглядит в основном здоровой коррекцией, а не новой поломкой;
- `reframe_zero` на tiny-event кейсах больше похоже на noise в audit, чем на реальную safety-проблему;
- prompt и cleaner переписывать не нужно.

Рекомендованные минимальные правки перед `facts.merge`:

1. ослабить `reframe_zero` флаг для very small stage-fact sets;
2. добавить простой aggregate diagnostic `covered_to_stage_ratio`.

### 3.2. Gemini

`Gemini` был запущен ровно один раз, но usable answer не вернул:

- provider-side `429`
- `RESOURCE_EXHAUSTED`
- `MODEL_CAPACITY_EXHAUSTED`

Следовательно, этот synthesis не трактует отсутствие Gemini-ответа как скрытое согласие или несогласие.

## 4. My Synthesis

С кодом и run artifacts лучше всего согласуется такой вывод:

- `iter3` действительно устранил mechanical blocker `iter2`;
- это был главный вопрос перед `facts.merge`, и теперь он закрыт;
- оставшиеся issues выглядят не как safety blocker, а как audit/measurement noise plus minor classification edge-cases.

Я не вижу сейчас основания делать ещё один `facts.dedup` prompt retune до первого `facts.merge` round.

То, что всё ещё требует внимания:

- `2673` как единственный meaningful `reframe_zero` signal;
- мониторинг over-anchoring на следующем слое;
- cheap metric для отслеживания доли `COVERED`.

Но это уже не причины держать `facts.merge` заблокированным.

## 5. Decision

Текущий статус:

- `facts.dedup iter3`: `GO`
- `facts.merge`: `GO`, но с небольшими harness-only guardrails

Что именно делать дальше:

1. Не rerun `facts.extract`.
2. Не retune `baseline_diff` prompt снова.
3. Добавить в dedup harness:
   - `reframe_zero_min_facts` guard;
   - `covered_to_stage_ratio` aggregate metric.
4. После этого переходить к первой family `facts.merge`.

## 6. Important Clarification

Текущий `iter3` run был:

- `dedup-only`;
- поверх сохранённого `facts.extract` casebook;
- без нового broad-run extraction.

То есть дальнейшая скорость/итеративность сохраняется: следующий шаг уже можно делать на `facts.merge`, не тратя новый полный extraction budget.

Update after harness refresh:

- `covered_to_stage_ratio` уже добавлен в `iter3` aggregate;
- `reframe_zero` остаётся purely diagnostic signal и не блокирует переход к `facts.merge`.
