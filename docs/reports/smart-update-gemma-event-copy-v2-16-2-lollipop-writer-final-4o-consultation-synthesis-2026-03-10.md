# Smart Update Gemma Event Copy V2.16.2 Lollipop Writer.Final_4o Consultation Synthesis

Дата: 2026-03-10

## 1. Inputs

- author brief: `artifacts/codex/tasks/smart-update-lollipop-v2-16-2-writer-final-4o-consultation-brief-2026-03-10.md`
- strict `Opus` consultation: `artifacts/codex/reports/smart-update-lollipop-v2-16-2-writer-final-4o-consultation-opus-2026-03-10.raw.json`
- `Gemini 3.1 Pro Preview` critique brief: `artifacts/codex/tasks/smart-update-lollipop-v2-16-2-writer-final-4o-consultation-gemini-brief-2026-03-10.md`
- `Gemini 3.1 Pro Preview` critique result: `artifacts/codex/reports/smart-update-lollipop-v2-16-2-writer-final-4o-consultation-gemini-3.1-pro-preview-2026-03-10.raw.json`

## 2. Agreement

- `writer.final_4o` остаётся единственным final writer call.
- final schema должен быть маленьким и focused on public text.
- главное product constraint: formalized `infoblock` не должен дублироваться в prose.
- `literal_items` нужно сохранять через markdown list, а не растворять в свободной прозе.
- post-run validator должен быть deterministic и retry должен включаться только по hard failures.

## 3. Disagreement

### `Opus`

- правильно настоял на unified `{title, description_md}` contract, чтобы future `title_strategy = enhance` не требовал отдельной downstream family;
- предложил `_debug.covered_fact_ids` как coverage signal;
- исходно хотел валидировать `strategy=keep` title как exact model behavior.

### `Gemini 3.1`

- подтвердил unified `{title, description_md}`;
- отверг `_debug.covered_fact_ids` как unreliable self-report;
- предложил не тратить retry на `keep` title mismatch: Python должен просто принудительно применять `original_title`;
- сузил hard validator до двух реально high-signal классов: infoblock leakage и literal list integrity.

## 4. Final Chosen Design

Собран гибрид с бoльшим весом у `Opus`, но с двумя критичными поправками от `Gemini 3.1`:

1. Runtime schema:

```json
{
  "title": "string",
  "description_md": "string"
}
```

2. `title_strategy = keep`:
   - model title still requested;
   - applied title deterministically equals `original_title`;
   - mismatch уходит в warning, не в retry.

3. `title_strategy = enhance`:
   - downstream stage остаётся тем же `writer.final_4o`;
   - validator блокирует unchanged title и infoblock leakage.

4. `_debug` contract dropped in `iter1`.

5. Hard validation focuses on:
   - JSON/schema;
   - infoblock duplication;
   - literal item loss / missing markdown list;
   - missing intro before `narrative_plus_literal_list`;
   - invented headings;
   - URL leakage.

6. Soft warnings stay for:
   - keep-title mismatch;
   - heading collapse;
   - long lead;
   - filler prose.

## 5. My Additional View

- infoblock leakage check нельзя делать слишком широко по venue-name fragments, иначе `2759` даст false positive на организатора `Музей Мирового океана`;
- поэтому location hard-block остаётся narrow: full address-like value / explicit logistics duplication, а не любое совпадение institution name;
- semantic coverage по ordinary narrative facts в `iter1` лучше оставить на human/LLM post-run review, чем притворяться deterministic NLP.

## 6. Verdict

`GO` для `writer.final_4o iter1` с каноническим направлением:

```text
writer_pack.select.v1
-> writer.final_4o.v1
-> deterministic validate
```

Следующий practical step после `iter1`: реальный full-family run на `12`-event casebook и post-run review по actual outputs.
