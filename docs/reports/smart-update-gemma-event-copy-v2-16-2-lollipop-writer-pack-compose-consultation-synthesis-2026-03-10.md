# Smart Update Gemma Event Copy V2.16.2 Lollipop Writer Pack Compose Consultation Synthesis

Date: `2026-03-10`

## Inputs

- Opus report: [smart-update-lollipop-v2-16-2-writer-pack-compose-consultation-opus-2026-03-10.md](/workspaces/events-bot-new/artifacts/codex/reports/smart-update-lollipop-v2-16-2-writer-pack-compose-consultation-opus-2026-03-10.md)
- Gemini review: [smart-update-lollipop-v2-16-2-writer-pack-compose-consultation-gemini-3.1-pro-preview-2026-03-10.raw.json](/workspaces/events-bot-new/artifacts/codex/reports/smart-update-lollipop-v2-16-2-writer-pack-compose-consultation-gemini-3.1-pro-preview-2026-03-10.raw.json)
- Main brief: [smart-update-lollipop-v2-16-2-writer-pack-compose-consultation-brief-2026-03-10.md](/workspaces/events-bot-new/artifacts/codex/tasks/smart-update-lollipop-v2-16-2-writer-pack-compose-consultation-brief-2026-03-10.md)

## Consensus

- `writer_pack.compose` должен остаться deterministic stage без нового LLM call.
- `writer_pack.select` в `v1` должен быть identity/no-op.
- `title` enhancement остаётся hint для финального writer, а не отдельным deterministic title generator.
- `infoblock` должен жить отдельно от prose и сортироваться по canonical order ещё до final writer.

## Main disagreement resolved

`Opus` справедливо настаивал на flat `sections` array как прямом continuation `editorial.layout`.
`Gemini` справедливо указала, что нельзя отдавать final writer и `literal_items`, и тот же список ещё раз внутри raw fact text.

Каноническое решение:

- root остаётся flat `sections`;
- для list-heavy facts compose разделяет:
  - `facts` как narrative-safe residue;
  - `literal_items` как verbatim list;
  - `coverage_plan` как явный ledger того, чем покрывается каждый `fact_id`;
- `absorbed_by_list` допустим только для фактов, которые после выноса literal items не добавляют нового narrative residue.

## Canonical v1 decision

Использовать такой chain:

```text
writer_pack.compose.v1
-> writer_pack.select.v1
-> writer.final_4o
```

Где:
- `compose.v1` собирает один canonical writer pack;
- `select.v1` не строит варианты;
- `writer.final_4o` получает один финальный pack.

## Schema decision

Предпочтительный contract:

- `title_context`
- flat `sections`
- отдельный `infoblock`
- deterministic `constraints`
- per-section `coverage_plan` для list partition

Не использовать:

- разнесение `lead/body/program` по разным top-level keys;
- `source_text` внутри downstream pack;
- pack-level creative toggles вроде `quote_allowed`;
- multi-variant `select` на этом шаге.

## Why this is the best next step

- Сохраняется точный document order из `editorial.layout`.
- Final writer получает меньше шумных полей и меньше поводов к hallucination-like repetition.
- Literal lists выживают буквально, но без двойной экспозиции в prose input.
- Coverage остаётся проверяемым детерминированно.

## Implementation verdict

`GO`.

Реализовать:

- flat `sections`;
- canonical infoblock ordering;
- deterministic list partition;
- identity `writer_pack.select`;
- явный `coverage_plan` как bridge между Opus flat-pack design и Gemini anti-double-exposure critique.
