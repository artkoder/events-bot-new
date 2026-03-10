# Smart Update Gemma Event Copy V2.16.2 Lollipop Facts.Merge Evidence Pack

Дата: 2026-03-09

## Scope

- family: `facts.merge`
- iteration: `iter1`
- source of truth for consultation: полный packet `source -> prompt -> Gemma raw output -> parsed result`

## Main Packet

- consultation packet JSON: `/workspaces/events-bot-new/artifacts/codex/smart_update_lollipop_facts_merge_family_v2_16_2_iter1_2026-03-09_consultation_packet.json`
- trace root: `/workspaces/events-bot-new/artifacts/codex/smart_update_lollipop_facts_merge_family_v2_16_2_iter1_2026-03-09`
- lab JSON: `/workspaces/events-bot-new/artifacts/codex/smart_update_lollipop_facts_merge_family_v2_16_2_iter1_2026-03-09.json`
- lab report: `/workspaces/events-bot-new/docs/reports/smart-update-gemma-event-copy-v2-16-2-lollipop-family-facts-merge-lab-iter1-2026-03-09.md`

## What The Packet Contains

Для каждого representative event packet хранит:

- event metadata
- source excerpts
- `facts.merge.bucket.v1`
- `facts.merge.resolve.v1`
- `facts.merge.emit.v1`

Для каждого stage лежит:

- `input.json`
- `prompt.txt`
- `raw_output.txt`
- `output.json`
- `result.json`

## Representative Events

- `2673` `Собакусъел`
- `2734` `Концерт Владимира Гудожникова «Ты… моя мелодия»`
- `2731` `Хоровая вечеринка «Праздник у девчат»`
- `2747` `Киноклуб: «Последнее метро»`
- `2759` `Выставка «Королева Луиза: идеал или красивая легенда?»`

## Quick Open Targets

- `2673 bucket input`: `/workspaces/events-bot-new/artifacts/codex/smart_update_lollipop_facts_merge_family_v2_16_2_iter1_2026-03-09/2673/facts.merge.bucket.v1/input.json`
- `2673 bucket prompt`: `/workspaces/events-bot-new/artifacts/codex/smart_update_lollipop_facts_merge_family_v2_16_2_iter1_2026-03-09/2673/facts.merge.bucket.v1/prompt.txt`
- `2673 bucket raw`: `/workspaces/events-bot-new/artifacts/codex/smart_update_lollipop_facts_merge_family_v2_16_2_iter1_2026-03-09/2673/facts.merge.bucket.v1/raw_output.txt`
- `2673 emit result`: `/workspaces/events-bot-new/artifacts/codex/smart_update_lollipop_facts_merge_family_v2_16_2_iter1_2026-03-09/2673/facts.merge.emit.v1/result.json`
- `2734 resolve result`: `/workspaces/events-bot-new/artifacts/codex/smart_update_lollipop_facts_merge_family_v2_16_2_iter1_2026-03-09/2734/facts.merge.resolve.v1/result.json`
- `2731 emit result`: `/workspaces/events-bot-new/artifacts/codex/smart_update_lollipop_facts_merge_family_v2_16_2_iter1_2026-03-09/2731/facts.merge.emit.v1/result.json`
- `2747 emit result`: `/workspaces/events-bot-new/artifacts/codex/smart_update_lollipop_facts_merge_family_v2_16_2_iter1_2026-03-09/2747/facts.merge.emit.v1/result.json`
- `2759 emit result`: `/workspaces/events-bot-new/artifacts/codex/smart_update_lollipop_facts_merge_family_v2_16_2_iter1_2026-03-09/2759/facts.merge.emit.v1/result.json`

## Why This Exists

Этот пакет нужен, чтобы консультанты и локальный review не опирались на пересказ. Для `facts.merge` консультация должна видеть:

`source/post material -> exact Gemma prompt -> raw Gemma output -> parsed merged result`

Именно этот packet использовался как factual base для post-run consultation round.
