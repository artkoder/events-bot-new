# Smart Update Gemma Event Copy V2.16.2 Lollipop Facts.Merge Family Lab

Дата: 2026-03-09

## 1. Scope

- family: `facts.merge`
- iteration: `iter6`
- design shift: `stored bucket.v2 -> link.v3 -> resolve.v3 -> emit.v1 -> deterministic audit.v3`
- carry: narrow `Gemini` consultation on the residual `2731` song-list blocker
- upstream reuse: stored `facts.extract` + stored `facts.dedup iter3` + stored `bucket.v2`
- casebook: `2731, 2673, 2447, 2759`

## 2. Aggregate Metrics

- events: `4`
- avg_merge_record_count: `13.75`
- avg_linked_items: `1.25`
- avg_resolved_cluster_count: `12.5`
- avg_emitted_cluster_count: `12.5`
- avg_total_bucket_items: `12.5`
- events_with_flags: `0`
- events_with_duplicate_suspects: `0`
- merge_record_stddev: `1.785`

## 3. Event Snapshot

### `2731` `Хоровая вечеринка «Праздник у девчат»`

- event_type: `party`
- sources: `2`
- merge records: `14`
- linked items: `2`
- resolved clusters: `12`
- emitted clusters: `12`
- bucket counts: `{'event_core': 1, 'program_list': 1, 'people_and_roles': 1, 'forward_looking': 0, 'logistics_infoblock': 4, 'support_context': 5, 'uncertain': 0}`

### `2673` `Собакусъел`

- event_type: `presentation`
- sources: `2`
- merge records: `16`
- linked items: `1`
- resolved clusters: `15`
- emitted clusters: `15`
- bucket counts: `{'event_core': 3, 'program_list': 1, 'people_and_roles': 0, 'forward_looking': 5, 'logistics_infoblock': 3, 'support_context': 3, 'uncertain': 0}`

### `2447` `🎨 Мастер-класс «Мирное небо»`

- event_type: `мастер-класс`
- sources: `3`
- merge records: `14`
- linked items: `0`
- resolved clusters: `14`
- emitted clusters: `14`
- bucket counts: `{'event_core': 4, 'program_list': 0, 'people_and_roles': 2, 'forward_looking': 0, 'logistics_infoblock': 5, 'support_context': 3, 'uncertain': 0}`

### `2759` `Выставка «Королева Луиза: идеал или красивая легенда?»`

- event_type: `выставка`
- sources: `4`
- merge records: `11`
- linked items: `2`
- resolved clusters: `9`
- emitted clusters: `9`
- bucket counts: `{'event_core': 2, 'program_list': 0, 'people_and_roles': 0, 'forward_looking': 0, 'logistics_infoblock': 3, 'support_context': 4, 'uncertain': 0}`

## 4. Findings

- `iter6` targets only the residual `2731` song-list linker inconsistency from `iter5`.
- `link.v3` now makes the merge contract explicit: merged rows must return the same `cluster_id`.
- a narrow cleaner guard force-collapses identical quoted song lists inside `program_list`.
- `2731` is now clean: the duplicate song-list lines collapse into one `program_list` cluster with shared evidence ids.
- guard-cases `2673`, `2447`, and `2759` stay clean, so the narrow fix did not reintroduce the earlier merge/audit failures on the regression subset.
