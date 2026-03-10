# Smart Update Gemma Event Copy V2.16.2 Lollipop Facts.Merge Family Lab

Дата: 2026-03-09

## 1. Scope

- family: `facts.merge`
- iteration: `iter5`
- design shift: `stored bucket.v2 -> link.v2 -> resolve.v3 -> emit.v1 -> deterministic audit.v3`
- carry: keep the Gemini-driven dedicated linkage stage and add bucket-aware duplicate audit guards
- upstream reuse: stored `facts.extract` + stored `facts.dedup iter3` + stored `bucket.v2`
- casebook: `2673, 2687, 2734, 2659, 2731, 2498, 2747, 2701, 2732, 2759, 2657, 2447`

## 2. Aggregate Metrics

- events: `12`
- avg_merge_record_count: `12.0`
- avg_linked_items: `1.333`
- avg_resolved_cluster_count: `10.667`
- avg_emitted_cluster_count: `10.667`
- avg_total_bucket_items: `10.667`
- events_with_flags: `1`
- events_with_duplicate_suspects: `1`
- merge_record_stddev: `2.309`

## 3. Event Snapshot

### `2673` `Собакусъел`

- event_type: `presentation`
- sources: `2`
- merge records: `16`
- linked items: `1`
- resolved clusters: `15`
- emitted clusters: `15`
- bucket counts: `{'event_core': 3, 'program_list': 1, 'people_and_roles': 0, 'forward_looking': 5, 'logistics_infoblock': 3, 'support_context': 3, 'uncertain': 0}`

### `2687` `📚 Лекция «Художницы»`

- event_type: `лекция`
- sources: `2`
- merge records: `11`
- linked items: `1`
- resolved clusters: `10`
- emitted clusters: `10`
- bucket counts: `{'event_core': 3, 'program_list': 0, 'people_and_roles': 0, 'forward_looking': 0, 'logistics_infoblock': 3, 'support_context': 4, 'uncertain': 0}`

### `2734` `Концерт Владимира Гудожникова «Ты… моя мелодия»`

- event_type: `concert`
- sources: `2`
- merge records: `13`
- linked items: `4`
- resolved clusters: `9`
- emitted clusters: `9`
- bucket counts: `{'event_core': 1, 'program_list': 1, 'people_and_roles': 1, 'forward_looking': 0, 'logistics_infoblock': 5, 'support_context': 1, 'uncertain': 0}`

### `2659` `Посторонний`

- event_type: `кинопоказ`
- sources: `2`
- merge records: `9`
- linked items: `0`
- resolved clusters: `9`
- emitted clusters: `9`
- bucket counts: `{'event_core': 0, 'program_list': 0, 'people_and_roles': 1, 'forward_looking': 0, 'logistics_infoblock': 5, 'support_context': 3, 'uncertain': 0}`

### `2731` `Хоровая вечеринка «Праздник у девчат»`

- event_type: `party`
- sources: `2`
- merge records: `14`
- linked items: `1`
- resolved clusters: `13`
- emitted clusters: `13`
- bucket counts: `{'event_core': 1, 'program_list': 2, 'people_and_roles': 1, 'forward_looking': 0, 'logistics_infoblock': 4, 'support_context': 5, 'uncertain': 0}`
- flags: `merge_duplicate_suspect`
- duplicate_suspects: `1`

### `2498` `Нюрнберг`

- event_type: `спектакль`
- sources: `3`
- merge records: `9`
- linked items: `0`
- resolved clusters: `9`
- emitted clusters: `9`
- bucket counts: `{'event_core': 4, 'program_list': 0, 'people_and_roles': 1, 'forward_looking': 0, 'logistics_infoblock': 2, 'support_context': 2, 'uncertain': 0}`

### `2747` `Киноклуб: «Последнее метро»`

- event_type: `кинопоказ`
- sources: `1`
- merge records: `8`
- linked items: `0`
- resolved clusters: `8`
- emitted clusters: `8`
- bucket counts: `{'event_core': 0, 'program_list': 0, 'people_and_roles': 2, 'forward_looking': 0, 'logistics_infoblock': 2, 'support_context': 4, 'uncertain': 0}`

### `2701` `«Татьяна танцует»`

- event_type: `party`
- sources: `1`
- merge records: `13`
- linked items: `3`
- resolved clusters: `10`
- emitted clusters: `10`
- bucket counts: `{'event_core': 3, 'program_list': 1, 'people_and_roles': 1, 'forward_looking': 0, 'logistics_infoblock': 3, 'support_context': 2, 'uncertain': 0}`

### `2732` `Вечер в русском стиле`

- event_type: `party`
- sources: `1`
- merge records: `13`
- linked items: `4`
- resolved clusters: `9`
- emitted clusters: `9`
- bucket counts: `{'event_core': 0, 'program_list': 1, 'people_and_roles': 0, 'forward_looking': 0, 'logistics_infoblock': 4, 'support_context': 4, 'uncertain': 0}`

### `2759` `Выставка «Королева Луиза: идеал или красивая легенда?»`

- event_type: `выставка`
- sources: `4`
- merge records: `11`
- linked items: `2`
- resolved clusters: `9`
- emitted clusters: `9`
- bucket counts: `{'event_core': 2, 'program_list': 0, 'people_and_roles': 0, 'forward_looking': 0, 'logistics_infoblock': 3, 'support_context': 4, 'uncertain': 0}`

### `2657` `Коллекция украшений 1930–1960-х годов`

- event_type: `выставка`
- sources: `2`
- merge records: `13`
- linked items: `0`
- resolved clusters: `13`
- emitted clusters: `13`
- bucket counts: `{'event_core': 4, 'program_list': 0, 'people_and_roles': 0, 'forward_looking': 0, 'logistics_infoblock': 5, 'support_context': 4, 'uncertain': 0}`

### `2447` `🎨 Мастер-класс «Мирное небо»`

- event_type: `мастер-класс`
- sources: `3`
- merge records: `14`
- linked items: `0`
- resolved clusters: `14`
- emitted clusters: `14`
- bucket counts: `{'event_core': 4, 'program_list': 0, 'people_and_roles': 2, 'forward_looking': 0, 'logistics_infoblock': 5, 'support_context': 3, 'uncertain': 0}`

## 4. Findings

- `iter5` keeps the dedicated linkage stage and retunes it for same-subject generic+specific exhibition lines.
- `audit.v3` is now bucket-aware so lexical template overlap alone should no longer flag different people or different forward-looking claims.
- `2759` is now clean: the generic exhibition theme line, the richer “through documents and objects” line, and the secondary enrichment fact collapsed into one `event_core` cluster.
- `2673` and `2447` are now clean: both prior flags were duplicate-audit false positives, not real merge failures.
- One residual blocker remains on `2731`: `facts.merge.link.v2` correctly explains that the two song-list facts should merge, but still returns two separate `cluster_id` values, so the duplicate survives into `program_list`.
