# Smart Update Gemma Event Copy V2.16.2 Lollipop Facts.Merge Family Lab

Дата: 2026-03-09

## 1. Scope

- family: `facts.merge`
- iteration: `iter3`
- design shift: `dedup outputs -> merge records -> bucket.v2 -> resolve.v3 -> emit.v1 -> deterministic audit.v2`
- upstream reuse: stored `facts.extract` + stored `facts.dedup iter3`, without a new extract run
- Gemma prompt carries applied in `resolve.v3`: self-contained sections, short algorithm, positive examples, no giant ban-list wall
- casebook: `2673, 2687, 2734, 2659, 2731, 2498, 2747, 2701, 2732, 2759, 2657, 2447`

## 2. Aggregate Metrics

- events: `12`
- avg_merge_record_count: `12.0`
- avg_resolved_cluster_count: `11.167`
- avg_emitted_cluster_count: `11.167`
- avg_total_bucket_items: `11.167`
- events_with_flags: `5`
- events_with_duplicate_suspects: `5`
- merge_record_stddev: `2.309`

## 3. Event Snapshot

### `2673` `Собакусъел`

- event_type: `presentation`
- sources: `2`
- raw facts: `32`
- baseline facts: `18`
- merge records: `16`
- resolved clusters: `15`
- emitted clusters: `15`
- bucket counts: `{'event_core': 3, 'program_list': 1, 'people_and_roles': 0, 'forward_looking': 5, 'logistics_infoblock': 3, 'support_context': 3, 'uncertain': 0}`
- flags: `merge_duplicate_suspect`
- duplicate_suspects: `1`

### `2687` `📚 Лекция «Художницы»`

- event_type: `лекция`
- sources: `2`
- raw facts: `24`
- baseline facts: `13`
- merge records: `11`
- resolved clusters: `11`
- emitted clusters: `11`
- bucket counts: `{'event_core': 4, 'program_list': 0, 'people_and_roles': 0, 'forward_looking': 0, 'logistics_infoblock': 3, 'support_context': 4, 'uncertain': 0}`

### `2734` `Концерт Владимира Гудожникова «Ты… моя мелодия»`

- event_type: `concert`
- sources: `2`
- raw facts: `25`
- baseline facts: `15`
- merge records: `13`
- resolved clusters: `11`
- emitted clusters: `11`
- bucket counts: `{'event_core': 1, 'program_list': 2, 'people_and_roles': 2, 'forward_looking': 0, 'logistics_infoblock': 5, 'support_context': 1, 'uncertain': 0}`
- flags: `merge_duplicate_suspect`
- duplicate_suspects: `1`

### `2659` `Посторонний`

- event_type: `кинопоказ`
- sources: `2`
- raw facts: `29`
- baseline facts: `15`
- merge records: `9`
- resolved clusters: `9`
- emitted clusters: `9`
- bucket counts: `{'event_core': 0, 'program_list': 0, 'people_and_roles': 1, 'forward_looking': 0, 'logistics_infoblock': 5, 'support_context': 3, 'uncertain': 0}`

### `2731` `Хоровая вечеринка «Праздник у девчат»`

- event_type: `party`
- sources: `2`
- raw facts: `31`
- baseline facts: `16`
- merge records: `14`
- resolved clusters: `14`
- emitted clusters: `14`
- bucket counts: `{'event_core': 2, 'program_list': 2, 'people_and_roles': 1, 'forward_looking': 0, 'logistics_infoblock': 4, 'support_context': 5, 'uncertain': 0}`
- flags: `merge_duplicate_suspect`
- duplicate_suspects: `1`

### `2498` `Нюрнберг`

- event_type: `спектакль`
- sources: `3`
- raw facts: `17`
- baseline facts: `12`
- merge records: `9`
- resolved clusters: `9`
- emitted clusters: `9`
- bucket counts: `{'event_core': 4, 'program_list': 0, 'people_and_roles': 1, 'forward_looking': 0, 'logistics_infoblock': 2, 'support_context': 2, 'uncertain': 0}`

### `2747` `Киноклуб: «Последнее метро»`

- event_type: `кинопоказ`
- sources: `1`
- raw facts: `13`
- baseline facts: `12`
- merge records: `8`
- resolved clusters: `8`
- emitted clusters: `8`
- bucket counts: `{'event_core': 0, 'program_list': 0, 'people_and_roles': 2, 'forward_looking': 0, 'logistics_infoblock': 2, 'support_context': 4, 'uncertain': 0}`

### `2701` `«Татьяна танцует»`

- event_type: `party`
- sources: `1`
- raw facts: `15`
- baseline facts: `13`
- merge records: `13`
- resolved clusters: `10`
- emitted clusters: `10`
- bucket counts: `{'event_core': 3, 'program_list': 1, 'people_and_roles': 1, 'forward_looking': 0, 'logistics_infoblock': 3, 'support_context': 2, 'uncertain': 0}`

### `2732` `Вечер в русском стиле`

- event_type: `party`
- sources: `1`
- raw facts: `12`
- baseline facts: `11`
- merge records: `13`
- resolved clusters: `10`
- emitted clusters: `10`
- bucket counts: `{'event_core': 0, 'program_list': 2, 'people_and_roles': 0, 'forward_looking': 0, 'logistics_infoblock': 4, 'support_context': 4, 'uncertain': 0}`

### `2759` `Выставка «Королева Луиза: идеал или красивая легенда?»`

- event_type: `выставка`
- sources: `4`
- raw facts: `46`
- baseline facts: `14`
- merge records: `11`
- resolved clusters: `10`
- emitted clusters: `10`
- bucket counts: `{'event_core': 3, 'program_list': 0, 'people_and_roles': 0, 'forward_looking': 0, 'logistics_infoblock': 3, 'support_context': 4, 'uncertain': 0}`
- flags: `merge_duplicate_suspect`
- duplicate_suspects: `1`

### `2657` `Коллекция украшений 1930–1960-х годов`

- event_type: `выставка`
- sources: `2`
- raw facts: `27`
- baseline facts: `16`
- merge records: `13`
- resolved clusters: `13`
- emitted clusters: `13`
- bucket counts: `{'event_core': 4, 'program_list': 0, 'people_and_roles': 0, 'forward_looking': 0, 'logistics_infoblock': 5, 'support_context': 4, 'uncertain': 0}`

### `2447` `🎨 Мастер-класс «Мирное небо»`

- event_type: `мастер-класс`
- sources: `3`
- raw facts: `29`
- baseline facts: `17`
- merge records: `14`
- resolved clusters: `14`
- emitted clusters: `14`
- bucket counts: `{'event_core': 4, 'program_list': 0, 'people_and_roles': 2, 'forward_looking': 0, 'logistics_infoblock': 5, 'support_context': 3, 'uncertain': 0}`
- flags: `merge_duplicate_suspect`
- duplicate_suspects: `1`

## 4. Findings

- `iter3` меняет только `resolve`; `bucket.v2` и `emit.v1` сохранены.
- Hypothesis check failed: `resolve.v3` не разблокировал `facts.prioritize`; aggregate flags выросли с `4` до `5`, а duplicate-heavy cases сохранились.
- `2734`: richer award line теперь канонизируется правильно внутри cluster, но duplicate survives in two separate `people_and_roles` clusters (`R_B07`, `R_B12`). Это уже bucket/linkage problem, а не intra-cluster resolve problem.
- `2731`: один и тот же song-list survives in two separate `program_list` clusters (`R_E02`, `R_E05`) по той же причине: records не были сведены в один cluster до `resolve`.
- `2673` и `2447` по-прежнему выглядят как audit false positives, а не как real merge regressions.
- `2759` дал новый duplicate-suspect между overlapping baseline `event_core` facts, что дополнительно указывает на baseline-overlap / bucket-linkage bottleneck.
- Next step is no longer `resolve.v4`, but `facts.merge bucket/linkage` retune or a small pre-resolve baseline-overlap linking pass.
