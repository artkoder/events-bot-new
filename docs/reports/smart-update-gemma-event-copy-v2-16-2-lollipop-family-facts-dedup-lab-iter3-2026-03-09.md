# Smart Update Gemma Event Copy V2.16.2 Lollipop Facts.Dedup Family Lab

Дата: 2026-03-09

## 1. Scope

- family: `facts.dedup`
- iteration: `iter3`
- design shift: `baseline_diff.v3(id-anchor) -> cross_enrich.v1 -> deterministic audit`
- pilot events: `2673, 2687, 2734, 2659, 2731, 2498, 2747, 2701, 2732, 2759, 2657, 2447`
- shortlisted extraction inputs: `facts.extract_subject.v1, facts.extract_card.v1, facts.extract_agenda.v1, facts.extract_support.v1, facts.extract_performer.v1, facts.extract_participation.v1, facts.extract_stage.v1`
- theme challenger included: `False`

## 2. Aggregate Metrics

- events: `12`
- avg_total_extraction_facts: `27.583`
- avg_enrichment_count: `2.083`
- avg_reframe_count: `1.667`
- avg_cross_duplicates: `0.0`
- avg_unique_enrichment_final: `3.75`
- avg_auto_reclassified_missing_match_count: `0.0`
- events_with_reframe: `9`
- events_with_cross_duplicates: `0`
- enrichment_stddev: `1.498`
- covered_to_stage_ratio: `0.864`
- events_with_flags: `3`

## 3. Event Snapshot

### `2673` `Собакусъел`

- event_type: `presentation`
- sources: `2`
- raw facts: `32`
- baseline facts: `18`
- extraction facts accounted: `38` / `38`
- enrichment count: `3`
- reframe count: `0`
- cross duplicates: `0`
- unique enrichment final: `3`
- auto reclassified missing match: `0`
- flags: `reframe_zero`

### `2687` `📚 Лекция «Художницы»`

- event_type: `лекция`
- sources: `2`
- raw facts: `24`
- baseline facts: `13`
- extraction facts accounted: `25` / `25`
- enrichment count: `1`
- reframe count: `1`
- cross duplicates: `0`
- unique enrichment final: `2`
- auto reclassified missing match: `0`

### `2734` `Концерт Владимира Гудожникова «Ты… моя мелодия»`

- event_type: `concert`
- sources: `2`
- raw facts: `25`
- baseline facts: `15`
- extraction facts accounted: `27` / `27`
- enrichment count: `2`
- reframe count: `1`
- cross duplicates: `0`
- unique enrichment final: `3`
- auto reclassified missing match: `0`

### `2659` `Посторонний`

- event_type: `кинопоказ`
- sources: `2`
- raw facts: `29`
- baseline facts: `15`
- extraction facts accounted: `23` / `23`
- enrichment count: `1`
- reframe count: `1`
- cross duplicates: `0`
- unique enrichment final: `2`
- auto reclassified missing match: `0`

### `2731` `Хоровая вечеринка «Праздник у девчат»`

- event_type: `party`
- sources: `2`
- raw facts: `31`
- baseline facts: `16`
- extraction facts accounted: `27` / `27`
- enrichment count: `5`
- reframe count: `2`
- cross duplicates: `0`
- unique enrichment final: `7`
- auto reclassified missing match: `0`

### `2498` `Нюрнберг`

- event_type: `спектакль`
- sources: `3`
- raw facts: `17`
- baseline facts: `12`
- extraction facts accounted: `26` / `26`
- enrichment count: `0`
- reframe count: `0`
- cross duplicates: `0`
- unique enrichment final: `0`
- auto reclassified missing match: `0`
- flags: `reframe_zero`

### `2747` `Киноклуб: «Последнее метро»`

- event_type: `кинопоказ`
- sources: `1`
- raw facts: `13`
- baseline facts: `12`
- extraction facts accounted: `22` / `22`
- enrichment count: `0`
- reframe count: `0`
- cross duplicates: `0`
- unique enrichment final: `0`
- auto reclassified missing match: `0`
- flags: `reframe_zero`

### `2701` `«Татьяна танцует»`

- event_type: `party`
- sources: `1`
- raw facts: `15`
- baseline facts: `13`
- extraction facts accounted: `32` / `32`
- enrichment count: `3`
- reframe count: `3`
- cross duplicates: `0`
- unique enrichment final: `6`
- auto reclassified missing match: `0`

### `2732` `Вечер в русском стиле`

- event_type: `party`
- sources: `1`
- raw facts: `12`
- baseline facts: `11`
- extraction facts accounted: `26` / `26`
- enrichment count: `4`
- reframe count: `7`
- cross duplicates: `0`
- unique enrichment final: `11`
- auto reclassified missing match: `0`

### `2759` `Выставка «Королева Луиза: идеал или красивая легенда?»`

- event_type: `выставка`
- sources: `4`
- raw facts: `46`
- baseline facts: `14`
- extraction facts accounted: `34` / `34`
- enrichment count: `1`
- reframe count: `3`
- cross duplicates: `0`
- unique enrichment final: `4`
- auto reclassified missing match: `0`

### `2657` `Коллекция украшений 1930–1960-х годов`

- event_type: `выставка`
- sources: `2`
- raw facts: `27`
- baseline facts: `16`
- extraction facts accounted: `21` / `21`
- enrichment count: `3`
- reframe count: `1`
- cross duplicates: `0`
- unique enrichment final: `4`
- auto reclassified missing match: `0`

### `2447` `🎨 Мастер-класс «Мирное небо»`

- event_type: `мастер-класс`
- sources: `3`
- raw facts: `29`
- baseline facts: `17`
- extraction facts accounted: `30` / `30`
- enrichment count: `2`
- reframe count: `1`
- cross duplicates: `0`
- unique enrichment final: `3`
- auto reclassified missing match: `0`

## 4. Findings

- Этот round сохраняет ту же `baseline_diff -> cross_enrich -> audit` architecture, но переводит anchoring на `baseline_match_id`, снимая brittle exact-text requirement.
- `facts.dedup` по-прежнему оценивается как factual accounting / quality uplift layer, а не как narrative layer.
- Ключевой вопрос этого iter3: исчезает ли `auto_reclassified_missing_match` без отката по `REFRAME` и без роста ложного `COVERED`.
