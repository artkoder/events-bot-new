# Smart Update Gemma Event Copy V2.16.2 Lollipop Facts.Dedup Family Lab

Дата: 2026-03-09

## 1. Scope

- family: `facts.dedup`
- design shift: `baseline_diff -> cross_enrich -> deterministic audit`
- pilot events: `2673, 2687, 2734, 2447`
- shortlisted extraction inputs: `facts.extract_subject.v1, facts.extract_card.v1, facts.extract_agenda.v1, facts.extract_support.v1, facts.extract_performer.v1, facts.extract_participation.v1, facts.extract_stage.v1`
- theme challenger included: `False`

## 2. Aggregate Metrics

- events: `4`
- avg_total_extraction_facts: `30.0`
- avg_enrichment_count: `3.25`
- avg_reframe_count: `0.0`
- avg_cross_duplicates: `0.0`
- avg_unique_enrichment_final: `3.25`
- events_with_flags: `1`

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
- flags: `missing_baseline_match`

### `2687` `📚 Лекция «Художницы»`

- event_type: `лекция`
- sources: `2`
- raw facts: `24`
- baseline facts: `13`
- extraction facts accounted: `25` / `25`
- enrichment count: `3`
- reframe count: `0`
- cross duplicates: `0`
- unique enrichment final: `3`

### `2734` `Концерт Владимира Гудожникова «Ты… моя мелодия»`

- event_type: `concert`
- sources: `2`
- raw facts: `25`
- baseline facts: `15`
- extraction facts accounted: `27` / `27`
- enrichment count: `3`
- reframe count: `0`
- cross duplicates: `0`
- unique enrichment final: `3`

### `2447` `🎨 Мастер-класс «Мирное небо»`

- event_type: `мастер-класс`
- sources: `3`
- raw facts: `29`
- baseline facts: `17`
- extraction facts accounted: `30` / `30`
- enrichment count: `4`
- reframe count: `0`
- cross duplicates: `0`
- unique enrichment final: `4`

## 4. Findings

- Этот round уже не использует global clustering. Входы короткие и pairwise, что лучше соответствует practical reliability `Gemma`.
- `facts.dedup` пока оценивается как factual accounting layer, а не как narrative layer.
- Следующий шаг после этого отчёта: post-run consultation и решение, достаточно ли quality для перехода к `facts.merge`, или нужно тюнить prompts `baseline_diff` / `cross_enrich`.
