# Smart Update Gemma Event Copy V2.16.2 Lollipop Editorial.Layout Family Lab

Дата: 2026-03-11c

## 1. Scope

- family: `editorial.layout`
- iteration: `iter2`
- upstream input: full `12`-event `facts.prioritize iter3` pack from `2026-03-10`
- active stages: `editorial.layout.precompute.v1 -> editorial.layout.plan.v1 -> editorial.layout.validate.v1`
- this round does not rerun `facts.extract`, `facts.dedup`, `facts.merge`, or `facts.prioritize`

## 2. Aggregate Metrics

- events: `12`
- avg_block_count: `3.667`
- events_with_flags: `0`
- events_with_program_block: `2`
- events_with_body_block: `12`
- events_with_headings: `11`
- events_with_title_enhance: `1`
- fallback_total: `0`
- auto_fixed_total: `4`
- missing_fact_total: `0`
- duplicate_fact_total: `0`

## 3. Event Snapshot

### `2673` `Собакусъел`

- event_type: `presentation`
- density: `standard`
- has_long_program: `False`
- title_strategy: `enhance`
- title_hint_ref: `FL01`
- layout: `lead:narrative(FL01,PL01) -> body:narrative [О проекте «Собакусъел»](EC01,EC02,FL03,FL04,FL05,FL02) -> infoblock:structured(LG01,LG02,LG03)`

### `2687` `📚 Лекция «Художницы»`

- event_type: `лекция`
- density: `standard`
- has_long_program: `False`
- title_strategy: `keep`
- layout: `lead:narrative(EC01,EC03) -> body:narrative [О лекции](EC02) -> body:narrative [Художницы](SC01,SC02,SC03,SC04) -> infoblock:structured(LG01,LG02,LG03)`

### `2734` `Концерт Владимира Гудожникова «Ты… моя мелодия»`

- event_type: `concert`
- density: `standard`
- has_long_program: `True`
- title_strategy: `keep`
- layout: `lead:narrative(EC01,PR01) -> body:narrative(SC01) -> program:list(PL01) -> infoblock:structured(LG01,LG02,LG03,LG04,LG05)`

### `2659` `Посторонний`

- event_type: `кинопоказ`
- density: `minimal`
- has_long_program: `False`
- title_strategy: `keep`
- layout: `lead:narrative(SC01,PR01) -> body:narrative [О фильме](SC02,SC03) -> infoblock:structured(LG01,LG02,LG03,LG04,LG05)`

### `2731` `Хоровая вечеринка «Праздник у девчат»`

- event_type: `party`
- density: `rich`
- has_long_program: `True`
- title_strategy: `keep`
- layout: `lead:narrative(EC01,PR01) -> body:narrative [О вечеринке](SC01,SC02,SC03,SC04,SC05) -> program:list [Репертуар](PL01,PL02) -> infoblock:structured(LG01,LG02,LG03,LG04)`

### `2498` `Нюрнберг`

- event_type: `спектакль`
- density: `standard`
- has_long_program: `False`
- title_strategy: `keep`
- layout: `lead:narrative(EC02,EC03) -> body:narrative [О чем спектакль](EC01,EC04) -> body:narrative(PR01,SC01) -> infoblock:structured(LG01,LG02)`

### `2747` `Киноклуб: «Последнее метро»`

- event_type: `кинопоказ`
- density: `minimal`
- has_long_program: `False`
- title_strategy: `keep`
- layout: `lead:narrative(SC03,PR01) -> body:narrative [О фильме](SC02,SC04) -> body:narrative(PR02,SC01) -> infoblock:structured(LG01,LG02)`

### `2701` `«Татьяна танцует»`

- event_type: `party`
- density: `standard`
- has_long_program: `False`
- title_strategy: `keep`
- layout: `lead:narrative(EC01,PR01) -> body:narrative [О вечеринке](EC02,EC03) -> body:narrative [Что ожидать](SC01,SC02,PL01) -> infoblock:structured(LG01,LG02,LG03)`

### `2732` `Вечер в русском стиле`

- event_type: `party`
- density: `standard`
- has_long_program: `True`
- title_strategy: `keep`
- layout: `lead:narrative(PL01) -> body:narrative [О вечере](SC01,SC02,SC03,SC04) -> infoblock:structured(LG01,LG02,LG03,LG04)`

### `2759` `Выставка «Королева Луиза: идеал или красивая легенда?»`

- event_type: `выставка`
- density: `standard`
- has_long_program: `False`
- title_strategy: `keep`
- layout: `lead:narrative(EC01,EC02) -> body:narrative [О выставке](SC01,SC02,SC03,SC04) -> infoblock:structured(LG01,LG02,LG03)`

### `2657` `Коллекция украшений 1930–1960-х годов`

- event_type: `выставка`
- density: `standard`
- has_long_program: `False`
- title_strategy: `keep`
- layout: `lead:narrative(EC01,EC02) -> body:narrative [О коллекции](EC03,EC04) -> body:narrative(SC05,SC02) -> infoblock:structured(LG01,LG02,LG03,LG04,LG05)`

### `2447` `🎨 Мастер-класс «Мирное небо»`

- event_type: `мастер-класс`
- density: `rich`
- has_long_program: `False`
- title_strategy: `keep`
- layout: `lead:narrative(EC01,PR01) -> body:narrative [О мастер-классе](EC03,EC02,EC04,SC01,SC02) -> body:narrative [Художники](PR02,SC03) -> infoblock:structured(LG01,LG02,LG03,LG04,LG05)`

## 4. Findings

- `editorial.layout` produces a structure plan only; no prose is generated here.
- `density` and `has_long_program` are precomputed deterministically and never inferred by `Gemma`.
- Validation enforces `lead first`, `infoblock last`, exact `LG*` containment, and full `fact_id` coverage.
- The trace directory is consultation-ready: every event has `input.json -> prompt.txt -> raw_output.txt -> result.json` plus deterministic `precompute` and `validate` outputs.
