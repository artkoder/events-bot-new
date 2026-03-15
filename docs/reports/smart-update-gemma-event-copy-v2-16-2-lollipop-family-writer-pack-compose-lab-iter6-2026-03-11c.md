# Smart Update Gemma Event Copy V2.16.2 Lollipop Writer_Pack.Compose Family Lab

Дата: 2026-03-11c

## 1. Scope

- family: `writer_pack.compose`
- iteration: `iter2`
- upstream input: full `12`-event `editorial.layout iter2` pack on top of `facts.prioritize iter3`
- active stages: `writer_pack.compose.v1 -> writer_pack.select.v1 -> deterministic audit`
- this round does not rerun `facts.extract`, `facts.dedup`, `facts.merge`, `facts.prioritize`, or `editorial.layout`

## 2. Aggregate Metrics

- events: `12`
- avg_section_count: `2.667`
- events_with_program_section: `2`
- events_with_literal_items: `3`
- literal_item_total: `19`
- absorbed_by_list_total: `1`
- events_with_flags: `0`
- missing_fact_total: `0`
- duplicate_fact_total: `0`
- infoblock_misc_total: `3`
- select_identity_failures: `0`

## 3. Event Snapshot

### `2673` `Собакусъел`

- title_strategy: `enhance`
- sections: `lead:narrative(FL01,PL01) -> body:narrative [О проекте «Собакусъел»](EC01,EC02,FL03,FL04,FL05,FL02)`
- infoblock_labels: `Дата, Локация, Цена`
- literal_items: `0`

### `2687` `📚 Лекция «Художницы»`

- title_strategy: `keep`
- sections: `lead:narrative(EC01,EC03) -> body:narrative [О лекции](EC02) -> body:narrative [Художницы](SC01,SC02,SC03,SC04)`
- infoblock_labels: `Дата, Локация, Возраст`
- literal_items: `0`

### `2734` `Концерт Владимира Гудожникова «Ты… моя мелодия»`

- title_strategy: `keep`
- sections: `lead:narrative(EC01,PR01) -> body:narrative(SC01) -> program:list(PL01) list=4 partial`
- infoblock_labels: `Дата, Время, Локация, Билеты, Возраст`
- literal_items: `4`

### `2659` `Посторонний`

- title_strategy: `keep`
- sections: `lead:narrative(SC01,PR01) -> body:narrative [О фильме](SC02,SC03)`
- infoblock_labels: `Дата, Билеты, Возраст, Прочее, Прочее`
- literal_items: `0`

### `2731` `Хоровая вечеринка «Праздник у девчат»`

- title_strategy: `keep`
- sections: `lead:narrative(EC01,PR01) -> body:narrative [О вечеринке](SC01,SC02,SC03,SC04,SC05) -> program:list [Репертуар](PL01,PL02) list=10`
- infoblock_labels: `Дата, Время, Локация, Цена`
- literal_items: `10`

### `2498` `Нюрнберг`

- title_strategy: `keep`
- sections: `lead:narrative(EC02,EC03) -> body:narrative [О чем спектакль](EC01,EC04) -> body:narrative(PR01,SC01)`
- infoblock_labels: `Дата, Локация`
- literal_items: `0`

### `2747` `Киноклуб: «Последнее метро»`

- title_strategy: `keep`
- sections: `lead:narrative(SC03,PR01) -> body:narrative [О фильме](SC02,SC04) -> body:narrative(PR02,SC01)`
- infoblock_labels: `Цена, Прочее`
- literal_items: `0`

### `2701` `«Татьяна танцует»`

- title_strategy: `keep`
- sections: `lead:narrative(EC01,PR01) -> body:narrative [О вечеринке](EC02,EC03) -> body:narrative [Что ожидать](SC01,SC02,PL01)`
- infoblock_labels: `Дата, Время, Локация`
- literal_items: `0`

### `2732` `Вечер в русском стиле`

- title_strategy: `keep`
- sections: `lead:narrative(PL01) list=5 -> body:narrative [О вечере](SC01,SC02,SC03,SC04)`
- infoblock_labels: `Дата, Время, Локация, Цена`
- literal_items: `5`

### `2759` `Выставка «Королева Луиза: идеал или красивая легенда?»`

- title_strategy: `keep`
- sections: `lead:narrative(EC01,EC02) -> body:narrative [О выставке](SC01,SC02,SC03,SC04)`
- infoblock_labels: `Дата, Локация, Локация`
- literal_items: `0`

### `2657` `Коллекция украшений 1930–1960-х годов`

- title_strategy: `keep`
- sections: `lead:narrative(EC01,EC02) -> body:narrative [О коллекции](EC03,EC04) -> body:narrative(SC05,SC02)`
- infoblock_labels: `Дата, Дата, Дата, Билеты, Билеты`
- literal_items: `0`

### `2447` `🎨 Мастер-класс «Мирное небо»`

- title_strategy: `keep`
- sections: `lead:narrative(EC01,PR01) -> body:narrative [О мастер-классе](EC03,EC02,EC04,SC01,SC02) -> body:narrative [Художники](PR02,SC03)`
- infoblock_labels: `Дата, Время, Локация, Цена, Билеты`
- literal_items: `0`

## 4. Findings

- `writer_pack.compose` keeps document order through a flat `sections` array rather than splitting lead/body/program into separate roots.
- `writer_pack.select` is an identity/no-op stage in `iter2`.
- Program sections separate verbatim `literal_items` from narrative facts and track deterministic coverage through `coverage_plan`.
- Infoblock entries are normalized into canonical labels and sorted deterministically before the final writer sees them.
