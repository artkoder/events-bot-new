# Smart Update Gemma Event Copy V2.16.2 Lollipop Writer_Pack.Compose Family Lab

Дата: 2026-03-10

## 1. Scope

- family: `writer_pack.compose`
- iteration: `iter1`
- upstream input: full `12`-event `editorial.layout iter1` pack on top of `facts.prioritize iter2`
- active stages: `writer_pack.compose.v1 -> writer_pack.select.v1 -> deterministic audit`
- this round does not rerun `facts.extract`, `facts.dedup`, `facts.merge`, `facts.prioritize`, or `editorial.layout`

## 2. Aggregate Metrics

- events: `12`
- avg_section_count: `2.167`
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

- title_strategy: `keep`
- sections: `lead:narrative(EC02,FL02) -> body:narrative [О проекте и презентации](EC01,EC03,FL01,FL03,FL04,FL05,PL01,SC01,SC02,SC03)`
- infoblock_labels: `Дата, Локация, Цена`
- literal_items: `0`

### `2687` `📚 Лекция «Художницы»`

- title_strategy: `keep`
- sections: `lead:narrative(EC01,EC03) -> body:narrative(EC02,SC01,SC02,SC03,SC04)`
- infoblock_labels: `Дата, Локация, Возраст`
- literal_items: `0`

### `2734` `Концерт Владимира Гудожникова «Ты… моя мелодия»`

- title_strategy: `keep`
- sections: `lead:narrative(EC01,PR01) -> body:narrative(SC01) -> program:list(PL01) list=4`
- infoblock_labels: `Дата, Время, Локация, Билеты, Возраст`
- literal_items: `4`

### `2659` `Посторонний`

- title_strategy: `keep`
- sections: `lead:narrative(PR01) -> body:narrative(SC01,SC02,SC03)`
- infoblock_labels: `Дата, Билеты, Возраст, Прочее, Прочее`
- literal_items: `0`

### `2731` `Хоровая вечеринка «Праздник у девчат»`

- title_strategy: `keep`
- sections: `lead:narrative(EC01,PR01) -> body:narrative [Что вас ждет](SC05,SC02,SC03,SC04,SC01) -> program:list [Репертуар](PL01,PL02) list=10`
- infoblock_labels: `Дата, Время, Локация, Цена`
- literal_items: `10`

### `2498` `Нюрнберг`

- title_strategy: `keep`
- sections: `lead:narrative(EC02,EC03) -> body:narrative(EC01,EC04,PR01,SC02,SC01)`
- infoblock_labels: `Дата, Локация`
- literal_items: `0`

### `2747` `Киноклуб: «Последнее метро»`

- title_strategy: `keep`
- sections: `lead:narrative(PR01,PR02) -> body:narrative(SC01,SC02,SC03,SC04)`
- infoblock_labels: `Цена, Прочее`
- literal_items: `0`

### `2701` `«Татьяна танцует»`

- title_strategy: `keep`
- sections: `lead:narrative(EC01,PR01) -> body:narrative(EC02,EC03,PL01,SC01,SC02)`
- infoblock_labels: `Дата, Время, Локация`
- literal_items: `0`

### `2732` `Вечер в русском стиле`

- title_strategy: `keep`
- sections: `lead:narrative(PL01) list=5 -> body:narrative(SC02,SC01,SC03,SC04)`
- infoblock_labels: `Дата, Время, Локация, Цена`
- literal_items: `5`

### `2759` `Выставка «Королева Луиза: идеал или красивая легенда?»`

- title_strategy: `keep`
- sections: `lead:narrative(EC02) -> body:narrative(EC01,SC01,SC02,SC03,SC04)`
- infoblock_labels: `Дата, Локация, Локация`
- literal_items: `0`

### `2657` `Коллекция украшений 1930–1960-х годов`

- title_strategy: `keep`
- sections: `lead:narrative(EC01) -> body:narrative(EC02,EC03,EC04,SC01,SC02,SC03,SC04)`
- infoblock_labels: `Дата, Дата, Дата, Билеты, Билеты`
- literal_items: `0`

### `2447` `🎨 Мастер-класс «Мирное небо»`

- title_strategy: `keep`
- sections: `lead:narrative(EC01,PR01) -> body:narrative [О чём мастер-класс](EC03,EC02,EC04,PR02,SC01,SC03,SC02)`
- infoblock_labels: `Дата, Время, Локация, Цена, Билеты`
- literal_items: `0`

## 4. Findings

- `writer_pack.compose` keeps document order through a flat `sections` array rather than splitting lead/body/program into separate roots.
- `writer_pack.select` is an identity/no-op stage in `iter1`.
- Program sections separate verbatim `literal_items` from narrative facts and track deterministic coverage through `coverage_plan`.
- Infoblock entries are normalized into canonical labels and sorted deterministically before the final writer sees them.
