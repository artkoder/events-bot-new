# Smart Update Gemma Event Copy V2.16.2 Lollipop Facts.Prioritize Family Lab

Дата: 2026-03-10

## 1. Scope

- family: `facts.prioritize`
- iteration: `iter3`
- upstream input: full `12`-event `facts.merge iter5` pack from `2026-03-09`
- active stages: `facts.prioritize.weight.v1 -> facts.prioritize.lead.v1 -> deterministic audit`
- this round does not rerun `facts.extract`, `facts.dedup`, or `facts.merge`

## 2. Aggregate Metrics

- events: `12`
- avg_merge_fact_count: `10.75`
- avg_high_count: `5.5`
- avg_medium_count: `3.083`
- avg_low_count: `2.167`
- events_with_flags: `0`
- events_with_lead_support: `8`
- events_with_fallback_lead: `0`
- events_with_bureaucratic_lead: `0`
- auto_added_weight_decisions_total: `0`

## 3. Event Snapshot

### `2673` `Собакусъел`

- event_type: `presentation`
- merge facts: `15`
- high / medium / low: `6 / 7 / 2`
- lead: `Проект «Собакусъел» представляет собой социальную сеть для профессионалов креативных индустрий.`
- lead support: `Откроется предрегистрация для участников проекта.`
- lead bucket: `event_core`

### `2687` `📚 Лекция «Художницы»`

- event_type: `лекция`
- merge facts: `10`
- high / medium / low: `5 / 1 / 4`
- lead: `Лекция посвящена творчеству Елены Поленовой, Марии Якунчиковой-Вебер, Зинаиды Серебряковой, Натальи Гончаровой, Ольги Розановой и Любови Поповой.`
- lead support: `Лекция расскажет о вкладе художниц в историю русского искусства.`
- lead bucket: `event_core`

### `2734` `Концерт Владимира Гудожникова «Ты… моя мелодия»`

- event_type: `concert`
- merge facts: `9`
- high / medium / low: `7 / 1 / 1`
- lead: `Концерт посвящен великой любви Муслима Магомаева и Тамары Синявской.`
- lead support: `Владимир Гудожников – лауреат всероссийских и международных конкурсов, включая «Янтарный соловей».`
- lead bucket: `event_core`

### `2659` `Посторонний`

- event_type: `кинопоказ`
- merge facts: `9`
- high / medium / low: `5 / 4 / 0`
- lead: `Режиссёр фильма – Франсуа Озон, современный классик французского кино.`
- lead bucket: `people_and_roles`

### `2731` `Хоровая вечеринка «Праздник у девчат»`

- event_type: `party`
- merge facts: `13`
- high / medium / low: `6 / 2 / 5`
- lead: `На вечеринке будет исполнено 10 песен о весне, любви и жизни.`
- lead support: `Мероприятие проводит музыкальная студия «Life.love.songs».`
- lead bucket: `event_core`

### `2498` `Нюрнберг`

- event_type: `спектакль`
- merge facts: `9`
- high / medium / low: `5 / 3 / 1`
- lead: `Спектакль о закулисных отношениях участников Нюрнбергского процесса и любви, разделенной идеологией.`
- lead support: `Спектакль основан на реальных событиях Нюрнбергского процесса.`
- lead bucket: `event_core`

### `2747` `Киноклуб: «Последнее метро»`

- event_type: `кинопоказ`
- merge facts: `8`
- high / medium / low: `2 / 2 / 4`
- lead: `В главных ролях: Катрин Денев и Жерар Депардье.`
- lead support: `Режиссёр фильма — Франсуа Трюффо.`
- lead bucket: `people_and_roles`

### `2701` `«Татьяна танцует»`

- event_type: `party`
- merge facts: `10`
- high / medium / low: `5 / 3 / 2`
- lead: `Вечеринка посвящена международному женскому дню.`
- lead support: `Организатор – бар «Татьяна». Татьяна - влиятельная женщина барной индустрии Калининграда.`
- lead bucket: `event_core`

### `2732` `Вечер в русском стиле`

- event_type: `party`
- merge facts: `9`
- high / medium / low: `5 / 4 / 0`
- lead: `В программе вечера: экскурсия, мастер-классы, викторина, фотозона, конкурс костюмов.`
- lead bucket: `program_list`

### `2759` `Выставка «Королева Луиза: идеал или красивая легенда?»`

- event_type: `выставка`
- merge facts: `9`
- high / medium / low: `5 / 4 / 0`
- lead: `Выставка посвящена биографии и личности королевы Луизы. Королева Луиза является ключевой фигурой выставки. Выставка расскажет о жизни и личности королевы Луизы через исторические документы и предметы быта.`
- lead bucket: `event_core`

### `2657` `Коллекция украшений 1930–1960-х годов`

- event_type: `выставка`
- merge facts: `14`
- high / medium / low: `7 / 3 / 4`
- lead: `Выставка рассказывает об истории зарубежного ювелирного и бижутерийного искусства.`
- lead bucket: `event_core`

### `2447` `🎨 Мастер-класс «Мирное небо»`

- event_type: `мастер-класс`
- merge facts: `14`
- high / medium / low: `8 / 3 / 3`
- lead: `Мастер-класс о жизни после войны в творчестве Л. А. Климентовской и К. М. Кишкина.`
- lead support: `Л. А. Климентовская - художник, чье творчество будет представлено на мастер-классе.`
- lead bucket: `event_core`

## 4. Findings

- `facts.prioritize` keeps all facts as a weighted JSON pack; no prose is generated here.
- `fact_id`-based prompting was used to avoid text drift and make downstream audit deterministic.
- The trace directory is consultation-ready: each event/stage contains `input.json -> prompt.txt -> raw_output.txt -> result.json`.
