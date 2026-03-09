# Smart Update Gemma Event Copy V2.16.2 Lollipop Facts.Extract Family Lab

Дата: 2026-03-09

## 1. Scope

- family: `facts.extract`
- events in casebook: `12`
- stage candidates: `15`
- merge/select ещё не выполнялись; это pre-merge family run
- infoblock/text generation в этот раунд не запускались

## 2. Casebook

- `2673` `presentation` | sources=`2` | facts=`32` | Собакусъел
- `2687` `лекция` | sources=`2` | facts=`24` | 📚 Лекция «Художницы»
- `2734` `concert` | sources=`2` | facts=`25` | Концерт Владимира Гудожникова «Ты… моя мелодия»
- `2659` `кинопоказ` | sources=`2` | facts=`29` | Посторонний
- `2731` `party` | sources=`2` | facts=`31` | Хоровая вечеринка «Праздник у девчат»
- `2498` `спектакль` | sources=`3` | facts=`17` | Нюрнберг
- `2747` `кинопоказ` | sources=`1` | facts=`13` | Киноклуб: «Последнее метро»
- `2701` `party` | sources=`1` | facts=`15` | «Татьяна танцует»
- `2732` `party` | sources=`1` | facts=`12` | Вечер в русском стиле
- `2759` `выставка` | sources=`4` | facts=`46` | Выставка «Королева Луиза: идеал или красивая легенда?»
- `2657` `выставка` | sources=`2` | facts=`27` | Коллекция украшений 1930–1960-х годов
- `2447` `мастер-класс` | sources=`3` | facts=`29` | 🎨 Мастер-класс «Мирное небо»

## 3. Priority View

- `facts.extract_card.v1` -> `broad_candidate` | grounded_events=`12`/12, nonempty=`12`, avg_grounded_ratio=`0.983`
- `facts.extract_theme.v1` -> `broad_candidate` | grounded_events=`12`/12, nonempty=`12`, avg_grounded_ratio=`0.979`
- `facts.extract_subject.v1` -> `broad_candidate` | grounded_events=`12`/12, nonempty=`12`, avg_grounded_ratio=`0.972`
- `facts.extract_concept.v1` -> `broad_candidate` | grounded_events=`12`/12, nonempty=`12`, avg_grounded_ratio=`0.931`
- `facts.extract_identity.v1` -> `broad_candidate` | grounded_events=`12`/12, nonempty=`12`, avg_grounded_ratio=`0.924`
- `facts.extract_cluster.v1` -> `broad_candidate` | grounded_events=`12`/12, nonempty=`12`, avg_grounded_ratio=`0.903`
- `facts.extract_stage.v1` -> `broad_candidate` | grounded_events=`11`/12, nonempty=`11`, avg_grounded_ratio=`0.97`
- `facts.extract_performer.v1` -> `broad_candidate` | grounded_events=`11`/12, nonempty=`11`, avg_grounded_ratio=`0.939`
- `facts.extract_agenda.v1` -> `broad_candidate` | grounded_events=`11`/12, nonempty=`11`, avg_grounded_ratio=`0.933`
- `facts.extract_support.v1` -> `broad_candidate` | grounded_events=`11`/12, nonempty=`11`, avg_grounded_ratio=`0.906`
- `facts.extract_participation.v1` -> `broad_candidate` | grounded_events=`10`/12, nonempty=`10`, avg_grounded_ratio=`0.933`
- `facts.extract_setlist.v1` -> `broad_candidate` | grounded_events=`10`/12, nonempty=`10`, avg_grounded_ratio=`0.832`
- `facts.extract_program.v1` -> `broad_candidate` | grounded_events=`8`/12, nonempty=`8`, avg_grounded_ratio=`0.854`
- `facts.extract_program_shape.v1` -> `broad_candidate` | grounded_events=`7`/12, nonempty=`7`, avg_grounded_ratio=`0.857`
- `facts.extract_profiles.v1` -> `broad_candidate` | grounded_events=`5`/12, nonempty=`5`, avg_grounded_ratio=`0.945`

## 4. Analyst Review

- Broad-run completed successfully on the full `12`-event casebook: `180/180` stage calls saved with pre-merge raw outputs, prompts, parsed JSON and per-stage metrics.
- Реально универсальный core bank уже виден: `facts.extract_card.v1`, `facts.extract_theme.v1`, `facts.extract_subject.v1`, `facts.extract_concept.v1`, `facts.extract_identity.v1` и `facts.extract_cluster.v1` стабильно дали nonempty + grounded outputs на всех `12` кейсах и покрыли все event types casebook'а.
- Второй слой широкого применения тоже подтвердился, но уже слабее: `facts.extract_stage.v1`, `facts.extract_performer.v1`, `facts.extract_agenda.v1`, `facts.extract_support.v1` уверенно работают на большинстве кейсов, однако не выглядят одинаково нужными для каждого shape.
- Узкие/specialized stages уже выделились, несмотря на auto-classification `broad_candidate`: `facts.extract_participation.v1`, `facts.extract_setlist.v1`, `facts.extract_program.v1`, `facts.extract_program_shape.v1`, `facts.extract_profiles.v1` надо дальше рассматривать как type-biased bank, а не как безусловный universal core.
- Текущая эвристика classification слишком доброжелательна: она смотрит на coverage и grounded quotes, но почти не видит semantic overreach. Поэтому `broad_candidate` в этом раунде нужно читать как `worth keeping`, а не как `готово к безусловному merge`.
- Самые явные overreach cases:
- `facts.extract_setlist.v1` вытаскивает не только literal repertoire, но и screening/exhibition card facts. На `2659` stage вернул `Фильм: «Посторонний»`, `Режиссёр фильма: Франсуа Озон`, `Экранизация романа...`; на `2759` stage превратил состав экспозиции в pseudo-setlist.
- `facts.extract_program.v1` на `2687` тащит lecture profiles и логистику как будто это program sequence. Groundedness есть лишь частично (`0.571`), но stage по смыслу уже расползается.
- `facts.extract_program_shape.v1` на `2732` даёт низкий grounded ratio (`0.333`) и показывает, что stage пока плохо держит границу между literal program items и перефразированной event activity prose.
- `facts.extract_profiles.v1` хорош на `2747`, `2759`, `2657`, но на `2673` раздувается в mixed bag из subject/agendas/support-service facts (`печенье`, `чаем напоим`, audience-facing lines). То есть stage полезен, но пока недостаточно profile-pure.
- Самые полезные positive signals для carry-forward:
- `facts.extract_card.v1` стал лучшим literal card extractor по всему кейсбуку.
- `facts.extract_theme.v1` и `facts.extract_subject.v1` дали стабильный canonical core для всех `12` кейсов.
- `facts.extract_concept.v1` оказался surprisingly transferable и для exhibition/screening/party, не только для concert-rich cases.
- `facts.extract_program_shape.v1` дал хорошие shape-rich outputs на `2734`, `2747`, `2447`, то есть stage стоит оставлять как specialized candidate для later select/merge.
- Рабочая priority grouping после первого family lab:
- `priority_core`: `facts.extract_card.v1`, `facts.extract_theme.v1`, `facts.extract_subject.v1`, `facts.extract_concept.v1`, `facts.extract_identity.v1`
- `secondary_broad`: `facts.extract_cluster.v1`, `facts.extract_stage.v1`, `facts.extract_performer.v1`, `facts.extract_agenda.v1`, `facts.extract_support.v1`
- `specialized_keep`: `facts.extract_participation.v1`, `facts.extract_setlist.v1`, `facts.extract_program.v1`, `facts.extract_program_shape.v1`, `facts.extract_profiles.v1`
- Следующий шаг для этой family уже не consultation всего пакета, а raw family review по нескольким representative кейсам с вопросом: какие prompts оставить как priority, какие сузить, и какие новые versions добавить именно для `setlist/program/program_shape/profiles`.

## 5. Stage Stats

### `facts.extract_card.v1`

- lineage: `normalize_card_v1 <- v2.15.8`
- classification: `broad_candidate`
- nonempty events: `12`
- grounded events: `12`
- total items: `55`
- avg grounded ratio on nonempty events: `0.983`
- grounded event types: `concert, party, presentation, выставка, кинопоказ, лекция, мастер-класс, спектакль`
- nonempty event ids: `2673, 2687, 2734, 2659, 2731, 2498, 2747, 2701, 2732, 2759, 2657, 2447`

### `facts.extract_theme.v1`

- lineage: `theme_v1_compact <- v2.15.6`
- classification: `broad_candidate`
- nonempty events: `12`
- grounded events: `12`
- total items: `51`
- avg grounded ratio on nonempty events: `0.979`
- grounded event types: `concert, party, presentation, выставка, кинопоказ, лекция, мастер-класс, спектакль`
- nonempty event ids: `2673, 2687, 2734, 2659, 2731, 2498, 2747, 2701, 2732, 2759, 2657, 2447`

### `facts.extract_subject.v1`

- lineage: `subject_v1_strict <- v2.15.5`
- classification: `broad_candidate`
- nonempty events: `12`
- grounded events: `12`
- total items: `43`
- avg grounded ratio on nonempty events: `0.972`
- grounded event types: `concert, party, presentation, выставка, кинопоказ, лекция, мастер-класс, спектакль`
- nonempty event ids: `2673, 2687, 2734, 2659, 2731, 2498, 2747, 2701, 2732, 2759, 2657, 2447`

### `facts.extract_concept.v1`

- lineage: `concept_v1_compact <- v2.15.7`
- classification: `broad_candidate`
- nonempty events: `12`
- grounded events: `12`
- total items: `40`
- avg grounded ratio on nonempty events: `0.931`
- grounded event types: `concert, party, presentation, выставка, кинопоказ, лекция, мастер-класс, спектакль`
- nonempty event ids: `2673, 2687, 2734, 2659, 2731, 2498, 2747, 2701, 2732, 2759, 2657, 2447`

### `facts.extract_identity.v1`

- lineage: `normalize_identity_v2_strict <- v2.15.8`
- classification: `broad_candidate`
- nonempty events: `12`
- grounded events: `12`
- total items: `47`
- avg grounded ratio on nonempty events: `0.924`
- grounded event types: `concert, party, presentation, выставка, кинопоказ, лекция, мастер-класс, спектакль`
- nonempty event ids: `2673, 2687, 2734, 2659, 2731, 2498, 2747, 2701, 2732, 2759, 2657, 2447`

### `facts.extract_cluster.v1`

- lineage: `cluster_v2_named_group <- v2.15.6`
- classification: `broad_candidate`
- nonempty events: `12`
- grounded events: `12`
- total items: `35`
- avg grounded ratio on nonempty events: `0.903`
- grounded event types: `concert, party, presentation, выставка, кинопоказ, лекция, мастер-класс, спектакль`
- nonempty event ids: `2673, 2687, 2734, 2659, 2731, 2498, 2747, 2701, 2732, 2759, 2657, 2447`

### `facts.extract_stage.v1`

- lineage: `stage_v2_compact <- v2.15.7`
- classification: `broad_candidate`
- nonempty events: `11`
- grounded events: `11`
- total items: `60`
- avg grounded ratio on nonempty events: `0.97`
- grounded event types: `concert, party, presentation, выставка, кинопоказ, лекция, мастер-класс, спектакль`
- nonempty event ids: `2673, 2687, 2734, 2731, 2498, 2747, 2701, 2732, 2759, 2657, 2447`

### `facts.extract_performer.v1`

- lineage: `performer_v1_awards <- v2.15.7`
- classification: `broad_candidate`
- nonempty events: `11`
- grounded events: `11`
- total items: `31`
- avg grounded ratio on nonempty events: `0.939`
- grounded event types: `concert, party, presentation, выставка, кинопоказ, лекция, мастер-класс, спектакль`
- nonempty event ids: `2673, 2687, 2734, 2659, 2731, 2498, 2747, 2701, 2759, 2657, 2447`

### `facts.extract_agenda.v1`

- lineage: `agenda_v2_prose_ready <- v2.15.5`
- classification: `broad_candidate`
- nonempty events: `11`
- grounded events: `11`
- total items: `51`
- avg grounded ratio on nonempty events: `0.933`
- grounded event types: `concert, party, presentation, выставка, кинопоказ, лекция, мастер-класс, спектакль`
- nonempty event ids: `2673, 2687, 2734, 2659, 2731, 2498, 2747, 2701, 2732, 2759, 2447`

### `facts.extract_support.v1`

- lineage: `normalize_support_v1 <- v2.15.8`
- classification: `broad_candidate`
- nonempty events: `11`
- grounded events: `11`
- total items: `55`
- avg grounded ratio on nonempty events: `0.906`
- grounded event types: `concert, party, presentation, выставка, кинопоказ, лекция, мастер-класс, спектакль`
- nonempty event ids: `2673, 2687, 2734, 2659, 2498, 2747, 2701, 2732, 2759, 2657, 2447`

### `facts.extract_participation.v1`

- lineage: `normalize_participation_v1 <- v2.15.8`
- classification: `broad_candidate`
- nonempty events: `10`
- grounded events: `10`
- total items: `36`
- avg grounded ratio on nonempty events: `0.933`
- grounded event types: `concert, party, presentation, выставка, кинопоказ, лекция, мастер-класс`
- nonempty event ids: `2673, 2687, 2734, 2659, 2731, 2701, 2732, 2759, 2657, 2447`

### `facts.extract_setlist.v1`

- lineage: `setlist_v1_grouped <- v2.15.7`
- classification: `broad_candidate`
- nonempty events: `10`
- grounded events: `10`
- total items: `40`
- avg grounded ratio on nonempty events: `0.832`
- grounded event types: `concert, party, presentation, выставка, кинопоказ, мастер-класс, спектакль`
- nonempty event ids: `2673, 2734, 2659, 2498, 2747, 2701, 2732, 2759, 2657, 2447`

### `facts.extract_program.v1`

- lineage: `program_v1_compact <- v2.15.5`
- classification: `broad_candidate`
- nonempty events: `8`
- grounded events: `8`
- total items: `53`
- avg grounded ratio on nonempty events: `0.854`
- grounded event types: `concert, party, presentation, выставка, кинопоказ, лекция, мастер-класс`
- nonempty event ids: `2673, 2687, 2734, 2747, 2701, 2732, 2657, 2447`

### `facts.extract_program_shape.v1`

- lineage: `normalize_program_v1 <- v2.15.8`
- classification: `broad_candidate`
- nonempty events: `7`
- grounded events: `7`
- total items: `42`
- avg grounded ratio on nonempty events: `0.857`
- grounded event types: `concert, party, presentation, выставка, кинопоказ, мастер-класс`
- nonempty event ids: `2673, 2734, 2747, 2701, 2732, 2657, 2447`

### `facts.extract_profiles.v1`

- lineage: `profiles_v1_literal <- v2.15.6`
- classification: `broad_candidate`
- nonempty events: `5`
- grounded events: `5`
- total items: `31`
- avg grounded ratio on nonempty events: `0.945`
- grounded event types: `concert, presentation, выставка, кинопоказ`
- nonempty event ids: `2673, 2734, 2747, 2759, 2657`

## 6. Event-Level Snapshot

### `2673` `Собакусъел`

- event_type: `presentation`
- sources: `2`
- raw facts: `32`
- nonempty stages: `facts.extract_subject.v1, facts.extract_agenda.v1, facts.extract_program.v1, facts.extract_cluster.v1, facts.extract_theme.v1, facts.extract_profiles.v1, facts.extract_concept.v1, facts.extract_setlist.v1, facts.extract_performer.v1, facts.extract_stage.v1, facts.extract_card.v1, facts.extract_support.v1, facts.extract_identity.v1, facts.extract_participation.v1, facts.extract_program_shape.v1`

### `2687` `📚 Лекция «Художницы»`

- event_type: `лекция`
- sources: `2`
- raw facts: `24`
- nonempty stages: `facts.extract_subject.v1, facts.extract_agenda.v1, facts.extract_program.v1, facts.extract_cluster.v1, facts.extract_theme.v1, facts.extract_concept.v1, facts.extract_performer.v1, facts.extract_stage.v1, facts.extract_card.v1, facts.extract_support.v1, facts.extract_identity.v1, facts.extract_participation.v1`

### `2734` `Концерт Владимира Гудожникова «Ты… моя мелодия»`

- event_type: `concert`
- sources: `2`
- raw facts: `25`
- nonempty stages: `facts.extract_subject.v1, facts.extract_agenda.v1, facts.extract_program.v1, facts.extract_cluster.v1, facts.extract_theme.v1, facts.extract_profiles.v1, facts.extract_concept.v1, facts.extract_setlist.v1, facts.extract_performer.v1, facts.extract_stage.v1, facts.extract_card.v1, facts.extract_support.v1, facts.extract_identity.v1, facts.extract_participation.v1, facts.extract_program_shape.v1`

### `2659` `Посторонний`

- event_type: `кинопоказ`
- sources: `2`
- raw facts: `29`
- nonempty stages: `facts.extract_subject.v1, facts.extract_agenda.v1, facts.extract_cluster.v1, facts.extract_theme.v1, facts.extract_concept.v1, facts.extract_setlist.v1, facts.extract_performer.v1, facts.extract_card.v1, facts.extract_support.v1, facts.extract_identity.v1, facts.extract_participation.v1`

### `2731` `Хоровая вечеринка «Праздник у девчат»`

- event_type: `party`
- sources: `2`
- raw facts: `31`
- nonempty stages: `facts.extract_subject.v1, facts.extract_agenda.v1, facts.extract_cluster.v1, facts.extract_theme.v1, facts.extract_concept.v1, facts.extract_performer.v1, facts.extract_stage.v1, facts.extract_card.v1, facts.extract_identity.v1, facts.extract_participation.v1`

### `2498` `Нюрнберг`

- event_type: `спектакль`
- sources: `3`
- raw facts: `17`
- nonempty stages: `facts.extract_subject.v1, facts.extract_agenda.v1, facts.extract_cluster.v1, facts.extract_theme.v1, facts.extract_concept.v1, facts.extract_setlist.v1, facts.extract_performer.v1, facts.extract_stage.v1, facts.extract_card.v1, facts.extract_support.v1, facts.extract_identity.v1`

### `2747` `Киноклуб: «Последнее метро»`

- event_type: `кинопоказ`
- sources: `1`
- raw facts: `13`
- nonempty stages: `facts.extract_subject.v1, facts.extract_agenda.v1, facts.extract_program.v1, facts.extract_cluster.v1, facts.extract_theme.v1, facts.extract_profiles.v1, facts.extract_concept.v1, facts.extract_setlist.v1, facts.extract_performer.v1, facts.extract_stage.v1, facts.extract_card.v1, facts.extract_support.v1, facts.extract_identity.v1, facts.extract_program_shape.v1`

### `2701` `«Татьяна танцует»`

- event_type: `party`
- sources: `1`
- raw facts: `15`
- nonempty stages: `facts.extract_subject.v1, facts.extract_agenda.v1, facts.extract_program.v1, facts.extract_cluster.v1, facts.extract_theme.v1, facts.extract_concept.v1, facts.extract_setlist.v1, facts.extract_performer.v1, facts.extract_stage.v1, facts.extract_card.v1, facts.extract_support.v1, facts.extract_identity.v1, facts.extract_participation.v1, facts.extract_program_shape.v1`

### `2732` `Вечер в русском стиле`

- event_type: `party`
- sources: `1`
- raw facts: `12`
- nonempty stages: `facts.extract_subject.v1, facts.extract_agenda.v1, facts.extract_program.v1, facts.extract_cluster.v1, facts.extract_theme.v1, facts.extract_concept.v1, facts.extract_setlist.v1, facts.extract_stage.v1, facts.extract_card.v1, facts.extract_support.v1, facts.extract_identity.v1, facts.extract_participation.v1, facts.extract_program_shape.v1`

### `2759` `Выставка «Королева Луиза: идеал или красивая легенда?»`

- event_type: `выставка`
- sources: `4`
- raw facts: `46`
- nonempty stages: `facts.extract_subject.v1, facts.extract_agenda.v1, facts.extract_cluster.v1, facts.extract_theme.v1, facts.extract_profiles.v1, facts.extract_concept.v1, facts.extract_setlist.v1, facts.extract_performer.v1, facts.extract_stage.v1, facts.extract_card.v1, facts.extract_support.v1, facts.extract_identity.v1, facts.extract_participation.v1`

### `2657` `Коллекция украшений 1930–1960-х годов`

- event_type: `выставка`
- sources: `2`
- raw facts: `27`
- nonempty stages: `facts.extract_subject.v1, facts.extract_program.v1, facts.extract_cluster.v1, facts.extract_theme.v1, facts.extract_profiles.v1, facts.extract_concept.v1, facts.extract_setlist.v1, facts.extract_performer.v1, facts.extract_stage.v1, facts.extract_card.v1, facts.extract_support.v1, facts.extract_identity.v1, facts.extract_participation.v1, facts.extract_program_shape.v1`

### `2447` `🎨 Мастер-класс «Мирное небо»`

- event_type: `мастер-класс`
- sources: `3`
- raw facts: `29`
- nonempty stages: `facts.extract_subject.v1, facts.extract_agenda.v1, facts.extract_program.v1, facts.extract_cluster.v1, facts.extract_theme.v1, facts.extract_concept.v1, facts.extract_setlist.v1, facts.extract_performer.v1, facts.extract_stage.v1, facts.extract_card.v1, facts.extract_support.v1, facts.extract_identity.v1, facts.extract_participation.v1, facts.extract_program_shape.v1`

## 6. Findings

- Это уже реальный new-run family lab, а не review исторических outputs.
- Статистика здесь относится только к extraction-candidates до merge/select.
- `classification` пока предварительная: она показывает, какие stage ids уже выглядят пригодными для broad-run, а какие пока слабые или слишком узкие.
- Следующий ход после этого отчёта: narrow family review по raw outputs, затем prompt expansion внутри `facts.extract`, и только потом проектирование `facts.merge.tier1/tier2`.
