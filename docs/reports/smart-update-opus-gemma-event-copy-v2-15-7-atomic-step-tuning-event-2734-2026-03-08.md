# Smart Update Gemma Event Copy V2.15.7 Atomic Step Tuning - Event 2734

Date: 2026-03-08

Это третий atomic cycle: после `2673` и `2687` разбор перенесён на `program_rich` кейс `2734`, где проблема живёт не только в генерации, но и в потере setlist facts ещё до generation.

## 1. Зачем этот кейс

- `2734` нужен как тест на program sharpness и на сохранение attendee-useful фактов.
- Здесь `v2.15.3` терял конкретику: вместо setlist оставалась абстрактная формула про произведения для Магомаева.
- Поэтому цикл разделён уже не только по generation, но и по pre-generation extraction contracts.

## 2. Fact Slots

- `performer_named`
- `laureate`
- `love_story`
- `program_core`
- `setlist`
- `muse_dancer`

## 3. Baseline

- Baseline missing deterministic: `3`
- Baseline slot score: `3/6`

## 4. Atomic Steps

### normalize_concept

- Win achieved: `yes`
- Chosen variant: `concept_v1_compact`

#### concept_v1_compact

- Note: Базовое сохранение love-story и program-core.
- Win: `yes`
- concept_facts: `['Концерт посвящен великой любви Муслима Магомаева и Тамары Синявской.', 'В программе концерта прозвучат произведения, писавшиеся для Муслима Магомаева.']`
- love_story: `True`
- program_core: `True`
- statement_count: `2`
- abstract_filler_hits: `0`
- score: `3`

### normalize_setlist

- Win achieved: `yes`
- Chosen variant: `setlist_v1_grouped`

#### setlist_v1_grouped

- Note: Базовое grouped-сохранение четырёх названий.
- Win: `yes`
- setlist_facts: `['В программе концерта: «Верни мне музыку», «Лучший город земли», «Королева красоты», «Разговор со счастьем» и другие композиции.']`
- all_titles_present: `True`
- grouped_fact: `True`
- abstract_filler_hits: `0`
- score: `2`

### normalize_performer

- Win achieved: `yes`
- Chosen variant: `performer_v1_awards`

#### performer_v1_awards

- Note: Базовый performer fact с конкурсным статусом.
- Win: `yes`
- performer_facts: `['Владимир Гудожников – лауреат всероссийских и международных конкурсов, включая «Янтарный соловей».']`
- performer_named: `True`
- laureate: `True`
- award_named: `True`
- abstract_filler_hits: `0`
- score: `3`

### normalize_stage

- Win achieved: `yes`
- Chosen variant: `stage_v2_compact`

#### stage_v1_basic

- Note: Базовый сценический fact про Музы.
- Win: `no`
- stage_facts: `['Владимир Гудожников - хореограф и постановщик концерта.', 'Концерт называется «Ты… моя мелодия».', 'В концерте представлена Муза.']`
- muse_dancer: `False`
- abstract_filler_hits: `0`
- hallucination_hits: `0`
- score: `0`

#### stage_v2_compact

- Note: То же, но без abstract embellishment.
- Win: `yes`
- stage_facts: `['Танцовщица в образе Музы']`
- muse_dancer: `True`
- abstract_filler_hits: `0`
- hallucination_hits: `0`
- score: `1`

### plan_lead

- Win achieved: `yes`
- Chosen variant: `plan_v1_basic`

#### plan_v1_basic

- Note: Lead через исполнителя и тему, concrete program facts в body.
- Win: `yes`
- lead_fact_ids: `[1, 2]`
- body_fact_ids: `[3, 4, 5]`
- lead_has_performer: `True`
- lead_has_love_story: `True`
- lead_has_setlist: `False`
- body_has_program: `True`
- body_has_setlist: `True`
- body_has_muse: `True`
- coverage_exact_once: `True`
- score: `6`

### generate_lead

- Win achieved: `yes`
- Chosen variant: `lead_v2_title_and_performer`

#### lead_v1_performer_theme

- Note: Lead через исполнителя и тему любви.
- Win: `no`
- first_sentence: `Лауреат всероссийских и международных конкурсов Владимир Гудожников приглашает вас на концерт «Ты…`
- mentions_performer: `True`
- mentions_concert: `True`
- mentions_love_story: `True`
- mentions_laureate: `False`
- mentions_program_core: `False`
- mentions_setlist: `False`
- abstract_filler_hits: `0`
- hallucination_hits: `0`
- sentence_count: `2`
- score: `3`

```md
Лауреат всероссийских и международных конкурсов Владимир Гудожников приглашает вас на концерт «Ты… моя мелодия», посвященный трогательной и вдохновляющей истории любви Муслима Магомаева и Тамары Синявской. Это вечер, пронизанный не только музыкой, но и искренними чувствами двух легенд.
```

#### lead_v2_title_and_performer

- Note: То же, но с более жёстким title/performer opening.
- Win: `yes`
- first_sentence: `Лауреат всероссийских и международных конкурсов, в том числе «Янтарный соловей», Владимир Гудожников приглашает вас на концерт «Ты…`
- mentions_performer: `True`
- mentions_concert: `True`
- mentions_love_story: `True`
- mentions_laureate: `True`
- mentions_program_core: `False`
- mentions_setlist: `False`
- abstract_filler_hits: `0`
- hallucination_hits: `0`
- sentence_count: `2`
- score: `4`

```md
Лауреат всероссийских и международных конкурсов, в том числе «Янтарный соловей», Владимир Гудожников приглашает вас на концерт «Ты… моя мелодия». Этот вечер станет музыкальным посвящением великой и трогательной любви Муслима Магомаева и Тамары Синявской.
```

### generate_body

- Win achieved: `yes`
- Chosen variant: `body_v1_inline_setlist`

#### body_v1_inline_setlist

- Note: Body с inline-покрытием program facts.
- Win: `yes`
- slots: `performer_named=yes, laureate=yes, love_story=yes, program_core=yes, setlist=yes, muse_dancer=yes`
- slot_score: `6`
- missing_deterministic: `['Владимир Гудожников – лауреат всероссийских и международных конкурсов, включая «Янтарный соловей».', 'Концерт посвящен великой любви Муслима Магомаева и Тамары Синявской.', 'В программе концерта: «Верни мне музыку», «Лучший город земли», «Королева красоты», «Разговор со счастьем» и другие композиции.', 'Танцовщица в образе Музы']`
- missing_count: `4`
- abstract_filler_hits: `0`
- hallucination_hits: `0`
- lead_ok: `True`
- score: `17`

```md
В программе концерта прозвучат произведения, писавшиеся для Муслима Магомаева. Вас ждут любимые многими «Верни мне музыку», «Лучший город земли», «Королева красоты», «Разговор со счастьем» и другие композиции, ставшие символами эпохи. Дополнит музыкальное впечатление выступление танцовщицы в образе Музы, вдохновлявшей великого певца.
```

### repair

- Win achieved: `no`
- Chosen variant: `n/a`

## 5. Final Text

```md
Лауреат всероссийских и международных конкурсов, в том числе «Янтарный соловей», Владимир Гудожников приглашает вас на концерт «Ты… моя мелодия». Этот вечер станет музыкальным посвящением великой и трогательной любви Муслима Магомаева и Тамары Синявской.

В программе концерта прозвучат произведения, писавшиеся для Муслима Магомаева. Вас ждут любимые многими «Верни мне музыку», «Лучший город земли», «Королева красоты», «Разговор со счастьем» и другие композиции, ставшие символами эпохи. Дополнит музыкальное впечатление выступление танцовщицы в образе Музы, вдохновлявшей великого певца.
```

## 6. Final Metrics

- Slot score: `6/6`
- Missing deterministic: `4`
- Abstract filler hits: `0`
- Hallucination hits: `0`
- Lead ok: `yes`

## 7. Findings

- Для `program_rich` кейса универсализация требует отдельного `setlist` шага, иначе конкретные attendee-useful facts теряются ещё до generation.
- Это подтверждает правило пользователя про `все факты`: в некоторых shape-ах надо fine-tune не только prose prompts, но и contracts подготовки fact-floor.
- Единым должен быть не giant prompt, а shape-aware atomic pipeline: одинаковая логика цикла, разные микроконтракты по типу события.
- Exact-match `missing_deterministic` здесь остаётся вторичным шумным индикатором: при полном slot coverage он всё ещё штрафует естественную переформулировку.
