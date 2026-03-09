# Smart Update Gemma Event Copy V2.12 Prompt Context

Дата: 2026-03-08

Связанные материалы:

- `artifacts/codex/experimental_pattern_dryrun_v2_12_2026_03_08.py`
- `artifacts/codex/gemma_event_copy_pattern_dry_run_v2_12_5events_2026-03-08.json`
- `docs/reports/smart-update-opus-gemma-event-copy-pattern-dry-run-v2-12-5-events-2026-03-08.md`

## 1. Зачем был собран `v2.12`

`v2.12` — это первая попытка уйти от почти исчерпанной линии:

- `subset extraction -> dirty merge/floor -> generation`.

Новая идея:

- не выбирать "лучшее подмножество" facts;
- а нормализовать всю релевантную `raw_facts`-базу в `clean narrative-ready floor`;
- потом уже делать deterministic hygiene/dedup;
- и только потом отдавать всё в generation.

Важно:

- это experimental harness, а не production runtime;
- цель та же: лучшее итоговое описание события, а не локальная победа на 5 кейсах;
- baseline `Style C`, research по telegram patterns, journalist lexicon и prose-wins `v2.3/v2.6` используются как источники сильных приёмов, но не как единственная основа архитектуры.

## 2. Архитектура `v2.12`

Pipeline:

1. `raw_facts`
2. cheap `shape detection`
3. `full-floor LLM normalization`
4. deterministic `cleanup / dedup / cap`
5. generation
6. optional editorial review pass
7. final policy-safe acceptance check

Shapes:

- `presentation_project`
- `lecture_person`
- `program_rich`
- `sparse_cultural`
- `generic`

Branching:

- `compact_fact_led` только для sparse;
- `fact_first_v2_12` для `presentation / lecture / program` cases.

## 3. Normalization Prompt

Ключевой контракт normalizer:

- обрабатывать ВЕСЬ список `candidate_facts` построчно;
- один input fact -> максимум один output `clean_fact`;
- не дробить grouped list в несколько facts;
- не выдумывать;
- не дропать содержательный fact только потому, что он выражен через `расскажут`, `посвящ`, `представит`;
- переписывать intent/metatext в clean fact;
- дропать только сервис, CTA, афиши, ссылки, билеты, логистику и truly non-publishable строки.

Ключевые generic rules:

- убрать дату, время, локацию, вход, билеты, регистрацию, возраст, ссылки;
- ban на label-style forms: `Тема: ...`, `Идея: ...`, `Цель: ...`;
- ban на корень `посвящ`;
- ban на metatext/intent frames, если смысл можно назвать напрямую;
- разрешён natural event framing вроде `лекция о ...`, `концерт с программой ...`, `выставка ...`.

## 4. Shape-Specific Intent

`presentation_project`

- преобразовывать `расскажут / покажут / представят` в noun-phrase facts;
- держать `agenda blocks` отдельно;
- не терять `problem / reasons / structure / opportunities`.

`lecture_person`

- не терять grouped names;
- не дробить person-rich blocks в шум;
- избегать музейного канцелярита.

`program_rich`

- не заменять названия программы vague-пересказом;
- не терять concrete program items.

`sparse_cultural`

- не раздувать facts;
- готовить плотный floor для короткого prose-текста.

## 5. Generation Prompt

Главная установка generation:

- писать как сильный культурный телеграм-пост;
- живо, профессионально, без рекламности и без шаблонности;
- каждый факт должен прозвучать, но не механическим списком;
- headings должны быть полезными и конкретными;
- при малом количестве фактов лучше короткий цельный текст, чем раздутый конструктор.

Ключевые bans:

- не начинать с `Это...`, `Лекция расскажет...`, `Спектакль рассказывает...`, `Автор ...`;
- не использовать бюрократические рамки;
- не использовать филлеры и рекламные клише;
- не использовать inline quotes;
- не делать generic headings вроде `О событии`, `Подробности`, `Основная идея`, `Формат мероприятия`.

Что разрешено:

- natural event framing;
- short `why it matters`, если он реально grounded;
- 2-3 meaningful sections для rich cases;
- compact 1-2 абзаца для sparse cases;
- concrete nouns и program details вместо общих эпитетов.

## 6. Editorial Review Pass

Editorial pass запускается:

- если есть `missing`;
- если есть `policy_issues`;
- всегда на `presentation / lecture / program` cases;
- на sparse — только при signal of trouble.

Его задача:

- добиться полного покрытия floor;
- убрать `посвящ`, `расскажут`, `представит`, `обсудят`;
- убрать дубли и inline quotes;
- не терять grouped items;
- сохранить естественный профессиональный тон.

Есть safe acceptance:

- кандидат принимается только если не ухудшает `missing / forbidden / policy issues`.

## 7. Что в `v2.12` реально оказалось полезным

- `full-floor normalization` реально снял проблему грязного `merge-back`;
- `2673` и `2687` заметно оздоровились против `v2.11`;
- `2660` вернулся к сильному sparse coverage;
- branch separation `sparse -> compact`, `rich -> standard` оказалась полезной.

## 8. Что осталось blocker

- `посвящ*` всё ещё доживает до финала на `2660 / 2734 / 2687`;
- `2745` звучит естественно, но literal coverage-чек считает, что все 5 facts paraphrased слишком свободно;
- `2673` стал заметно лучше `v2.11`, но prose всё ещё explanation-heavy и частично бюрократичен;
- `program/presentation` cases всё ещё тянут Gemma к generic high-level rewrite.
