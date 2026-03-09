# Smart Update Gemma Event Copy V2.13 Prompt Context

Дата: 2026-03-08

Связанные материалы:

- `artifacts/codex/experimental_pattern_dryrun_v2_13_2026_03_08.py`
- `artifacts/codex/gemma_event_copy_pattern_dry_run_v2_13_5events_2026-03-08.json`
- `docs/reports/smart-update-opus-gemma-event-copy-pattern-dry-run-v2-13-5-events-2026-03-08.md`

## 1. Зачем был собран `v2.13`

`v2.13` — это следующая итерация после `v2.12`, но не новый полный redesign.

Смысл был такой:

- сохранить выигрыш `v2.12` от `full-floor normalization`;
- не потерять prose-силу baseline `Style C` и лучших `v2.6` кейсов;
- убрать тяжёлый `same-model editorial pass`, который делал систему слишком хрупкой;
- сделать generation короче, человечнее и ближе к сильным event-copy patterns;
- заменить full rewrite на `targeted repair only`.

Цель осталась прежней:

- не “победить” ровно эти 5 кейсов любой ценой;
- а найти архитектуру, которая на многих тысячах разных исходников будет стабильно давать чистый, профессиональный, естественный текст описания события.

## 2. Архитектура `v2.13`

Pipeline:

1. `raw_facts`
2. cheap `shape detection`
3. `full-floor LLM normalization`
4. deterministic `cleanup / dedup / cap`
5. `exemplar-driven generation`
6. deterministic validation
7. `targeted repair`
8. optional narrow anti-`посвящ*` cleanup

Shapes:

- `presentation_project`
- `lecture_person`
- `program_rich`
- `sparse_cultural`
- `generic`

Branches:

- `compact_fact_led` для sparse cases;
- `fact_first_v2_13` для richer shapes.

## 3. Главное отличие от `v2.12`

### Что сохранено

- `full-floor normalization`, а не возврат к `subset extraction`;
- deterministic hygiene / dedup / caps;
- shape-aware contracts;
- `anti-quote` / `anti-metatext` / self-contained body;
- targeted anti-`посвящ*` cleanup как narrow fallback.

### Что изменено

- normalizer стал более `mechanical`, а не editorial:
  - удалять рамку `лекция посвящена / расскажут / представят / обсудят`;
  - сохранять только предметное содержание;
  - не добавлять атмосферу, интерпретации и `why it matters`.
- generation prompt стал заметно короче:
  - не большая стена из bans;
  - а shape-specific exemplars для compact / presentation / lecture / program cases.
- full editorial rewrite pass убран;
- вместо него добавлен `targeted repair`:
  - минимально исправить missing/policy issues;
  - сохранить уже найденную структуру и удачные фразы;
  - не переписывать весь текст заново без необходимости.

## 4. Normalization Contract

`v2.13` normalizer должен:

- обработать весь `candidate_facts`-floor;
- работать механически, а не копирайтерски;
- переписывать metatext / intent-style facts в subject-content facts;
- не терять coverage;
- не дробить сгруппированные блоки в шум;
- не добавлять новые оценки, атмосферу или “редакторские смыслы”.

Shape-specific intent:

`presentation_project`

- превращать `расскажут / покажут / представят` в noun-phrase facts;
- держать `problem / reasons / structure / opportunities` раздельно;
- не делать platform story слишком рекламной.

`lecture_person`

- не терять grouped names;
- не превращать person-rich block в общий пересказ;
- не уходить в музейный канцелярит.

`program_rich`

- не заменять program details vague-summary формулой;
- держать concrete program/visual element blocks.

`sparse_cultural`

- не раздувать floor;
- готовить компактный, publishable набор facts для короткого prose.

## 5. Generation Contract

`v2.13` generation должен:

- писать человеческий, профессиональный event-copy текст;
- быть ближе к живому Telegram/journalistic pattern writing;
- сохранять groundedness и coverage;
- не дублировать факты механическим списком;
- не перегружать sparse cases подзаголовками;
- использовать заголовки только там, где они реально помогают.

В generation явно поддержаны:

- short, strong sparse prose;
- self-contained main body;
- cleaner leads;
- less generic headings;
- avoidance of inline quote-lead echo.

## 6. Targeted Repair Contract

Repair stage в `v2.13` не должен делать новый “второй текст”.

Его задача:

- минимально исправить `missing`;
- убрать policy issues;
- не ломать уже найденный natural flow;
- сохранить структуру, если она уже работает;
- не возвращать `посвящ*`, `расскажут`, `представят`, `обсудят`.

## 7. Что реально улучшилось в `v2.13`

- `forbidden = 0` на всех 5 кейсах;
- total missing `14`, то есть лучше baseline `22`;
- `2745` улучшился и против baseline, и против `v2.12` по coverage;
- `2660` сохранил сильный sparse-case result;
- `2673` стал чище, чем ряд более ранних explanation-heavy версий;
- `2687` остался без `посвящ*`, что ранее стабильно ломалось.

## 8. Что осталось blocker

- `v2.13` не улучшил total missing против `v2.12`: оба дают `14`;
- `2734` не продвинулся относительно `v2.12` по missing и всё ещё склонен к украшательному high-level rewrite;
- `2687` регресснул против `v2.12` по coverage (`1 -> 2`);
- `2673` по coverage не лучше baseline (`5`) и хуже `v2.12` (`4`);
- `presentation/project` cases всё ещё тянут Gemma к explanation-heavy prose;
- `lecture/person` cases всё ещё рискуют терять часть person-rich specifics при попытке писать более гладко.

## 9. Как сейчас надо смотреть на `v2.13`

`v2.13` — это не провал и не готовый winner.

Сильные стороны:

- он чище по hygiene;
- он уже не тащит `посвящ*` в финальный текст;
- он доказал, что `full-floor normalization + shorter generation + targeted repair` жизнеспособны.

Слабые стороны:

- coverage plateau по сравнению с `v2.12`;
- недобор на `2673` и частично `2687`;
- не до конца восстановлена сила `v2.6` там, где prose уже был заметно лучше baseline.
