# Smart Update Gemma Event Copy V2.15 Design Brief

Дата: 2026-03-08

Основание:

- [retrospective baseline -> v2.14](/workspaces/events-bot-new/docs/reports/smart-update-gemma-event-copy-retrospective-baseline-v2-14-2026-03-08.md)
- [baseline dry-run](/workspaces/events-bot-new/docs/reports/smart-update-opus-gemma-event-copy-dry-run-5-events-2026-03-07.md)
- [v2.6 review](/workspaces/events-bot-new/docs/reports/smart-update-opus-gemma-event-copy-pattern-dry-run-v2-6-review-2026-03-07.md)
- [v2.13 review](/workspaces/events-bot-new/docs/reports/smart-update-opus-gemma-event-copy-pattern-dry-run-v2-13-review-2026-03-08.md)
- [v2.14 review](/workspaces/events-bot-new/docs/reports/smart-update-opus-gemma-event-copy-pattern-dry-run-v2-14-review-2026-03-08.md)

## 0. Нумерация итераций

Начиная с этой точки фиксируется такой принцип версионирования:

- `2.15` — базовая версия новой итерационной линии;
- если дальше идёт доработка этой же основы без нового архитектурного перелома, увеличивается младший номер:
  - `2.15.2`
  - `2.15.3`
  - `2.15.4`
- если происходит новый существенный architectural redesign, увеличивается номер второго порядка:
  - `2.16`

То есть:

- `2.15.x` = refinement family одной и той же архитектурной базы;
- `2.16` = новая большая версия, а не просто ещё один patch round.

## 1. Цель `v2.15`

`v2.15` нужен не как ещё один patch pack, а как первая версия, где одновременно фиксируется целевой профиль качества текста и архитектура, способная этот профиль держать на тысячах входов.

### 1.0. Сводный чек-лист качества текста

Ниже — короткий единый список того, каким должен быть итоговый текст.

Это reference checklist для любой следующей итерации `2.15.x`.

- Естественный: звучит как человеческий анонс, а не как AI-summary.
- Профессиональный: стиль культурного журналиста, а не пресс-релиз и не канцелярит.
- Нешаблонный: без одинаковых лидов, generic headings и механического ритма.
- Grounded: без выдумки, embellishment и unsupported praise.
- Полный по смыслу: не теряет subject, program items, grouped names, agenda blocks, project specifics.
- Ясный: быстро объясняет, что это за событие, о чём оно и чем наполнено.
- Уместно мотивирующий: может показать, почему событие важно или интересно, но без рекламного CTA.
- Вариативный по подаче: использует разные macro-patterns, а не один универсальный шаблон.
- С сильным лидом: не начинает с `Лекция расскажет...`, `Спектакль рассказывает...`, `Это событие...`.
- С хорошими заголовками: headings конкретные и смысловые, а не `О событии` / `Подробности`.
- С функциональным форматированием: epigraph, списки и секции используются по делу, а не декоративно.
- Чистый от AI clichés: нет `посвящ*`, label-style facts, bureaucratic self-reference, `это не ..., а ...`, question-headings ради псевдохука.
- Чистый от сервисного мусора: без URL, цен, билетов, регистрации, контактов, age labels и логистики.
- Лёгкий по ритму: без needless repetition, без чувства “модель закрывает coverage”.
- Масштабируемый: это качество достигается не ручными regex-костылями, а LLM-first архитектурой.

### 1.1. Текст должен быть естественным

Описание должно звучать как человеческий культурный анонс, а не как:

- LLM summary;
- пересказ пресс-релиза;
- формальная карточка события;
- набор склеенных фактов.

### 1.2. Текст должен быть профессиональным

Ориентир — аккуратный стиль профессионального культурного журналиста:

- уверенный;
- чистый;
- в меру выразительный;
- без дешёвой рекламности;
- без тяжёлой литературщины.

### 1.3. Текст должен быть менее шаблонным, чем baseline

Мы хотим уйти от baseline-дефектов:

- одинаковых лидов;
- generic headings;
- однотипного ритма предложений;
- flat fact-first packaging без живого editorial shaping.

### 1.4. Текст должен быть grounded

Каждое содержательное утверждение должно опираться на source-backed facts.

Нельзя:

- дорисовывать смысл;
- добавлять unsupported praise;
- создавать псевдо-журналистскую образность без опоры в фактах.

### 1.5. Coverage не должен проседать

Хороший текст не должен оплачиваться потерей ключевой конкретики.

Особенно важно сохранять:

- subject / автор / коллектив;
- program items;
- grouped names;
- agenda blocks;
- project-specific details;
- факты, объясняющие, что именно будет происходить.

### 1.6. Текст должен ясно объяснять, что это за событие

Читатель быстро должен понимать:

- что это вообще за формат;
- о чём событие;
- кто или что в центре;
- чем оно наполнено.

Без этого текст может звучать красиво, но оставаться пустым.

### 1.7. Текст должен объяснять, почему событие может быть интересно, но только когда это реально подтверждено

`Why it matters` и `why go` — это не обязательный шаблон.

Он допустим только когда:

- источник реально даёт материал для такого хода;
- этот ход не заменяет факты;
- текст не скатывается в рекламный CTA.

### 1.8. Подача должна быть вариативной через крупные паттерны

Живые редакторы не пишут все тексты одинаково.

Поэтому generation должен поддерживать несколько устойчивых macro-patterns, чтобы:

- sparse cases не выглядели как rich cases;
- person-led события не звучали как project launch;
- program-rich события не теряли list logic;
- тексты не схлопывались в один шаблон.

### 1.9. Лиды должны быть сильными

Мы хотим лиды, которые:

- быстро вводят в предмет;
- не начинаются с мёртвого метатекста;
- не повторяют карточку события;
- могут строиться через деталь, тему, субъекта, программу или эпиграф.

Нежелательные паттерны:

- `Лекция расскажет...`
- `Спектакль рассказывает...`
- `Это событие...`
- `Мероприятие будет интересно...`

### 1.10. Headings должны быть осмысленными

Headings нужны не всегда, но если они есть, то должны:

- реально помогать навигации;
- отражать смысл секции;
- быть конкретными;
- не быть structural filler.

Нежелательные headings:

- `О событии`
- `Подробности`
- `Формат мероприятия`
- `Основная идея`

### 1.11. Форматирование должно быть функциональным, а не декоративным

Нужно сохранять и использовать сильные formatting moves:

- epigraph / blockquote при наличии реально сильной цитаты или tagline;
- списки при наличии program/list payload;
- секции для rich cases;
- компактную подачу без headings для sparse cases.

Но formatting не должен:

- раздувать текст;
- создавать микроабзацы;
- превращаться в mechanical Telegraph template.

### 1.12. Текст должен быть чистым от AI clichés

Нельзя допускать:

- `посвящ*`;
- label-style facts;
- bureaucratic self-reference;
- generic promo filler;
- `это не ..., а ...`;
- service/meta headings;
- question-headings ради псевдохука;
- outline-style bureaucracy.

### 1.13. Текст должен быть чистым от сервисного мусора

В публичное описание не должны протекать:

- логистика;
- билеты;
- цены;
- регистрация;
- контакты;
- URL;
- age labels;
- иной служебный хвост.

### 1.14. Текст не должен быть тяжёлым

Даже при высокой factual density он должен оставаться:

- читаемым;
- ритмичным;
- без needless repetition;
- без чувства, что модель просто "отрабатывает coverage".

### 1.15. Всё это должно масштабироваться

Итоговый quality profile должен достигаться через архитектуру, которая:

- не зависит от длинного хвоста ручных regex corrections;
- работает на многих source shapes;
- не требует fragile per-case routing tricks;
- остаётся operationally defendable для тысяч heterogeneous posts.

## 2. Non-Negotiables

### 2.1. Что `v2.15` обязан сохранить

- `LLM-first` semantic core.
- `full-floor normalization` из `v2.12+`.
- fact discipline baseline.
- cleaner hygiene и `forbidden = 0` как ориентир `v2.13`.
- prose ambition `v2.6`, а не baseline flatness.
- optional epigraph/list benefits из baseline.

### 2.2. Что `v2.15` не должен повторять

- semantic regex shaping;
- dirty merge clean+dirty facts;
- full editorial rewrite pass;
- prose-like outline с headings/focus notes;
- расширяющийся wall of bans как главный механизм управления стилем;
- pattern routing, который сам по себе способен обрушить coverage.

## 3. Главная идея `v2.15`

Pattern library надо вернуть, но переместить в правильное место.

Patterning должен жить в generation-layer:

- как выбор крупного narrative mold;
- как выбор структуры лида;
- как выбор section logic;
- как решение про epigraph / list / compact mode.

Patterning не должен жить в:

- semantic extraction;
- aggressive `copy_assets` interpretation;
- prose-like outline before generation.

Коротко:

- `facts` должны быть чистыми и предметными;
- `patterns` должны управлять тем, **как** писать;
- deterministic слой должен только страховать и валидировать.

## 4. Архитектура `v2.15`

### 4.1. Scalable Core

Базовая цепочка:

1. `raw_facts`
2. cheap `shape metadata`
3. `full-floor normalization` через LLM
4. deterministic cleanup / dedup / forbidden scan
5. lightweight pattern selection
6. pattern-aware generation
7. deterministic validation
8. narrow targeted repair only on validated issues

Это и есть основная production-like схема.

### 4.2. Step 1: `raw_facts`

Оставляем LLM extraction как semantic intake.

Задача extraction:

- собрать максимум релевантного содержательного материала;
- не заниматься prose;
- не выбирать narrative angle;
- не пытаться "сразу красиво писать".

### 4.3. Step 2: cheap `shape metadata`

Это не branching tree как в старых pattern rounds.

Это короткий metadata layer:

- `is_sparse`
- `has_strong_quote`
- `has_person_cluster`
- `has_program_list`
- `is_project_presentation`
- `is_performance_or_exhibition`

Источник:

- mostly deterministic from normalized facts and event type;
- optional tiny LLM fallback только для ambiguous rich cases.

Эта metadata не должна сама генерировать prose и не должна менять смысл.

### 4.4. Step 3: `full-floor normalization`

Это ключевой semantic layer.

Нормализатор должен:

- пройти по всей fact base, а не по "лучшему подмножеству";
- убирать metatext frame (`расскажут`, `представят`, `лекция посвящена`);
- превращать intent-style facts в content facts;
- сохранять grouped names, program items, agenda blocks;
- не раздувать facts;
- не делать prose;
- не создавать label-style facts (`Тема:`, `Цель:`, `Формат:`).

Нормализатор должен быть скорее mechanical, чем editorial.

### 4.5. Step 4: deterministic cleanup / dedup / forbidden scan

Этот слой допускается только как support:

- exact/near dedup;
- remove obvious service/logistics residue;
- cap runaway fact inflation;
- catch forbidden markers (`посвящ*`, CTA, generic service phrases).

Важно:

- этот слой не должен придумывать смысловые rewrites;
- он не должен заменять LLM semantic decisions;
- он не должен превращаться в regex-first pseudo-parser.

### 4.6. Step 5: lightweight pattern selection

Pattern selection должен быть safe by construction.

Он должен выбирать **только macro-pattern**, а не писать outline.

Рекомендуемые паттерны `v2.15`:

1. `compact_scene_led`
2. `quote_scene_led`
3. `person_led`
4. `program_led`
5. `project_led`
6. `theme_led`

Если выбор неуверенный:

- fallback в `theme_led` или `compact_scene_led`;
- не делать отдельный risky branch.

### 4.7. Step 6: pattern-aware generation

Generation должен стать главным местом, где возвращается human-like variability.

Именно здесь должны жить:

- variation of leads;
- section logic;
- heading palette;
- list formatting;
- epigraph decision.

Не через 20 branches, а через 5-6 больших patterns.

### 4.8. Step 7: deterministic validation

Validation должен проверять:

- forbidden markers;
- service contamination;
- heading/content mismatch;
- obvious duplication;
- presence of required named items / grouped blocks.

Этот слой не должен оценивать стиль как semantic oracle.

### 4.9. Step 8: targeted repair

Repair остаётся только как narrow fallback.

Его задачи:

- убрать validated forbidden issue;
- восстановить missing named/program element;
- поправить heading mismatch;
- убрать duplicate sentence.

Repair не должен делать полный rewrite текста.

## 5. Pattern Library для генерации

Именно это нужно вернуть осознанно, потому что живые редакторы действительно пишут не по одному шаблону.

### 5.1. `compact_scene_led`

Для:

- sparse cultural cases;
- лаконичных выставок/спектаклей/малых анонсов;
- событий с 3-6 плотными фактами.

Форма:

- 1-2 плотных абзаца;
- headings optional;
- без forced sectioning.

Что даёт:

- naturalness;
- меньше AI-structure;
- хороший match с `2660`/`2745`-style cases.

### 5.2. `quote_scene_led`

Для:

- случаев, где есть реальная сильная quote-like fact или tagline;
- только если цитата/эпиграф действительно сильны и source-backed.

Форма:

- короткий epigraph / blockquote;
- затем самодостаточный body;
- эпиграф не должен удерживать missing subject.

Что возвращаем из baseline:

- сильные эпиграфы;
- blockquote как visual hook.

### 5.3. `person_led`

Для:

- lecture/person-rich cases;
- авторских выставок;
- событий, где вход в текст лучше строить через человека или группу людей.

Форма:

- лид на субъекте;
- потом grouped specifics;
- headings конкретные, не generic.

Что важно:

- не превращать в biography-summary;
- не дробить person cluster на шум.

### 5.4. `program_led`

Для:

- концертов;
- лекций/событий с явной программой;
- program-rich и list-rich cases.

Форма:

- лид даёт общий смысл события;
- затем section/list по программе;
- список обязателен, если факты содержат конкретные named items.

Что возвращаем из baseline:

- силу списков;
- структурированную подачу program facts.

### 5.5. `project_led`

Для:

- презентаций проектов;
- платформ;
- публичных запусков и product/project explainers.

Форма:

- сразу называть `презентацию проекта`, если это подтверждено;
- лид должен говорить, что это за проект и зачем он нужен;
- потом grouped blocks:
  - что это;
  - как устроено / что включает;
  - что будет в разговоре / программе.

Важно:

- не скатываться в `мероприятие анонсирует`, `будет интересно`, `проект призван`;
- не превращать текст в пресс-релиз.

### 5.6. `theme_led`

Fallback pattern для:

- concept-led cases;
- выставок и культурных событий, где нет явного person/program/project entry point.

Форма:

- лид через предмет/тему;
- дальше конкретика и supporting details;
- headings конкретные, без generic meta labels.

## 6. Что именно надо вернуть из baseline и сильных раундов

### 6.1. Из baseline

- epigraph / blockquote as optional strong opener;
- lists for real program payload;
- уверенную структурированность, когда facts реально сложные;
- самодостаточный body.

### 6.2. Из `v2.6`

- less-template feel;
- более живые headings;
- культурно-журналистский тон;
- более естественный lead rhythm.

### 6.3. Из `v2.13`

- shorter exemplar-driven generation;
- zero-forbidden ambition;
- отказ от full editorial rewrite;
- cleaner hygiene.

## 7. Что должно быть strictly scalable

Это core, без которого `v2.15` нельзя будет считать production-track.

### 7.1. Scalable by default

- LLM full-floor normalization;
- small pattern library;
- short generation prompts with exemplars;
- deterministic hygiene/validation;
- optional quote/list modules only when strongly supported by facts;
- minimal shape metadata instead of huge branching tree.

### 7.2. Safe optional boosters

Это можно использовать только если неудача не может сломать весь output.

Допустимые boosters:

- epigraph only when strong quote exists;
- list rendering only when grouped named items exist;
- `why it matters` micro-move only when evidence is explicit;
- fact grouping call only for richer cases;
- optional pattern-fallback reroll only when validation proves hard failure.

### 7.3. Unsafe and should stay out of core

- semantic regex rewrites;
- prose-like outline with headings/focus notes;
- full editorial review pass on every event;
- giant routing tree;
- noisy `copy_assets` like `tone_hint/core_angle` if they are not strongly evidence-backed.

## 8. Prompt Strategy для Gemma

### 8.1. Normalizer prompt

Нужен короткий, mechanical, anti-metatext contract:

- transform to clean content facts;
- preserve grouped details;
- no labels;
- no `посвящ*`;
- no prose.

### 8.2. Pattern selector prompt or rule

По возможности deterministic/lightweight.

Если нужен LLM:

- only choose one macro-pattern;
- no free-text plan;
- no headings;
- no focus note prose.

### 8.3. Generation prompt

Нужен не wall of bans, а:

- 5-6 pattern exemplars;
- heading palette examples;
- epigraph/list rules;
- direct anti-bureaucracy / anti-cliche rules;
- explicit instruction to preserve all required facts.

### 8.4. Repair prompt

Очень narrow:

- fix one validated issue;
- do not rewrite the whole text;
- keep structure;
- keep facts.

## 9. Как использовать Opus

`Opus` стоит использовать не для выбора всей architecture, а как сильный prompt engineer reviewer после того, как черновик `v2.15` prompt pack уже написан.

Что именно ему давать:

- normalizer prompt;
- pattern selector contract;
- generation prompt по каждому pattern;
- repair prompt;
- 5-10 реальных кейсов с baseline / strongest experimental outputs.

Что именно просить:

- critique Gemma-specific prompt behavior;
- concrete prompt rewrites, not abstract advice;
- anti-cliche / anti-bureaucracy improvements;
- lead and heading improvements;
- guardrails against template collapse.

Чего не просить:

- придумывать regex-core;
- ломать LLM-first architecture;
- подтверждать уже выбранные решения "на доверии".

## 10. Success Criteria для `v2.15`

### 10.1. Against baseline

`v2.15` должен быть лучше baseline не только по prose, но и по operational quality.

Минимум:

- total missing < baseline;
- forbidden = 0;
- headings менее шаблонны, чем baseline;
- наличие epigraph/list advantages там, где они реально уместны.

### 10.2. Against `v2.13`

`v2.15` должен быть не хуже `v2.13` по hygiene и не хуже лучших `v2.13`/`v2.6` кейсов по naturalness.

### 10.3. Human reading goals

При human reading тексты должны:

- звучать как культурный анонс, а не как AI-summary;
- не тянуться к `лекция расскажет`, `мероприятие будет интересно`, `это не ..., а ...`;
- использовать headings как смысловые labels, а не как structural filler;
- не терять subject/program/project sharpness.

## 11. Bottom line

`v2.15` надо строить не как отказ от patterns, а как их правильное возвращение.

Правильная формула:

- semantic core = `LLM normalization`
- structure safety = deterministic validation
- prose variety = small pattern library
- richer formatting = optional epigraph/list modules
- scalability = no regex semantics, no prose-outline, no giant branching tree

Если коротко:

- baseline надо взять как disciplined floor;
- `v2.6` и `v2.13` — как prose reference;
- `v2.12+` — как правильную LLM-first architecture;
- patterning вернуть в generation-layer, где оно действительно улучшает текст, а не ломает систему.
