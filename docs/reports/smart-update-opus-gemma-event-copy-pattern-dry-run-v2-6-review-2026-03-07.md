# Smart Update Opus Gemma Event Copy Pattern Dry-Run V2.6 Review

Дата: 2026-03-07

Связанные материалы:

- `docs/reports/smart-update-opus-gemma-event-copy-pattern-dry-run-v2-6-5-events-2026-03-07.md`
- `artifacts/codex/gemma_event_copy_pattern_dry_run_v2_6_5events_2026-03-07.json`
- `artifacts/codex/experimental_pattern_dryrun_v2_6_2026_03_07.py`
- `docs/reports/smart-update-gemini-event-copy-v2-6-hypotheses-consultation-response.md`
- `docs/reports/smart-update-gemini-event-copy-v2-6-hypotheses-consultation-response-review.md`

## 1. Краткий verdict

`v2.6` — это mixed-positive round и, вероятно, самый полезный после `v2.3`, но всё ещё не runtime candidate.

Что реально подтвердилось:

- stricter extraction/generation contract помог sparse и performance cases;
- `2660` стал лучшим compact кейсом цикла по coverage;
- `2745` вернулся к уровню `v2.3`;
- `2734` дал strongest recovery за весь цикл: `missing=7 -> 3` против `v2.3`, без forbidden leakage.

Что осталось blocker:

- `2687` жёстко регресснул: `missing=1 -> 5`, вернулся `посвящ*`, а label-style facts всё ещё просачиваются;
- `2673` numerically улучшился, но prose остаётся бюрократичной и снова вернулось forbidden `посвящ*`.

Итог:

- в runtime переносить нельзя;
- новый Gemini round оправдан и нужен уже по полному `v2.6` evidence set;
- главный вопрос теперь не “работает ли направление вообще”, а “как не ломать lecture/presentation cases, сохранив gains на sparse/performance”.

## 2. Что в `v2.6` реально улучшилось

### 2.1. `2660`: первый действительно сильный compact win

Это лучший numeric outcome раунда:

- `missing=1 -> 0` против `v2.5`;
- forbidden markers нет;
- текст снова стал плотным и не развалился на сервисные блоки.

Именно здесь хорошо видно пользу:

- explicit merge permission для compact branch;
- более строгого extraction contract;
- отказа от буквального fact dump.

Это ещё не идеальный prose masterpiece, но это уже уверенный quality gain.

### 2.2. `2745`: sparse case снова живой и без мусора

`2745` не стал новым лучшим абсолютным кейсом, но round полезный:

- `missing=4 -> 3` против `v2.5`;
- forbidden leakage нет;
- compact ветка звучит спокойно и без service/meta drift.

То есть rollback к короткой форме здесь сохранился, а coverage частично оздоровился.

### 2.3. `2734`: это главный успех `v2.6`

Именно этот кейс подтверждает, что часть гипотез Gemini была полезной не на словах, а на практике.

Что получилось:

- `missing=4 -> 3` против `v2.5`;
- forbidden `посвящ*` исчез;
- branch ушёл в `fact_first_v2_6`, а не в компактное переупрощение;
- в тексте снова есть:
  - Магомаев / Синявская;
  - Муза;
  - program framing;
  - статус Владимира Гудожникова.

Это первый раунд после длинной серии, где `2734` снова выглядит как рабочий person/program-led case.

## 3. Где `v2.6` всё ещё провалился

### 3.1. `2687`: главный blocker раунда

Это самая тяжёлая проблема `v2.6`.

Фактически:

- `missing=1 -> 5` против `v2.5`;
- forbidden `посвящ*` вернулся;
- в `facts_text_clean` снова появились `Тема: ...`;
- lecture case ушёл в отстранённый музейный канцелярит.

Это особенно неприятно потому, что:

- именно этот раунд должен был жёстче банить label-style facts;
- именно здесь anti-`посвящ*` должен был стать сильнее.

Практический вывод:

- текущий extraction contract для lecture cases всё ещё конфликтный;
- revise severity не принуждает к реальной rewrite;
- gains на performance cases не переносятся автоматически на non-performance events.

### 3.2. `2673`: coverage лучше, но тон всё ещё служебный

`2673` numerically improved:

- `missing=6 -> 4` против `v2.5`.

Но качественно проблема остаётся:

- forbidden `посвящ*` вернулся;
- в тексте много bureaucratic transfer:
  - `мероприятие анонсирует запуск`
  - `презентация посвящена устройству платформы`
- prose лучше структурно, но не стала по-настоящему живой и профессиональной.

То есть anti-bureaucracy contract пока работает слишком слабо именно там, где он нужен больше всего.

## 4. Что `v2.6` доказал про реальный bottleneck

### 4.1. Проблема больше не в общей архитектуре

Это уже довольно надёжный вывод.

Пайплайн:

- `extract -> route -> generate -> revise`

сам по себе жизнеспособен.

Скачки качества определяются:

- точностью extraction contract;
- качеством narrative-ready facts;
- branch selection;
- тем, насколько revise реально blocking, а не advisory.

### 4.2. Performance/sparse moves переносимы, lecture/presentation quality — нет

`2660`, `2745`, `2734` подтверждают, что:

- compact merge;
- grouped preservation;
- content-aware blocker

могут быть полезны.

Но `2687` и `2673` показывают, что:

- lecture/presentation cases требуют отдельной калибровки;
- одни и те же rules по-разному ведут себя на разных content shapes.

### 4.3. Label-style ban пока не стал реально enforced

Это один из самых важных практических выводов раунда.

Мы уже просили:

- не извлекать `Тема: ...`;
- не использовать `посвящ*`;
- не тащить intent/metatext.

Но `2687` показывает, что текущий wording всё ещё:

- либо недостаточно жёсткий;
- либо конфликтует с другими инструкциями;
- либо стоит не в том месте prompt contract.

## 5. Что теперь нужно спросить у Gemini

Новый round должен быть узким и grounded.

Нужно попросить Gemini:

1. Разобрать, почему `2660/2745/2734` реально оздоровились, а `2687/2673` нет.
2. Сказать, где именно ломается contract:
   - extraction;
   - revise;
   - routing;
   - branch-specific prompt language.
3. Дать **конкретные prompt-level rewrites для Gemma** по:
   - extraction;
   - compact generation;
   - standard generation;
   - revise/policy wording.
4. Предложить, как отдельно стабилизировать:
   - lecture-led cases;
   - presentation/value-led cases.
5. Сказать, нужен ли ещё один theory round после этого или уже сразу строить `v2.7`.

## 6. Bottom line

`v2.6` не провальный round.

Наоборот, это сильный evidence set:

- направление частично подтвердилось практикой;
- у нас появились реальные wins;
- но и появился очень точный failure map.

Практический verdict:

- **accept as evidence**: да;
- **accept as runtime candidate**: нет;
- **нужен новый Gemini round по реальному `v2.6` bundle**: да.
