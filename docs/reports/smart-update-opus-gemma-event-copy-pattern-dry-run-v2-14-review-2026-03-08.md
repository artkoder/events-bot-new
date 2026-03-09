# Smart Update Opus Gemma Event Copy Pattern Dry Run V2.14 Review

Дата: 2026-03-08

Связанные материалы:

- `artifacts/codex/experimental_pattern_dryrun_v2_14_2026_03_08.py`
- `artifacts/codex/gemma_event_copy_pattern_dry_run_v2_14_5events_2026-03-08.json`
- `docs/reports/smart-update-opus-gemma-event-copy-pattern-dry-run-v2-14-5-events-2026-03-08.md`

## Verdict

`v2.14` — mixed round.

Фиксирую явно:

- относительно `v2.13`: `v2.14` не лучше overall;
- относительно baseline: `v2.14` лучше по суммарному coverage (`14 < 22`), но не даёт чистого overall-win по production-quality текста;
- в runtime переносить рано.

Чётко по totals:

- baseline total missing = `22`
- `v2.12` total missing = `14`
- `v2.13` total missing = `14`
- `v2.14` total missing = `14`

Но при этом:

- `v2.14` вернул forbidden issues (`2687`: `посвящ*`, `2673`: `cliche_not_about_but_about`);
- значит quality-safe plateau `v2.13` не удержан.

## Что реально улучшилось

### 1. `2734` стал лучше и `v2.13`, и baseline

`2734`

- baseline missing = `5`
- `v2.13` missing = `3`
- `v2.14` missing = `2`

Плюсы:

- outline дал понятную структуру;
- headings соответствуют секциям лучше, чем в части прошлых раундов;
- `program-rich` кейс не распался на vague prose.

Минус:

- текст всё ещё слегка editorialized;
- но это уже скорее refinement, а не blocker.

### 2. `2673` частично оздоровился как presentation-case

`2673`

- baseline missing = `5`
- `v2.13` missing = `5`
- `v2.14` missing = `4`

Реальные gains против `v2.13`:

- лид наконец прямо называет `презентацию проекта`, а не абстрактную `встречу`;
- лучше выражено, что речь о платформе и проекте;
- coverage немного вырос.

Но этот кейс всё ещё плохой по prose quality:

- heading-и бюрократические (`Цель и задачи проекта`, `Формат мероприятия`, `Для кого это будет интересно`);
- в теле снова появился explanation-heavy tone;
- финальный текст звучит как переработанный пресс-релиз, а не как сильный телеграм-анонс;
- `cliche_not_about_but_about` подтверждает это формально.

### 3. `2745` не сломался

`2745`

- baseline missing = `5`
- `v2.13` missing = `3`
- `v2.14` missing = `3`

То есть новый outline-path не повредил sparse branch, потому что `2745` остался в compact routing.

## Что стало хуже

### 1. `2687` снова развалился

`2687`

- baseline missing = `5`
- `v2.12` missing = `1`
- `v2.13` missing = `2`
- `v2.14` missing = `3`

Плюс вернулся forbidden:

- `посвящ*`

Главная проблема здесь очень показательна:

- outline для lecture-case сам сгенерировал фокус-формулы вроде `Лекция посвящена...`;
- generation потом подхватил этот язык;
- то есть новый LLM layer начал заносить обратно тот самый machine-like framing, с которым мы до этого долго боролись.

### 2. `2660` локально регресснул против `v2.13`

`2660`

- baseline missing = `2`
- `v2.13` missing = `1`
- `v2.14` missing = `2`

Сам текст publishable, но сильного выигрыша нет. На sparse case новый раунд не добавил качества.

## Главная инженерная находка

`v2.14` подтвердил, что split-call сам по себе не ошибка.

Проблема не в том, что появился отдельный `outline` вызов, а в том, что его контракт пока слишком свободный:

- outline пишет свои `focus_note` уже как маленький редактор;
- туда пролезают бюрократические или запрещённые рамки (`посвящена роли`, `для кого это будет интересно`, `формат мероприятия`);
- generation затем воспринимает этот outline как авторитетную структуру и частично копирует его язык.

То есть bottleneck сдвинулся:

- раньше semantic drift рождался в deterministic shaping;
- теперь часть drift рождается уже в самом LLM outline.

## Что это означает для следующего шага

### 1. `outline` не надо откатывать сразу

На `2734` и частично на `2673` он помог.

Но его надо сделать более узким и mechanical:

- меньше свободных `focus_note`;
- больше работы через `fact_ids` и named blocks;
- меньше готовых канцелярских heading-паттернов.

### 2. Следующий рост качества — не в новых regex

Пользовательская критика тут справедлива.

`v2.14` полезен именно тем, что снова подтвердил:

- semantic core должен оставаться у LLM;
- deterministic слой допустим только как validation/hygiene/support;
- добавлять ещё more semantic regexes сейчас было бы ложным направлением.

### 3. На `presentation_project` и `lecture_person` нужны более точные prompt contracts

Сейчас я бы смотрел на `v2.15` так:

- сохранить `full-floor normalization`;
- сохранить split-call только для rich cases;
- переделать outline так, чтобы он возвращал меньше prose и больше structure;
- отдельно запретить outline-level канцелярит и heading templates;
- просить внешние модели предлагать именно prompt-level changes для Gemma, а не уход в regex-first semantics.

## Bottom Line

`v2.14`:

- лучше baseline по суммарному coverage;
- хуже `v2.13` как hygiene-safe candidate;
- не лучше baseline как clean overall production-quality result;
- полезен как evidence round, потому что показал новый точный failure mode: `outline-generated bureaucracy`.
