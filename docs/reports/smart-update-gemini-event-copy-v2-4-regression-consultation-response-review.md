# Smart Update Gemini Event Copy V2.4 Regression Consultation Response Review

Дата: 2026-03-07

Связанные материалы:

- `docs/reports/smart-update-gemini-event-copy-v2-4-regression-consultation-response.md`
- `docs/reports/smart-update-gemini-event-copy-v2-4-regression-consultation-brief.md`
- `docs/reports/smart-update-opus-gemma-event-copy-pattern-dry-run-v2-4-5-events-2026-03-07.md`
- `docs/reports/smart-update-opus-gemma-event-copy-pattern-dry-run-v2-4-review-2026-03-07.md`
- `artifacts/codex/experimental_pattern_dryrun_v2_4_2026_03_07.py`

## 1. Краткий verdict

Это сильный и заметно более точный ответ Gemini, чем прошлый.

Главное:

- она правильно признала, что `v2.4` по сумме хуже `v2.3`;
- точно локализовала главный regression как `fact inflation -> routing drift`;
- хорошо самокорректировала свой прошлый overly strong advice про “сохранять каждое название”;
- предложила practical next step без нового redesign.

Мой итог:

- **новый Gemini-раунд сейчас не нужен**;
- response можно использовать как good-quality second opinion;
- но в локальный следующий patch pack надо брать не всё подряд, а только хорошо защищённые части.

## 2. Что в ответе Gemini принимаю

### 2.1. Возврат к `v2.3` base с selective carry-over

Это я принимаю.

Практически это означает:

- не продолжать строить следующий round поверх `v2.4` как будто это healthy base;
- вернуть в основу `v2.3`;
- поверх неё накатить только verified improvements.

Это совпадает с нашим собственным чтением `v2.4`.

### 2.2. Routing drift как главный regression vector

Это сильная часть ответа.

Gemini правильно увидела:

- `2660` и `2745` сломались не потому, что внезапно стали хуже prompts per se;
- их сломал переход из `compact_fact_led` в standard branch после роста fact count.

Это не весь root cause, но именно главный системный side effect `v2.4`.

### 2.3. Human-readable anti-`посвящ*` policy issue

Это тоже принимаю.

Она точно попала в слабое место:

- `forbidden_marker(посвящ*)` слишком opaque;
- Gemma не знает, как из этого построить rewriting action.

Замена на human-readable instruction выглядит правильной.

### 2.4. Grouping program items instead of one-title-per-line

Это тоже strong recommendation.

Она хорошо исправляет собственную прошлую ошибку:

- сохранять named items нужно;
- но не ценой fact explosion.

Это особенно важно для `2734`.

## 3. Что принимаю только с поправками

### 3.1. Threshold `<= 6`

Беру только как experiment-first move, а не как новую каноническую истину.

Причина:

- для текущего набора это почти наверняка вернёт `2660` и `2745` в compact;
- но сам по себе magic threshold still brittle.

То есть:

- для следующего локального round это good pragmatic patch;
- но в долгую лучше держать в голове density-aware routing.

### 3.2. `_near_dup_signature` tweak через добавление конкретных глаголов

Идея полезная, но в текущем виде слишком узкая.

Не хочется превращать dedup в endless list of verbs.
Поэтому:

- accept with modification;
- лучше добавить немного более широкий normalization block для metatext verbs, а не single-word patch only.

## 4. Что в ответе Gemini считаю неполным

### 4.1. Недостаточно внимания к `2687`

Gemini правильно заметила regression на `2687`, но недооценила, насколько он важен.

Это не случайный кейс:

- `2687` был одним из лучших `v2.3`;
- значит если patch pack его ломает, это сильный сигнал против reliability branch.

То есть next patch pack надо специально проверять не только на `2660/2745/2734/2673`, но и на `2687`.

### 4.2. Acceptance criteria всё ещё чуть слишком metric-oriented

Ответ уже намного лучше прошлых, но даже здесь:

- `2734 <= 4-5 missing` звучит полезно;
- однако итоговый quality gate всё равно должен быть редакторским, а не только count-based.

Это не большой дефект, просто важная оговорка.

## 5. Что точно пойдёт в следующий локальный round

Из этого ответа я беру следующее:

1. База — `v2.3`, не `v2.4`.
2. Keep anti-conversational headings.
3. Replace opaque anti-`посвящ*` marker with human-readable revise issue.
4. Replace one-title-per-line preservation with grouped program-item preservation.
5. Test `<= 6` as a local routing recovery experiment.
6. Slightly strengthen dedup normalization for metatext verb variants.

## 6. Что не беру как готовую истину

- Не принимаю `<= 6` как final routing law.
- Не считаю, что проблема уже полностью объяснена только routing.
- Не считаю, что после одного удачного next round ветка автоматически ready for runtime.

## 7. Bottom line

Этот Gemini response полезен и practical.

Если коротко:

- **accept**: rollback to `v2.3` base, human-readable anti-`посвящ*`, grouped program preservation;
- **accept with modification**: routing `<= 6`, dedup tweak;
- **keep critical distance**: final decision всё равно только после следующего live dry-run.

Мой рабочий вывод по всей задаче сейчас такой:

- проблема не в том, что pattern-driven direction неверен;
- проблема в том, что ветка стала слишком чувствительной к мелким изменениям facts layer и routing;
- следующий шаг должен быть narrow corrective round с quality-first целью и очень ограниченным scope.
