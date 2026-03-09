# Smart Update Opus Gemma Event Copy Pattern Dry-Run V2.12 Review

Дата: 2026-03-08

Связанные материалы:

- `artifacts/codex/experimental_pattern_dryrun_v2_12_2026_03_08.py`
- `artifacts/codex/gemma_event_copy_pattern_dry_run_v2_12_5events_2026-03-08.json`
- `docs/reports/smart-update-opus-gemma-event-copy-pattern-dry-run-v2-12-5-events-2026-03-08.md`
- `docs/reports/smart-update-gemma-event-copy-v2-12-prompt-context.md`
- `docs/reports/smart-update-opus-gemma-event-copy-pattern-dry-run-v2-11-review-2026-03-08.md`
- `docs/reports/smart-update-opus-gemma-event-copy-pattern-dry-run-v2-6-review-2026-03-07.md`

## 1. Краткий verdict

`v2.12` — первый раунд новой architecture, который реально оторвался от `v2.11`.

Что это значит по факту:

- `v2.12` лучше `v2.11`;
- `v2.12` лучше baseline по coverage;
- но `v2.12` всё ещё не runtime candidate.

Суммарно по 5 кейсам:

- baseline total missing = `22`
- `v2.11` total missing = `33`
- `v2.12` total missing = `14`

То есть `v2.12` впервые подтверждает, что новая `full-floor normalization` architecture способна beat baseline по coverage, а не только локально чинить одну поломку.

Но quality bar ещё не взят, потому что:

- на `2660`, `2734`, `2687` остался forbidden `посвящ*`;
- `2745` звучит естественнее baseline, но формально не закрыл literal coverage floor;
- `2673` заметно оздоровился, но prose всё ещё explanation-heavy и partly bureaucratic.

Итог:

- это strong architectural recovery round;
- это не final quality win.

## 2. Baseline-Relative Verdict

Этот пункт фиксирую явно, потому что это отдельный пользовательский приоритет.

### 2.1. Лучше или хуже `v2.11`

Лучше.

Не по настроению, а по сухому evidence:

- `v2.11 total missing = 33`
- `v2.12 total missing = 14`

Особенно сильный recovery:

- `2673`: `12 -> 4`
- `2687`: тяжёлый lecture case вернулся к `missing=1`
- `2660`: sparse case снова near-best

### 2.2. Лучше или хуже baseline

Смешанная, но в целом положительная картина.

По coverage:

- baseline total missing = `22`
- `v2.12 total missing = 14`

По prose quality:

- `2660`, `2687`, `2673` часто выглядят редакторски живее baseline;
- headings в rich cases менее шаблонны, чем старый baseline Style C;
- но `2734` и `2687` всё ещё портятся из-за `посвящ*`;
- `2745` по тону собранный, но всё ещё слишком свободно перефразирует sparse floor.

Практический вывод:

- `v2.12` уже нельзя назвать "хуже baseline вообще";
- но и назвать его качественной победой над baseline пока рано.

## 3. Что в `v2.12` реально подтвердилось

### 3.1. `full-floor normalization` лучше старого dirty merge

Главный structural win:

- `clean facts` больше не конфликтуют так разрушительно с грязным floor;
- `presentation / lecture / sparse` cases заметно стабилизировались;
- `v2.11`-уровня катастрофы с `missing=9-12` ушли.

Это важнее любой одной prompt-правки.

### 3.2. `2660` снова выглядит как сильный compact кейс

Фактически:

- baseline `missing=2`
- `v2.11` заметно деградировал
- `v2.12` дал `missing=1`

Текст стал компактным и собранным.
Но успех не полный:

- фраза `посвященную теме` всё ещё делает кейс policy-dirty.

### 3.3. `2687` вернулся в рабочую зону

Это один из самых сильных сигналов раунда:

- baseline `missing=5`
- `v2.12 missing=1`

При этом текст всё ещё не идеален:

- он лучше baseline по coverage и по тематической собранности;
- но блок `Лекция посвящена...` показывает, что revise/policy layer не дожимает narrative cleanup.

### 3.4. `2673` больше не разваливается

`2673` был одним из самых устойчивых blockers почти всего цикла.

Теперь:

- baseline `missing=5`
- `v2.6 missing=4`
- `v2.11 missing=12`
- `v2.12 missing=4`

То есть новая architecture вернула этот кейс минимум к лучшему known range по coverage.
Но по prose проблема осталась:

- текст всё ещё слишком explanatory;
- `устройство / причины / задачи / проблема` звучат как переработанный питч, а не как действительно сильный редакторский анонс.

## 4. Где `v2.12` всё ещё провалился

### 4.1. `посвящ*` остаётся системным blocker

Это важнейший unresolved issue раунда.

Forbidden leakage сохранился на:

- `2660`
- `2734`
- `2687`

Причём в этих кейсах проблема уже не только stylistic.
Она бьёт по ощущению живого текста:

- фраза становится музейно-канцелярской;
- текст перестаёт звучать естественно;
- часть gains против baseline обнуляется на уровне impression.

### 4.2. `2745` показал предел literal coverage metric

Формально:

- baseline `missing=5`
- `v2.12 missing=5`

Но это misleading case.
Текст звучит естественнее baseline и реально собирает большую часть фактуры.
Проблема здесь двойная:

- либо compact prose слишком свободно переписывает sparse facts;
- либо current deterministic missing checker слишком literal.

На практике я бы не считал `2745` главным quality failure раунда, но как evaluation case он всё ещё unresolved.

### 4.3. `2734` всё ещё не стал clean person/program-led кейсом

Фактически:

- baseline `missing=5`
- `v2.12 missing=3`

Coverage лучше baseline.
Но quality victory нет:

- `посвящён` остался;
- sections всё ещё слегка generic;
- program/person material пересобран аккуратно, но не по-настоящему editorially sharp.

## 5. Что `v2.12` показал про настоящий bottleneck

Теперь bottleneck уже не тот, что был в `v2.11`.

Сейчас основной узкий участок:

- не dirty merge;
- а переход `normalized floor -> final prose`.

Особенно это видно на двух families:

- `presentation_project`
- `lecture/person-led / program-led`

Gemma уже умеет:

- удерживать coverage лучше baseline;
- удерживать structure лучше старых degraded rounds.

Gemma всё ещё плохо умеет:

- превращать clean noun-phrase facts в по-настоящему живой текст без канцелярита;
- делать это, не скатываясь в `посвящ`, `в центре внимания`, `проект позволяет` и другие semi-template ходы.

## 6. Что я бы не потерял из прошлых удачных раундов

Это важно для `v2.13+`, чтобы не увлечься архитектурой и не забыть про итоговый текст.

Из baseline / Style C:

- собранность fact-first prose;
- дисциплина по coverage;
- самодостаточный body.

Из `v2.6`:

- более живые headings;
- менее шаблонный lead;
- лучший feel на `2734` и части rich cases;
- ощущение, что текст уже ближе к редакторскому анонсу, а не к чистому fact composer.

Из research / casebook:

- конкретные существительные и program details работают лучше общих эпитетов;
- `why it matters` уместен не всегда, а только при реальной фактуре;
- headings должны быть содержательными, а не служебными.

## 7. Практический вывод

`v2.12` встраивать в runtime рано.

Но unlike `v2.11`, это уже не тупиковый раунд.
Он дал два важных результата сразу:

1. новая architecture beat baseline по coverage;
2. теперь можно обсуждать уже не "нужна ли новая архитектура вообще", а какой следующий слой действительно поднимет итоговый текст.

Следующий шаг рационально делать не как blind local patch, а как конкурентную post-run консультацию:

- `Opus` в strict one-shot;
- затем `Gemini 3.1 Pro`;
- с полным фокусом не только на coverage, но и на prose quality.

## 8. Operational Note

Markdown-отчёт `v2.12` отрендерился не полностью по кейсам.
Поэтому канонический source of truth для этого раунда:

- raw JSON `artifacts/codex/gemma_event_copy_pattern_dry_run_v2_12_5events_2026-03-08.json`
- этот review

А не только markdown report.
