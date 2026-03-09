# Smart Update Opus Gemma Event Copy Pattern Dry Run V2.13 Review

Дата: 2026-03-08

Связанные материалы:

- `artifacts/codex/experimental_pattern_dryrun_v2_13_2026_03_08.py`
- `artifacts/codex/gemma_event_copy_pattern_dry_run_v2_13_5events_2026-03-08.json`
- `docs/reports/smart-update-opus-gemma-event-copy-pattern-dry-run-v2-13-5-events-2026-03-08.md`
- `docs/reports/smart-update-gemma-event-copy-v2-13-prompt-context.md`

## Verdict

`v2.13` — mixed-positive round.

Что можно утверждать уверенно:

- `v2.13` лучше baseline по coverage: `14 < 22`;
- `v2.13` лучше baseline по hygiene: `forbidden = 0`;
- `v2.13` не лучше `v2.12` по total missing: оба дают `14`;
- `v2.13` местами лучше `v2.12` по чистоте и naturalness, но это не uniform win.

Итог:

- `v2.13` ещё не готов к переносу в runtime;
- но это уже более сильная база для следующего шага, чем `v2.12`, если приоритетом считать не только literal coverage, но и общую publishability текста.

## Главные наблюдения

### 1. `v2.13` наконец чисто закрыл forbidden hygiene

По сравнению с `v2.12` ушли `посвящ*`-срывы:

- `2660`: `['посвящ*'] -> []`
- `2734`: `['посвящ*'] -> []`
- `2687`: `['посвящ*'] -> []`

Это не косметика, а реальный quality gain: текст перестал спотыкаться о самые заметные machine-like формулы.

### 2. Coverage теперь лучше baseline, но plateau против `v2.12`

Суммарно:

- baseline total missing = `22`
- `v2.6` total missing = `15`
- `v2.12` total missing = `14`
- `v2.13` total missing = `14`

То есть новая exemplar-driven ветка не пробила coverage plateau `v2.12`, но удержала его без forbidden leakage.

### 3. Лучшие реальные wins — `2660` и `2745`

`2660`

- `missing`: baseline `2`, `v2.12` `1`, `v2.13` `1`
- prose у `v2.13` сильнее baseline и естественнее `v2.12`;
- это хороший sparse-case result.

`2745`

- `missing`: baseline `5`, `v2.12` `5`, `v2.13` `3`
- текст краткий, чистый, не раздутый;
- здесь `v2.13` однозначно лучше baseline и `v2.12`.

### 4. Главный blocker остался в `2673`

`2673`

- baseline `missing=5`
- `v2.12 missing=4`
- `v2.13 missing=5`

Текст `v2.13` уже чище baseline stylistically, но он всё ещё explanation-heavy и теряет предметность:

- хуже донесены `задачи / возможности / устройство платформы`;
- исчезла часть project-specific content;
- prose звучит гладко, но уже начинает расплачиваться factual sharpness.

Это важный сигнал: `presentation/project` shape всё ещё требует отдельной инженерной доработки.

### 5. `2687` — явный локальный regression против `v2.12`

`2687`

- baseline `5`
- `v2.12` `1`
- `v2.13` `2`

Текст стал чище по формулировкам, но coverage упал: ослабли `british-roots / Shanks` сигналы.

Это ещё одно подтверждение, что cleaner generation сама по себе не решает задачу; для lecture/person-rich кейсов нужен stronger preservation contract.

### 6. `2734` перестал спотыкаться о `посвящ*`, но не стал убедительным quality win

`2734`

- baseline `5`
- `v2.12` `3`
- `v2.13` `3`

Плюс:

- текст чище `v2.12` по hygiene;
- нет прямого forbidden leakage.

Минус:

- всё ещё high-level rewrite;
- всё ещё недостаточно programme-specific;
- местами prose становится слишком “объясняющим”.

## Event-by-Event Verdict

### 2660 — Дуальность этого мира

- Лучше baseline: да.
- Лучше `v2.12`: скорее да по prose, примерно равно по coverage.
- Вердикт: один из strongest current results.

### 2745 — Сёстры

- Лучше baseline: да.
- Лучше `v2.12`: да.
- Вердикт: уверенный win.

### 2734 — Концерт Владимира Гудожникова

- Лучше baseline: да.
- Лучше `v2.12`: нет убедительного доказательства.
- Вердикт: hygiene improved, but not enough.

### 2687 — Лекция «Художницы»

- Лучше baseline: да.
- Лучше `v2.12`: нет, coverage хуже.
- Вердикт: локальный regression against best recent version.

### 2673 — Собакусъел

- Лучше baseline: по чистоте да, по coverage нет.
- Лучше `v2.12`: нет.
- Вердикт: главный blocker итерации.

## Что, по моему мнению, означает `v2.13`

`v2.13` подтвердил 3 вещи:

1. `full-floor normalization` откатывать не надо.
2. `shorter exemplar-driven generation` работает лучше wall-of-rules prompts.
3. `full editorial rewrite pass` действительно был лишним и хрупким.

Но `v2.13` также подтвердил 3 ограничения:

1. `presentation/project` cases не лечатся только более красивым generation.
2. `lecture/person` cases всё ещё требуют stronger preservation of grouped specifics.
3. Следующий рост качества надо искать не в общем усложнении пайплайна, а в точных shape-specific contracts.

## Предварительная рамка для следующего шага

Если идти дальше после внешней консультации, я бы смотрел на `v2.14` так:

- сохранить `v2.13` core architecture;
- не возвращать full editorial pass;
- отдельно усилить `presentation_project` preservation;
- отдельно усилить `lecture_person` grouped-content preservation;
- держать generation коротким и exemplar-driven;
- добавлять новые LLM calls только если они реально бьют в blocker, а не просто расширяют pipeline.
