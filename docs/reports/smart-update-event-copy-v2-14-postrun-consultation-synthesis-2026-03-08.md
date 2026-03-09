# Smart Update Event Copy V2.14 Post-Run Consultation Synthesis

Дата: 2026-03-08

Связанные материалы:

- `artifacts/codex/tasks/event-copy-v2-14-postrun-complex-consultation-brief.md`
- `artifacts/codex/reports/event-copy-v2-14-postrun-complex-consultation-claude-opus.md`
- `artifacts/codex/reports/event-copy-v2-14-postrun-complex-consultation-gemini-3.1-pro.md`
- `docs/reports/smart-update-opus-gemma-event-copy-pattern-dry-run-v2-14-review-2026-03-08.md`
- `docs/reports/smart-update-gemma-event-copy-v2-14-prompt-context.md`

## 1. Bottom Line

Мой итог после `v2.14` dry-run, `Opus`, `Gemini` и локальной проверки такой:

- `v2.14` лучше baseline по aggregate coverage;
- `v2.14` хуже `v2.13` overall как quality-safe branch;
- `v2.14` не готов к production rollout;
- новый правильный следующий шаг — не откат architecture, а `v2.15` как subtraction-release.

Чётко:

- baseline total missing = `22`
- `v2.13` total missing = `14`, forbidden = `0`
- `v2.14` total missing = `14`, forbidden = `2` кейса (`2687`, `2673`)

То есть:

- относительно baseline progress real;
- относительно `v2.13` current branch regressed in hygiene and consistency.

## 2. Где Opus и Gemini совпали

Есть сильная общая зона согласия, и я её принимаю.

### 2.1 `split-call` сам по себе не ошибка

Обе модели считают, что current problem не в самом `normalize -> outline -> generate`.

Проблема в том, что:

- outline делает не structural planning;
- outline пишет mini-prose;
- generation потом начинает пересказывать outline, а не писать напрямую по фактам.

Это локально подтверждается по коду и артефактам:

- `2673` outline породил heading-и `Цель и задачи проекта`, `Формат мероприятия`, `Для кого это будет интересно`;
- `2687` outline `focus_note` прямо занёс формулу `Лекция посвящена...`;
- final generation это подхватил.

### 2.2 Outline надо делать structural-only

И `Opus`, и `Gemini` сходятся:

- убрать `focus_note`;
- убрать `heading` из outline schema;
- оставить только grouping by `fact_ids`.

Это сильный и practically useful consensus.

### 2.3 Generation prompt надо укорачивать

Обе модели считают, что current generation contract слишком длинный и перегружен bans.

Их общий совет:

- меньше exhaustive negative constraints;
- больше concise positive examples;
- headings и prose должны рождаться в generation stage, а не в outline.

### 2.4 `2687` — доказательство outline leakage

Обе модели видят один и тот же failure:

- forbidden framing попал не из deterministic layer, а из LLM outline;
- значит следующий шаг не в новых regex-патчах, а в том, чтобы перестать пускать bureaucratic prose в intermediate plan.

## 3. Где ответы расходятся

### 3.1 Universal normalization vs. risk of hallucination

`Opus` жёстче настаивает:

- normalization должен запрещать hollow labels типа `задачи платформы`;
- shape-specific normalization contracts надо убрать.

`Gemini` согласен по направлению, но делает важную поправку:

- если source сам даёт только topic label и не раскрывает detail,
- нельзя заставлять Gemma всегда раскрывать label в concrete fact,
- иначе модель начнёт hallucinate.

Я принимаю именно версию `Gemini` как более сильную:

- universal normalization да;
- hollow labels should be discouraged;
- но only when real detail is present in source;
- если в source detail нет, label можно сохранить как high-level fact, не выдумывая содержание.

### 3.2 Shape detection

Обе внешние модели склоняются к ослаблению shape-specific branching.

Мой вывод:

- как pipeline router shape действительно надо ослаблять;
- как lightweight metadata hint (`презентация проекта`, `лекция`, `концерт`) shape всё ещё полезен.

То есть:

- shape stays as prompt hint;
- shape should stop being a large branching tree.

## 4. Что я дополнительно проверил сам

### 4.1 В `v2.14` реальный problem channel — именно outline

Это не гипотеза, а подтверждённый факт по артефактам:

- `2673` outline сгенерировал бюрократические headings;
- `2687` outline сгенерировал `focus_note` с `посвящена`;
- final descriptions их использовали.

### 4.2 User concern about regex drift was correct

После последних раундов реальный следующий рост качества нельзя искать в:

- ещё большем числе regex bans;
- semantic source reconstruction through deterministic helpers.

Это не масштабируется на many-thousands-of-posts scenario.

Поэтому:

- semantic core must stay LLM-first;
- deterministic layer should remain support-only.

## 5. Comparative verdict

### Against baseline

`v2.14` лучше baseline по aggregate coverage.

Но я **не называю его clean overall win against baseline**, потому что:

- `2687` вернул forbidden issue;
- `2673` всё ещё prose-poor;
- по production-quality ветка ещё unstable.

### Against `v2.13`

`v2.14` хуже `v2.13` overall.

Почему:

- same total missing;
- worse hygiene;
- `2687` regressed;
- `2673` improved some structure, but still not enough to offset the regressions.

## 6. What I take into `v2.15`

### Must keep

- `full-floor normalization`
- split-call for richer cases
- exemplar-driven generation
- deterministic validation/hygiene
- baseline and `v2.6` as prose reference sources, not templates

### Must change

- outline schema must become `fact-ids only`
- generation must create its own headings from facts
- generation prompt must get shorter
- normalization should use one universal contract
- universal normalization must prefer concrete detail over topic labels when detail exists
- deterministic hygiene should strip forbidden framing before generation, not only after

### Reject

- new regex-first semantic shaping
- free-text `focus_note`
- heading generation inside outline
- ever-growing lists of negative constraints as the main lever

### Experiment-only

- skip outline/grouping when `facts_count <= 4`
- outline/grouping temperature = `0.0`, generation temperature tuned separately
- minimal shape hint only as generation metadata

## 7. Working image of `v2.15`

На сегодня я вижу `v2.15` так:

1. `raw_facts`
2. `universal normalization` through LLM
3. deterministic cleanup/dedup/forbidden sweep
4. `group_facts` LLM call only for richer cases, returning pure `fact_ids`
5. shorter generation prompt with exemplars and generated headings
6. deterministic validation
7. very narrow repair only for validated policy issues

For simple cases:

- skip step 4 when fact count is small

## 8. Final decision

Консультационный цикл по `v2.14` закрыт.

Мой следующий шаг вижу так:

- не делать ещё один external round before code;
- собрать `v2.15` по этой subtraction-логике;
- прогнать те же 5 кейсов;
- после этого снова сравнить against baseline and `v2.13`, explicitly both on coverage and on prose quality.
