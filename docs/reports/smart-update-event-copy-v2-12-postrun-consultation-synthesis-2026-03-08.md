# Smart Update Event Copy V2.12 Post-Run Consultation Synthesis

Дата: 2026-03-08

Связанные материалы:

- `artifacts/codex/tasks/event-copy-v2-12-postrun-complex-consultation-brief.md`
- `artifacts/codex/reports/event-copy-v2-12-postrun-complex-consultation-claude-opus.json`
- `artifacts/codex/reports/event-copy-v2-12-postrun-complex-consultation-claude-opus.md`
- `artifacts/codex/reports/event-copy-v2-12-postrun-complex-consultation-gemini-3.1-pro.md`
- `docs/reports/smart-update-gemma-event-copy-v2-12-prompt-context.md`
- `docs/reports/smart-update-opus-gemma-event-copy-pattern-dry-run-v2-12-review-2026-03-08.md`

## 1. Проверка моделей

`Opus` был вызван в strict one-shot режиме:

- `claude -p --model claude-opus-4-6 --tools "" --output-format json`

Проверка `modelUsage` подтвердила:

- `claude-opus-4-6`

`Gemini` была вызвана через:

- `gemini -m gemini-3.1-pro-preview`

Session log подтвердил:

- `gemini-3.1-pro-preview`

## 2. Baseline-Relative Verdict

Этот пункт фиксирую отдельно, потому что он обязателен для всех следующих раундов.

На тех же 5 кейсах:

- baseline total missing = `22`
- `v2.11` total missing = `33`
- `v2.12` total missing = `14`

Значит:

- `v2.12` лучше `v2.11`
- `v2.12` лучше baseline по coverage

Но overall quality verdict не равен "победа":

- baseline всё ещё чище по forbidden patterns;
- `v2.12` всё ещё не runtime-ready;
- prose quality mixed: местами лучше baseline, местами всё ещё bureaucratic.

## 3. Где Opus и Gemini сошлись

Обе внешние модели независимо пришли к нескольким сильным выводам:

1. `v2.12` — реальный coverage recovery round, а не косметический patch.
2. `full-floor normalization` структурно сильнее старой линии `subset extraction -> dirty merge`.
3. длинные negative ban-lists плохо работают на Gemma и partly poison generation.
4. same-model editorial review pass даёт мало пользы: модель в review сохраняет те же blind spots, что и в generation.
5. следующий раунд должен быть проще, а не сложнее.

Это я принимаю.

## 4. Где я спорю с внешними моделями

### 4.1. С Opus

Главное расхождение:

- Opus жёстко трактует итог как `baseline > v2.12`.

Я это не принимаю в таком виде.

Почему:

- для production event-copy coverage реально критичен;
- `22 -> 14` это не minor improvement;
- часть `v2.12` текстов уже редакторски живее baseline;
- проблема `v2.12` не в том, что baseline "всё ещё лучше", а в том, что `v2.12` пока не довёл quality win до clean production threshold.

Вторая сильная оговорка:

- Opus слишком охотно тянет к deterministic stripping.

В чистом виде это unsafe для русского текста.
Regex-like смысловая замена не должна становиться semantic core.

### 4.2. С Gemini

Gemini дал полезный ответ, но он оказался менее независимым, чем хотелось бы:

- часть его diagnosis почти зеркалит линию Opus;
- он слишком быстро делает вывод `factual coverage is king`, недооценивая пользовательский приоритет по качеству prose;
- его optimistичный тезис про ultra-short exemplar-driven generation полезен как hypothesis, но пока не доказан на наших кейсах.

Именно поэтому я не беру его few-shot line на веру, а только как next experiment.

## 5. Что реально подтверждено по практике

По коду, dry-run и outputs сейчас защищаемы следующие тезисы:

### 5.1. Новую architecture откатывать не надо

`v2.12` впервые beat baseline по coverage без возврата в `v2.11` chaos.

Поэтому:

- откат обратно к `subset extraction -> merge-back` был бы шагом назад.

### 5.2. Full editorial pass больше не выглядит justified

Это подтверждают и внешние модели, и кейсы:

- `2660`, `2734`, `2687` всё равно пропускают `посвящ*`;
- same-model review не дал качественного скачка по prose;
- latency/cost при этом заметные.

### 5.3. Следующий основной рычаг — не ещё одна ban-list, а better prose assembly

Это ключевой вывод.

Теперь проблема уже не в удержании facts как таковом, а в том, как Gemma собирает из них финальный живой текст.

Особенно чувствительные families:

- `presentation_project`
- `lecture_person`
- `program_rich`

### 5.4. Нельзя забывать лучшие prose wins прошлых раундов

Это отдельная защита от локального overfit.

Нужно сохранить:

- fact discipline baseline / Style C;
- живость headings и less-template feel из `v2.6`;
- research-derived preference for concrete nouns, useful headings and grounded `why it matters`.

## 6. Что я вижу как `v2.13`

### 6.1. Архитектура

Не делать `v2.12 + ещё ban-layers`.

Делать:

1. `full-floor normalization`
2. deterministic hygiene / dedup
3. generation
4. deterministic validation
5. только при реальном signal — targeted repair, а не full editorial rewrite

То есть:

- `v2.13` = `v2.12 - full editorial pass + targeted corrective loop`

### 6.2. Что менять в normalization

Нормализатор должен стать более mechanical и менее двусмысленным.

Практически:

- contentful metatext вида `лекция посвящена ...`, `расскажут о ...`, `проект представит ...` не должен доживать до `clean_fact` как frame;
- в `clean_fact` должен оставаться только предмет/содержание, а не бюрократическая оболочка;
- grouped items терять нельзя.

Это принимаю и от Opus, и от Gemini.

### 6.3. Что менять в generation

Generation prompt надо не расширять, а упрощать.

Рациональный ход:

- убрать часть negative microrules;
- сделать prompt короче;
- опереться на 1 strong shape-specific exemplar, а не на стену запретов;
- сохранить only hard invariants:
  - no hallucination
  - reflect all facts
  - no service/logistics
  - no generic headings

Это не означает "забыть baseline" или "забыть `v2.6`".
Наоборот:

- exemplars надо собирать именно из лучших human-judged moves baseline / `v2.6` / research.

### 6.4. Что менять в post-check

Вместо полного editorial pass:

- deterministic forbidden scan;
- deterministic coverage check;
- если нарушение найдено:
  - targeted sentence/section rewrite;
  - а не полный re-generation whole body.

Именно здесь я соглашаюсь с внешними моделями.

## 7. Что не беру в `v2.13`

- pure regex semantic rewriting as core mechanism;
- тезис `baseline overall still better`, как будто coverage secondary;
- ещё более длинные generation prompts;
- ещё один full same-model editorial pass;
- резкий architecture rollback к baseline Style C.

## 8. Что нужно проверить дополнительно

### 8.1. Evaluation quality

`2745` показывает, что current missing checker частично literal and brittle.

Поэтому next round надо оценивать по двум осям:

- current deterministic missing
- human semantic coverage judgement

### 8.2. Gemma generation settings

Нужно проверить текущие generation params.
Если generation идёт на сверхнизкой temperature, это plausibly подталкивает Gemma к safest bureaucratic register.

### 8.3. Generalization

Следующая итерация не должна тестироваться только на старых 5 кейсах.

Минимум:

- те же 5 для regression tracking
- плюс новые 5-10 unseen cases

## 9. Итоговое решение

Консультационный цикл по `v2.12` можно считать закрытым.

Следующий рациональный шаг:

- собрать `v2.13` локально

Его рабочая формула:

- keep `full-floor normalization`
- simplify normalizer framing
- move generation to shorter exemplar-driven prompts
- drop full editorial pass
- add deterministic validation + targeted repair only where needed

Это выглядит самым defensible путём, если главная цель остаётся прежней:

- не просто better coverage,
- а сильный, естественный, профессиональный итоговый текст на тысячах heterogeneous sources.
