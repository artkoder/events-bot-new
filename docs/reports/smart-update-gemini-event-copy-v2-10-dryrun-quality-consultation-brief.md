# Smart Update Gemini Event Copy V2.10 Dry-Run Quality Consultation Brief

Дата: 2026-03-08

Связанные материалы:

- `docs/reports/smart-update-gemini-event-copy-v2-9-dryrun-quality-consultation-response.md`
- `docs/reports/smart-update-gemini-event-copy-v2-9-dryrun-quality-consultation-response-review.md`
- `docs/reports/smart-update-gemini-event-copy-v2-10-prompt-context.md`
- `docs/reports/smart-update-opus-gemma-event-copy-pattern-dry-run-v2-10-5-events-2026-03-08.md`
- `docs/reports/smart-update-opus-gemma-event-copy-pattern-dry-run-v2-10-review-2026-03-08.md`

## 1. Зачем нужна эта консультация

Это post-run consultation уже после реального `v2.10` dry-run.

`v2.10` был собран как narrow round поверх `v2.9`, без нового pre-consultation.

Новый шаг тестировал:

- `list consolidation`;
- compact `[плохо] -> [хорошо]` intent examples;
- action-oriented issue hints.

Теперь нужен не opinion round, а evidence-based critique по реальным outputs.

## 2. Главная цель

Нужно понять:

- почему `list consolidation` реально помог `2734` и `2687`;
- почему тот же patch pack сломал `2660` и особенно `2673`;
- какие exact prompt-level changes для Gemma worth testing дальше;
- нужно ли разделять extraction contract по source shape.

## 3. Что показал `v2.10`

Коротко:

- `2660`: `missing 4 -> 6`, `forbidden none -> none`
- `2745`: `missing 5 -> 5`, `forbidden none -> none`, branch вернулся в `compact_fact_led`
- `2734`: `missing 3 -> 1`, `forbidden none -> посвящ*`
- `2687`: `missing 4 -> 2`, `forbidden посвящ* -> посвящ*`
- `2673`: `missing 6 -> 9`, `forbidden посвящ* -> none`, `facts 12 -> 15`

Практическая картина mixed:

- lecture / concert rich cases partly improved;
- project / presentation case развалился сильнее;
- `посвящ*` всё ещё не добивается стабильно.

## 4. Наш текущий рабочий диагноз

Это hypothesis, которую Gemini должен критически проверить.

### 4.1. `list consolidation` itself is real

`2734` и `2687` это подтверждают.

### 4.2. Current realization of `list consolidation` is too broad

Похоже, что:

- для lecture / concert cases это useful compression;
- для project / presentation cases Gemma начинает ещё сильнее размножать service/metatext framing.

### 4.3. `[ПЛОХО] -> [ХОРОШО]` examples могли быть too close to source phrasing

Особенно для `2673`.

Вместо real transformation модель местами закрепила саму рамку `расскажут о ...`.

### 4.4. `посвящ*` остаётся отдельным stubborn failure

Даже при более action-oriented wording:

- `2734` improved strongly, but forbidden marker came back;
- `2687` improved strongly, but forbidden marker stayed.

## 5. Что Gemini должен увидеть

В dry-run report уже есть:

- source texts;
- raw_facts;
- extraction hints;
- extracted_facts_initial;
- facts_text_clean;
- copy_assets;
- side-by-side descriptions.

Мы также передаём:

- `v2.10` prompt context;
- предыдущий `v2.9` Gemini response и наш review;
- grounded `v2.10` review.

## 6. Что мы хотим от Gemini сейчас

Нужно:

1. Критически прочитать реальный `v2.10` evidence set.
2. Объяснить, почему `2734/2687` выиграли, а `2660/2673` проиграли.
3. Сказать, не нужен ли split extraction contract by source shape.
4. Дать только такие next-step prompt changes, которые реально можно защитить evidence.

## 7. Самое важное требование

Нужны **конкретные prompt-level правки для Gemma**, особенно для:

- extraction prompt;
- `_pre_extract_issue_hints`;
- post-extract hints;
- revise / policy wording.

При этом нельзя:

- возвращать `v2.7`-style narrative shaping;
- предлагать большой новый pipeline без сильного evidence;
- игнорировать, что `v2.10` уже дал real gains на части кейсов.
