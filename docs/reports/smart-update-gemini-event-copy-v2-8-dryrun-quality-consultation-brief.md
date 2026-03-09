# Smart Update Gemini Event Copy V2.8 Dry-Run Quality Consultation Brief

Дата: 2026-03-07

Связанные материалы:

- `docs/reports/smart-update-gemini-event-copy-v2-8-hypotheses-consultation-response.md`
- `docs/reports/smart-update-gemini-event-copy-v2-8-hypotheses-consultation-response-review.md`
- `docs/reports/smart-update-gemini-event-copy-v2-8-prompt-context.md`
- `docs/reports/smart-update-opus-gemma-event-copy-pattern-dry-run-v2-8-5-events-2026-03-07.md`
- `docs/reports/smart-update-opus-gemma-event-copy-pattern-dry-run-v2-8-review-2026-03-07.md`

## 1. Зачем нужна эта консультация

Нужен новый Gemini round уже после реального `v2.8` dry-run.

Это post-run consultation по mixed failure / partial gain outcome.

Мы уже:

- собрали `v2.8 hypothesis pack`;
- критически проконсультировались по нему с Gemini;
- локально внедрили `v2.8` как rollback-to-`v2.6` round;
- прогнали live Gemma на тех же 5 событиях.

Итог неоднородный:

- `2734` реально улучшился;
- остальные 4 кейса не стали quality win;
- `2687` всё ещё не избавился от `посвящ*`.

## 2. Главная цель

Нужно понять:

- почему rollback помог `2734`, но не помог `2660`, `2745`, `2687`, `2673`;
- какие идеи `v2.8` worth saving;
- какие exact prompt-level изменения всё ещё нужны для Gemma;
- какой minimal `v2.9` patch pack вообще имеет смысл тестировать дальше.

## 3. Что показал `v2.8`

Коротко:

- `2660`: `missing 0 -> 4` vs `v2.6`
- `2745`: `missing 3 -> 6` vs `v2.6`
- `2734`: `missing 5 -> 2` vs `v2.7`, явный gain
- `2687`: `missing 4`, `посвящ*` всё ещё жив
- `2673`: `missing 6`, service tone в prose всё ещё жив

Системный сигнал:

- все 5 кейсов ушли в `fact_first_v2_8`;
- даже sparse-кейсы не вернулись в compact branch.

## 4. Наш текущий рабочий диагноз

Это hypothesis, которую Gemini должен критически проверить, а не принимать автоматически.

### 4.1. `v2.8` убрал worst-case packaging из `v2.7`, но не дал controlled dense extraction

То есть:

- narrative shaping в Extraction действительно был ошибкой;
- но простой rollback оказался слишком слабым;
- extraction снова пропускает junk вроде label-style / intent-style facts.

### 4.2. Simplified hints убрали вредные sentence templates, но потеряли forcing power

Это видно по:

- `2660`: `Тема: теме противоречий мира.`
- `2687`: `Лекция расскажет о ...`
- `2673`: несколько `На презентации расскажут ...`

### 4.3. Human-readable revise для `посвящ*` недостаточен

`2687` показывает, что even stronger policy wording всё равно не гарантирует clean final text.

### 4.4. `2734` важен как positive outlier

Нельзя рассматривать `v2.8` только как failure.

`2734` suggests:

- сам rollback direction не был ошибкой целиком;
- где-то мы всё же попали в более правильное поведение модели.

## 5. Что Gemini должен увидеть

Мы передаём полный operational context:

- source texts;
- raw facts;
- extracted facts;
- facts_text_clean;
- copy_assets;
- final descriptions;
- prompts и algorithm contract `v2.8` в docs-only форме;
- pre-run consultation response и наш review;
- grounded `v2.8` dry-run review.

## 6. Что мы хотим от Gemini сейчас

Нужен не общий opinion и не повторение pre-run advice.

Нужно:

1. Критически прочитать реальный `v2.8` evidence set.
2. Объяснить, почему `2734` выиграл, а остальные кейсы нет.
3. Отделить:
   - good rollback direction;
   - weak wording;
   - still-wrong architecture or stage placement.
4. Дать только такие next-step changes, которые реально можно защитить evidence.

## 7. Самое важное требование

Нужны **конкретные prompt-level правки для Gemma**, особенно для:

- extraction prompt;
- `_pre_extract_issue_hints`;
- post-extract correction hints, если это нужно;
- standard generation prompt;
- revise/policy wording.

При этом нельзя:

- возвращать в Extraction новый narrative shaping;
- предлагать уже опровергнутые `v2.7`-стилем safe-positive templates;
- маскировать проблему новыми сложными слоями без явной необходимости.
