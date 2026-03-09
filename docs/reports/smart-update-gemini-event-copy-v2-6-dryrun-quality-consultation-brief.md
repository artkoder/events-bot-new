# Smart Update Gemini Event Copy V2.6 Dry-Run Quality Consultation Brief

Дата: 2026-03-07

Связанные материалы:

- `docs/reports/smart-update-gemini-event-copy-v2-6-hypotheses-consultation-response.md`
- `docs/reports/smart-update-gemini-event-copy-v2-6-hypotheses-consultation-response-review.md`
- `docs/reports/smart-update-opus-gemma-event-copy-pattern-dry-run-v2-6-5-events-2026-03-07.md`
- `docs/reports/smart-update-opus-gemma-event-copy-pattern-dry-run-v2-6-review-2026-03-07.md`
- `artifacts/codex/experimental_pattern_dryrun_v2_6_2026_03_07.py`

## 1. Зачем нужна эта консультация

Нам нужен новый Gemini round уже после реального `v2.6` dry-run.

Это не theoretical consultation.
Мы уже:

- сформулировали свои гипотезы;
- обсудили их с Gemini;
- локально внедрили `v2.6`;
- прогнали live Gemma на тех же 5 событиях.

Теперь нужен разбор именно фактического результата.

## 2. Главная цель

Цель не меняется:

- максимально естественный;
- профессиональный;
- связный;
- точный;
- grounded;
- не шаблонный текст описания события.

Но теперь вопрос стоит так:

- почему `v2.6` дал сильные wins на части кейсов;
- почему он всё ещё ломается на `2687` и частично `2673`;
- какие конкретные правки промптов для Gemma реально worth testing next.

## 3. Что уже показал `v2.6`

Коротко:

- `2660`: лучший compact result цикла (`missing=0`);
- `2745`: частичный recovery (`missing=3`);
- `2734`: самый важный success case (`missing=3`, forbidden cleared);
- `2687`: главный blocker (`missing=5`, `посвящ*`, label-style leak);
- `2673`: mixed improvement (`missing=4`), но tone всё ещё бюрократичный и `посвящ*` вернулся.

Итог:

- это уже не blind theory;
- у нас есть реальный failure map;
- следующий шаг должен быть очень точным.

## 4. Что Gemini должен увидеть

Мы передаём полный operational context, а не summary:

- source texts;
- raw facts;
- extracted facts;
- facts_text_clean;
- copy_assets;
- branch routing;
- final descriptions по версиям;
- prompts;
- algorithm contract всего `v2.6` harness;
- наш review hypothesis-response;
- наш review самого `v2.6`.

То есть Gemini видит:

- и сами тексты;
- и весь путь их получения.

## 5. Наш текущий рабочий диагноз

Это не догма. Это гипотеза, которую Gemini должен критически проверить.

### 5.1. Направление частично подтверждено

Stricter contract реально помог:

- sparse cases;
- performance/program-led case `2734`.

### 5.2. Проблема сместилась в lecture/presentation branch behavior

Именно там сейчас основной failure:

- `2687`
- `2673`

То есть вопрос уже не “нужен ли redesign вообще”, а “как стабилизировать слабые content shapes, не ломая уже подтверждённые gains”.

### 5.3. Label-style / `посвящ*` / intent transfer всё ещё не реально enforced

Мы уже ужесточали contract, но практика показывает:

- ban ещё не стал исполненным;
- revise не всегда делает real rewrite;
- extraction всё ещё пропускает narrative-poor facts.

## 6. Что мы хотим от Gemini сейчас

Нужен не общий opinion и не автоматическая защита прошлых советов.

Нужно:

1. Критически разобрать реальные `v2.6` outputs.
2. Отделить:
   - working idea;
   - bad implementation;
   - idea that itself is weak.
3. Дать только такие next-step changes, которые реально могут улучшить качество текста.

## 7. Самое важное требование

Нужны **конкретные правки промптов для Gemma**.

Нас особенно интересуют:

- extraction prompt;
- compact generation prompt;
- standard generation prompt;
- revise / policy issue wording;
- небольшие routing / contract tweaks.

Нужны:

- patchable changes;
- rewritten blocks;
- before/after style guidance;
- Gemma-friendly формулировки;
- без needless complexity.
