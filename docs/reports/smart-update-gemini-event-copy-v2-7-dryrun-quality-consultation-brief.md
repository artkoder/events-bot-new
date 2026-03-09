# Smart Update Gemini Event Copy V2.7 Dry-Run Quality Consultation Brief

Дата: 2026-03-07

Связанные материалы:

- `docs/reports/smart-update-gemini-event-copy-v2-7-hypotheses-consultation-response.md`
- `docs/reports/smart-update-gemini-event-copy-v2-7-hypotheses-consultation-response-review.md`
- `docs/reports/smart-update-opus-gemma-event-copy-pattern-dry-run-v2-7-5-events-2026-03-07.md`
- `docs/reports/smart-update-opus-gemma-event-copy-pattern-dry-run-v2-7-review-2026-03-07.md`
- `artifacts/codex/experimental_pattern_dryrun_v2_7_2026_03_07.py`

## 1. Зачем нужна эта консультация

Нам нужен новый Gemini round уже после реального `v2.7` dry-run.

Это failure consultation.

Мы уже:

- обсудили hypothesis pack;
- критически откалибровали safe-positive direction;
- локально внедрили `v2.7`;
- прогнали live Gemma на тех же 5 событиях.

Итог оказался плохим.

## 2. Главная цель

Нужно понять:

- почему safe-positive patch pack не только не решил `2687` и `2673`, но ещё и сломал `2660`, `2745`, `2734`;
- какие идеи из `v2.7` worth saving;
- что exactly надо откатить;
- какие минимальные prompt edits для Gemma можно тестировать дальше без повторения этого regression.

## 3. Что показал `v2.7`

Коротко:

- `2660`: `missing 0 -> 2`
- `2745`: `missing 3 -> 6`
- `2734`: `missing 3 -> 5`
- `2687`: `missing 5 -> 4`, но `посвящ*` жив
- `2673`: `missing 4 -> 5`, `facts 11 -> 14`, `посвящ*` жив

То есть:

- current `v2.7` patch pack не сработал;
- main targets не закрылись;
- collateral damage высокий.

## 4. Наш текущий рабочий диагноз

Это hypothesis, которую Gemini должен критически проверить.

### 4.1. Safe-positive transformation в текущем wording порождает fact inflation

То есть:

- мы убрали канцелярит;
- но взамен начали порождать packaging facts;
- representation стала более generic и рыхлой.

### 4.2. Prompt examples оказались слишком формульными

Конструкции вроде:

- `В центре встречи — ...`
- `Выставка носит название ...`

оказались редакторски безопасными, но quality-negative.

### 4.3. `посвящ*` и intent transfer не лечатся только examples

`2687` и `2673` показывают, что:

- новые examples сами по себе не достаточны;
- возможно, проблема уже требует более точного support layer или другого revise behavior.

## 5. Что Gemini должен увидеть

Мы передаём полный operational context:

- source texts;
- raw facts;
- extracted facts;
- facts_text_clean;
- copy_assets;
- final descriptions;
- prompts и algorithm contract `v2.7`;
- pre-run consultation response и наш review;
- наш grounded `v2.7` review.

## 6. Что мы хотим от Gemini сейчас

Нужен не общий opinion и не защита прошлых советов.

Нужно:

1. Критически прочитать реальный failure set.
2. Отделить:
   - good idea / bad implementation;
   - bad wording;
   - bad idea in principle.
3. Дать только такие next-step changes, которые реально можно защитить evidence.

## 7. Самое важное требование

Нужны **конкретные prompt-level правки для Gemma**, но:

- без unsafe rewrites;
- без new complexity for its own sake;
- без optimistic abstractions.

Нас особенно интересуют:

- extraction prompt;
- `_pre_extract_issue_hints`;
- standard generation prompt;
- revise/policy wording;
- возможно, very small deterministic support, если без него никак.
