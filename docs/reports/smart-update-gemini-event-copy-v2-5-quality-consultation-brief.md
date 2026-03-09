# Smart Update Gemini Event Copy V2.5 Quality Consultation Brief

Дата: 2026-03-07

Связанные материалы:

- `docs/reports/smart-update-opus-gemma-event-copy-pattern-dry-run-v2-4-5-events-2026-03-07.md`
- `docs/reports/smart-update-opus-gemma-event-copy-pattern-dry-run-v2-4-review-2026-03-07.md`
- `docs/reports/smart-update-gemini-event-copy-v2-4-regression-consultation-response.md`
- `docs/reports/smart-update-gemini-event-copy-v2-4-regression-consultation-response-review.md`
- `docs/reports/smart-update-opus-gemma-event-copy-pattern-dry-run-v2-5-5-events-2026-03-07.md`
- `docs/reports/smart-update-opus-gemma-event-copy-pattern-dry-run-v2-5-review-2026-03-07.md`
- `artifacts/codex/experimental_pattern_dryrun_v2_5_2026_03_07.py`

## 1. Зачем нужна эта консультация

Нам нужен новый Gemini round уже не после теоретического патча, а после реального `v2.5` dry-run на live Gemma.

Главная цель не изменилась:

- естественный;
- профессиональный;
- связный;
- фактически grounded;
- не шаблонный;
- редакторски сильный текст описания события для Telegraph.

Но теперь вопрос стоит конкретнее:

- почему `v2.5` оказался полезным corrective round, но не новым quality winner;
- какие идеи из `v2.5` реально подтверждены;
- что именно ещё ломает качество текста;
- какие точные Gemma-friendly prompt edits стоит тестировать дальше.

## 2. Что было до этого

`v2.3` был первой mixed-positive версией:

- живее sparse cases;
- лучше self-sufficiency;
- меньше brittle lexical forcing.

`v2.4` стал regression round:

- routing drift;
- fact inflation;
- ухудшение prose balance;
- потеря naturalness на нескольких кейсах.

После этого был Gemini regression round.
Его рекомендации мы не приняли на веру, а критически разобрали и собрали на их основе узкий `v2.5`.

## 3. Что проверял `v2.5`

`v2.5` строился от `v2.3`, а не от `v2.4`.

В нём проверялись только узкие changes:

1. `human-readable anti-посвящ*` policy issue вместо opaque marker;
2. `anti-question headings`;
3. более мягкие grouped hints для program items;
4. routing cushion `<= 6` как локальный experiment;
5. небольшой metatext dedup tweak.

То есть это не новый redesign, а narrow corrective round.

## 4. Что показал `v2.5`

Коротко:

- `v2.5` лучше `v2.4`;
- но по сумме всё ещё не лучше `v2.3`.

Самые важные outcomes:

- `2660`: `missing=1`, но prose стала тяжелее и дублирующей.
- `2745`: routing восстановился, но `v2.3` всё ещё лучше.
- `2734`: заметный recovery, `missing=4` и без `посвящ*`.
- `2687`: coverage сильный (`missing=1`), но forbidden `посвящ*` вернулся.
- `2673`: generic structure drift остаётся; anti-question heading не решил template feel.

Итог:

- ветка полезна как evidence;
- в runtime переносить рано;
- нужен новый grounded external critique.

## 5. Что именно Gemini должен увидеть

Мы передаём не summary, а полный operational context:

- source texts;
- raw facts;
- extracted facts;
- facts_text_clean;
- copy_assets;
- side-by-side descriptions по версиям;
- prompts и algorithm contract текущего harness;
- наш критический review.

То есть Gemini видит:

- не только финальные тексты;
- но и полный путь их формирования.

## 6. Наш текущий рабочий диагноз

Это не догма, а гипотеза, которую Gemini должен критически проверить.

### 6.1. Главный bottleneck всё ещё в раннем слое

Основные swings приходят не из “красивых” generation hints, а из:

- extraction;
- fact shaping;
- routing;
- revise policy wording.

### 6.2. `<= 6` помог sparse cases, но слишком груб для richer compact-looking events

`2660` и `2745` выиграли от routing rollback.

Но `2734` тоже ушёл в compact branch, и это может быть:

- либо lucky exception;
- либо признак недостаточно точного gate.

### 6.3. Human-readable anti-`посвящ*` ещё не достаточно strong

Да, это лучше старого marker.
Но `2687` показывает, что текущая формулировка всё ещё не forced enough для Gemma.

### 6.4. `2673` показывает, что anti-question headings не решает generic structure drift

Ветка стала чуть чище, но prose всё ещё легко уходит в объясняющую и шаблонную структуру.

## 7. Что мы хотим от Gemini сейчас

Нужен не общий opinion и не новый большой redesign.

Нужно:

1. Критически прочитать реальный `v2.5` evidence set.
2. Не защищать автоматически прошлые рекомендации Gemini.
3. Отделить:
   - верную идею;
   - неудачную реализацию;
   - слабую идею.
4. Дать только такие next-step changes, которые реально worth testing.

## 8. Самое важное требование

Нужны **конкретные правки промптов для Gemma**.

Особенно интересуют:

- extraction prompt;
- compact generation prompt;
- standard generation prompt;
- revise prompt;
- policy issue wording;
- small routing / contract adjustments.

Нас интересуют:

- patchable changes;
- rewritten blocks;
- before/after wording;
- Gemma-friendly rules;
- без needless complexity.
