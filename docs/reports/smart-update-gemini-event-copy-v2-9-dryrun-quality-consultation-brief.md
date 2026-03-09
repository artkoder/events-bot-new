# Smart Update Gemini Event Copy V2.9 Dry-Run Quality Consultation Brief

Дата: 2026-03-08

Связанные материалы:

- `docs/reports/smart-update-gemini-event-copy-v2-9-hypotheses-consultation-response.md`
- `docs/reports/smart-update-gemini-event-copy-v2-9-hypotheses-consultation-response-review.md`
- `docs/reports/smart-update-gemini-event-copy-v2-9-prompt-context.md`
- `docs/reports/smart-update-opus-gemma-event-copy-pattern-dry-run-v2-9-5-events-2026-03-08.md`
- `docs/reports/smart-update-opus-gemma-event-copy-pattern-dry-run-v2-9-review-2026-03-08.md`

## 1. Зачем нужна эта консультация

Это post-run consultation уже после реального `v2.9` dry-run.

Мы уже:

- собрали `v2.9 hypothesis pack`;
- критически проконсультировались по нему с Gemini;
- локально внедрили `v2.9`;
- прогнали live Gemma на тех же 5 событиях.

Теперь нужен не opinion round, а evidence-based review по реальным outputs.

## 2. Главная цель

Нужно понять:

- какие именно hypotheses `v2.9` реально подтвердились;
- почему synthetic label cleanup помог лишь частично;
- почему `2687` и `2673` почти не сдвинулись;
- какие exact prompt-level changes для Gemma действительно worth testing дальше.

## 3. Что показал `v2.9`

Коротко по dry-run:

- `2660`: `missing 4 -> 4`, `forbidden none -> none`
- `2745`: `missing 6 -> 5`, `forbidden none -> none`
- `2734`: `missing 2 -> 3`, `forbidden none -> none`
- `2687`: `missing 4 -> 4`, `forbidden посвящ* -> посвящ*`
- `2673`: `missing 6 -> 6`, `forbidden none -> посвящ*`

Практическая картина:

- synthetic `Тема: ...` pollution почти исчезла;
- но intent-style / service-style facts всё ещё живут;
- dense lecture / presentation cases всё ещё раздувают facts layer;
- `посвящ*` всё ещё surviving failure в сложных кейсах.

## 4. Наш текущий рабочий диагноз

Это hypothesis, которую Gemini должен критически проверить, а не принимать на веру.

### 4.1. Узкий sanitizer fix был правильным, но слишком локальным

Да, branch `<event> посвящена ... -> Тема: ...` действительно вредила.

Но после её bypass стало видно, что deeper problem осталась:

- weak fact-unit shaping;
- intent/metatext wrappers;
- insufficient repair of dense facts.

### 4.2. `ОШИБКА:` hints повысили signal, но не стали blocking behavior

Это видно по:

- `2687`, где `посвящ*` остался;
- `2673`, где остались `На презентации расскажут ...`;
- overall lack of strong change on dense cases.

### 4.3. `Anti-splitting` wording всё ещё слишком слабая

Даже после `v2.9`:

- `2687` дошёл до `11` facts;
- `2673` дошёл до `12` facts.

То есть Gemma всё ещё не удерживает плотные fact units там, где source сам даёт много пересекающихся сигналов.

### 4.4. `2734` важен как calibration case

`v2.9` не развалил его, но и не удержал `v2.8` gain.

Это полезно, потому что показывает:

- sanitizer bypass safe;
- но stronger extraction wording itself не прибавила качества.

## 5. Что Gemini должен увидеть

Мы передаём полный quality context:

- source texts;
- raw_facts;
- extracted_facts_initial;
- facts_text_clean;
- copy_assets;
- final texts;
- `v2.9` prompt context;
- pre-run Gemini consultation response и наш review;
- grounded `v2.9` dry-run review.

## 6. Что мы хотим от Gemini сейчас

Нужен не general brainstorming и не blind repetition прошлых советов.

Нужно:

1. Критически прочитать реальный `v2.9` evidence set.
2. Отделить:
   - what actually improved;
   - what stayed broken;
   - what regressed.
3. Понять, где bottleneck:
   - extraction wording;
   - hints forcing power;
   - stage placement;
   - or still-overlooked support layer interaction.
4. Дать только такие next-step changes, которые реально можно защитить evidence.

## 7. Самое важное требование

Нужны **конкретные prompt-level правки для Gemma**, особенно для:

- extraction prompt;
- `_pre_extract_issue_hints`;
- post-extract hints;
- standard generation prompt;
- revise / policy wording.

При этом нельзя:

- возвращать `v2.7`-style narrative shaping;
- предлагать уже проверенные и не сработавшие blunt fixes;
- уходить в большой новый pipeline без сильного evidence.
