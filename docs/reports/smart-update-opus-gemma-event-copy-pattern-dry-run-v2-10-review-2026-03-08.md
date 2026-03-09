# Smart Update Opus Gemma Event Copy Pattern Dry-Run V2.10 Review

Дата: 2026-03-08

Связанные материалы:

- `docs/reports/smart-update-opus-gemma-event-copy-pattern-dry-run-v2-10-5-events-2026-03-08.md`
- `docs/reports/smart-update-gemini-event-copy-v2-9-dryrun-quality-consultation-response-review.md`
- `artifacts/codex/experimental_pattern_dryrun_v2_10_2026_03_08.py`

## 1. Краткий verdict

`v2.10` не стал quality win и не готов к переносу в runtime.

Но это не пустой раунд.

`v2.10` впервые подтвердил, что `list consolidation` может реально помочь dense lecture / concert cases:

- `2734`: `missing 3 -> 1` vs `v2.9`
- `2687`: `missing 4 -> 2` vs `v2.9`

Одновременно он показал, что тот же patch pack ломается на:

- `2660`: `missing 4 -> 6`
- `2673`: `missing 6 -> 9`, `facts 12 -> 15`

То есть новая гипотеза не опровергнута полностью, но она пока unsafe и case-sensitive.

## 2. Что реально улучшилось

### 2.1. `2734` дал strongest gain

Это лучший кейс раунда:

- `facts=5` остались компактными;
- `missing 3 -> 1`;
- текст по структуре стал более собранным.

Но quality still incomplete:

- `forbidden=['посвящ*']` вернулся;
- final prose чище, но revise всё ещё не дожимает forbidden marker.

### 2.2. `2687` впервые заметно сжался

Это ещё один реальный gain:

- `facts 11 -> 10`;
- `missing 4 -> 2`.

Значит `list consolidation` для lecture/person-rich cases не была теоретической идеей.

Но blocker остался:

- `forbidden=['посвящ*']`;
- intent-style residue в facts layer всё ещё жив.

### 2.3. `2745` вернулся в compact branch

Практический плюс:

- `branch=fact_first_v2_9 -> compact_fact_led`;
- `facts 7 -> 6`;
- prose звучит легче и компактнее.

Но coverage не улучшился:

- `missing 5 -> 5`.

Это stylistic gain, но не quality breakthrough.

## 3. Что ухудшилось

### 3.1. `2660` стал явным regression case

Итог:

- `facts 7 -> 8`;
- `missing 4 -> 6`;
- текст стал хуже не только по coverage, но и по prose hygiene.

Самые неприятные симптомы:

- искусственные кавычки / almost-literal quote leakage;
- section inflation;
- awkward repetition of already-known facts.

Практический вывод:

- новые examples/hints местами провоцируют не consolidation,
- а extra packaging / quoting behavior.

### 3.2. `2673` развалился сильнее всего

Итог:

- `facts 12 -> 15`;
- `missing 6 -> 9`;
- intent-style service framing усилилось, а не ослабло.

Это самый важный negative result всего раунда.

`presentation`-кейсы по-прежнему ломают extraction:

- `Презентация расскажет ...`
- `На презентации расскажут ...`
- duplicated project-intent facts

То есть Gemma не просто игнорирует anti-intent rule.
Она при таком source shape иногда начинает размножать эту рамку ещё сильнее.

## 4. Что показал `v2.10` по гипотезам

### 4.1. `list consolidation` — real but unstable

Это главное.

Идея подтверждена на `2734` и `2687`.
Но текущая prompt realization unsafe для `2660` и `2673`.

Следствие:

- отвергать саму гипотезу нельзя;
- текущую формулировку принимать в runtime тоже нельзя.

### 4.2. `action-oriented hints` — частично полезны

Они явно не были пустыми.

Иначе `2687` не дал бы такого сдвига.

Но они не стали robust forcing layer:

- `2673` их фактически проигнорировал;
- `посвящ*` по-прежнему surviving issue.

### 4.3. `compact intent examples` — mixed to negative

Похоже, что для project/presentation cases они слишком близки к source metatext.

Вместо transform effect часть examples могла закрепить сам шаблон `расскажут о ...`.

## 5. Текущий рабочий диагноз

После `v2.10` diagnosis стал точнее:

1. `list consolidation` нужен, но не как universal rule.
2. Dense lecture / concert cases реагируют на него лучше, чем project/presentation cases.
3. `presentation`-style sources требуют другой transformation contract:
   - не просто anti-intent,
   - а более жёсткое превращение `что расскажут` в `что представляет собой предмет`.
4. `посвящ*` остаётся отдельным stubborn failure, который extraction + revise пока не добивают.
5. Generation не выглядит root cause; основной leverage всё ещё в extraction/fact shaping.

## 6. Что хочу проверить с Gemini дальше

Нужен уже не общий review, а very narrow corrective reading:

1. Почему `list consolidation` помог `2734/2687`, но сломал `2660/2673`?
2. Не слишком ли close-to-source были наши `[ПЛОХО] / [ХОРОШО]` examples?
3. Нужно ли разделять extraction contract по source shape:
   - lecture/person-rich
   - presentation/project-rich
4. Как дать Gemma более надёжный anti-`посвящ*` repair without deterministic смысловая порча.

## 7. Bottom line

`v2.10` — valuable diagnostic round, но не candidate branch.

Практический итог:

- **save for future**: сама идея `list consolidation`;
- **save for future**: compact/action-oriented hints как direction;
- **rework**: examples for intent transformation;
- **rework**: project/presentation extraction contract;
- **rework**: `посвящ*` suppression.
