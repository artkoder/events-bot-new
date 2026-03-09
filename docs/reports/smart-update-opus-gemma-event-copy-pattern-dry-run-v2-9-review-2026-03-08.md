# Smart Update Opus Gemma Event Copy Pattern Dry-Run V2.9 Review

Дата: 2026-03-08

Связанные материалы:

- `docs/reports/smart-update-opus-gemma-event-copy-pattern-dry-run-v2-9-5-events-2026-03-08.md`
- `docs/reports/smart-update-gemini-event-copy-v2-9-hypotheses-consultation-response.md`
- `docs/reports/smart-update-gemini-event-copy-v2-9-hypotheses-consultation-response-review.md`
- `docs/reports/smart-update-gemini-event-copy-v2-9-prompt-context.md`
- `artifacts/codex/experimental_pattern_dryrun_v2_9_2026_03_08.py`

## 1. Краткий verdict

`v2.9` не стал quality win и не готов к переносу в runtime.

Это полезный corrective round, но только как evidence-building iteration.

Что подтвердилось:

- synthetic `Тема: ...` contamination действительно была реальной проблемой;
- узкий bypass этой ветки в prompt-facing sanitizer полезен;
- `v2.9` убрал часть label-style мусора и немного оздоровил `2745`.

Что не подтвердилось:

- одного sanitizer fix недостаточно;
- stronger extraction wording сама по себе не лечит dense lecture / presentation cases;
- `ОШИБКА:`-style hints не дали enough forcing power для `2687` и `2673`.

## 2. Что реально улучшилось

### 2.1. `2745` стал немного лучше

Самый явный локальный gain:

- `missing 6 -> 5` vs `v2.8`;
- текст остаётся generic, но чуть чище;
- явного label-style мусора нет.

Это слабый, но реальный сигнал в пользу:

- bypass synthetic label rewrite;
- content-aware anti-splitting как направления, а не как завершённого решения.

### 2.2. `2660` очистился stylistically, но не по coverage

`missing` остался `4`, то есть quality gain не случился.

Но важная деталь:

- `Тема: ...` больше не всплывает;
- prose стал менее артефактным;
- проблема осталась в coverage и section inflation, а не в самом forbidden junk.

### 2.3. `2734` не развалился

Это тоже важно.

`v2.9` хуже `v2.8` (`missing 2 -> 3`), но не вернулся к более тяжёлым ранним failure modes.

То есть:

- убрать synthetic `Тема:` rewrite было безопасно;
- generation-side hygiene не сломалась;
- но gain `v2.8` удержать полностью не удалось.

## 3. Что осталось сломанным

### 3.1. `2687` снова показывает core bottleneck

Итог:

- `facts=11`;
- `missing=4`;
- `forbidden=['посвящ*']`.

Это плохой сигнал по трём слоям сразу:

- extraction всё ещё раздувает facts;
- `посвящ*` всё ещё доживает до финального текста;
- generation не превращает такой fact layer в clean publishable copy.

Практический вывод:

- проблема не только в sanitize branch;
- главный bottleneck всё ещё в shape самих facts.

### 3.2. `2673` почти не сдвинулся

Итог:

- `facts=12`;
- `missing=6`;
- `forbidden=['посвящ*']`.

Intent-style pollution выживает даже после `v2.9`:

- `На презентации расскажут ...`
- `как устроена платформа`
- `какую проблему решает проект`

То есть `presentation`-кейсы по-прежнему тянут Gemma в service/metatext framing.

### 3.3. `2734` частично регресснул

По отношению к `v2.8`:

- `missing 2 -> 3`;
- в facts layer снова живёт duplicated core angle;
- one of the prompt-facing facts опять содержит `посвящ...`.

Да, финальный текст clean.
Но сам факт того, что pollution живёт в fact layer, остаётся risk marker.

## 4. Практический разбор гипотез `v2.9`

### 4.1. H1 `build from v2.8`

Подтверждено.

Это был правильный base.

### 4.2. H2 `remove only synthetic Тема rewrite`

Подтверждено частично.

Да, это нужно сохранить.
Но как standalone fix оно слишком узкое.

### 4.3. H3 `content-aware anti-splitting`

Подтверждено как направление, но не как достаточная реализация.

`2687` и `2673` показывают, что wording ещё слишком слабая.

### 4.4. H4 `harder label / intent shape constraints`

Подтверждено частично.

Label-style facts реально стало меньше.
Intent-style facts всё ещё живы.

То есть:

- label part сработала лучше;
- intent part всё ещё недожата.

### 4.5. H5 `ОШИБКА`-style hints

Полезно, но недостаточно.

Gemma увидела stronger issue framing, но это не стало blocking behavior.

### 4.6. H6 `keep v2.8 generation-side hygiene`

Подтверждено.

Это не дало quality jump, но и не стало новой причиной регрессии.

### 4.7. H7 `no extra pre-generation fact gate`

Предварительное решение Gemini не добавлять отдельный gate выглядит пока правильным.

`v2.9` не доказывает, что нужен новый stage.
Он скорее доказывает, что extraction contract всё ещё слаб.

## 5. Текущий рабочий диагноз

После `v2.9` main diagnosis такой:

1. Synthetic label rewrite надо оставить выключенной.
2. Main failure moved deeper:
   - не `Тема: ...`,
   - а weak fact-unit shaping.
3. Dense cases всё ещё получают:
   - redundant fact units;
   - metatext / intent wrappers;
   - insufficiently transformed `посвящ*`.
4. Generation prompt уже mostly hygiene-preserving, но upstream facts всё ещё слишком грязные.

## 6. Что хочу проверить с Gemini дальше

Нужна уже не общая консультация, а post-run critical reading по реальному evidence set.

Самые важные вопросы:

1. Почему label contamination ушла, а intent-style metatext остался?
2. Почему `2687` и `2673` почти не сдвинулись?
3. Почему `2734` не сломался, но всё же потерял часть gain?
4. Какие exact Gemma prompt changes нужны теперь:
   - в extraction;
   - в issue hints;
   - возможно в revise;
   - без возврата к `v2.7`.

## 7. Bottom line

`v2.9` полезен как corrective evidence round.
Но candidate-веткой он не стал.

Практический итог:

- **save**: bypass synthetic `Тема:` rewrite;
- **save**: generation-side hygiene from `v2.8`;
- **rework**: intent-style suppression;
- **rework**: anti-splitting forcing power;
- **rework**: `посвящ*` handling for dense lecture / presentation cases.
