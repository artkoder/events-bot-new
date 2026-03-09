# Smart Update Gemini Event Copy V2.6 Hypotheses Consultation Response Review

Дата: 2026-03-07

Связанные материалы:

- `docs/reports/smart-update-gemini-event-copy-v2-6-hypotheses-consultation-response.md`
- `docs/reports/smart-update-gemini-event-copy-v2-6-hypotheses-consultation-brief.md`
- `docs/reports/smart-update-opus-gemma-event-copy-pattern-dry-run-v2-5-5-events-2026-03-07.md`
- `docs/reports/smart-update-opus-gemma-event-copy-pattern-dry-run-v2-5-review-2026-03-07.md`
- `artifacts/codex/experimental_pattern_dryrun_v2_6_2026_03_07.py`

## 1. Краткий verdict

Это сильный pre-run response.

Главное:

- Gemini правильно подтвердил, что bottleneck всё ещё в раннем слое, а не в полном architectural redesign;
- не потащил нас в ещё один длинный theory loop;
- дал набор prompt-level moves, которые реально стоило проверить локально в `v2.6`.

Мой итог на этом этапе был и остаётся таким:

- **новый до-кодовый раунд с Gemini был не нужен**;
- идти в локальный `v2.6` было правильно;
- но несколько его советов нужно было брать только как направление, а не как literal wording.

## 2. Что в ответе Gemini было действительно сильным

### 2.1. Строже, но спокойнее формулировать contract

Это принимаю полностью.

Gemini правильно зафиксировал, что Gemma плохо исполняет vague hints вроде:

- `по возможности`
- `предпочтительно`
- `старайся`

Нужны были:

- явные запреты;
- явные replacement patterns;
- чёткое разделение blocking issues и stylistic advice.

### 2.2. Performance-only grouped pattern

Это тоже было сильной рекомендацией.

Ключевое достоинство:

- Gemini не навязывал `В программе: ...` универсально;
- он ограничил это performance / repertory cases.

Именно такая граница и была нужна.

### 2.3. Compact branch должен уметь смысловое слияние

Это был один из самых полезных тезисов.

`2660` уже в `v2.5` показывал, что compact ветка может выиграть по coverage и проиграть по prose, если модель вынуждена механически раскладывать каждый факт отдельно.

Разрешение на смысловое слияние близких facts было правильным move.

### 2.4. Anti-bureaucracy нужен уже на extraction/generation contract

Gemini и здесь попал точно.

`2673` страдает не только от headings, а от самого типа facts:

- `На презентации расскажут...`
- `мероприятие анонсирует...`
- `будет представлен обзор...`

Без явного anti-bureaucracy rules это почти неизбежно переезжает в финальный текст.

## 3. Что я брал только с поправками

### 3.1. `<= 6` как временный gate

Это был полезный pragmatic move, но не more than that.

Правильная трактовка:

- для ближайшего раунда threshold можно оставить;
- но обязательно дополнять content-aware blocker.

Без этого count-driven routing остаётся brittle.

### 3.2. Анти-`посвящ*`

Gemini правильно усиливал severity, но literal hard-ban брать в лоб было нельзя.

Правильная версия:

- issue должен быть blocking;
- должны быть допустимые replacement patterns;
- нельзя заменять проблему новым metatext вроде `событие рассказывает о...`.

### 3.3. Positive examples

Gemini прав, что мы недооценивали короткие положительные примеры.

Но и тут нужна мера:

- примеры должны направлять трансформацию;
- они не должны превращать prompt в few-shot essay.

## 4. Что в ответе Gemini не стоило копировать буквально

### 4.1. ALL CAPS / shouting tone

Это не надо было переносить как prompt style.

Полезна не истеричность, а ясная обязательность действия.

### 4.2. Универсализация performance patterns

`В программе: ...` нельзя нормализовать как общий шаблон для lecture / exhibition / person-led cases.

### 4.3. Сведение проблемы к одному слову

Даже когда Gemini говорил о `посвящ*` правильно, риск overfitting оставался:

- реальная проблема шире;
- это отстранённый академический тон и метатекст;
- бороться нужно не только с одним корнем, а с whole pattern.

## 5. Bottom line

Этот Gemini response был достаточно сильным и actionable, чтобы:

- не делать ещё один theoretical round;
- собрать `v2.6`;
- и проверять уже реальные outputs.

То есть решение после него было правильным:

- **consult again before coding**: нет;
- **go to local `v2.6` dry-run**: да.

Дальше judge должен идти уже не по красоте самого ответа Gemini, а по фактическим результатам `v2.6`.
