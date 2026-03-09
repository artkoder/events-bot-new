# Smart Update Gemini Event Copy V2.7 Hypotheses Consultation Response Review

Дата: 2026-03-07

Связанные материалы:

- `docs/reports/smart-update-gemini-event-copy-v2-7-hypotheses-consultation-response.md`
- `docs/reports/smart-update-gemini-event-copy-v2-7-hypotheses-consultation-brief.md`
- `docs/reports/smart-update-opus-gemma-event-copy-pattern-dry-run-v2-6-5-events-2026-03-07.md`
- `docs/reports/smart-update-opus-gemma-event-copy-pattern-dry-run-v2-6-review-2026-03-07.md`
- `artifacts/codex/experimental_pattern_dryrun_v2_6_2026_03_07.py`

## 1. Краткий verdict

Это очень сильный pre-run response.

Главное:

- Gemini полностью признал, что literal unsafe rewrites из прошлого ответа были dangerous overreach;
- подтвердил наш `agenda-safe` direction;
- не утащил нас в новый theory loop;
- дал usable wording для `v2.7`, не ломая factual grounding как принцип.

Мой итог:

- **ещё один pre-run consultation round не нужен**;
- можно идти в локальный `v2.7`;
- но даже теперь часть формулировок нужно брать не literally, а с небольшой санитарной правкой.

## 2. Что в ответе Gemini принимаю

### 2.1. `agenda-safe positive transformations` как ядро `v2.7`

Это полностью принимаю.

Самый важный вывод:

- бороться нужно не просто против `посвящ*` и `расскажут о...`;
- нужно давать Gemma безопасные формы, которые:
  - не канцелярские;
  - не выдумывают детали;
  - всё ещё честно отражают формат события.

### 2.2. Нельзя банить форматный субъект целиком

Это важная и правильная калибровка.

То есть нельзя превращать правило в:

- `не начинай с "Лекция ..."`
- `не используй "Спектакль ..." как подлежащее`

Правильно запрещать:

- weak bureaucratic predicates;

а не сами слова `лекция`, `концерт`, `спектакль`, `презентация`.

### 2.3. `Routing not now`

Согласен.

На этом этапе adding routing variables только усложнит интерпретацию `v2.7`.

## 3. Что беру только с поправками

### 3.1. Даже agenda-safe examples должны быть предельно нейтральными

Gemini сам уже лучше откалибровался, но риск всё ещё есть.

Например:

- `На встрече разберут причины появления проекта...`

может быть слишком конкретно, если в raw fact есть только `зачем появился проект`.

Это не критичная ошибка, но я всё же буду предпочитать максимально нейтральные safe forms:

- `В центре встречи — причины появления проекта, его задачи и возможности платформы.`
- `Разговор о ...`
- `Лекция о ...`

### 3.2. `Встреча посвящена... (заменить на ...)` в самом prompt block лучше не оставлять

Идея понятна, но сам пример может визуально засорять prompt.

Лучше дать сразу чистые allowed patterns, без parenthetical corrective chatter.

## 4. Что не беру буквально

- Не беру overly specific agenda nouns, если они могут подтолкнуть к over-interpretation.
- Не беру лишнюю пояснительную болтовню внутри rule blocks.

## 5. Что реально пойдёт в `v2.7`

Из этого ответа беру следующее:

1. agenda-safe transformation block в extraction prompt;
2. переписанные `_pre_extract_issue_hints`;
3. standard generation rule с разделением:
   - ban on bureaucratic metatext;
   - allow natural event framing;
4. более жёсткое revise issue для `посвящ*`.

## 6. Bottom line

Этот Gemini response достаточно сильный, чтобы закрыть pre-run consultation stage по `v2.7`.

Практический вывод:

- **ещё один pre-run раунд не нужен**;
- следующий шаг уже локальный: собрать `v2.7` и прогнать live dry-run на тех же 5 событиях.
