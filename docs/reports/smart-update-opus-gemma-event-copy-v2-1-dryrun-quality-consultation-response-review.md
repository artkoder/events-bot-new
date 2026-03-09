# Smart Update Opus Gemma Event Copy V2.1 Dry-Run Quality Consultation Response Review

Дата: 2026-03-07

Связанные материалы:

- `docs/reports/smart-update-opus-gemma-event-copy-v2-1-dryrun-quality-consultation-response.md`
- `docs/reports/smart-update-opus-gemma-event-copy-pattern-dry-run-v2-1-5-events-2026-03-07.md`
- `docs/reports/smart-update-opus-gemma-event-copy-pattern-dry-run-v2-1-review-2026-03-07.md`
- `artifacts/codex/gemma_event_copy_pattern_dry_run_v2_1_5events_2026-03-07.json`
- `artifacts/codex/experimental_pattern_dryrun_v2_1_2026_03_07.py`

## 1. Краткий verdict

Это один из самых сильных ответов Opus за весь `event-copy` цикл.

Главное:

- он честно признал, что его прошлые рекомендации локально не сработали;
- он правильно локализовал главный failure source: `extraction repair pass`;
- он перестал расширять complexity и вместо этого предлагает subtractive `v2.2`;
- и он даёт понятный decision tree, а не ещё один abstract redesign.

Мой итог:

- **ещё один консультационный раунд сейчас не нужен**;
- ответ достаточно силён, чтобы переходить к локальному `v2.2` dry-run;
- но часть его deterministic fixes нужно брать **с уточнениями**.

## 2. Что в ответе Opus особенно сильное

### 2.1. Correct root cause: repair pass сломал `v2.1`

Это главное.

Opus правильно увидел, что:

- проблема не только в wording prompt'ов;
- основной регресс дал именно новый `repair`-вызов;
- Gemma 27b плохо справляется с meta-task вида
  `прочитай текущий JSON + issue hints + raw_facts -> пересобери extraction`.

Это полностью совпадает с тем, что видно по dry-run кейсам.

### 2.2. Subtractive `v2.2` — правильное направление

Это сильная self-correction.

Вместо новой сложности Opus предлагает:

- убрать repair pass;
- сохранить useful prompt improvements;
- добавить минимальные deterministic fixes;
- проверить результат одним dry-run.

Это инженерно гораздо здоровее, чем ещё один широкий redesign.

### 2.3. Baseline-first fallback сформулирован правильно

Это тоже сильный момент.

Он не пытается любой ценой спасать `v2.x`, а прямо говорит:

- если `v2.2` не beat'ит `v2` / не приближается к `v1`,
- нужно брать deterministic wins и возвращаться к narrow baseline-first tuning.

Такой stop-condition сейчас реально нужен.

## 3. Что я бы принял в `v2.2`

### 3.1. Accept now

1. Удалить extraction repair pass целиком.
2. Перенести issue hints в initial extraction prompt, а не в отдельный repair call.
3. Re-filter floor output после preservation.
4. Сохранить current quality block / lead guidance / whole-body metatext detection.
5. Считать `v2.2` только локальным experimental dry-run, без production integration.
6. Использовать stop-conditions Opus как decision gate после `v2.2`.

### 3.2. Accept with modification

1. `Near-dedup` с очень высоким порогом

Direction полезный.
Но:

- только в experimental branch;
- только с высоким threshold;
- и с явным исключением facts, которые содержат quoted program items или list-like content.

Иначе можно снова открыть coverage loss на program-heavy events.

2. Anti-merge refinement

Формулировка Opus `if quotes -> separate facts` сильная и практичная.
Её стоит брать почти буквально.

3. Baseline-first fallback

Полностью согласен как с резервным планом.
Но сначала всё же стоит сделать один локальный `v2.2`, потому что теперь речь идёт о subtractive fix, а не о новом architectural branch.

## 4. Что в ответе Opus всё ещё рискованно

### 4.1. Его новый `посвящ*` strip всё ещё недостаточно безопасен

Это моя главная оговорка.

Предложенный паттерн:

```python
re.compile(r'(?i)^(.+?)\s+посвящён[аоы]?\s+(.+)$'), r'\1 о \2'
```

всё ещё может давать грамматически кривые фразы:

- `Лекция посвящена творчеству ...` -> `Лекция о творчеству ...`

То есть direction правильный, но реализация всё ещё слишком грубая.

Я бы принял это только так:

- либо очень narrow whitelist of safe patterns;
- либо prompt-facing sanitize, а не raw rewrite stored facts;
- либо selective rewrite only when output is grammatically safe.

### 4.2. Direct production placement для raw fact rewrite — пока рано

В ответе Opus звучит идея тащить `_strip_posvyash_from_fact` прямо в production post-filter.

Я бы не делал этого пока в raw persisted fact path.

Сначала:

- локальный `v2.2` dry-run;
- проверка на real cases;
- потом решение, можно ли такое deterministic rewrite поднимать в production.

## 5. Где я только частично согласен с micro-verdicts

Есть несколько локальных моментов, где я бы не принимал его ranking буквально.

Например по `2660`:

- tone-wise `v2` действительно самый “честный”;
- но baseline всё ещё safer по структуре и coverage.

Это не ломает общий вывод ответа, но важно не превращать отдельные editorial preferences в главный критерий.

## 6. Нужен ли ещё один этап консультаций

**Нет.**

Сейчас у нас уже есть:

- full `baseline / v1 / v2 / v2.1` competitive set;
- наш `v2.1` failure review;
- self-corrected response Opus;
- очень узкий `v2.2` proposal.

Новый внешний round до локального `v2.2` даст меньше пользы, чем просто проверить этот subtractive plan на тех же 5 кейсах.

## 7. Следующий разумный шаг

Правильный следующий шаг сейчас:

1. Локально собрать `v2.2 = v2.1 - repair pass + deterministic cleanup`.
2. Не трогать production runtime.
3. Повторить dry-run на тех же 5 событиях.
4. После `v2.2` уже принять одно из двух решений:
   - либо переносить только deterministic wins в production;
   - либо останавливать `v2.x` ветку и идти в baseline-first tuning.

## 8. Bottom line

Этот ответ Opus **принимается в основном**.

Самая ценная его часть:

- не новые идеи,
- а правильный отказ от лишней сложности и переход к `subtractive v2.2`.

Моя позиция:

- консультационную фазу на этом этапе можно остановить;
- дальше — локальный `v2.2` dry-run;
- и только по его итогам решать судьбу всей `v2.x` ветки.
