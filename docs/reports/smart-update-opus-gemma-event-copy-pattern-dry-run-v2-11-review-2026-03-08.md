# Smart Update Opus Gemma Event Copy Pattern Dry-Run V2.11 Review

Дата: 2026-03-08

Связанные материалы:

- `docs/reports/smart-update-opus-gemma-event-copy-pattern-dry-run-v2-11-5-events-2026-03-08.md`
- `docs/reports/smart-update-opus-gemini-event-copy-v2-11-complex-consultation-review.md`
- `artifacts/codex/experimental_pattern_dryrun_v2_11_2026_03_08.py`
- `artifacts/codex/gemma_event_copy_pattern_dry_run_v2_11_5events_2026-03-08.json`

## 1. Краткий verdict

`v2.11` не стал runtime candidate.

Это полезный diagnostic round, но не quality win.

Лучший результат:

- `2734`: `missing 3 -> 1` vs `v2.10`, `forbidden: посвящ* -> none`

Частичный плюс:

- `2660`: `missing 6 -> 5` vs `v2.10`, ушла цитатная деградация как главный symptom

Но сумма regressions слишком велика:

- `2745`: `missing 5 -> 6`
- `2687`: `missing 2 -> 9`
- `2673`: `missing 9 -> 12`, `forbidden=посвящ*`

## 2. Что реально сработало

### 2.1. Anti-quote control был не пустым

На `2660` больше нет самого неприятного `v2.10`-симптома:

- длинных inline-цитат фактов;
- секций, построенных вокруг `«Это ...»`.

Это подтверждает, что generation-side hygiene действительно была нужна, а не была ложной гипотезой.

### 2.2. Merge-cleanup помог на `2734`

Именно здесь новая логика дала лучший результат:

- `facts=5` сохранились компактными;
- `forbidden=посвящ*` исчез;
- coverage улучшился до `missing=1`.

Значит консультационный диагноз про `merge-back contamination` был реальным.

## 3. Что сломалось

### 3.1. `2687` разлетелся из-за over-granular extraction

Вместо плотного grouped fact про художниц `v2.11` extraction дал:

- `творчество Елены Поленовой`
- `творчество Марии Якунчиковой-Вебер`
- `творчество Зинаиды Серебряковой`
- и т.д.

Дальше preserve-floor добавил назад:

- `Лекция посвящена ...`
- `Лекция посвящена жизни и творчеству ...`
- другие более общие baseline facts

Итог:

- `facts=16`
- `missing=9`

Практический вывод:

- scoped `list consolidation` было сформулировано слишком слабо;
- Gemma ушла не в правильную группировку, а в person-splitting.

### 3.2. `2673` остался главным blocker

Хотя extraction впервые породил clean noun-phrase items:

- `задачи платформы`
- `устройство платформы`
- `возможности платформы`
- `причины появления проекта`
- `проблема, которую решает проект`

floor/merge вернул обратно:

- `На презентации расскажут ...`
- `Презентация посвящена ...`
- другие service-style facts

Итог:

- `facts=18`
- `missing=12`
- `forbidden=посвящ*`

То есть новый semantic dedup оказался недостаточно сильным именно там, где clean fact и dirty fact лексически различаются сильнее всего.

### 3.3. `2745` снова подтвердил fragility sparse branch

Branch остался `compact_fact_led`, но coverage всё равно просел:

- `missing 5 -> 6`

Это означает, что узкий `v2.11` patch pack не дал sparse-case quality gain и частично ухудшил уже рабочую compact-подачу.

## 4. Что показал `v2.11` по гипотезам

### 4.1. `merge-back contamination` — подтверждено

Это уже не гипотеза.

`2734` показал, что чистка merge/floor действительно может убрать `посвящ*` и улучшить result без нового LLM-stage.

### 4.2. `semantic dedup` в текущем виде — недостаточно точен

Он помогает, когда clean и dirty facts близки по токенам.

Он не помогает, когда:

- clean extraction слишком granular (`2687`);
- dirty baseline fact выражен через другой syntactic frame (`2673`).

### 4.3. `anti-quote` — оставить

Даже несмотря на общий regression round, generation-side anti-quote слой надо сохранять.

На `2660` это реальный practical gain.

## 5. Bottom line

`v2.11` не переносить в runtime.

Что сохранить:

- anti-quote rule;
- сам диагноз `merge-back contamination`;
- идея semantic preference `clean fact > dirty metatext fact`.

Что откатить или переписать:

- текущий `semantic dedup` threshold/shape;
- текущее scoped list consolidation wording;
- текущий merge/floor interaction для lecture/presentation cases.

Следующий раунд должен идти уже не в сторону ещё одного общего “улучшающего” patch pack, а в сторону более точного ответа на два конкретных сбоя:

1. как не допустить person-splitting на lecture/person-rich cases;
2. как матчить noun-phrase extraction с clause-style dirty baseline facts на presentation cases.
