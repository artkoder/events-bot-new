# Smart Update Opus Gemma Event Copy V2.2 Dry-Run Quality Consultation Brief

Дата: 2026-03-07

Связанные материалы:

- `docs/reports/smart-update-opus-gemma-event-copy-pattern-dry-run-v2-2-5-events-2026-03-07.md`
- `docs/reports/smart-update-opus-gemma-event-copy-pattern-dry-run-v2-2-review-2026-03-07.md`
- `artifacts/codex/gemma_event_copy_pattern_dry_run_v2_2_5events_2026-03-07.json`
- `artifacts/codex/experimental_pattern_dryrun_v2_2_2026_03_07.py`
- `docs/reports/smart-update-opus-gemma-event-copy-pattern-dry-run-v2-1-5-events-2026-03-07.md`
- `docs/reports/smart-update-opus-gemma-event-copy-pattern-dry-run-v2-1-review-2026-03-07.md`
- `docs/reports/smart-update-opus-gemma-event-copy-v2-1-dryrun-quality-consultation-response.md`
- `docs/reports/smart-update-opus-gemma-event-copy-v2-1-dryrun-quality-consultation-response-review.md`

## 1. Контекст

После `v2.1` мы не пошли в новый redesign.
Вместо этого локально реализовали **subtractive `v2.2`**:

- убрали extraction repair pass;
- перенесли issue hints в initial extraction;
- сохранили полезные quality blocks;
- оставили deterministic cleanup / floor;
- повторили live dry-run на тех же 5 событиях.

То есть `v2.2` — это уже не теоретическая ветка, а grounded попытка восстановить качество после неудачного `v2.1`.

## 2. Что получилось на реальном dry-run

Результат снова mixed.

Плюсы:

- `2745` стал заметно чище;
- `2687` дал лучший локальный recovery (`missing: 4 -> 2`);
- runtime снизился с `538.8s` до `446.9s`;
- часть forbidden leakage ушла.

Но overall quality win всё равно не случился:

- `2660`: текст чистый, но слишком thin;
- `2734`: clumsy lead и слабое удержание музыкальной программы;
- `2687`: всё ещё `посвящ*` и unsupported broadening;
- `2673`: representation грязный, coverage снова провален по фактам и/или по metric contract.

Иными словами:

- `v2.2` подтвердил, что repair-pass стоило убрать;
- но он **не доказал**, что сама pattern-driven ветка уже способна выигрывать по сумме качества.

## 3. Что особенно важно проанализировать

### 3.1. Где реальный regression, а где metric artifact

Это критично.

Например:

- `2660` редакторски читается лучше, чем сухой count `missing=4`;
- `2673` может быть частично завален из-за duplicated facts в `facts_text_clean`, а не только из-за слабого prose.

Нужен честный разбор:

- что здесь настоящий text-quality failure;
- что representation failure;
- что weakness deterministic evaluation.

### 3.2. `facts_text_clean` как bottleneck

После `v2.2` это выглядит всё более вероятным.

Особенно на:

- `2734` — weak program representation;
- `2673` — дубли и почти-дубли;
- частично `2660` — факты есть, но generation слишком легко их обобщает.

Нужно понять:

- проблема уже не в patterns как таковых;
- или без stronger fact representation branch quality не вырастет.

### 3.3. Sparse elegance vs fact coverage

`2660` и `2745` показывают важную развилку.

`compact_fact_led` даёт cleaner prose, но:

- начинает терять явные опорные факты;
- иногда слишком быстро заменяет точные детали на общую формулировку.

Нужно понять:

- это правильная цена за better text;
- или branch contract надо менять.

### 3.4. Remaining forbidden / unsupported behavior

На `2687` `посвящ*` всё ещё живёт.

На `2734` и `2687` generation всё ещё допускает broadening beyond evidence.

Нужно оценить:

- это prompt wording issue;
- routing issue;
- asset grounding issue;
- или revise layer still too weak.

### 3.5. `2673` как stress test

Этот кейс сейчас особенно важен.

Там одновременно видны:

- noisy fact representation;
- service-like duplication;
- branch compression;
- mismatch between structured output and coverage contract.

Если `2673` не разобрать правильно, можно принять неверное архитектурное решение по всей ветке.

## 4. Чего мы хотим от нового ответа Opus

Нужна максимально критичная quality-first консультация, уже по **реальному `v2.2` evidence set**.

### 1. `Event-by-event corrected verdict`

Для всех 5 кейсов:

- baseline vs `v1` vs `v2` vs `v2.1` vs `v2.2`;
- где `v2.2` реально лучше;
- где хуже;
- почему именно.

### 2. `Text quality review`

Просим смотреть редакторски:

- естественность;
- связность;
- логичность повествования;
- чистоту и профессиональность;
- отсутствие synthetic drift;
- уместность headings / sections / hooks;
- реальную human readability.

### 3. `Failure attribution map`

Просим разложить regressions по стадиям:

- extraction / representation;
- cleanup / floor;
- routing;
- generation;
- revise / repair;
- evaluation mismatch.

### 4. `Keep / Modify / Rollback / Remove`

Нужен честный разбор:

- что из `v2.2` и более ранних v-веток стоит сохранить;
- что модифицировать;
- что откатить;
- что удалить как тупиковое.

### 5. `V2.3 patch plan or stop verdict`

Нужен прагматичный следующий шаг:

- если существует сильный и узкий `v2.3`, пусть Opus его предложит;
- если нет, пусть прямо скажет, что pattern branch в текущем виде не оправдана и надо возвращаться к narrower baseline-first tuning.

## 5. Важные рамки

- Приоритет: естественный и качественный текст описания события.
- `Полнота фактов = P0`.
- Качество текста, логика и профессиональность важнее, чем искусственная минимальность схемы.
- Moderate рост runtime / token usage допустим, если он реально повышает качество.
- Не нужно защищать прошлые рекомендации, если dry-run их опровергает.
- Можно свободно спорить и с нашей локальной реализацией, и с прошлой линией консультаций.

## 6. Bottom line

Это уже консультация не про теоретические patterns, а про вопрос:

**может ли `v2.x` вообще выйти на реально более сильный текст, или текущая ветка упёрлась в representation / routing / coverage ограничения Gemma.**

Нужен максимально критичный и practically useful ответ.
