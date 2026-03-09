# Smart Update Opus Gemma Event Copy Pattern Dry-Run V2.2 Review

Дата: 2026-03-07

Связанные материалы:

- `docs/reports/smart-update-opus-gemma-event-copy-pattern-dry-run-v2-2-5-events-2026-03-07.md`
- `artifacts/codex/gemma_event_copy_pattern_dry_run_v2_2_5events_2026-03-07.json`
- `artifacts/codex/experimental_pattern_dryrun_v2_2_2026_03_07.py`
- `docs/reports/smart-update-opus-gemma-event-copy-pattern-dry-run-v2-1-5-events-2026-03-07.md`
- `docs/reports/smart-update-opus-gemma-event-copy-pattern-dry-run-v2-1-review-2026-03-07.md`
- `docs/reports/smart-update-opus-gemma-event-copy-v2-1-dryrun-quality-consultation-response-review.md`

## 1. Краткий verdict

`v2.2` не стал quality win и не готов к переносу в runtime.

При этом dry-run нельзя считать пустым:

- subtractive откат repair-pass действительно убрал часть мусора и часть forbidden leakage;
- `2745` и особенно `2687` стали лучше, чем в `v2.1`;
- runtime снизился с `538.8s` до `446.9s`.

Но по сумме качества ветка всё ещё не выигрывает:

- `2660` стал чище, но тоньше и беднее по coverage;
- `2734` остался слабым и по тексту, и по traceability программы;
- `2687` улучшился, но всё ещё течёт через `посвящ*` и unsupported generalization;
- `2673` остаётся нестабильным и по representation, и по coverage.

Мой вывод:

- **ещё один Opus-раунд теперь нужен**;
- но это должен быть уже очень узкий `quality-first corrective round` по реальным `v2.2` outputs.

## 2. Что в `v2.2` реально улучшилось

### 2.1. Hygiene recovery против `v2.1`

Это главный полезный результат.

`v2.2` убрал часть проблем, внесённых `v2.1`:

- исчезли `tickets/date_ru_words` leakage на `2745`;
- на `2734` ушёл прямой `посвящ*` leak из финального текста;
- на `2660` пропал сервисный хвост про длительность;
- общая ветка стала менее noisy и менее meta-heavy.

То есть диагноз Opus про вред repair-pass в целом подтвердился.

### 2.2. `2745` стал редакторски чище

Для sparse case это уже неплохой результат:

- текст компактный;
- без сервисного хвоста;
- без лишних секций;
- без явного synthetic drift.

Да, coverage всё ещё слабый, но prose заметно здоровее, чем в `v2.1`.

### 2.3. `2687` — лучший локальный recovery-кейс

Тут есть реальный прогресс:

- `missing: 4 -> 2`;
- текст стал спокойнее и собраннее;
- ушла часть лишней перегрузки `v2.1`.

Но кейс всё ещё не production-ready из-за `посвящ*` и unsupported формулировок.

## 3. Где `v2.2` всё ещё провалился

### 3.1. `2660`: clean prose, но branch слишком сжимает факты

`2660` — показательный кейс.

Текст в `v2.2` читается чище, чем `v2.1`, но компактная ветка:

- теряет имя автора в теле текста;
- обобщает технику и фактуру;
- не удерживает coverage в формулировках, которые проходит deterministic check.

Это не catastrophic output, но это всё ещё regress по полноте и explainability.

### 3.2. `2734`: subtractive fix не решил core weakness

Этот кейс всё ещё проблемный.

Что осталось плохо:

- clumsy lead: `в центре внимания великой любви` звучит неестественно;
- track coverage всё ещё слабый;
- образ Музы опять ушёл слишком глубоко;
- текст стал длиннее, но не убедительнее.

То есть `v2.2` убрал symptom, но не починил сам narrative contract.

### 3.3. `2687`: главный remaining leak — уже не repair-pass

На `2687` видно, что после subtractive rollback root cause сместился.

Проблема теперь не в отдельном repair-call, а в том, что:

- `посвящ*` всё ещё проходит в generation;
- prompt допускает unsupported историческое усиление;
- branch всё ещё любит делать broad cultural framing без достаточного evidence contract.

### 3.4. `2673`: representation и coverage всё ещё сломаны

Это самый сложный кейс.

Формально `missing=10`, но тут важно не впасть в ложную строгость:

- часть missing вызвана duplicated / merged facts в самом `facts_text_clean`;
- deterministic metric штрафует повторяющиеся service-like facts отдельно;
- но и реальный coverage loss тоже есть: текст не удерживает достаточно operational details.

То есть это не просто “метрика шумит” и не просто “текст плохой”.
Это гибридная проблема:

- upstream fact representation остаётся грязной;
- generation сжимает детали;
- итоговый текст выглядит структурно аккуратно, но не закрывает fact contract.

## 4. Быстрый разбор по кейсам

### 4.1. `2660`

- Текст чище `v2.1`, но слишком thin.
- Похоже, compact branch слишком агрессивно жертвует точными фактами ради гладкости.
- Хороший кейс для вопроса Opus: где должна проходить граница между sparse elegance и fact undercoverage.

### 4.2. `2745`

- Это, вероятно, лучший текст `v2.2` среди sparse-кейсов.
- Он естественный и без мусора.
- Но до сих пор не решён главный вопрос: достаточно ли он informative по меркам нашего fact-first contract.

### 4.3. `2734`

- Текст профессионально не держится.
- Лид синтаксически слабый.
- Музыкальная программа всё ещё сведена слишком грубо.
- Этот кейс требует не косметики, а более точного narrative routing и coverage contract.

### 4.4. `2687`

- Самый promising case в `v2.2`.
- Но именно здесь видно, что даже после успеха по counts ветка ещё не готова:
  - `посвящена` живёт;
  - появляются broad unsupported phrases вроде `определило облик русского искусства`.

### 4.5. `2673`

- Текст заметно организованнее, чем `v2.1`.
- Но representation facts остаётся некачественным: много почти-дублей, много service-like facts.
- Пока этот слой не станет чище, честно оценивать branch quality по `2673` будет трудно.

## 5. Что теперь особенно важно спросить у Opus

Новый раунд должен быть не про “придумай ещё один redesign”, а про следующее:

1. Где здесь реальный text-quality regression, а где metric artifact.
2. Является ли `facts_text_clean` representation главным bottleneck теперь.
3. Надо ли продолжать pattern branch вообще, или пора откатываться ближе к tuned baseline.
4. Какие ровно 3-6 changes worth testing в `v2.3`.
5. Что нужно explicitly не трогать, чтобы не потерять уже работающие части current fact-first flow.

## 6. Нужен ли ещё один этап консультаций

**Да.**

Теперь это оправдано.

Причина:

- `v2.2` уже дал новый evidence set;
- локально следующий шаг не настолько очевиден, чтобы безопасно идти без внешней калибровки;
- при этом раунд должен быть узким, grounded и конкурентным именно по качеству текста.

## 7. Bottom line

`v2.2` полезен как corrective experiment, но не как ready branch.

Самые важные выводы:

- repair-pass действительно стоило убрать;
- этого оказалось недостаточно;
- нынешний bottleneck сдвинулся в сторону `facts representation + branch behavior + coverage contract`.

Значит следующий шаг:

- не интеграция в бот;
- а ещё один критический Opus-раунд по реальным `v2.2` outputs и по вопросу, есть ли вообще сильный `v2.3`, который оправдан quality-wise.
