# Smart Update Opus Gemma Event Copy V2.1 Dry-Run Quality Consultation Brief

Дата: 2026-03-07

Связанные материалы:

- `docs/reports/smart-update-opus-gemma-event-copy-pattern-dry-run-v2-1-5-events-2026-03-07.md`
- `docs/reports/smart-update-opus-gemma-event-copy-pattern-dry-run-v2-1-review-2026-03-07.md`
- `artifacts/codex/gemma_event_copy_pattern_dry_run_v2_1_5events_2026-03-07.json`
- `artifacts/codex/experimental_pattern_dryrun_v2_1_2026_03_07.py`
- `docs/reports/smart-update-opus-gemma-event-copy-pattern-dry-run-v2-5-events-2026-03-07.md`
- `artifacts/codex/gemma_event_copy_pattern_dry_run_v2_5events_2026-03-07.json`
- `docs/reports/smart-update-opus-gemma-event-copy-v2-dryrun-quality-consultation-response.md`
- `docs/reports/smart-update-opus-gemma-event-copy-v2-dryrun-quality-consultation-response-review.md`

## 1. Контекст

После критической консультации Opus по `v2 dry-run` мы локально реализовали `v2.1` experimental patch pack.

Что именно было добавлено:

- stronger extraction rules:
  - anti-`посвящ*`
  - anti-merge
  - anti-inflate
- whole-body metatext checks
- stronger anti-embellishment / lead rules
- exact fact dedup
- extraction repair pass before generation

То есть `v2.1` уже не теория, а **реальная локальная реализация** рекомендаций по итогам прошлой консультации.

## 2. Что получилось на реальном dry-run

Результат получился отрицательным.

Коротко:

- `v2.1` не стал quality win;
- на `2660`, `2745`, `2734`, `2687` он хуже `v2`;
- на `2673` он лучше `v2`, но всё равно хуже `v1`;
- runtime вырос до `538.8s`.

То есть вопрос теперь не “как ещё усилить ту же линию”, а:

- что именно в `v2.1` сломалось;
- какие рекомендации Opus в локальной реализации не сработали;
- и какой следующий шаг вообще оправдан.

## 3. Что особенно важно проанализировать

### 3.1. Extraction repair failure

Это главный вопрос.

Repair-pass должен был:

- убрать `посвящ*`;
- сжать inflated facts;
- вернуть program items;
- убрать service contamination.

Но на реальных кейсах он:

- оставил `посвящ*` в `2734` и `2687`;
- не вернул треклист в `2734`;
- протащил сервисный факт в `2745`;
- оставил дубли в `2673`;
- добавил service/meta fact в `2687`.

Нужно понять:

- проблема в prompt formulation;
- в schema / contract;
- в runtime placement;
- или в самой идее отдельного repair-pass для Gemma.

### 3.2. Branch inflation on sparse cases

`2660` и `2745` ушли в standard branch.

Нужно оценить:

- это действительно richer fact set;
- или polluted fact set;
- и не надо ли routing опирать на более жёстко очищенный subset.

### 3.3. `copy_assets` vs `facts_text_clean`

Сейчас есть расхождение:

- `copy_assets` уже может знать сильные program details;
- `facts_text_clean` их не содержит;
- generation всё равно опирается на более слабый facts set.

Нужно оценить:

- это prompt bug;
- schema bug;
- или pipeline design bug.

### 3.4. Text quality itself

Важно смотреть не только на counts, а на сами тексты.

По `v2.1` особенно бросаются в глаза:

- editorial drift на `2660`;
- service leakage на `2745`;
- unresolved `посвящ*` на `2734` и `2687`;
- generic section noise на `2673`.

## 4. Чего мы хотим от нового ответа Opus

Нужен уже не opinion, а **corrective consultation** по реальному failed iteration.

### 1. `Event-by-event corrected verdict`

Для всех 5 кейсов:

- baseline vs `v1` vs `v2` vs `v2.1`
- где именно `v2.1` проиграл
- где именно есть partial gain
- что является root cause в каждом кейсе

### 2. `Failure attribution map`

Просим разложить regressions по ownership:

- extraction prompt
- extraction repair prompt
- post-filter / floor
- routing
- generation prompt
- revise / repair
- cleanup

Не общий verdict, а именно stage attribution.

### 3. `Keep / Modify / Rollback / Remove`

По `v2.1` changes нужен честный разбор:

- что стоит сохранить;
- что надо модифицировать;
- что откатить;
- что удалить как неработающее.

Особенно по:

- extraction repair pass
- anti-`посвящ*`
- anti-merge / anti-inflate
- stronger lead rules
- whole-body metatext checks
- exact fact dedup

### 4. `V2.2 patch plan`

Нужен очень узкий следующий шаг.

Не широкий redesign, а:

- минимальный набор changes, который worth trying;
- expected quality impact;
- expected risk;
- что deliberately не трогать.

### 5. `Decision point`

Просим Opus дать прагматичный ответ:

- есть ли смысл делать локальный `v2.2`;
- или текущая линия исчерпана и надо возвращаться к более узкому baseline-first tuning.

## 5. Важные рамки

- Приоритет всё ещё: естественный, профессиональный, точный текст.
- `Полнота фактов = P0`.
- Moderate runtime growth допустим только если есть явный quality win.
- Не нужно защищать прошлые рекомендации, если `v2.1` их опровергает.
- Можно свободно предлагать rollback, simplification или отказ от части `v2.1`.

## 6. Bottom line

Это уже консультация не “что попробовать”, а **почему локально реализованный `v2.1` не сработал и что делать дальше**.

Нужен максимально критичный, технически точный и редакторски жёсткий ответ.
