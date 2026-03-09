# Smart Update Opus Gemma Event Copy Pattern Dry-Run V2.5 Review

Дата: 2026-03-07

Связанные материалы:

- `docs/reports/smart-update-opus-gemma-event-copy-pattern-dry-run-v2-5-5-events-2026-03-07.md`
- `artifacts/codex/gemma_event_copy_pattern_dry_run_v2_5_5events_2026-03-07.json`
- `artifacts/codex/experimental_pattern_dryrun_v2_5_2026_03_07.py`
- `docs/reports/smart-update-gemini-event-copy-v2-4-regression-consultation-response-review.md`

## 1. Краткий verdict

`v2.5` лучше `v2.4`, но всё ещё не лучше `v2.3` по суммарному качеству текста.

Это полезный corrective round, а не quality win.

Что реально подтвердилось:

- rollback к `v2.3` базе был правильным;
- widened sparse routing вернул `2660` и `2745` в `compact_fact_led`;
- human-readable anti-`посвящ*` помог убрать forbidden leakage в `2734`;
- anti-question heading line не навредила и убрала разговорный heading drift.

Что не подтвердилось как достаточное решение:

- human-readable anti-`посвящ*` всё ещё слишком слаб в revise для `2687`;
- `<= 6` как routing cushion полезен для sparse cases, но слишком груб для `2734`;
- grouped program-preservation hints сами по себе не дают уверенной fact preservation;
- `2673` всё ещё скатывается в generic structural prose.

Итог:

- в runtime переносить нельзя;
- следующий шаг оправдан как ещё один grounded Gemini round;
- Gemini нужно показать не только metrics, а конкретные prose regressions/partial wins `v2.5`.

## 2. Что в `v2.5` реально улучшилось

### 2.1. `2660`: coverage улучшился, но prose стала тяжелее

Формально это лучший numeric result раунда:

- `missing=2 -> 1` против `v2.3`;
- branch снова `compact_fact_led`;
- forbidden markers нет.

Но редакторски кейс не стал лучшим:

- текст стал заметно дублировать один и тот же факт про дебютную выставку и технику барельефа;
- исчезла сжатость `v2.3`;
- compact branch написал слишком буквальный fact dump.

То есть improvement здесь partial:

- coverage лучше;
- naturalness хуже.

### 2.2. `2734`: это главный practical recovery round

Это самый важный успех `v2.5`.

После провального `v2.3` и тяжёлого `v2.4` здесь получилось:

- `missing=7 -> 4` против `v2.3`;
- forbidden `посвящ*` исчез;
- текст снова читается как человеческий компактный анонс;
- не потеряны Магомаев / Синявская / Муза / program focus.

Ограничение остаётся:

- extraction representation всё ещё слабая и допускает label-style fact `Тема: ...`;
- кейс ушёл в `compact_fact_led`, и это выглядит скорее удачным исключением, чем надёжным routing rule.

### 2.3. `v2.5` реально исправил часть `v2.4` regressions

По сумме видно:

- `2660`: `missing 3 -> 1`
- `2734`: `missing 5 -> 4`
- `2673`: `missing 7 -> 6`

То есть ветка действительно отыграла часть вреда `v2.4`.

Это важно, потому что подтверждает:

- проблема была не в общей idea quality-first;
- проблема была в слишком грубом `v2.4` patch pack.

## 3. Где `v2.5` всё ещё проиграл

### 3.1. `2745`: `v2.3` остаётся лучшим вариантом

`v2.5` вернул кейс в compact branch, но не вернул лучший quality balance:

- `v2.3 missing=3`
- `v2.5 missing=4`

Prose почти такая же, но фактической полноты меньше.

Это важный сигнал:

- routing rollback сработал;
- extraction / revise quality не дали дополнительного выигрыша.

### 3.2. `2687`: coverage сохранён, но anti-`посвящ*` всё ещё не работает как надо

Это mixed case:

- `missing=1`, то есть coverage снова сильный;
- но forbidden `посвящ*` вернулся прямо в body.

Причём leakage произошёл не потому, что проблема незаметна, а потому что:

- facts layer всё ещё допускает label-like / lecture-template phrasing;
- revise issue формально human-readable, но слишком “советующий”, а не repair-driving.

То есть здесь `v2.5` не закрыл главный hygiene bug.

### 3.3. `2673`: generic structure остаётся сильнее редакторского качества

`v2.5` лучше `v2.4`, но хуже `v2.3`.

Симптомы:

- `missing=5 -> 6` против `v2.3`;
- heading `### О платформе` generic;
- body снова звучит объясняюще и немного бюрократично;
- часть problem/value facts схлопнута в общую формулу.

То есть anti-question heading сам по себе не решил проблему template feel.

## 4. Что `v2.5` доказал про root cause

### 4.1. Ранний слой действительно критичнее позднего prose tuning

Пользовательский тезис подтвердился:

- главные swings происходят в extraction / fact shaping / routing;
- generation-only tweaks не вытягивают branch, если upstream representation плохой;
- даже хороший revise prompt не спасает слабую fact packaging.

### 4.2. `<= 6` как routing patch работает, но не универсален

Это хороший локальный recovery move для sparse cases:

- `2660`
- `2745`

Но этого недостаточно как долговременного правила:

- `2734` тоже оказался в compact, хотя structurally он richer, чем типичный sparse case.

Следовательно:

- нужен не просто length threshold;
- нужен более content-aware compact gate.

### 4.3. Human-readable anti-`посвящ*` лучше opaque marker, но пока недостаточно принудителен

Это уже лучше, чем `forbidden_marker(посвящ*)`.

Но `2687` показывает:

- issue wording должно не просто запрещать корень;
- оно должно указывать, как именно переписать sentence-level fragment.

Иначе Gemma слишком часто оставляет старую формулу.

## 5. Что теперь нужно спросить у Gemini

Следующий Gemini round должен быть не про ещё один redesign, а про quality repair на реальном `v2.5` evidence.

Нужно попросить:

1. Разобрать, почему `2660` выиграл по coverage, но проиграл по редакторской лёгкости.
2. Сказать, был ли `2734 -> compact` удачным исключением или признаком неверного gate.
3. Дать более сильные Gemma-friendly rewrites для anti-`посвящ*` именно на уровне extraction / revise.
4. Предложить, как сохранить `2745` на уровне `v2.3`, не возвращая `v2.4`.
5. Отдельно разобрать `2673` как case generic-structure drift.

## 6. Bottom line

`v2.5` — это полезный corrective round, который:

- доказал, что откат к `v2.3` был правильным;
- вернул часть качества после `v2.4`;
- но не дал нового лучшего кандидата.

Практически:

- **accept as evidence**: да;
- **accept as runtime candidate**: нет;
- **нужен ещё один Gemini consultation round**: да, и уже по полному `v2.5` bundle с фокусом на конкретные prompt edits для Gemma.
