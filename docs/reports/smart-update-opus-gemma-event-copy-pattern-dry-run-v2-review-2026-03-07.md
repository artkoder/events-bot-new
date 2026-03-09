# Smart Update Opus Gemma Event Copy Pattern Dry Run V2 Review — 2026-03-07

Дата: 2026-03-07

Связанные материалы:

- `docs/reports/smart-update-opus-gemma-event-copy-pattern-dry-run-v2-5-events-2026-03-07.md`
- `artifacts/codex/gemma_event_copy_pattern_dry_run_v2_5events_2026-03-07.json`
- `artifacts/codex/experimental_pattern_dryrun_v2_2026_03_07.py`
- `docs/reports/smart-update-opus-gemma-event-copy-pattern-dry-run-5-events-2026-03-07.md`
- `artifacts/codex/gemma_event_copy_pattern_dry_run_5events_2026-03-07.json`
- `docs/reports/smart-update-opus-gemma-event-copy-dry-run-5-events-2026-03-07.md`

## 1. Краткий verdict

Этот `v2 dry-run` **не даёт основания переносить новый подход в код бота**.

После реального прогона на тех же 5 событиях итог такой:

- `v2` действительно убрал часть самых неприятных `v1`-дефектов;
- но в текущем виде он **не даёт достаточно сильного общего quality win**;
- на важных кейсах он всё ещё теряет слишком много фактов и продолжает ловить `посвящ*` leaks;
- по сумме результатов `v2` сейчас выглядит **хуже `v1` как experimental candidate**.

## 2. Что в `v2` реально стало лучше

### 2.1. Ушли катастрофические повторы уровня `v1`

Это главное положительное изменение.

Особенно заметно на:

- `2687`
- `2673`

Там `v2` уже не разваливается на повтор одной и той же детали 2-5 раз, как это было в `v1`.

### 2.2. Тексты стали ближе к current fact-first discipline

По тону `v2` чаще:

- менее рекламный;
- чуть чище структурно;
- реже уходит в откровенный CTA;
- меньше опирается на шумные `copy_assets`.

То есть как анти-шаблонный и anti-dup слой это движение в правильную сторону.

### 2.3. `2660` стал честнее по тону

После повторного прогона без source-driven epigraph текст стал заметно чище.

Но:

- это не стало quality win по coverage;
- baseline всё ещё лучше по полноте и Telegraph-структуре.

## 3. Что в `v2` остаётся blocker'ом

### 3.1. Sparse branch пока слабый

Это главный вывод по `2660` и `2745`.

На sparse-кейсах `compact_fact_led` пока:

- слишком легко схлопывает факты;
- не даёт стабильного quality lift;
- может звучать чище, но не становится содержательнее;
- всё ещё ловит `посвящ*` leaks (`2745`).

По факту:

- `2660`: `missing=3` против baseline `2`
- `2745`: `missing=5`, то есть вообще без выигрыша

### 3.2. `посвящ*` остаётся системной проблемой

Это видно сразу на нескольких кейсах:

- `2745`
- `2734`
- `2673`

Даже после:

- prompt bans;
- sanitize;
- `remove_posv`;
- final policy revise

Gemma всё ещё норовит вернуться к этой формуле.

Это уже не “единичный глитч”, а устойчивый failure mode текущего prompt pack.

### 3.3. Coverage regressions слишком дорогие

Самый плохой кейс тут — `2673`.

Сравнение:

- baseline: `missing=5`
- `v1`: `missing=1`
- `v2`: `missing=6`

То есть `v2` убрал CTA leak, но фактически проиграл по ключевой задаче: донести содержательную программу и смысл проекта.

### 3.4. `2734` показал, что branching ещё нестабилен

После ужесточения preservation/filtering этот кейс ушёл в `compact_fact_led`.

Это дало:

- `missing=3` как у `v1`;
- но слишком схлопнутый текст;
- и всё равно `посвящ*` leak.

То есть branch selection тут пока нельзя считать удачным.

### 3.5. `2687` стал чище, но всё ещё не лучший

Это наиболее двусмысленный кейс.

Плюсы:

- нет catastrophic duplication;
- текст стал аккуратнее.

Минусы:

- `missing=3` против `v1 missing=1`;
- prose уже чище, но coverage просел;
- по сумме `v1` всё ещё сильнее как output candidate.

## 4. Сводный итог по пяти кейсам

| Event | Baseline | V1 | V2 | Итог |
|---|---|---|---|---|
| 2660 | baseline лучше по coverage | mixed | tone чище, но coverage не лучше | baseline > v2 ≈ v1 |
| 2745 | weak | weak | weak + `посвящ*` | no win |
| 2734 | mixed | strongest in this trio by balance | over-compressed + `посвящ*` | v1 > v2 |
| 2687 | readable but incomplete | strongest by coverage, but dirty by duplication | cleaner, but coverage weaker | v1 > v2 on net |
| 2673 | decent baseline | promising but CTA leak | cleaner CTA-wise, but big coverage loss | baseline > v2, v1 still more promising |

## 5. Финальная позиция

`v2 patch pack` в этой локальной реализации **не готов ни к внедрению, ни к переносу в основной runtime как экспериментальной ветки**.

Что уже ясно:

- сама идея `anti-dup + anti-metatext + sparse branch` полезна;
- но текущая сборка rules ещё не даёт качественного результата на реальных кейсах;
- лучший следующий шаг сейчас — **не внедрение**, а ещё один узкий round по real outputs.

## 6. Следующий разумный шаг

Сейчас есть два рациональных варианта:

1. Отправить этот `v2 dry-run` пакет в Opus на критическую консультацию уже по реальным output-кейсам.
2. Локально сделать ещё один очень узкий `v2.1` iteration только по трём направлениям:
   - stubborn `посвящ*` leakage,
   - sparse branch quality,
   - coverage loss на `2673` / `2687`.

Мой приоритетный вариант:

- **сначала отправить именно этот пакет в Opus**, потому что теперь спор уже идёт не по теории, а по конкретным текстам и реальным regressions.
