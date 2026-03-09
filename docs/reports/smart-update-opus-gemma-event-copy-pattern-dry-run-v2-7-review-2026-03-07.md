# Smart Update Opus Gemma Event Copy Pattern Dry-Run V2.7 Review

Дата: 2026-03-07

Связанные материалы:

- `docs/reports/smart-update-opus-gemma-event-copy-pattern-dry-run-v2-7-5-events-2026-03-07.md`
- `artifacts/codex/gemma_event_copy_pattern_dry_run_v2_7_5events_2026-03-07.json`
- `artifacts/codex/experimental_pattern_dryrun_v2_7_2026_03_07.py`
- `docs/reports/smart-update-gemini-event-copy-v2-7-hypotheses-consultation-response.md`
- `docs/reports/smart-update-gemini-event-copy-v2-7-hypotheses-consultation-response-review.md`

## 1. Краткий verdict

`v2.7` — это regression round.

Он не решил main targets:

- `2687` всё ещё с `посвящ*` и слабым coverage;
- `2673` всё ещё с `посвящ*` и служебным тоном.

И при этом он сломал уже подтверждённые gains:

- `2660`: `missing 0 -> 2`
- `2745`: `missing 3 -> 6`
- `2734`: `missing 3 -> 5`

Итог:

- runtime candidate: нет;
- current `v2.7` patch pack: reject;
- новый Gemini round: да, и уже именно как failure consultation по реальному regression bundle.

## 2. Что в `v2.7` сломалось

### 2.1. Safe-positive transformation породил new fact inflation

Это главный structural failure.

На практике `v2.7` начал:

- превращать один смысл в несколько формально “аккуратных”, но избыточных facts;
- добавлять weak packaging вроде:
  - `Выставка носит название ...`
  - `Автор выставки — ...`
  - `В центре встречи — ...`
- раздувать representation на cases, где `v2.6` был уже компактным и рабочим.

Признаки:

- `2660`: `facts 4 -> 5`
- `2745`: `facts 5 -> 6`
- `2673`: `facts 11 -> 14`

### 2.2. Agenda-safe wording стало слишком generic

Вместо канцелярита ветка часто начала давать:

- редакторски безопасный, но слабый generic phrasing;
- менее яркий и менее dense текст;
- лишние event-framing statements.

Именно это видно на:

- `2660`
- `2745`
- `2734`

### 2.3. Главные bugs вообще не закрылись

Самое неприятное:

- `2687` всё ещё с `посвящ*`;
- `2673` всё ещё с `посвящ*`;
- то есть ради устранения этих проблем мы внесли patch pack, но core target не достигнут.

## 3. Case-by-case

### 3.1. `2660`: regression after a real win

`v2.6` был лучшим round этого кейса.

В `v2.7` произошло следующее:

- эпиграф стал слабее;
- extraction добавил low-value fact `Выставка носит название ...`;
- prose стала более объясняющей и менее плотной;
- `missing 0 -> 2`.

То есть safe-positive extraction тут не оздоровил факты, а ухудшил их.

### 3.2. `2745`: catastrophic sparse regression

Это один из худших outcomes раунда.

- `missing 3 -> 6`
- текст стал короче, но беднее;
- compact branch потеряла informational density;
- часть тем и оттенков схлопнулась в generic family-drama summary.

### 3.3. `2734`: performance recovery потерян

`v2.6` наконец вернул working `2734`.

`v2.7` снова просел:

- `missing 3 -> 5`
- prose стала более safe и generic;
- richer program/person balance ослаб.

### 3.4. `2687`: target не закрыт

Да, numerically это лучше `v2.6`:

- `missing 5 -> 4`

Но practically это всё ещё провал:

- `посвящ*` остался;
- часть лекционного канцелярита всё ещё жива;
- coverage далека от `v2.5 missing=1`.

### 3.5. `2673`: target не закрыт

`2673` тоже не вылечен:

- `facts 11 -> 14`
- `missing 4 -> 5`
- `посвящ*` остался;
- prose снова служебная и explanatory.

Это важный сигнал, что positive transformation pack в текущем виде не просто слабый, а контрпродуктивный.

## 4. Что `v2.7` доказал про root cause

### 4.1. Не вся positive transformation полезна

Это главный practical вывод.

Да, one-sided negative bans были проблемой.
Но замена их на явные transformation examples сама по себе не спасает quality.

Если examples:

- слишком формульные;
- слишком explanatory;
- слишком safe;

то Gemma начинает плодить:

- packaging facts;
- generic “в центре ...” конструкции;
- новый template feel.

### 4.2. Extraction contract слишком сильно влияет на fact count

`v2.7` хорошо показал, что даже seemingly-safe guidance может:

- увеличивать число фактов;
- ухудшать compact branch;
- ломать downstream prose.

### 4.3. `посвящ*` нельзя вылечить только новыми examples

`2687` и `2673` показывают:

- проблема глубже;
- одних safe positive examples недостаточно;
- возможно, нужен более узкий deterministic/sanitizing support around extraction output, но без смысловой подмены.

## 5. Что теперь нужно спросить у Gemini

Новый round должен быть очень конкретным.

Нужно попросить Gemini:

1. Разобрать, почему safe-positive pack дал fact inflation.
2. Отделить:
   - good direction;
   - bad wording;
   - idea that itself should be rolled back.
3. Сказать, какие exact lines в prompts/hints спровоцировали:
   - `Выставка носит название ...`
   - `В центре встречи ...`
   - generic safe paraphrases.
4. Предложить, как лечить `посвящ*` и intent transfer без раздувания facts.
5. Дать very small `v2.8` patch pack, если такой вообще есть.

## 6. Bottom line

`v2.7` — не mixed round, а именно regression.

Практический verdict:

- **accept as evidence**: да;
- **accept as candidate**: нет;
- **нужен post-run Gemini round**: да, и именно с фокусом на failure analysis, а не на ещё один optimistic redesign.
