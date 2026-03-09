# Smart Update Event Copy V2.12 Consultation Synthesis

Дата: 2026-03-08

Связанные материалы:

- `artifacts/codex/tasks/event-copy-v2-12-results-consultation-brief.md`
- `artifacts/codex/reports/event-copy-v2-12-results-consultation-claude-opus.json`
- `artifacts/codex/reports/event-copy-v2-12-results-consultation-claude-opus.md`
- `artifacts/codex/reports/event-copy-v2-12-results-consultation-gemini-3.1-pro.md`
- `docs/reports/smart-update-opus-gemma-event-copy-pattern-dry-run-v2-11-5-events-2026-03-08.md`
- `docs/reports/smart-update-opus-gemma-event-copy-pattern-dry-run-v2-11-review-2026-03-08.md`

## 1. Проверка моделей

`Opus` был вызван в strict режиме:

- `claude -p --model claude-opus-4-6 --tools "" --output-format json`

`modelUsage` в raw JSON подтвердил:

- `claude-opus-4-6`

`Gemini` была вызвана через:

- `gemini -m gemini-3.1-pro-preview`

Session log подтвердил модель:

- `gemini-3.1-pro-preview`

## 2. Где внешние модели сошлись

Обе консультации независимо пришли к одному общему выводу:

1. текущая линия `subset extraction -> merge/floor -> generation` близка к architectural ceiling;
2. проблема уже не в ещё одной порции extraction-rules, а в том, что clean facts конфликтуют с floor/raw facts;
3. baseline structurally выигрывает не потому, что текст у него лучше, а потому, что он не ломает fact coverage через destructive intermediate layer;
4. если хотим beat baseline, нужно менять unit of work:
   не извлекать "лучшее подмножество" фактов, а нормализовать всю fact base.

## 3. Где Opus и Gemini разошлись

Главное расхождение:

- `Opus` предлагает радикально убрать LLM extraction и перейти к deterministic cleaning + generation + editorial review.
- `Gemini` предлагает сохранить LLM-first, но заменить current extraction на LLM normalization всей `raw_facts` базы с роутингом по source shape.

Для этого проекта я принимаю позицию ближе к `Gemini`, потому что она лучше согласуется с каноникой `LLM-first`:

- deterministic support rules допустимы как hygiene/dedup/filtering;
- но primary смысловая трансформация фактов не должна переезжать в жёсткие regex-правила.

## 4. Baseline-relative verdict

На текущем наборе из 5 событий:

- baseline total missing = `22`
- `v2.11` total missing = `33`

То есть:

- `v2.11` хуже baseline;
- и хуже не только локально, а по сумме качества.

Но важная оговорка:

- некоторые старые experimental rounds уже beat baseline по coverage.

Самый сильный пример:

- `v2.6` total missing = `15`

Значит baseline не является потолком качества.
Потолок текущей single-pass extraction architecture — да, похоже, близок.

## 5. Что я принимаю в качестве направления `v2.12`

### 5.1. Новый основной unit of work

Не `subset extraction`, а:

- `raw_facts -> normalized_floor`

То есть LLM должна не выбирать часть фактов и не изобретать новую абстракцию поверх них, а переписывать всю входящую fact base в clean, publishable, narrative-ready form без потери coverage.

### 5.2. Shape-routed normalization

`v2.12` должен иметь не один универсальный normalizer prompt, а хотя бы 2-3 контракта:

- `presentation/project-rich`
  - упор на nominalization и удаление metatext/intent frames
- `lecture/person-rich`
  - упор на сохранение grouped people blocks и запрет person-splitting
- `concert/program-rich` / `sparse cultural`
  - упор на сохранение grouped program items и компактность

### 5.3. Deterministic слой остаётся, но уже как support

Допустимо и нужно оставить детерминированно:

- service/logistics filtering;
- exact/near dedup;
- hard forbidden scans;
- fact count cap / ranking;
- simple structural cleanup.

Но не primary смысловую rewrite-трансформацию.

### 5.4. Generation-side hygiene сохраняется

Из `v2.11` в `v2.12` надо переносить:

- anti-quote rule;
- anti-metatext / anti-bureaucratic framing;
- самодостаточность body;
- compact/sparse hygiene.

## 6. Что я не принимаю в `v2.12`

- ещё один раунд “чуть-чуть лучше extraction prompt” на старой architecture;
- pure deterministic replacement LLM normalization как основной путь;
- универсальный prompt для всех source shapes;
- blind merge/floor of dirty and clean facts в одном списке.

## 7. Как я вижу `v2.12`

### Proposed architecture

1. `raw_facts`
2. shape detection
3. shape-routed `LLM normalization` of the full floor
4. deterministic dedup / hygiene / cap
5. generation
6. optional small editorial review pass only when needed

### Почему optional editorial pass

Здесь я беру лучшее из обеих консультаций:

- `Opus` прав, что узкий editorial pass может быть эффективнее ещё одного broad revise;
- но делать его unconditional пока рано.

Для `v2.12` разумнее:

- сначала протестировать `normalized_floor -> generation`;
- потом при необходимости добавить маленький editorial pass только на dense/high-risk cases.

## 8. Minimal experiment plan for `v2.12`

### Phase A

Собрать новый experimental harness, где:

- полностью убран current subset-extraction + preserve-floor merge logic;
- вместо этого added `shape-routed normalized_floor`.

### Phase B

Сначала прогнать 3 hardest cases:

- `2673`
- `2687`
- `2745`

Потому что именно они лучше всего покажут:

- лечится ли project-rich intent persistence;
- исчезает ли person-splitting;
- не ломается ли sparse coverage.

### Phase C

Если на 3 hardest cases видно signal, прогнать все 5 событий.

## 9. Baseline-relative success criteria for `v2.12`

`v2.12` нельзя считать успехом, если он только “лучше `v2.11`”.

Минимальный success bar:

- total missing < `22` on the 5-event set;
- forbidden = `0`;
- no catastrophic outlier worse than `baseline max missing=5` by large margin.

Stronger target:

- total missing around `15` or below;
- forbidden = `0`;
- prose quality not weaker than strongest `v2.3/v2.6` cases.

## 10. Bottom line

Мой synthesis такой:

- текущая линия single-pass extraction действительно почти исчерпана;
- но полный уход в deterministic primary cleaning я не принимаю;
- `v2.12` должен стать first round новой architecture:
  `full-floor LLM normalization -> deterministic cleanup -> generation`,
  а не `v2.11 + ещё patch pack`.
