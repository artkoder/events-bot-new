# Smart Update Opus Gemma Event Copy Quality Consultation Response Review

Дата: 2026-03-07

Связанные материалы:

- `docs/reports/smart-update-opus-gemma-event-copy-quality-consultation-response.md`
- `docs/reports/smart-update-opus-gemma-event-copy-pattern-dry-run-5-events-2026-03-07.md`
- `docs/reports/smart-update-opus-gemma-event-copy-dry-run-5-events-2026-03-07.md`
- `artifacts/codex/gemma_event_copy_pattern_dry_run_5events_2026-03-07.json`
- `artifacts/codex/experimental_pattern_dryrun_v1_2026_03_07.py`

## 1. Краткий verdict

Этот ответ Opus **полезный и содержательно сильнее, чем первый quality-consultation request сам по себе мог гарантировать**.

Главное, что в нём ценно:

- он увидел, что prototype v1 пока не production-ready;
- он точно поймал несколько реальных blockers;
- он дал уже не abstract ideas, а рабочий `v2 patch list`.

Но пользователь прав и в другом:

- **Opus местами действительно плохо откалибровал чтение текущего dry-run**;
- он местами слишком переоценил structural cues baseline (`blockquote`, headings, Telegraph-readability);
- и недооценил, что часть проблем prototype = не “неправильный pattern”, а сырой harness / revise-loop / extraction contract.

Итоговая позиция:

- ответ **принимается частично**;
- многие предложения стоит брать уже сейчас;
- но нужен **ещё один узкий раунд консультации**:
  - не про новую архитектуру;
  - а про `recalibration + v2 patch planning`.

## 2. Что в ответе Opus реально сильное

### 2.1. Он правильно назвал главный blocker: duplication

Это самый полезный practical finding во всём ответе.

Сильные пункты:

- anti-duplication rule в generation prompt;
- anti-duplication check после revise/repair;
- признание, что current revise-loop может лечить coverage ценой повторов.

Это действительно P0 improvement.

### 2.2. Он точно увидел loss нескольких baseline strengths

Это хороший and grounded diagnosis:

- prototype потерял blockquote/epigraph opening;
- prototype слишком часто сваливается в generic headings;
- prototype местами теряет Telegraph readability;
- sparse routing сейчас реально промахивается.

Это не полный диагноз качества, но как v2-fix input — очень полезно.

### 2.3. Routing critique по sparse events в целом верная

Особенно полезная мысль:

- `value_led` сейчас побеждает слишком часто;
- sparse / semi-sparse events нельзя автоматически тащить в value framing;
- `compact_fact_led` должен быть намного важнее в routing tree.

Это похоже на реальный root cause хотя бы для части regressions.

### 2.4. Он правильно сместил разговор с “pattern beauty” на pipeline blockers

Сильный момент ответа:

- Opus не пытается защищать prototype как концепцию;
- он смотрит на extraction;
- routing;
- generation;
- revise;
- cleanup;
- evaluation.

Это полезнее, чем очередной prompt-эстетический round.

## 3. Где Opus действительно misread текущий прогон

### 3.1. Он слишком переоценил blockquote и headings как quality proxy

Это заметный calibration issue.

Да, baseline blockquote/opening часто визуально лучше.
Да, generic heading `О событии` — слабое место prototype.

Но Opus местами делает из этого слишком сильный вывод:

- не каждый текст без blockquote — regression;
- не каждый compact single-paragraph output = “wall of text”;
- для `compact_fact_led` отсутствие headings может быть нормальным design choice, а не автоматически defect.

То есть structural Telegraph readability важна, но её нельзя ставить выше:

- grounding;
- fact retention;
- hygiene;
- duplication control;
- service leakage control.

### 3.2. Он местами смешал “prototype text bad” с “pattern idea bad”

Это особенно важно по `2687`.

Там у prototype действительно catastrophic duplication.
Но из этого **не следует**, что:

- `program_led` как pattern неверен;
- или baseline direction в целом лучше.

Скорее наоборот:

- там routing может быть нормальным;
- но generation/revise implementation сырой.

Если это не разделять, легко сделать слишком широкий rollback.

### 3.3. Он слишком жёстко трактует `copy_assets not in facts_text_clean = editorialized`

Это важное conceptual смещение.

Если `copy_assets` действительно source-backed и traceable, то они не обязаны быть “нелегитимными” только потому, что не совпали с `facts_text_clean`.

Правильнее формулировать не так:

- “всё, чего нет в facts_text_clean, запрещено”

а так:

- `copy_assets` допустимы только если они evidence-backed;
- их роль должна быть ясной;
- они не должны незаметно расширять factual contract без traceability.

То есть проблема не в самом существовании richer copy assets, а в слабом contract между:

- facts
- copy_assets
- generation source of truth

### 3.4. `Don't reduce fact count below baseline` полезно, но в таком виде слишком грубо

Как safety heuristic это сильная мысль.
Но в literally таком виде правило слишком жёсткое.

Почему:

- baseline fact inventory сам по себе не идеален;
- baseline может содержать weak/meta/service-like facts;
- literal count floor может консервировать шум.

Правильнее:

- prototype extraction не должен снижать **publishable content coverage** по сравнению с baseline без явной и безопасной причины;
- если baseline fact dropped, drop должен быть объясним:
  - service-like
  - duplicated
  - low-value
  - policy-forbidden

То есть не raw count floor, а `content-preservation floor`.

### 3.5. Он местами слишком щедр к baseline просто потому, что тот лучше оформлен

Это заметно на `2745` и partly `2687`.

Baseline действительно:

- чище структурно;
- Telegraph-readable;
- более аккуратно оформлен.

Но baseline тоже системно слаб:

- service leakage;
- unsupported promo-like phrasing;
- пустые generic closing lines;
- “правильный вид” при неидеальном factual discipline.

Если этого не проговаривать, можно случайно начать optimizе `for looking like baseline`, а не `for actual text quality`.

## 4. Что из предложений Opus стоит принять уже сейчас

### 4.1. Принять почти без спора

Ниже — предложения, которые я считаю strong and directly useful:

1. Добавить explicit anti-duplication rule в generation prompt.
2. Добавить anti-duplication guard после revise/repair.
3. Вернуть blockquote/epigraph как cross-pattern enhancement.
4. Исправить sparse routing: `<= 5 facts -> compact_fact_led` или equivalent sparse-safe branch.
5. Запретить generic headings вроде `О событии`, `О лекции`, `О концерте`.
6. Добавить CTA detection в quality/forbidden checks.
7. Прогнать ещё один v2 dry-run до любой интеграции в код.

Это очень похоже на правильный v2 starter set.

### 4.2. Принять, но с важной правкой

Следующие идеи полезны, но требуют уточнения:

1. `format_signal` deriving from `event_type`
- Да, runtime-first derivation полезна.
- Но не как полная замена semantic hints.
- Скорее:
  - `event_type` = hard default
  - LLM `subformat` / refinements = optional layer

2. `don't reduce fact count below baseline`
- Принимать только как `content-preservation floor`, не как raw literal count.

3. `constrain credibility_signals`
- Да, нужно.
- Но не “убрать всё, чего нет в facts_text_clean”.
- А:
  - `source-backed only`
  - evidence-backed
  - role-bounded in generation

4. `experience_signals mostly editorialized`
- Это ценный warning.
- Но я бы не удалял поле сразу.
- Лучше:
  - резко tighten grounding contract;
  - проверить utility на v2.

### 4.3. Пока не принимать как факт

Здесь я бы не спешил:

1. `merge value_led + topic_led`
- возможно;
- но пока evidence недостаточно.

2. `reduce pattern set from 6 to 4`
- потенциально разумно;
- но сперва надо разделить:
  - design issue
  - prototype implementation issue

3. `baseline significantly better` как глобальный narrative
- это полезный editorial impression,
- но как architecture-level conclusion пока слишком рано.

## 5. Нужен ли ещё один этап консультаций

**Да.**

Но это должен быть уже **узкий recalibration round**, а не новая свободная brainstorm-консультация.

Почему он нужен:

- у Opus уже есть много полезных v2 fixes;
- но часть оценки перекошена в сторону structural readability и baseline optics;
- нам нужно привести evaluation frame к более точному quality contract;
- и получить от него уже не просто review, а `corrected v2 patch plan`.

## 6. Что должен решить следующий раунд

Следующая консультация должна закрыть 5 конкретных вещей:

### 6.1. Recalibrated evaluation frame

Opus должен заново оценить кейсы по явным осям:

- factual coverage
- unsupported prose
- service leakage
- duplication
- readability
- structural fit
- professional tone

Нужно отделить:

- “baseline выглядит аккуратнее”
- от
- “baseline реально лучше по quality priorities”

### 6.2. Pattern vs harness failure

Нужно жёстко разделить:

- неправильный pattern choice;
- плохой routing;
- слабый extraction contract;
- сырой generation prompt;
- broken revise loop.

Без этого можно откатить полезную идею просто потому, что v1 harness сделал её плохо.

### 6.3. Exact v2 patch set

Нам нужен не просто список идей, а конкретный `v2 implementation subset`, который Opus считает наилучшим next move:

- anti-dup prompt rule
- anti-dup runtime repair
- epigraph recovery
- sparse routing correction
- weak-heading ban
- CTA check
- extraction floor / preservation rule
- evidence-backed copy_assets contract

### 6.4. `copy_assets` contract

Нужно отдельно дожать:

- что может legitimately жить в `copy_assets`;
- что должно жить только в `facts_text_clean`;
- что generation имеет право использовать как factual material;
- где нужна evidence traceability.

### 6.5. V2 success criteria

Нужны явные критерии следующего prototype round:

- no catastrophic duplication;
- sparse events no longer over-patterned;
- epigraph restored where justified;
- no generic headings;
- no new service leakage;
- no unjustified fact loss relative to baseline content coverage.

## 7. Bottom line

Ответ Opus полезный и practically valuable.

Но:

- он **ещё не даёт полностью правильно откалиброванного reading** текущего dry-run;
- и его нельзя просто принять целиком как окончательный verdict по prototype.

Правильное next step:

- взять его strongest v2 ideas;
- оспорить его miscalibrated assumptions;
- и провести ещё один **узкий quality recalibration round** с целью получить уже corrected v2 plan.
