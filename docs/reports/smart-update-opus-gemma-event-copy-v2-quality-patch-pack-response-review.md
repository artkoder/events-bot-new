# Smart Update Opus Gemma Event Copy V2 Quality Patch Pack Response Review

Дата: 2026-03-07

Связанные материалы:

- `docs/reports/smart-update-opus-gemma-event-copy-v2-quality-patch-pack-response.md`
- `docs/reports/smart-update-opus-gemma-event-copy-v2-quality-patch-pack-brief.md`
- `docs/reports/smart-update-opus-gemma-event-copy-quality-consultation-followup-response-review.md`
- `docs/reports/smart-update-opus-gemma-event-copy-text-quality-improvements-review.md`
- `docs/reports/smart-update-opus-gemma-event-copy-pattern-dry-run-5-events-2026-03-07.md`
- `smart_event_update.py`

## 1. Краткий verdict

Это пока самый сильный и самый implementation-ready ответ Opus по всей ветке `Gemma event copy quality`.

Главное:

- он действительно сделал то, что требовалось на этом этапе;
- не ушёл в новый idea catalog;
- сжал предложения в компактный `v2 patch pack`;
- дал prompt blocks, runtime gates и sparse contract в форме, близкой к реальному внедрению.

Мой итоговый вывод:

- **ещё один consultation round перед локальной v2 итерацией уже не нужен**;
- этот ответ достаточно хорош, чтобы переходить к локальной implementation/dry-run фазе;
- но внедрять его нужно **не буквально**, а с несколькими важными поправками.

## 2. Что в ответе Opus особенно сильное

### 2.1. Наконец получен именно compact implementation subset

Это ключевой успех ответа.

Opus не пытается снова обсуждать всё подряд и даёт то, что действительно нужно:

- shortlist high-signal changes;
- компактные prompt blocks;
- короткий runtime gate set;
- понятный sparse contract;
- список `do not include yet`.

Это именно тот формат, который нужен для controlled v2 prototype.

### 2.2. Приоритизация в целом здравая

Особенно удачно, что в `must_include_v2` попали:

- anti-duplication;
- anti-embellishment;
- anti-metatext;
- filler denylist;
- sparse routing;
- epigraph recovery.

Это действительно самые вероятные источники quality lift при умеренном prompt overhead.

### 2.3. Prompt compression выглядит реалистично для Gemma

Это тоже сильная часть ответа.

`generation_quality_block`, `lead_block` и `revise_quality_block`:

- короткие;
- директивные;
- без лишней философии;
- соответствуют реальным проблемам dry-run.

В сравнении с более ранними раундами это уже не “красивые идеи”, а usable prompt surface.

### 2.4. Runtime gate shortlist хорошо ложится на текущий runtime

Ответ хорошо попал в существующие точки расширения:

- `_collect_policy_issues`
- prompt generation
- revise loop
- `_pick_epigraph_fact`

То есть с инженерной точки зрения это не redesign с нуля, а разумный слой поверх уже работающего fact-first pipeline.

## 3. Что нужно принять с обязательными поправками

### 3.1. Patch pack должен быть additive, а не replacement

Это главный implementation caveat.

Нельзя потерять сильные части текущего runtime, которые уже работают:

- `_sanitize_fact_text_clean_for_prompt`
- `_pick_epigraph_fact`
- правила про visitor conditions / format facts / numeric fidelity / list fidelity
- coverage check + revise loop
- `_fact_first_remove_posv_prompt`
- `_cleanup_description`
- существующие forbidden-marker guardrails

То есть `v2 patch pack` надо **накладывать поверх current fact-first flow**,
а не переписывать prompt family как будто существующей сильной базы нет.

### 3.2. `generic heading ban` я бы поднял из `include_if_compact` в `must_include_v2`

Это почти бесплатное улучшение.

Причины:

- проблема реально видна и в baseline, и в prototype;
- rule очень короткий;
- ложноположительный риск низкий;
- он хорошо работает как prompt rule и как revise signal.

Это уже не optional polish, а cheap quality win.

### 3.3. `dedup runtime guard` нельзя применять как “немую автопилу”

Идея правильная, но proposed placement рискован.

Если просто удалять near-duplicate sentences после revise,
можно случайно выбросить coverage-critical sentence,
а повторной coverage-проверки уже не будет.

Правильнее для v2:

- либо делать dedup как `policy issue` и отправлять в revise;
- либо применять dedup post-process только очень консервативно;
- либо после dedup делать ещё один cheap coverage sanity check.

Иначе можно починить `2687` и `2673`, но открыть новый класс latent fact-loss regressions.

### 3.4. Sparse contract нужен чуть свободнее по длине

`150-400 chars` — полезный ориентир, но слишком жёсткий как hard target.

Проблема:

- даже sparse event может содержать 5 плотных publishable facts;
- если жёстко зажать объём, модель начнёт drop'ать детали или схлопывать факты слишком грубо.

Поэтому лучше:

- оставить logic `sparse = no headings / compact prose`;
- но длину привязывать к текущему budget estimator и `content-preservation floor`,
  а не к слишком узкому hard cap.

### 3.5. `content-preservation floor` нельзя потерять только потому, что он не в центре этого ответа

Это важная дыра в финальной таблице.

Opus раньше сам правильно поднял `content-preservation floor`,
но в этом ответе extraction-side fix уже не так явно встроен в final pack.

Для v2 это всё ещё обязательно:

- concrete details нельзя терять ради “красивой компактности”;
- sparse routing и anti-filler не заменяют extraction discipline;
- кейс `2734` без этого не считается закрытым.

То есть этот patch pack нужно реализовывать **вместе** с уже принятым extraction-preservation rule, а не вместо него.

## 4. Что я бы принял в локальную v2 итерацию

### 4.1. Prompt changes

Принять:

- `generation_quality_block`
- `lead_block`
- `heading_quality_rule`
- `revise_quality_block`
- отдельный sparse prompt variant

С оговоркой:

- встраивать их поверх current prompt contract;
- не терять правила про visitor conditions, format, lists, numbers и epigraph handling.

### 4.2. Runtime changes

Принять:

- CTA detection
- metatext lead detection
- weak heading detection
- epigraph recovery plumbing

Принять с осторожностью:

- dedup guard

Для него нужен более безопасный placement, чем “удалить и сразу вернуть”.

### 4.3. Routing changes

Принять:

- `len(facts_text_clean) <= 5` -> compact sparse branch
- `> 5` -> standard fact-first branch with v2 blocks

Но:

- это не должно отменять future pattern routing work;
- для текущей локальной v2 это acceptable narrow improvement.

### 4.4. Что пока не брать

Согласен не брать в v2:

- question-led openings;
- few-shot examples;
- 3-tier density system;
- paragraph quality gate;
- broad sentence-level prose gate;
- pattern set reduction;
- merge `value_led + topic_led`;
- direct use of `experience_signals` in generation.

## 5. Нужен ли ещё один этап консультаций

**Нет.**

На этой стадии дополнительные консультации дадут убывающую отдачу.

У нас уже есть:

- реальный dry-run baseline;
- реальный dry-run prototype;
- калибровка quality failures;
- сжатый v2 patch pack;
- понятные caveats для implementation.

Следующий рациональный шаг:

- локально внедрить v2 subset;
- прогнать новый dry-run;
- сравнить baseline / v1 prototype / v2 prototype;
- и уже потом, если нужно, отправить обновлённые реальные промпты и outputs в Opus на fine-tuning round.

## 6. Bottom line

Этот ответ Opus **достаточен для выхода из консультационного цикла и перехода к локальной v2 итерации**.

Моя финальная позиция:

- consultation phase по `v2 quality patch pack` можно считать закрытой;
- implementation phase можно начинать;
- но с сохранением сильных частей текущего runtime и с осторожной реализацией dedup / sparse budget / extraction preservation.
