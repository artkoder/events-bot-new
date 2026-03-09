# Smart Update Opus Gemma Event Copy V2 Dry-Run Quality Consultation Response Review

Дата: 2026-03-07

Связанные материалы:

- `docs/reports/smart-update-opus-gemma-event-copy-v2-dryrun-quality-consultation-response.md`
- `docs/reports/smart-update-opus-gemma-event-copy-pattern-dry-run-v2-5-events-2026-03-07.md`
- `docs/reports/smart-update-opus-gemma-event-copy-pattern-dry-run-v2-review-2026-03-07.md`
- `artifacts/codex/gemma_event_copy_pattern_dry_run_v2_5events_2026-03-07.json`
- `artifacts/codex/experimental_pattern_dryrun_v2_2026_03_07.py`

## 1. Краткий verdict

Это самый полезный ответ Opus после `v2 dry-run`.

Главное:

- он не спорит с самим verdict `v2 пока не годится`;
- он достаточно честно признаёт, что quality win не получен;
- он хорошо разделяет `idea failure` и `engineering failure`;
- и даёт уже не abstract advice, а конкретный `v2.1 patch set`.

Мой итог:

- **ещё один консультационный раунд сейчас не нужен**;
- этот ответ достаточно силён, чтобы переходить к локальной `v2.1` итерации в experimental harness;
- но часть его fixes нужно брать **с уточнениями**, потому что в raw виде некоторые из них сами рискованны.

## 2. Что в ответе Opus особенно сильное

### 2.1. Он правильно увидел `посвящ*` как lifecycle problem

Это очень сильная коррекция.

Самая полезная мысль во всём ответе:

- `посвящ*` leak возникает не только на generation layer;
- он часто приходит уже из extraction;
- дальше generation, которому велят “строго по фактам”, просто копирует этот паттерн назад.

Это точнее и полезнее, чем смотреть на `посвящ*` только как prompt hygiene issue.

### 2.2. Он правильно ударил в extraction anti-merge / anti-inflate

Это сильный root-cause diagnosis.

Особенно полезно по:

- `2734`, где теряется track/program detail;
- `2745`, где extraction раздувает один смысл в 5 квази-фактов;
- `2673`, где остаются дублирующие и слишком близкие fact lines.

Это действительно P0 слой для следующей локальной итерации.

### 2.3. Whole-body metatext detection — правильный ход

Это сильная practical correction.

Lead-only detection уже явно недостаточен.
На `2687` и других кейсах проблема не только в первом абзаце, а в том, что:

- `лекция расскажет...`
- `спектакль рассказывает...`
- `презентация посвящена...`

могут размазываться по всему тексту.

### 2.4. Concrete anti-embellishment examples — тоже правильный ход

Это хорошо бьёт именно по тому классу ошибок, который мы увидели в `2687`:

- факты называют имя;
- Gemma дорисовывает характеристику, которой не было;
- текст становится “красивее”, но менее точным.

Абстрактное “не достраивай смысл” действительно оказалось слишком слабым.

## 3. Что я бы принял в `v2.1`

### 3.1. Accept now

1. Ban `посвящ*` на extraction layer.
2. Extraction anti-merge rule для program-like items.
3. Extraction anti-inflate rule: `one fact = one detail`.
4. Whole-body metatext detection, а не только lead detection.
5. Более конкретный anti-embellishment block с примерами.
6. Rule `blockquote != lead repeat`.
7. Exact duplicate fact removal.
8. Anti-thin-section revise rule.

Это выглядит как действительно high-signal `v2.1 core`.

### 3.2. Accept with modification

1. Compact lead hook guidance

Да, её стоит брать.
Но я бы не фиксировал её слишком жёстко как “обязательно blockquote из яркого факта”.

Причина:

- не каждый длинный факт хорошо выглядит как epigraph;
- иногда лучше просто начать с конкретной детали без blockquote;
- иначе можно снова уйти в декоративность.

То есть guidance нужна, но без превращения blockquote в обязательный sparse ornament.

2. Weak heading detection expansion

Идея правильная.
Но это должен быть:

- revise signal,
- а не hard rejection.

Иначе можно слишком агрессивно penalize headings формата `О музыканте` / `О платформе`, даже когда они ещё salvageable.

3. “Routing is correct, fixes not needed”

Я бы согласился только частично.

Скорее так:

- routing tree как концепция сейчас не главный blocker;
- но routing quality по-прежнему зависит от качества extraction output;
- значит routing сам по себе не “closed”, просто сейчас он не первый приоритет.

4. Post-extraction `посвящ*` sanitize

Direction правильный, но не в той raw-реализации, которую предлагает Opus.

Это должно быть:

- либо узкое deterministic rewrite по известным safe-паттернам;
- либо reuse / extension текущего sanitize-слоя;
- но не blind replace уровня `посвящ* -> о`.

## 4. Что в ответе Opus всё ещё рискованно

### 4.1. Naive regex replace `посвящ* -> о` брать нельзя

Это важная оговорка.

Его пример:

```python
pattern = re.compile(r'(?i)\bпосвящ\w*\s+')
return [pattern.sub('о ', f).replace('  ', ' ').strip() for f in facts]
```

в raw виде слишком грубый.

Проблема:

- `Лекция посвящена творчеству...` -> `Лекция о творчеству...`
- `Концерт посвящён великой любви...` -> иногда нормально, иногда грамматически криво

То есть сама идея правильная, но реализация должна быть не таким regex-hack, а:

- через extraction ban;
- через расширение уже существующего `_sanitize_fact_text_clean_for_prompt`;
- или через более узкий deterministic rewrite по известным паттернам.

### 4.2. `_dedup_thin_facts` heuristic слишком рискован

Это слабое место ответа.

Идея semantic dedup правильная, но предложенный heuristic:

- легко сольёт разные близкие факты;
- плохо контролируется;
- может незаметно открыть ещё больше coverage loss.

На этой стадии я бы **не брал** такой runtime heuristic.

Правильнее:

- сначала улучшить extraction prompt;
- затем смотреть, что остаётся после exact dedup и better prompt discipline.

## 5. Что важно не потерять из текущего runtime

Это отдельная оговорка, потому что новые `v2.1` fixes должны быть **additive**, а не replacement по отношению к уже полезным fact-first guardrails.

Сохранять нужно:

1. Текущий sanitize-подход к `facts_text_clean`, а не заменять его одним новым regex-хаком.
2. Existing cleanup pipeline после generation, включая late hygiene и нормализацию структуры.
3. Coverage-first discipline и `content-preservation floor` как обязательный invariant.
4. Conditional epigraph / blockquote как optional feature, а не как обязательную sparse-декорацию.
5. Existing targeted anti-`посвящ*` repair как fallback guard, пока extraction-layer fix не доказан на dry-run.

Иначе есть риск “починить” новые `v2` regressions ценой потери уже полезных страховок из текущего fact-first runtime.

## 6. Нужен ли ещё один этап консультаций

**Нет.**

Сейчас у нас уже есть:

- real `baseline / v1 / v2` outputs;
- honest dry-run review;
- root-cause analysis от Opus;
- компактный `v2.1` patch direction.

Ещё один round сейчас даст меньше пользы, чем локальная реализация этих fixes и повторный rerun.

## 7. Следующий разумный шаг

Правильный следующий шаг сейчас:

1. Локально сделать `v2.1` только в experimental harness.
2. Не трогать основной runtime.
3. Внести узкий subset fixes:
   - extraction anti-merge / anti-inflate / anti-`посвящ*`
   - exact fact dedup
   - whole-body metatext detection
   - stronger anti-embellishment
   - blockquote/lead non-repeat
4. Повторить dry-run на тех же 5 событиях.

Только после этого можно решать:

- есть ли production candidate;
- или нужен ещё один external fine-tuning round уже по `v2.1` outputs.

## 8. Bottom line

Этот ответ Opus **принимается в основном**.

Он не закрывает задачу сам по себе,
но даёт уже хороший, узкий и практичный план для локального `v2.1`.

Моя позиция:

- консультационную фазу на этом этапе можно остановить;
- дальше — локальная `v2.1` итерация и новый dry-run;
- без переноса в код бота, пока `v2.1` реально не beat'ит текущие варианты на тех же кейсах.
