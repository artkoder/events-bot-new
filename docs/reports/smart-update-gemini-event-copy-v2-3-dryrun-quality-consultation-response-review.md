# Smart Update Gemini Event Copy V2.3 Dry-Run Quality Consultation Response Review

Дата: 2026-03-07

Связанные материалы:

- `docs/reports/smart-update-gemini-event-copy-v2-3-dryrun-quality-consultation-response.md`
- `docs/reports/smart-update-gemini-event-copy-v2-3-dryrun-quality-consultation-brief.md`
- `docs/reports/smart-update-opus-gemma-event-copy-pattern-dry-run-v2-3-5-events-2026-03-07.md`
- `docs/reports/smart-update-opus-gemma-event-copy-pattern-dry-run-v2-3-review-2026-03-07.md`
- `artifacts/codex/experimental_pattern_dryrun_v2_3_2026_03_07.py`

## 1. Краткий verdict

Ответ Gemini сильный и practically useful, но принимать его надо не целиком, а выборочно.

Главное:

- он правильно увидел, что `v2.3` наконец стал real step forward;
- хорошо попал в проблему conversational/generic headings;
- правильно настаивает на сохранении named program items;
- нормально калибрует ценность `compact_fact_led` и body self-sufficiency.

Но есть и важные оговорки:

- он частично переоценил deterministic metric `missing` как quality proxy, особенно на `2734`;
- его диагноз про leak через `copy_assets.core_angle` для текущего `v2.3` не до конца точен, потому что generation/revise prompts сейчас почти не опираются на `copy_assets`;
- его stop condition “если `v2.4` решает `2734` и `2673`, можно сливать в production” слишком оптимистична.

Мой итог:

- **новый Gemini-раунд сейчас не нужен**;
- нужен локальный `v2.4` dry-run;
- в `v2.4` стоит взять только те рекомендации, которые реально подтверждаются нашим harness и кейсами.

## 2. Что в ответе Gemini принимаю

### 2.1. `compact_fact_led` и body self-sufficiency

Это принимаю без возражений.

Он правильно фиксирует, что:

- `2660` и `2745` стали живее;
- отказ от forced headings на sparse cases был сильным ходом;
- правило про самодостаточный body действительно улучшило тексты.

### 2.2. Anti-conversational headings

Это high-signal рекомендация.

`2673` действительно показывает, что headings типа:

- `Что в программе?`
- `Что ещё нужно знать?`
- `Зачем создан проект ...?`

ломают профессиональный tone.

Такой guardrail стоит добавить:

- headings только назывные;
- без вопросов;
- без разговорных формул.

### 2.3. Program preservation

Это тоже принимаю.

Особенно на `2734` видно, что system needs stronger protection for:

- quoted song titles;
- named program items;
- named participants.

Но реализовывать это надо не только через generation rules, а и через extraction/facts layer.

## 3. Что принимаю только с поправками

### 3.1. Anti-`посвящ*`

Идея полезная, но не в той форме, как её формулирует Gemini.

Принимаю только в modified version:

- не делать тупой deterministic rewrite;
- не полагаться только на один hard lexical ban;
- усилить extraction/generation/revise contract;
- вернуть `посвящ*` как policy issue для revise;
- оставить positive replacements, чтобы Gemma не ломала русский синтаксис.

### 3.2. Pre-consolidation worth keeping

Да, но с уточнением.

Gemini прав, что её не надо выкидывать.
Но issue в `2734` не доказывает, что проблема именно в pre-consolidation.

По текущему `v2.3` harness более вероятно следующее:

- часть потерь уже возникает на extraction;
- generation contract всё ещё позволяет слишком свободно компрессировать named program details;
- `copy_assets` в standard generation почти не участвуют, так что сводить провал к ним нельзя.

То есть:

- pre-consolidation сохраняем;
- но не считаем её главным виновником без дополнительного evidence.

## 4. Что в ответе Gemini считаю miscalibrated

### 4.1. `copy_assets.core_angle` как главный источник утечки

Это слабое место ответа.

Для текущего `v2.3` это не основной диагноз, потому что:

- non-compact generation prompt строится поверх `facts_text_clean`;
- revise prompt тоже почти не использует `copy_assets`;
- значит `посвящён` и content compression сейчас с высокой вероятностью приходят из facts layer и из prompt contract, а не из direct `copy_assets` routing.

Иными словами:

- сам symptom Gemini прочитал правильно;
- root cause он локализовал не совсем точно.

### 4.2. Acceptance / stop conditions

Тут Gemini слишком агрессивен.

Проблемы:

- `2734 missing=7` уже сейчас partly metric artifact, потому что несколько named items реально есть в тексте, но mismatch идёт на уровне exact fact wording;
- поэтому target вида “уронить missing до 2-3 и можно сливать в прод” слишком грубый;
- даже хороший `v2.4` будет требовать ещё одной оценки, а не auto-merge.

## 5. Что именно пойдёт в `v2.4`

В локальный patch pack беру только такое:

1. Stronger anti-conversational heading contract:
   - headings не вопросами;
   - headings не разговорными формулами;
   - revise должен это чинить.

2. Stronger program preservation:
   - extraction prompt;
   - facts reinforcement for quoted program items;
   - generation/revise rules не схлопывать names/titles.

3. Modified anti-`посвящ*`:
   - stronger extraction hint;
   - stronger generation/revise rule;
   - `посвящ*` снова становится policy issue;
   - без наивного regex-rewrite финального текста.

## 6. Что не беру в `v2.4`

- Не принимаю тезис, что `copy_assets` надо срочно redesignить ради этого раунда.
- Не принимаю merge-to-production после одного успешного `v2.4`.
- Не беру overly literal interpretation его `missing` thresholds как главную мету.

## 7. Bottom line

Ответ Gemini полезен и в целом сильнее, чем средний внешний feedback, но его надо читать инженерно, а не как final oracle.

Мой practical verdict:

- **accept with modification** для heading contract;
- **accept with modification** для anti-`посвящ*`;
- **accept** для stronger program preservation;
- **reject** idea that one `v2.4` success already justifies production merge;
- **reject** root-cause over-attribution to `copy_assets`.

Следующий шаг:

- локально собрать `v2.4`;
- прогнать те же 5 событий;
- уже потом смотреть, нужен ли новый consultation round.
