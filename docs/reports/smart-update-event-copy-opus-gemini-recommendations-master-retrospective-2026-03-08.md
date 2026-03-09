# Smart Update Event Copy Opus + Gemini Recommendations Master Retrospective

Дата: 2026-03-08

Цель документа:

- собрать в одном месте не только synthesis, но и ретроспективную базу рекомендаций `Opus` и `Gemini` по всей линии улучшения Gemma event-copy;
- дать один canonical knowledge base перед реализацией `2.15.2`;
- зафиксировать не только "что советовали", но и что из этого позже подтвердилось dry-run-практикой, а что нет.

Важно:

- это не новый consultation round;
- это meta-index и retrospective summary уже существующих раундов;
- здесь приоритет у рекомендаций, которые реально влияли на архитектуру, prompts и качество текста.

Связанные канонические материалы:

- [full retrospective baseline -> v2.14](/workspaces/events-bot-new/docs/reports/smart-update-gemma-event-copy-retrospective-baseline-v2-14-2026-03-08.md)
- [external consultation retrospective](/workspaces/events-bot-new/docs/reports/smart-update-event-copy-external-consultation-retrospective-2026-03-08.md)
- [v2.15 design brief](/workspaces/events-bot-new/docs/reports/smart-update-gemma-event-copy-v2-15-design-brief-2026-03-08.md)
- [v2.15.2 design brief](/workspaces/events-bot-new/docs/reports/smart-update-gemma-event-copy-v2-15-2-design-brief-2026-03-08.md)

## 1. Как пользоваться этим документом

Это не просто список ссылок.

У каждой фазы ниже есть:

- ключевые source docs;
- что именно советовал `Opus` или `Gemini`;
- что было принято;
- что позже подтвердилось или не подтвердилось в dry-run.

То есть это не "архив ради архива", а рабочая база решений.

## 2. Самые устойчивые recurring signals across the whole cycle

Это самые важные рекомендации, которые повторялись в разных раундах и не развалились при практической проверке.

### 2.1. Semantic core должен оставаться LLM-first

Повторяли и `Opus`, и `Gemini`.

Смысл:

- регулярки и deterministic support полезны;
- но они не должны становиться semantic core для event-copy на тысячах heterogeneous posts.

Практический итог:

- это подтверждено всей историей dry-run;
- каждый уход в regex-heavy drift давал локальные fixes и новые long-tail проблемы.

### 2.2. Качество текста нельзя жертвовать ради одной coverage-метрики

Повторялось в разных формулировках.

Смысл:

- coverage P0;
- но если ради literal-missing текст становится плоским, бюрократичным или шаблонным, это не победа.

Практический итог:

- baseline был стабильнее по coverage;
- `v2.6`, `v2.12`, `v2.13` показывали, что prose можно реально поднять выше baseline;
- значит оптимизироваться надо по обеим осям: coverage + publishability.

### 2.3. Большие wall-of-bans prompt'ы плохо работают на Gemma

Повторяли `Gemini`, `Opus` и наши own reviews.

Смысл:

- Gemma плохо держит длинные запретительные списки;
- часть правил начинает игнорироваться;
- остальная часть исполняется уродливо и механически.

Практический итог:

- anti-`посвящ*`, anti-metatext и anti-cliche должны жить в более компактной форме;
- deterministic validation лучше длинной inline prohibition wall.

### 2.4. Patterns полезны, но только если живут в generation-layer

Повторяли и `Opus`, и `Gemini`, и retrospective это подтверждает.

Смысл:

- pattern family нужна для вариативности и human-like writing behavior;
- но pattern layer не должен рано формировать смысл или prose-outline.

Практический итог:

- patterning в extraction/outline repeatedly ломал систему;
- patterning в generation оставалось сильной идеей.

### 2.5. Anti-duplication — это central quality issue

Особенно последовательно это продвигал `Opus`.

Практический итог:

- duplication действительно repeatedly портил тексты;
- anti-dup нужен не только в revise, но и в generation + validation.

## 3. Фаза A: ранний Opus-цикл до pattern-driven redesign

Ключевые документы:

- [smart-update-opus-gemma-event-copy-response-review.md](/workspaces/events-bot-new/docs/reports/smart-update-opus-gemma-event-copy-response-review.md)
- [smart-update-opus-gemma-event-copy-followup-response-review.md](/workspaces/events-bot-new/docs/reports/smart-update-opus-gemma-event-copy-followup-response-review.md)
- [smart-update-opus-gemma-event-copy-pattern-redesign-response-review.md](/workspaces/events-bot-new/docs/reports/smart-update-opus-gemma-event-copy-pattern-redesign-response-review.md)
- [smart-update-opus-gemma-event-copy-pattern-redesign-followup-response-review.md](/workspaces/events-bot-new/docs/reports/smart-update-opus-gemma-event-copy-pattern-redesign-followup-response-review.md)

### 3.1. Что тогда продвигал Opus

- pattern-driven redesign вместо одного flat fact-first prose mold;
- routing между несколькими text patterns;
- idea `why go / why it matters`, но только when justified;
- richer extraction, potentially including `copy_assets`;
- уход от излишне однотипного baseline prose;
- importance of scene/value/person/program distinctions.

### 3.2. Что из этого потом подтвердилось

- pattern idea itself подтвердилась;
- flat baseline generation действительно был слишком шаблонным;
- `why it matters` полезен как optional move, не как mandatory block;
- grouped pattern logic для разных shapes событий — правильное направление.

### 3.3. Что оказалось слабее, чем тогда казалось

- слишком широкий `copy_assets` schema;
- ранняя вера в rich routing without enough guardrails;
- часть abstract pattern names была слишком размыта для Gemma.

## 4. Фаза B: поздний Opus-цикл по quality patch packs и preservation

Ключевые документы:

- [smart-update-opus-gemma-event-copy-quality-first-calibration-response-review.md](/workspaces/events-bot-new/docs/reports/smart-update-opus-gemma-event-copy-quality-first-calibration-response-review.md)
- [smart-update-opus-gemma-event-copy-final-impl-calibration-response-review.md](/workspaces/events-bot-new/docs/reports/smart-update-opus-gemma-event-copy-final-impl-calibration-response-review.md)
- [smart-update-opus-gemma-event-copy-preservation-matrix-response-review.md](/workspaces/events-bot-new/docs/reports/smart-update-opus-gemma-event-copy-preservation-matrix-response-review.md)
- [smart-update-opus-gemma-event-copy-v2-quality-patch-pack-response-review.md](/workspaces/events-bot-new/docs/reports/smart-update-opus-gemma-event-copy-v2-quality-patch-pack-response-review.md)

### 4.1. Что тогда было самым полезным

- `full facts / coverage = P0`;
- migration audit: нельзя терять полезное из текущего runtime;
- preserve baseline strengths:
  - epigraph
  - list logic
  - cleanup
  - policy guardrails
- additive changes safer than full replacement;
- compact quality blocks better than giant rewrite.

### 4.2. Что из этого стало долгоживущим carry

- preservation matrix;
- additive mentality;
- baseline strengths as modules;
- coverage floor in a smarter form.

### 4.3. Где рекомендации Opus потом были miscalibrated

- иногда слишком жёстко тянул в narrowing-first mode;
- иногда переоценивал "safety over prose ambition";
- позже в `v2.12` тянул сильнее к deterministic stripping, чем было полезно.

## 5. Фаза C: ранний Gemini-цикл around v2.3-v2.6

Ключевые документы:

- [smart-update-gemini-event-copy-first-consultation-response-review.md](/workspaces/events-bot-new/docs/reports/smart-update-gemini-event-copy-first-consultation-response-review.md)
- [smart-update-gemini-event-copy-v2-3-dryrun-quality-consultation-response-review.md](/workspaces/events-bot-new/docs/reports/smart-update-gemini-event-copy-v2-3-dryrun-quality-consultation-response-review.md)
- [smart-update-gemini-event-copy-v2-4-regression-consultation-response-review.md](/workspaces/events-bot-new/docs/reports/smart-update-gemini-event-copy-v2-4-regression-consultation-response-review.md)
- [smart-update-gemini-event-copy-v2-5-quality-consultation-response-review.md](/workspaces/events-bot-new/docs/reports/smart-update-gemini-event-copy-v2-5-quality-consultation-response-review.md)
- [smart-update-gemini-event-copy-v2-6-hypotheses-consultation-response-review.md](/workspaces/events-bot-new/docs/reports/smart-update-gemini-event-copy-v2-6-hypotheses-consultation-response-review.md)
- [smart-update-gemini-event-copy-v2-6-dryrun-quality-consultation-response-review.md](/workspaces/events-bot-new/docs/reports/smart-update-gemini-event-copy-v2-6-dryrun-quality-consultation-response-review.md)

### 5.1. Самые полезные советы Gemini в этой фазе

- sparse cases не надо forced дробить заголовками;
- меньше brittle lexical bans, больше positive transformations;
- fact fragmentation — реальный prose bottleneck;
- body должен быть self-contained, а не жить за счёт epigraph;
- anti-bureaucracy нужно формулировать explicitly;
- text quality is shaped by sentence-level rules, not by abstract persona.

### 5.2. Что позже подтвердилось практикой

- `v2.3` действительно выиграл от sparse no-heading instinct;
- positive transformation framing было полезнее тупых bans;
- sentence-level anti-bureaucracy rules later repeatedly remained useful.

### 5.3. Где Gemini тогда переоценивал свои советы

- pre-consolidation не оказалась универсальным fix;
- не все bans вредны, вредны прежде всего brittle lexical bans;
- часть рекомендаций была слишком style-centric и слабее по architecture discipline.

## 6. Фаза D: Gemini-цикл around v2.7-v2.11

Ключевые документы:

- [smart-update-gemini-event-copy-v2-7-hypotheses-consultation-response-review.md](/workspaces/events-bot-new/docs/reports/smart-update-gemini-event-copy-v2-7-hypotheses-consultation-response-review.md)
- [smart-update-gemini-event-copy-v2-7-dryrun-quality-consultation-response-review.md](/workspaces/events-bot-new/docs/reports/smart-update-gemini-event-copy-v2-7-dryrun-quality-consultation-response-review.md)
- [smart-update-gemini-event-copy-v2-8-dryrun-quality-consultation-response-review.md](/workspaces/events-bot-new/docs/reports/smart-update-gemini-event-copy-v2-8-dryrun-quality-consultation-response-review.md)
- [smart-update-gemini-event-copy-v2-8-sanitizer-followup-response-review.md](/workspaces/events-bot-new/docs/reports/smart-update-gemini-event-copy-v2-8-sanitizer-followup-response-review.md)
- [smart-update-gemini-event-copy-v2-9-dryrun-quality-consultation-response-review.md](/workspaces/events-bot-new/docs/reports/smart-update-gemini-event-copy-v2-9-dryrun-quality-consultation-response-review.md)
- [smart-update-gemini-event-copy-v2-10-dryrun-quality-consultation-response-review.md](/workspaces/events-bot-new/docs/reports/smart-update-gemini-event-copy-v2-10-dryrun-quality-consultation-response-review.md)

### 6.1. Самые полезные советы Gemini в этой фазе

- positive transformation > giant prohibition wall;
- label-style contamination реально разрушает фактовый слой;
- anti-quote control нужен;
- nominalization examples полезны для `presentation/project` cases;
- stop-phrase/anti-filler bank нужен как отдельный knowledge layer;
- `Тема: ...` / `Цель: ...` rewrites опасны и synthetic.

### 6.2. Что особенно подтвердилось

- `sanitizer bypass` для synthetic `Тема:` был реальным исправлением;
- anti-quote и anti-bureaucracy rules оказались долгоживущими carries;
- sentence-level prompt language worked better than abstract persona language.

### 6.3. Что repeatedly не подтверждалось

- идея, что ещё один intermediary review/quality gate magically всё починит;
- слишком широкая вера в deterministic phrase governance;
- слишком уверенные советы по routing-law without enough evidence.

## 7. Фаза E: combined Opus + Gemini around v2.12-v2.14

Ключевые документы:

- [smart-update-event-copy-v2-12-consultation-synthesis-2026-03-08.md](/workspaces/events-bot-new/docs/reports/smart-update-event-copy-v2-12-consultation-synthesis-2026-03-08.md)
- [smart-update-event-copy-v2-12-postrun-consultation-synthesis-2026-03-08.md](/workspaces/events-bot-new/docs/reports/smart-update-event-copy-v2-12-postrun-consultation-synthesis-2026-03-08.md)
- [smart-update-event-copy-v2-13-postrun-consultation-synthesis-2026-03-08.md](/workspaces/events-bot-new/docs/reports/smart-update-event-copy-v2-13-postrun-consultation-synthesis-2026-03-08.md)
- [smart-update-event-copy-v2-14-postrun-consultation-synthesis-2026-03-08.md](/workspaces/events-bot-new/docs/reports/smart-update-event-copy-v2-14-postrun-consultation-synthesis-2026-03-08.md)

### 7.1. Где Opus и Gemini в этой фазе реально сошлись

- `full-floor normalization` правильнее старого subset+merge;
- same-model full editorial pass даёт мало пользы;
- semantic core должен оставаться LLM-first;
- patterning полезно, но не должно создавать новый semantic drift;
- outline должен быть structural-only;
- deterministic support layer хорош только как support.

### 7.2. Где они расходились

`Opus` чаще тянул:

- к меньшему числу moving parts;
- к более жёсткому content-preservation;
- местами к deterministic stripping.

`Gemini` чаще тянул:

- к syntax-level prose rules;
- к anti-bureaucracy / anti-filler focus;
- к более мягким positive transformations вместо bans.

### 7.3. Что из combined phase стало основой current direction

- `v2.12+` architecture shift;
- `v2.13` cleaner quality-safe branch;
- no prose-outline;
- no full editorial pass;
- need for better prompt decomposition.

## 8. Late-stage text-quality consultation on v2.15

Ключевые документы:

- [smart-update-gemini-event-copy-v2-15-text-quality-consultation-review-2026-03-08.md](/workspaces/events-bot-new/docs/reports/smart-update-gemini-event-copy-v2-15-text-quality-consultation-review-2026-03-08.md)
- [smart-update-opus-event-copy-v2-15-2-prompt-profiling-review-2026-03-08.md](/workspaces/events-bot-new/docs/reports/smart-update-opus-event-copy-v2-15-2-prompt-profiling-review-2026-03-08.md)

### 8.1. Что полезного принёс поздний Gemini

- syntax-level prose rules instead of abstract "journalist" persona;
- stronger lead rules;
- semantic heading rules;
- epigraph/list gating must be meaningful;
- stop-phrase bank for Russian AI-prose;
- anti-bureaucracy as explicit layer.

### 8.2. Что полезного принёс поздний Opus

- dynamic prompt assembly;
- register over persona;
- shrink the main prose prompt;
- stop-phrase enforcement should not be duplicated inside generation;
- repair must receive facts;
- planning should be structural-only and very small.

### 8.3. Combined takeaway for `2.15.2`

- not one giant generation prompt;
- not five always-on LLM calls either;
- rather `2-3` core LLM calls with tiny/hybrid planning and deterministic validation;
- patterns stay in generation;
- prompts become shorter, more role-separated, and more concrete.

### 8.4. Что добавили уже самые поздние `2.15.3` консультации

Ключевые документы:

- [smart-update-opus-v2-15-3-prompt-pack-review-2026-03-08.md](/workspaces/events-bot-new/docs/reports/smart-update-opus-v2-15-3-prompt-pack-review-2026-03-08.md)
- [smart-update-gemini-event-copy-v2-15-3-dryrun-text-consultation-review-2026-03-08.md](/workspaces/events-bot-new/docs/reports/smart-update-gemini-event-copy-v2-15-3-dryrun-text-consultation-review-2026-03-08.md)
- [smart-update-opus-gemma-event-copy-v2-15-3-step-profile-event-2673-review-2026-03-08.md](/workspaces/events-bot-new/docs/reports/smart-update-opus-gemma-event-copy-v2-15-3-step-profile-event-2673-review-2026-03-08.md)

Новый useful carry:

- `Opus` хорошо соблюдает Gemma-research carry, если brief уже собран дисциплинированно, но всё равно требует локальной калибровки schema/contracts;
- поздний `Gemini` особенно полезен не по architecture, а по prose defects:
  - weak openings
  - AI-cliches
  - heading quality
  - agenda recap
  - over-abstract wording;
- step profiling показал, что для project/presentation кейсов проблема часто живёт не в одном prompt, а в связке:
  - weak normalized fact hierarchy
  - overly literal deterministic lead selection
  - obedient but wrong generation.

Практический итог:

- следующий consultation/use-case для внешних моделей должен опираться не только на final text, но и на step-level trace;
- `Opus` стоит просить переписывать prompts по шагам, а не предлагать regex scaffolding;
- `Gemini` стоит продолжать использовать как critic именно по naturalness, headings, openings, clichés и micro-style rules.

## 9. Что из всех рекомендаций уже можно считать stable carry into implementation

Это condensed list того, что уже не надо каждый раз переоткрывать.

### 9.1. Text-quality carries

- strong direct lead;
- no generic headings;
- no question-headings;
- no `посвящ*`;
- no `лекция расскажет / мероприятие будет интересно`;
- no promo filler;
- no label-style prose;
- no decorative formatting;
- self-contained body;
- professional but non-bureaucratic register.

### 9.2. Prompt-design carries

- smaller prompts;
- role-separated prompts;
- syntax-level rules;
- positive transformations where possible;
- deterministic validation for lexical bans;
- anti-duplication in both prompt and validation.

### 9.3. Architecture carries

- LLM-first semantic core;
- full-floor normalization;
- pattern layer in generation;
- tiny structural planning only;
- no prose-outline;
- no full editorial pass;
- repair only when validator says so;
- support-layer determinism only.

## 10. Что стоит считать repeatedly weak or dangerous

- regex-first semantic shaping;
- giant routing tree;
- giant inline ban-list inside generation prompt;
- same-model full rewrite pass;
- prose-like planning before generation;
- treating one metric as sole truth;
- chasing local fixes that do not scale beyond the 5-case set.

## 11. Bottom line

Если нужен один документ, который держать перед глазами перед следующими `2.15.x` раундами, то это именно он.

Он нужен для трёх вещей:

- не потерять полезное из старых Opus/Gemini rounds;
- не повторять уже опровергнутые идеи;
- иметь под рукой не только design brief, но и evidence-backed knowledge base по prompt and architecture decisions.
