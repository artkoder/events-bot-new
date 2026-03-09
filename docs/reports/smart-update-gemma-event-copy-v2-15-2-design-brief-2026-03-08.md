# Smart Update Gemma Event Copy V2.15.2 Design Brief

Дата: 2026-03-08

Основание:

- [v2.15 design brief](/workspaces/events-bot-new/docs/reports/smart-update-gemma-event-copy-v2-15-design-brief-2026-03-08.md)
- [external consultation retrospective](/workspaces/events-bot-new/docs/reports/smart-update-event-copy-external-consultation-retrospective-2026-03-08.md)
- [Opus + Gemini master retrospective](/workspaces/events-bot-new/docs/reports/smart-update-event-copy-opus-gemini-recommendations-master-retrospective-2026-03-08.md)
- [Gemini text-quality review for v2.15](/workspaces/events-bot-new/docs/reports/smart-update-gemini-event-copy-v2-15-text-quality-consultation-review-2026-03-08.md)
- [Opus prompt profiling review for 2.15.2](/workspaces/events-bot-new/docs/reports/smart-update-opus-event-copy-v2-15-2-prompt-profiling-review-2026-03-08.md)
- [Gemma deep research impact on v2.15.2](/workspaces/events-bot-new/docs/reports/smart-update-gemma-deep-research-impact-on-v2-15-2-2026-03-08.md)

## 1. Цель `2.15.2`

`2.15.2` — это не новый redesign, а refinement round архитектуры `2.15`.

Главная идея:

- разложить generation flow на достаточно маленькие, атомарные задачи;
- не перегружать один prompt сразу стилем, coverage, patterns, headings, epigraphs, lists, stopwords и repair rules;
- оставить Gemma на каждом шаге только одну главную обязанность.

Практическая оговорка после prompt-profiling `Opus`:

- атомарность не должна превращаться в бессмысленный extra-call budget;
- поэтому `2.15.2` проектируется не как "обязательно 5 LLM вызовов", а как `2-3` основных LLM-вызова в happy path;
- classification и format gating должны быть cheap/hybrid там, где это можно сделать без semantic drift.

Canonical recommendation knowledge base for this round:

- [smart-update-event-copy-opus-gemini-recommendations-master-retrospective-2026-03-08.md](/workspaces/events-bot-new/docs/reports/smart-update-event-copy-opus-gemini-recommendations-master-retrospective-2026-03-08.md)

## 2. Почему нужен атомарный multi-step flow

Практический диагноз по Gemma:

- на длинных промптах с большим числом правил она начинает исполнять их unevenly;
- негативные ограничения конкурируют между собой;
- сложные style+coverage+formatting prompts легко дают:
  - bureaucratic drift;
  - partial instruction loss;
  - new template collapse.

Следствие:

- лучше несколько маленьких prompt passes, чем один "всемогущий" prompt;
- но число шагов должно быть ограничено и role-separated, чтобы не потерять TPM discipline.

## 3. `2.15.2` pipeline

### Step A. `normalize_floor`

Задача:

- из всей relevant fact base сделать clean factual floor;
- убрать metatext;
- сохранить names / program / agenda / subject.

Output:

- `normalized_facts[]`
- `has_direct_quote`
- `quote_text`
- `quote_speaker`

Один prompt, один JSON contract.
Никакого prose.

### Step B. `shape_and_format_plan`

Задача:

- дешёво и структурно определить:
  - какой pattern реально допустим;
  - нужны ли `epigraph`, `headings`, `list_block`;
  - нужно ли вообще отдельное LLM-planning вмешательство.

Default mode:

- `quote` / `headings` / `list` gates считаются детерминистически;
- pattern candidates сначала сужаются rule-based;
- tiny LLM planner вызывается только если после cheap gating остаётся реальная неоднозначность rich case.

Output:

```json
{
  "pattern": "scene_led | quote_led | person_led | program_led",
  "use_epigraph": true,
  "use_headings": false,
  "use_list_block": true,
  "blocks": [
    {"kind": "lead", "fact_ids": [1,2]},
    {"kind": "section", "fact_ids": [3,4,5]}
  ]
}
```

Важно:

- никакого free-text outline;
- никакого prose;
- only structural decisions;
- `blocks` optional и нужны только для richer cases.

### Step C. `generate_description`

Задача:

- написать сам текст по:
  - `normalized_facts`
  - выбранному `pattern`
  - `format_plan`

Prompt responsibilities:

- lead quality;
- section/heading quality;
- epigraph/list use;
- anti-cliche;
- anti-bureaucracy;
- prose rhythm.

Это единственный главный prose step.

Ключевой принцип `2.15.2`:

- prompt собирается динамически в Python;
- в него попадают только активные blocks;
- inline wall-of-bans не дублирует deterministic validation;
- pattern должен влиять на structure, а не просто передаваться как строка "на удачу".

## 3.1. Editorial patterns vs execution patterns

Для `2.15.2` нужно явно развести два уровня pattern logic.

### Editorial pattern vocabulary

Это более богатый человеческий слой, который нужен нам как design language:

- `scene`
- `quote`
- `person`
- `program`
- `project`
- `theme`
- optional `why_it_matters`

Этот слой полезен:

- для brainstorm/design;
- для анализа качества текста;
- для будущего расширения;
- для понимания, как реально варьируется human editorial writing.

### Execution patterns for Gemma

Gemma в runtime не должна выбирать между слишком близкими abstract cards.

В execution-layer `2.15.2` использует более короткий и sharper set:

- `scene_led`
- `quote_led`
- `person_led`
- `program_led`

Причина:

- эти 4 pattern дают Gemma более чёткие structural contracts;
- они меньше overlap-ят;
- они лучше подходят для короткого atomic planning/generation flow.

### Mapping principle

Важный architectural principle:

- editorial vocabulary не теряется;
- но в runtime rich editorial intent при необходимости мапится на компактный execution pattern set;
- то есть `project` и `theme` не исчезают как редакторские идеи, а чаще мапятся в `scene_led`;
- `why_it_matters` не является отдельным execution pattern и живёт как optional move внутри generation only when justified.

Итог:

- мы не теряем вариативность человеческой подачи;
- но и не заставляем Gemma работать с чрезмерно расплывчатой pattern taxonomy.

### Step D. `validate_description`

Задача:

- deterministic checks:
  - forbidden phrases
  - generic headings
  - heading/content mismatch heuristics
  - duplicate sentences
  - service leakage
  - epigraph/body duplication

Output:

- compact issue list only.

### Step E. `targeted_repair`

Запускается только если Step D реально нашёл issue.

Задача:

- локально исправить конкретные проблемы;
- не переписывать whole body.
- опираться на исходные facts, а не на "ремонт по vibes".

Repair types:

- forbidden phrase cleanup;
- heading cleanup;
- duplicate sentence cleanup;
- missing named/program item insertion if clearly validated.

## 4. Почему это TPM-friendly

Промптов больше, чем в single-pass flow, но они:

- короче;
- проще;
- устойчивее;
- меньше конкурируют за instruction budget.

Практически:

- A — основной semantic-normalization prompt;
- B — mostly deterministic / tiny-hybrid planner;
- C — основной prose prompt;
- D mostly deterministic;
- E optional and narrow.

То есть median event не обязательно получит все 5 LLM calls:

- sparse events often: A -> C -> D
- normal events often: A -> B(optional) -> C -> D
- rich/problem events: A -> B(optional) -> C -> D -> E

## 5. Что должно войти в brief как selected external-model takeaways

### 5.1. From both `Opus` and `Gemini`

- `full-floor normalization`
- no wall-of-bans prompts
- no prose-outline
- no full editorial pass
- deterministic support only
- preserve baseline strengths like epigraph/list capability

### 5.2. Strong Gemini carries

- syntax-level prose rules
- stop-phrase bank
- stronger lead rules
- semantic heading rules
- anti-filler / anti-evaluation layer
- fewer abstract personas, more concrete sentence-level guidance

### 5.3. Strong Opus carries

- anti-duplication as P0
- content-preservation discipline
- prompt simplification
- structural-only planning
- dynamic prompt assembly
- register over persona
- quote metadata in normalization
- repair must receive facts

### 5.4. Gemma deep-research carries

- each independent Gemma call must be self-contained; do not rely on hidden `system` behavior
- JSON extraction / planning / narrow repair should use deterministic decoding defaults
- main prose generation may use a slightly more creative decoding profile, but only there
- planner and normalization must stay structure-first, without chain-of-thought output
- short positive transformations and compact anti-pattern examples are preferred over long few-shot walls
- prompts should be clearly sectioned into short labeled blocks rather than one dense instruction paragraph
- JSON contracts should carry semantic guidance, not only bare types, especially on tricky planner fields
- repetition-control may be worth testing as an optional prose-quality lever, but not as a new core dependency
- if prompt-only refinement plateaus, fine-tuning becomes a legitimate next step rather than a taboo

## 6. Prompt pack that should be profiled with `Opus`

`Opus` нужно показывать уже не весь history, а compact `2.15.2` pack:

1. `normalize_floor` prompt
2. `shape_and_format_plan` prompt
3. `generate_description` prompt
4. `targeted_repair` prompt
5. stop-phrase / stopword layer
6. pattern cards

Что просить у `Opus`:

- profile prompt-by-prompt failure risks for Gemma;
- point out instruction overload risks;
- rewrite prompts toward stronger compliance;
- improve positive transformations;
- improve lead / heading / epigraph / list behavior;
- keep architecture feedback secondary to prompt quality.

## 7. Prompt-design defaults for `2.15.2`

### 7.1. What must be true

- each prompt has one job;
- each prompt is self-contained for Gemma and does not rely on hidden `system` state;
- each prompt should use clear labeled sections instead of one dense instruction blob;
- core prose prompt must stay compact;
- optional formatting rules are injected only when active;
- deterministic validation enforces lexical bans and generic-heading bans;
- repair gets facts plus issue list;
- positive transformations dominate over giant negative enumerations;
- JSON / planner / repair prompts must prefer deterministic decoding defaults;
- planner and normalization outputs must stay structural, not explanatory;
- schema-bearing prompts should explain tricky fields semantically, not only by type.

### 7.2. What should be avoided

- one giant generation prompt with every rule always present;
- duplicated stop-phrase lists in both prompt and validator;
- free-text outline or focus-note prose before generation;
- same-model full editorial rewrite;
- semantic regex rewrites as core behavior.

## 7.3. Pattern defaults

- pattern richness для design-level thinking может быть выше, чем для runtime;
- Gemma execution-layer должен оставаться компактным;
- execution patterns должны различаться по structure, а не только по названию или "настроению";
- optional editorial moves не должны превращаться в отдельные noisy runtime branches без strong evidence.

## 8. Success bar for `2.15.2`

`2.15.2` should:

- keep `LLM-first` core;
- preserve the pattern-library idea;
- make prompt responsibilities smaller and clearer;
- improve compliance on anti-cliche / anti-bureaucracy;
- preserve epigraph/list strengths;
- stay safer than old pattern-family overfit;
- keep happy-path LLM usage close to `2-3` calls, not "5 calls by default";
- keep the main prose prompt compact enough that Gemma does not start dropping rules unpredictably;
- keep `normalize_floor` and planner JSON validity rate as close to `100%` as practical;
- keep repair loop rate low enough that repair stays an exception, not the main path;
- keep plan-compliance high enough that `use_headings/use_epigraph/use_list_block` are actually enforced in final output.

## 9. Current architectural choice after Gemini + Opus reviews

`2.15.2` takes a combined position:

- `Gemini` is right that text quality needs more concrete syntax-level guidance and fewer brittle bans;
- `Opus` is right that the main failure mode is prompt overload, especially in generation;
- therefore the next iteration should not choose between "one huge prompt" and "five always-on calls";
- it should use a small number of atomic, role-separated steps with hybrid gating where LLM planning is not always necessary.

Working shape:

- Step A = LLM normalization
- Step B = deterministic / tiny-hybrid shape and format plan
- Step C = dynamic generation prompt
- Step D = deterministic validation
- Step E = narrow repair with facts

## 10. Bottom line

`2.15.2` should be the first version where:

- patterns are back;
- prompts are smaller;
- Gemma gets one job at a time;
- optional boosters cannot destroy the base output;
- and the system is still scalable enough for many-thousands-of-posts reality.
