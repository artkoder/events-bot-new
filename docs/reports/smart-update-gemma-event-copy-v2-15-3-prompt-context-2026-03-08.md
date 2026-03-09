# Smart Update Gemma Event Copy V2.15.3 Prompt Context

Дата: 2026-03-08

Основание:

- [v2.15.3 design brief](/workspaces/events-bot-new/docs/reports/smart-update-gemma-event-copy-v2-15-3-design-brief-2026-03-08.md)
- [Opus prompt pack](/workspaces/events-bot-new/docs/reports/smart-update-opus-v2-15-3-prompt-pack-2026-03-08.md)
- [Opus prompt pack review](/workspaces/events-bot-new/docs/reports/smart-update-opus-v2-15-3-prompt-pack-review-2026-03-08.md)
- Experimental harness: [experimental_pattern_dryrun_v2_15_3_2026_03_08.py](/workspaces/events-bot-new/artifacts/codex/experimental_pattern_dryrun_v2_15_3_2026_03_08.py)
- [v2.15.3 dry-run report](/workspaces/events-bot-new/docs/reports/smart-update-opus-gemma-event-copy-pattern-dry-run-v2-15-3-5-events-2026-03-08.md)

## 1. Задача round

`2.15.3` был уже не narrow bugfix round, а full prompt-pack repack для Gemma внутри той же `LLM-first` architecture.

Цель:

- не менять semantic core на regex-first;
- перепаковать все main prompts под Gemma-specific discipline;
- сохранить сильные стороны `2.15.2` по coverage/hygiene;
- уменьшить:
  - false quote / false epigraph;
  - plan-to-output leakage;
  - agenda-like project prose;
  - giant-ban-list drift.

## 2. Pipeline

### Step A. `normalize_floor`

Тип:

- LLM JSON pass

Форма prompt:

- self-contained
- sectioned:
  - `РОЛЬ`
  - `ПРАВИЛА`
  - `ВЫХОД`
  - `ФАКТЫ И КОНТЕКСТ`

Задача:

- переписать raw narrative / intent facts в короткие publishable `clean_fact`;
- убрать service / link / logistics мусор;
- сохранять names, grouped people blocks, program payload, project payload;
- для presentation/project cases убирать frame `расскажут / представят / обсудят` и оставлять предмет разговора;
- для lecture/person-rich cases не дробить grouped people block без необходимости.

Что изменилось против `2.15.2`:

- prompt стал более секционированным и self-contained;
- positive transformations встроены прямо в step;
- hints переписаны в action-oriented форму;
- quote-related metadata начали влиять downstream уже через более жёсткий contract.

### Step B. `shape_and_format_plan`

Тип:

- deterministic / tiny-hybrid

Задача:

- выбрать:
  - `pattern`
  - `use_epigraph`
  - `use_headings`
  - `use_list_block`
- сделать это как structural plan, а не prose outline.

Runtime execution patterns:

- `scene_led`
- `quote_led`
- `person_led`
- `program_led`

Что изменилось:

- prompt переписан в self-contained JSON planner;
- schema fields получили semantic descriptions;
- planner теперь опирается на:
  - `has_verified_quote`
  - `quote_speaker`
  - `shape_name`
  - `facts_count`
- `presentation_project` cases по умолчанию сильнее склоняются к `use_headings=false`;
- эпиграф не должен включаться без verified quote.

### Step C. `generate_description`

Тип:

- main prose LLM pass

Форма prompt:

- self-contained
- sectioned:
  - `РОЛЬ`
  - `ЗАДАЧА`
  - `ОБЩИЕ ПРАВИЛА`
  - `ANTI-PATTERNS`
  - `СТОП-ФРАЗЫ`
  - `OUTPUT`
  - `КОНТЕКСТ`
  - `FACTS_TEXT_CLEAN`

Dynamic assembly:

- один core block;
- одна pattern card;
- только активные optional blocks:
  - headings
  - epigraph
  - list block
  - shape-specific hints.

Gemma-facing execution logic:

- pattern library в runtime узкая:
  - `scene_led`
  - `quote_led`
  - `person_led`
  - `program_led`
- editorial richness остаётся за prompt design, а не за раздуванием runtime router.

Ключевые правила генерации:

- lead должен сразу называть субъект / событие / действие;
- body должен быть self-contained;
- нельзя выдумывать цитаты;
- нельзя использовать loose meta-openers;
- запрещены weak/bureaucratic openers и stop-phrases;
- headings и epigraph разрешены только если это включено в plan;
- для project/presentation cases добавлен отдельный execution note;
- prose должен быть короче, конкретнее и менее template-heavy, чем в baseline.

### Step D. `validate_description`

Тип:

- deterministic support layer

Проверяет:

- forbidden reasons;
- missing facts;
- policy issues;
- plan/format mismatches.

Что изменилось:

- validation теперь смотрит на `format_plan`, а не только на branch heuristics;
- запрещённые headings / list / epigraph оцениваются относительно planner contract;
- enforcement остаётся support-layer, а не semantic author.

### Step E. `targeted_repair`

Тип:

- optional LLM pass

Форма prompt:

- self-contained
- sectioned
- issue-specific

Задача:

- не переписать текст заново;
- исправить только validated issues;
- опираться на `facts_text_clean` и текущий текст;
- не сочинять новые формулировки без опоры на факты.

## 3. Что было явно взято из Gemma research

Из deep research и последующей калибровки были сознательно встроены:

- self-contained prompts без надежды на скрытую `system`-магии;
- короткая multi-step chain вместо giant prompt;
- sectioned packaging:
  - `ROLE / RULES / OUTPUT / FACTS`;
- per-step decoding discipline:
  - normalization / planner / repair более детерминированы;
  - generation немного свободнее;
- короткие positive transformations вместо длинной стены few-shot примеров;
- structure-first JSON contract для planner / normalize steps.

## 4. Что было явно взято из Opus prompt pack

Из strict `Opus` round были приняты и адаптированы:

- full prompt repack across all steps;
- dynamic prompt assembly;
- pattern cards;
- facts-backed repair;
- stronger epigraph discipline;
- sharper role separation between steps.

Что было принято только с поправками:

- richer quote provenance;
- typed/support metadata;
- compact stop-phrase bank;
- optional formatting modules вместо unconditional formatting.

## 5. Что показал сам `2.15.3` dry-run

`2.15.3` доказал, что repack действительно повлиял на систему:

- `2745` заметно улучшился;
- `2687` больше не сломан false quote / false epigraph leakage как в `2.15.2`;
- format discipline стала сильнее;
- forbidden total снизился по ряду старых проблем.

Но одновременно round показал remaining blockers:

- `2660` регресснул в сторону объясняющего и более общего текста;
- `2734` потерял часть program sharpness;
- `2687` всё ещё пропускает `посвящ*`;
- `2673` всё ещё explanation-heavy и не дотягивает до хорошей project-presentation prose.

## 6. Practical takeaway

`2.15.3` — это не новый architecture pivot, а первый полноценный Gemma-specific repack внутри `2.15.x`.

Его фактическая роль:

- подтвердить, что prompt-pack redesign влияет на результат;
- отделить prompt discipline gains от старых architecture gains;
- понять, какие remaining проблемы ещё сидят в prompt contracts, а какие уже в fact quality / planning / repair interaction.

Главное ограничение остаётся прежним:

- deterministic layer поддерживает и валидирует;
- смысл, стиль и editorial form должны оставаться в `LLM-first` path.
