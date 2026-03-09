# Smart Update Gemma Event Copy V2.15.2 Prompt Context

Дата: 2026-03-08

Основание:

- [v2.15.2 design brief](/workspaces/events-bot-new/docs/reports/smart-update-gemma-event-copy-v2-15-2-design-brief-2026-03-08.md)
- [v2.15.2 dry-run report](/workspaces/events-bot-new/docs/reports/smart-update-opus-gemma-event-copy-pattern-dry-run-v2-15-2-5-events-2026-03-08.md)
- [v2.15.2 review](/workspaces/events-bot-new/docs/reports/smart-update-opus-gemma-event-copy-pattern-dry-run-v2-15-2-review-2026-03-08.md)
- Experimental harness: [experimental_pattern_dryrun_v2_15_2_2026_03_08.py](/workspaces/events-bot-new/artifacts/codex/experimental_pattern_dryrun_v2_15_2_2026_03_08.py)

## 1. Задача round

`2.15.2` должен был:

- оставить semantic core `LLM-first`;
- разложить flow на короткие role-separated steps;
- вернуть pattern variability в generation-layer;
- не повторить regex-heavy drift;
- улучшить и coverage, и чистоту итогового текста.

## 2. Pipeline

### Step A. `normalize_floor`

Тип:

- LLM JSON pass

Вход:

- `normalization_input_facts`
- `shape_name`
- `issue_hints`

Задача:

- переписать raw narrative/intent facts в короткие publishable `clean_fact`;
- убрать service/logistics/meta framing;
- не терять names / program items / project agenda blocks.

Особенности:

- shape-specific positive transformations;
- запрет на `Тема: ...`, `Идея: ...`, `Цель: ...`;
- запрет на `посвящ*`;
- для presentation cases трансформация вида:
  - `на презентации расскажут о задачах платформы` -> `задачи платформы`
- для lecture cases сохраняется grouped people block, а не отдельный fact на каждого.

### Step B. `shape_and_format_plan`

Тип:

- deterministic by default
- tiny LLM JSON planner only for richer ambiguous cases

Output:

- `pattern`
- `use_epigraph`
- `use_headings`
- `use_list_block`
- `blocks[]` only with `fact_ids`

Execution patterns в runtime:

- `scene_led`
- `quote_led`
- `person_led`
- `program_led`

Planner принципиально structural-only:

- никакого prose;
- никаких heading texts;
- никаких focus notes.

### Step C. `generate_description`

Тип:

- main prose LLM pass

Prompt assembly:

- короткий core rule block;
- pattern-specific structural hint;
- optional formatting blocks только если они активны в plan;
- numbered `facts_text_clean`.

Ключевые generation rules:

- первое предложение — конкретный субъект/факт/действие;
- основной текст должен быть самодостаточным;
- не использовать `посвящ`, `расскажет`, `расскажут`, `представит`, `обсудят`, `не просто ..., а ...`;
- не повторять одну мысль в разных формулировках;
- без CTA, логистики, билетов, возраста;
- для compact branch — без `###`, 1-2 абзаца;
- headings разрешаются только если plan их включил;
- epigraph разрешается только при source-backed quote.

Generation не получает длинную стену ban-rules; enforcement должен добиваться коротким prompt + downstream validation.

### Step D. `validate_description`

Тип:

- deterministic checks

Проверяет:

- forbidden reasons;
- missing facts;
- policy issues;
- epigraph/body conflicts;
- некоторые heading issues.

### Step E. `targeted_repair`

Тип:

- optional LLM text pass

Принцип:

- исправить только validated issues;
- опираться на `facts_text_clean`;
- не делать full rewrite без необходимости.

## 3. Что подтвердилось по `v2.15.2`

- `full-floor normalization` реально лучше старого `subset extraction`;
- dynamic generation prompt в целом помог coverage и hygiene;
- tiny planner не сломал весь round и дал полезный structural bias для rich cases;
- `2673` в этом round впервые нормально называет событие презентацией проекта.

## 4. Что сейчас явно не работает

### 4.1. Quote extraction too loose

`2687` показал, что quote extractor может взять нерелевантную цитату из соседнего digest item, если в source text есть другие афиши/сеансы.

Следствие:

- ложный `has_direct_quote`;
- ложный `quote_led`;
- ложный epigraph.

### 4.2. Plan enforcement still incomplete

`2660` показал:

- `use_headings=false`
- branch=`compact_fact_led`

Но финальный текст всё равно сгенерировал headings.

Значит:

- generation prompt пока не всегда исполняет format gate;
- validation не ловит все `plan -> output` mismatch cases.

### 4.3. Quote block policy still too permissive

`2734` показал, что source-backed fragment ещё не гарантирует сильный editorial epigraph.

Проблема не в наличии кавычек как таковых, а в том, что:

- fragment может быть технически true;
- но риторически слабым или случайным.

### 4.4. Presentation prose still drifts into agenda sections

`2673` уже лучше factual framing-wise, но prose ещё частично звучит как agenda unpacking:

- `устройство и цели платформы`
- `формат встречи`

То есть project case всё ещё тянет к explanatory block structure.

## 5. Текущий practical takeaway

`2.15.2` полезен как base branch для следующего refinement round, потому что он:

- лучше baseline по aggregate coverage;
- лучше `v2.13` и `v2.14` по `missing + forbidden`;
- всё ещё требует text-quality work именно на:
  - quote gating
  - epigraph discipline
  - heading quality
  - presentation prose naturalness

Главное ограничение:

- дальнейшие улучшения не должны превращаться в regex-driven semantic core;
- deterministic layer остаётся support/validation, а не автором смысла.
