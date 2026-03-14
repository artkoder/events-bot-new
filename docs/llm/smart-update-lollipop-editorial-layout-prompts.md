# Smart Update Lollipop Editorial Layout Prompts

Канонический `v1` контракт для следующего downstream family: `editorial.layout.plan`.

## Stage shape

```text
editorial.layout.precompute
-> editorial.layout.plan.v1
-> editorial.layout.validate
```

Где:
- `precompute` — детерминированный код;
- `plan.v1` — единственный `Gemma` call;
- `validate` — детерминированный код.

## Deterministic precompute

До вызова `Gemma` код должен вычислить:

- `density`: `minimal | standard | rich`
- `has_long_program`: `true | false`
- `non_logistics_total`: integer
- `body_cluster_count`: integer
- `body_block_floor`: integer
- `multi_body_split_recommended`: `true | false`
- `title_is_bare`: `true | false`
- `title_needs_format_anchor`: `true | false`
- `allow_semantic_headings`: `true | false`
- `heading_guardrail_recommended`: `true | false`

Идея:
- `Gemma` не должна считать количество фактов или длину списка;
- она должна только разложить уже известный state по блокам.
- title opacity и право на semantic headings тоже должны рождаться детерминированно, а не угадываться моделью каждый раз заново.
- Любые heading labels в примерах ниже — это только illustrative model outputs; runtime code не должен детерминированно подставлять такие подписи сам.

## Output schema for `editorial.layout.plan.v1`

```json
{
  "title_strategy": "keep|enhance",
  "title_hint_ref": "EC02",
  "blocks": [
    {
      "role": "lead|body|program|infoblock",
      "fact_refs": ["EC02", "FL02"],
      "style": "narrative|list|structured",
      "heading": null
    }
  ]
}
```

## `editorial.layout.plan.v1`

```text
ROLE
You do one small step: editorial.layout.plan.v1.

GOAL
Plan the structure of the final event text using the already-prioritized fact pack.

INPUT
- event title
- event type
- lead_fact_id
- lead_support_id
- density (precomputed)
- has_long_program (precomputed)
- non_logistics_total (precomputed)
- body_cluster_count (precomputed)
- body_block_floor (precomputed)
- multi_body_split_recommended (precomputed)
- title_is_bare (precomputed)
- title_needs_format_anchor (precomputed)
- allow_semantic_headings (precomputed)
- heading_guardrail_recommended (precomputed)
- all_fact_ids (precomputed checklist)
- prioritized facts JSON

OUTPUT
Return only JSON:
{
  "title_strategy": "keep|enhance",
  "title_hint_ref": "fact_id or null",
  "blocks": [
    {
      "role": "lead|body|program|infoblock",
      "fact_refs": ["fact_id"],
      "style": "narrative|list|structured",
      "heading": null
    }
  ]
}

RULES
1. Lead block is always first. Its first fact_ref must be lead_fact_id. If lead_support_id is not empty, put it second in the lead block.
2. Infoblock is always last. It contains all and only logistics_infoblock facts. Its style must be "structured". Heading must be null.
3. If has_long_program is true, create a separate program block with style "list" for the program_list facts.
4. If density is minimal and allow_semantic_headings is false, do not create headings. Use only lead and infoblock unless a program block is required by rule 3.
5. Every fact_id from the input must appear exactly once across all blocks. Do not omit facts and do not repeat facts.
6. Use heading only on body or program blocks, only when allow_semantic_headings is true, and only when the heading is short and factual.
7. If title_needs_format_anchor is true, keep the lead focused on event format/action clarity and move film/project/background detail into body blocks.
8. If title_needs_format_anchor is true and there is a semantic shift after the lead, you may create a body heading, but choose it from the facts yourself; do not fall back to generic filler headings.
9. title_strategy = "enhance" only when title_is_bare is true and one fact clearly makes the title more informative. If you choose enhance, title_hint_ref must contain exactly one fact_id. Otherwise use "keep" and null.
10. Use all_fact_ids as the explicit checklist for exact once-only coverage across the blocks.
11. Order fact_refs inside body blocks by importance: high before medium, medium before low.
12. Use event_type to choose heading vocabulary that sounds native to the material. Screening, lecture, concert, and exhibition cases should not all receive the same generic heading logic.
13. If body_block_floor = 2, plan at least two post-lead narrative sections unless a separate program block already carries one of those semantic clusters.
14. If multi_body_split_recommended is true, do not collapse all post-lead facts into one long body blob when event/core facts and forward/context/people facts clearly separate.
15. If non_logistics_total >= 4 and there is more than one thematic cluster after the lead, prefer at least one heading or split block instead of one long undifferentiated body blob.
16. If heading_guardrail_recommended is true, heading recovery is expected unless the material is genuinely one-theme and sparse.
17. Do not write prose. Do not paraphrase facts. Only plan structure.

EXAMPLE: screening
- if lead clarifies that this is a screening, a body heading like `О фильме` or `Контекст` is acceptable when later facts switch from event framing to synopsis/cast.

EXAMPLE: lecture
- if the lecture has one opening fact about the topic and several follow-up facts about names, schools, or contributions, a short heading like `О чем лекция` or `В центре внимания` is acceptable.

AVOID
- generic filler headings like `О событии`, `Подробности`, `Основная идея`
- collapsing all non-logistics facts into one body blob when there are clear thematic shifts
EXAMPLE 1
Input:
- title: Собакусъел
- event_type: presentation
- lead_fact_id: FL01
- lead_support_id: FL02
- density: rich
- has_long_program: true
- title_needs_format_anchor: true
- allow_semantic_headings: true

Output:
{
  "title_strategy": "enhance",
  "title_hint_ref": "FL01",
  "blocks": [
    {
      "role": "lead",
      "fact_refs": ["FL01", "FL02"],
      "style": "narrative",
      "heading": null
    },
    {
      "role": "body",
      "fact_refs": ["EC01", "EC03", "FL03", "FL04", "SC01", "SC02"],
      "style": "narrative",
      "heading": "О проекте"
    },
    {
      "role": "program",
      "fact_refs": ["PL01"],
      "style": "list",
      "heading": "В программе"
    },
    {
      "role": "infoblock",
      "fact_refs": ["LG01", "LG02", "LG03"],
      "style": "structured",
      "heading": null
    }
  ]
}

EXAMPLE 2
Input:
- title: Хоровая вечеринка «Праздник у девчат»
- event_type: party
- lead_fact_id: EC01
- lead_support_id: PR01
- density: standard
- has_long_program: true

Output:
{
  "title_strategy": "keep",
  "title_hint_ref": null,
  "blocks": [
    {
      "role": "lead",
      "fact_refs": ["EC01", "PR01"],
      "style": "narrative",
      "heading": null
    },
    {
      "role": "program",
      "fact_refs": ["PL01"],
      "style": "list",
      "heading": null
    },
    {
      "role": "infoblock",
      "fact_refs": ["LG01", "LG02", "LG03", "LG04"],
      "style": "structured",
      "heading": null
    }
  ]
}

EXAMPLE 3
Input:
- title: Киноклуб: «Последнее метро»
- event_type: screening
- lead_fact_id: SC01
- lead_support_id: PR01
- density: standard
- has_long_program: false
- title_needs_format_anchor: true
- allow_semantic_headings: true

Output:
{
  "title_strategy": "keep",
  "title_hint_ref": null,
  "blocks": [
    {
      "role": "lead",
      "fact_refs": ["SC01", "PR01"],
      "style": "narrative",
      "heading": null
    },
    {
      "role": "body",
      "fact_refs": ["SC02", "SC03"],
      "style": "narrative",
      "heading": "О фильме"
    },
    {
      "role": "infoblock",
      "fact_refs": ["LG01", "LG02"],
      "style": "structured",
      "heading": null
    }
  ]
}
```

## Deterministic validation

После `Gemma` код должен проверить:

- все входные `fact_id` использованы ровно один раз;
- первый block = `lead`;
- последний block = `infoblock`;
- `infoblock` содержит все и только `LG*`;
- `title_hint_ref = null`, если `title_strategy = keep`;
- `title_hint_ref != null`, если `title_strategy = enhance`;
- `heading = null` для `lead` и `infoblock`;
- если `allow_semantic_headings = false`, headings должны быть удалены;
- при `has_long_program = true` program facts не должны растворяться в body.
- если `heading_guardrail_recommended = true` и headings не осталось, audit должен выставить `missing_headings_for_dense_case`, а не silently считать такой plan нормой.
