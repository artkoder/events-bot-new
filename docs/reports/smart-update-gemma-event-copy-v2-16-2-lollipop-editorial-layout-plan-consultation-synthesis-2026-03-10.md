# Smart Update Gemma Event Copy V2.16.2 Lollipop Editorial Layout Plan Consultation Synthesis

Date: `2026-03-10`

## Inputs

- Opus report: [smart-update-lollipop-v2-16-2-editorial-layout-plan-consultation-opus-2026-03-10.md](/workspaces/events-bot-new/artifacts/codex/reports/smart-update-lollipop-v2-16-2-editorial-layout-plan-consultation-opus-2026-03-10.md)
- Gemini report: [smart-update-lollipop-v2-16-2-editorial-layout-plan-consultation-gemini-3.1-pro-preview-2026-03-10.md](/workspaces/events-bot-new/artifacts/codex/reports/smart-update-lollipop-v2-16-2-editorial-layout-plan-consultation-gemini-3.1-pro-preview-2026-03-10.md)
- Main brief: [smart-update-lollipop-v2-16-2-editorial-layout-plan-consultation-brief-2026-03-10.md](/workspaces/events-bot-new/artifacts/codex/tasks/smart-update-lollipop-v2-16-2-editorial-layout-plan-consultation-brief-2026-03-10.md)

## Consensus

- `editorial.layout.plan` should stay a single `Gemma` stage.
- The output should stay grounded through `fact_refs`, not abstract prose slots.
- Deterministic validation after the LLM call is mandatory.
- `title` planning belongs to `layout.plan`, not to the final writer.
- `program_list` should remain structurally visible as a list-oriented block when required.

## Main disagreement resolved

`Opus` wrote a strong prompt contract, but `Gemini` correctly identified one structural flaw: the prompt asks `Gemma` to count facts and infer density from raw counts and enumerations.

That logic should not live in the prompt.

Canonical decision:
- pre-compute `density` in Python;
- pre-compute `has_long_program` in Python;
- pass both into the prompt as explicit input state;
- remove density/list counting from the LLM decision burden.

## Canonical v1 decision

Use this shape for `editorial.layout.plan.v1`:

- one `Gemma` stage;
- deterministic `precompute_layout_state`;
- deterministic `layout.validate`;
- no second `Gemma` stage yet.

Recommended stage chain:

```text
editorial.layout.precompute
-> editorial.layout.plan.v1
-> editorial.layout.validate
```

Where:
- `precompute` is deterministic;
- `plan.v1` is the only Gemma call;
- `validate` is deterministic.

## Schema decision

Prefer:

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

Not:
- `density` returned by Gemma;
- `title_hint_refs` array;
- freeform section text;
- abstract content slots detached from fact ids.

## Why this is the best next step

- Keeps the stage small enough for `Gemma`.
- Preserves grounding and full-fact traceability.
- Avoids brittle small-LLM counting behavior.
- Makes validation cheap and explicit.
- Leaves writer stages downstream constrained by structure rather than creative guesswork.

## Implementation verdict

`GO`, but not with the raw Opus prompt as-is.

Implement:
- Opus prompt structure;
- Gemini correction on deterministic density/list precompute;
- singular `title_hint_ref`;
- deterministic validation.
