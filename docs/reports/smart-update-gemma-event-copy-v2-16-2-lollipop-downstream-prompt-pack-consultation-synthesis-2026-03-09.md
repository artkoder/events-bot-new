# Smart Update Gemma Event Copy V2.16.2 Lollipop Downstream Prompt-Pack Consultation Synthesis

Дата: 2026-03-09

## 1. Scope

- branch: `Smart Update lollipop`
- consultation target: all remaining downstream stages after `facts.merge`
- order: `Opus` strict single launch -> `Gemini` single launch
- goal: decide the real implementation spine for downstream planning/writing before opening the next active family

## 2. Inputs

- canonical funnel doc: `/workspaces/events-bot-new/docs/llm/smart-update-lollipop-funnel.md`
- downstream brief: `/workspaces/events-bot-new/artifacts/codex/tasks/smart-update-lollipop-v2-16-2-downstream-prompt-pack-consultation-brief-2026-03-09.md`
- `Opus` report: `/workspaces/events-bot-new/artifacts/codex/reports/smart-update-lollipop-v2-16-2-downstream-prompt-pack-consultation-opus-2026-03-09.md`
- `Gemini` report: `/workspaces/events-bot-new/artifacts/codex/reports/smart-update-lollipop-v2-16-2-downstream-prompt-pack-consultation-gemini-3.1-pro-preview-2026-03-09.md`

## 3. Convergence

Both models agree on the core downstream shape:

- `facts.prioritize.infoblock_split` should not exist as a separate stage;
- `writer.spec` should not exist as a separate LLM stage;
- final implementation should start from a smaller spine, not from all originally listed downstream stages at once;
- `prioritize.weight` and `prioritize.lead` are the right first active family;
- `writer.final_4o` remains a single final public-writing call only.

Both models also agree that the original `14`-stage downstream plan is too wide for a first implementation round.

## 4. Main Disagreement

`Opus` leaves `pack.compose` as a Gemma stage.

`Gemini` says this is the wrong boundary:

- semantic decisions should stay in LLM stages;
- data hydration / packing / pointer resolution should be deterministic;
- otherwise `pack.compose` becomes the highest-probability fact-loss stage.

I agree with `Gemini` here. This fits the whole `lollipop` philosophy better:
- Gemma decides meaning and structure;
- code performs exact copying / hydration / assembly.

## 5. Revised Downstream Spine

The downstream spine I will treat as canonical for implementation is:

```text
facts.prioritize.weight     (Gemma)
facts.prioritize.lead       (Gemma)
editorial.hooks.seed        (Gemma, later)
editorial.hooks.select      (deterministic)
editorial.patterns.route    (deterministic)
editorial.layout.plan       (Gemma)
writer_pack.compose         (deterministic)
writer_pack.select          (deterministic)
writer.final_4o             (4o)
```

Implications:

- `facts.prioritize.roles` becomes `facts.prioritize.weight`
- `facts.prioritize.lead_vs_body` becomes `facts.prioritize.lead`
- `facts.prioritize.infoblock_split` is removed
- `patterns.seed/select` collapses into one deterministic `pattern.route`
- `pack.compose.v1/v2` collapses into one deterministic `writer_pack.compose`
- `writer.spec` is removed

## 6. Implementation Order

The next implementation order should be:

1. `facts.prioritize.weight`
2. `facts.prioritize.lead`
3. `editorial.layout.plan`
4. deterministic `writer_pack.compose`
5. `writer.final_4o`
6. only after that: `hooks.seed`
7. deterministic `hooks.select`
8. deterministic `pattern.route`

Reason:

- this builds the minimal end-to-end spine from merged facts to final text;
- it postpones stylistic variation until the structural/public-writing path is proven;
- it keeps the highest fact-loss risk (`pack.compose`) out of Gemma.

## 7. Immediate Next Step

The next active family should be `facts.prioritize`, but with the corrected shape:

- `facts.prioritize.weight`
- `facts.prioritize.lead`
- deterministic audit

I do **not** plan to implement the original downstream list literally anymore. The consultation round is strong enough to justify tightening the architecture first.
