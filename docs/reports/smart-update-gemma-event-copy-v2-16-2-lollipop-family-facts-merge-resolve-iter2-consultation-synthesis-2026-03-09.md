# Smart Update Gemma Event Copy V2.16.2 Lollipop Facts.Merge.Resolve Iter2 Consultation Synthesis

Дата: 2026-03-09

## 1. Scope

- family: `facts.merge`
- target stage: `facts.merge.resolve`
- current stage under review: `facts.merge.resolve.v2`
- objective: decide whether remaining `iter2` failures require a stronger `resolve` prompt or mostly audit refinement

Inputs:

- `facts.merge iter2` lab: `/workspaces/events-bot-new/docs/reports/smart-update-gemma-event-copy-v2-16-2-lollipop-family-facts-merge-lab-iter2-2026-03-09.md`
- `Opus`: `/workspaces/events-bot-new/artifacts/codex/reports/smart-update-lollipop-v2-16-2-facts-merge-resolve-iter2-consultation-opus-2026-03-09.md`
- `Gemini`: `/workspaces/events-bot-new/artifacts/codex/reports/smart-update-lollipop-v2-16-2-facts-merge-resolve-iter2-consultation-gemini-3.1-pro-preview-2026-03-09.md`

## 2. Consultant Consensus

Strong consensus:

- `resolve.v2` is still the main remaining prompt bottleneck
- true `resolve` failures are:
  - `2734`
  - `2731`
- likely audit false positives are:
  - `2673`
  - `2447`
- next move should be `resolve.v3`, not another architecture rewrite
- audit should be refined too, but after the `resolve` fix

## 3. What To Change In `resolve.v3`

Accepted carry from `Opus`, reinforced by `Gemini`:

1. Replace flat heuristic bullets with an explicit decision procedure.
2. Use pairwise labels:
   - `SUBSUMES`
   - `SUBSUMED`
   - `DISTINCT`
3. Explicitly compare content, not JSON field shape.
4. Add hard `DISTINCT` guards:
   - different named entities => `DISTINCT`
   - different semantic claims => `DISTINCT`
5. Add mandatory `subsumption_log` in output for auditability.

## 4. Why This Is Safe Enough

Main safety concern was over-collapse.

Both consultants converged on the same answer:

- the new framework is safer than current `near-duplicate` language if `DISTINCT` is spelled out sharply;
- `2673` and `2447` show exactly why a fuzzy collapse rule is dangerous;
- a structured `SUBSUMES / DISTINCT` procedure is safer than “collapse harder”.

## 5. Implementation Decision

I will use `Opus` prompt structure as the base for `facts.merge.resolve.v3`, with these carry points explicitly preserved:

- same list inline vs in `literal_items` must count as the same content
- different people/entities must never be collapsed
- different semantic claims must remain distinct
- pairwise log must be emitted before final canonical item

## 6. Next Step

Immediate next move:

- implement `facts.merge.resolve.v3`
- keep `bucket.v2`
- refine duplicate audit only after the new `resolve` rerun

So the next run should be:

`facts.merge iter3 = bucket.v2 + resolve.v3 + existing emit + current audit`

Then:

- if `2734` and `2731` clear while `2673` and `2447` remain only as audit noise, refine the duplicate audit instead of touching `resolve` again.
