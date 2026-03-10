# Smart Update Gemma Event Copy V2.16.2 Lollipop Facts.Merge Iter1 Consultation Synthesis

Дата: 2026-03-09

## 1. Scope

- family: `facts.merge`
- iteration: `iter1`
- run mode: reuse stored `facts.extract` outputs + stored `facts.dedup iter3` outputs
- consultation evidence: explicit `source/post -> prompt -> Gemma raw output -> parsed result`

Main evidence:

- merge lab report: `/workspaces/events-bot-new/docs/reports/smart-update-gemma-event-copy-v2-16-2-lollipop-family-facts-merge-lab-iter1-2026-03-09.md`
- merge evidence pack: `/workspaces/events-bot-new/docs/reports/smart-update-gemma-event-copy-v2-16-2-lollipop-family-facts-merge-evidence-pack-2026-03-09.md`
- consultation packet: `/workspaces/events-bot-new/artifacts/codex/smart_update_lollipop_facts_merge_family_v2_16_2_iter1_2026-03-09_consultation_packet.json`

Consultation inputs:

- `Opus`: `/workspaces/events-bot-new/artifacts/codex/reports/smart-update-lollipop-v2-16-2-facts-merge-family-postrun-consultation-iter1-opus-2026-03-09.md`
- `Gemini`: `/workspaces/events-bot-new/artifacts/codex/reports/smart-update-lollipop-v2-16-2-facts-merge-family-postrun-consultation-iter1-gemini-3.1-pro-preview-2026-03-09.md`

## 2. Run Result

Aggregate:

```json
{
  "events": 12,
  "avg_merge_record_count": 12.0,
  "avg_resolved_cluster_count": 11.083,
  "avg_emitted_cluster_count": 11.083,
  "avg_total_bucket_items": 11.083,
  "events_with_flags": 0,
  "merge_record_stddev": 2.309
}
```

Representative quality problems seen in emitted output:

- `2673`: `печенье` / `чай` land in `logistics_infoblock`; program-like support line is isolated in `support_context`
- `2734`: duplicate/near-duplicate `people_and_roles` items survive; generic + literal program lines both survive
- `2731`: the same song list survives twice in `program_list`; `event_core` is overstuffed
- `2747`: screening content / plot-like material sits in `event_core`
- `2759`: `event_core` is bloated and partially repetitive

## 3. Consultant Consensus

Strong consensus:

- `facts.prioritize`: `NO-GO`
- `facts.merge iter1` is structurally promising but not semantically clean enough
- primary bottleneck: `facts.merge.resolve`
- secondary bottleneck: `facts.merge.bucket`
- `facts.merge.emit` is not the main problem
- next step should be `facts.merge iter2`, not a jump to `facts.prioritize`

## 4. Where Consultants Diverge

### 4.1. Bucket Taxonomy

`Opus`:

- current buckets are sufficient for `v1`
- do not add a new bucket yet
- tighten the screening/content rule instead

`Gemini`:

- current buckets are not fully sufficient
- add one new bucket like `background_lore` / `content_synopsis`

### 4.2. My Decision

I am not adding a new bucket in mainline `iter2`.

Reason:

- a new bucket would force immediate downstream expansion into `facts.prioritize`, `editorial.patterns`, `layout`, and `writer_pack`;
- the current failure can still plausibly be fixed by a stricter `bucket` contract plus a stronger `resolve` collapse rule;
- this is the smallest defensible next step.

So for `iter2` I will keep the current bucket set and treat `background_lore` as a challenger idea only if tightened contracts still leave screening/exhibition contamination on the same `12`-event casebook.

## 5. Accepted Next Changes

### 5.1. `facts.merge.resolve` retune

Add an explicit collapse/subsumption rule:

- if two items express the same fact but one is more detailed, keep only the more detailed one;
- if a generic program line is subsumed by a literal list, keep only the literal one;
- if the same literal list appears with different lead-in wording, emit only one canonical item.

This is the primary fix.

### 5.2. `facts.merge.bucket` retune

Add explicit routing rules:

- screening / cine-club / work-content facts:
  - plot, award history, synopsis, character-description facts should not go to `event_core` by default
  - `event_core` should describe the event itself, not summarize the work
- named activities, shows, performances, repertoire units should prefer `program_list`
- organizer / partner / institution lines should not default to `support_context`

This is the secondary fix.

### 5.3. Deterministic duplicate audit

Add a post-emit duplicate-suspect audit:

- compare bucket items pairwise inside the same bucket
- if overlap is too high, flag `MERGE_DUPLICATE_SUSPECT`

This is a cheap guardrail and should stop silent propagation of obvious duplicate emissions.

## 6. Verdict

- `facts.merge architecture`: `GO`
- `facts.merge iter1 prompt pack`: `NO-GO`
- `facts.prioritize`: still blocked

## 7. Next Step

Proceed to `facts.merge iter2` on the same `12`-event casebook with:

1. `resolve` collapse/subsumption retune
2. tighter `bucket` rules
3. deterministic duplicate-suspect audit

If `iter2` clears duplicate survival and screening/event-core contamination, then `facts.prioritize` can become the next active family.
