# Smart Update Lollipop Funnel

## Purpose

This is the canonical map for the experimental `Smart Update lollipop` branch.

`lollipop` is a `dry-run` shadow pipeline layered on top of the current `Smart Update` fact-first runtime. The baseline runtime is preserved. New logic is explored as a funnel of small LLM steps, with one final `4o` call reserved for public text writing.

Companion docs:

- casebook: [smart-update-lollipop-casebook.md](/workspaces/events-bot-new/docs/llm/smart-update-lollipop-casebook.md)
- mixed-phase prompt pack: [smart-update-lollipop-mixed-phase-prompts.md](/workspaces/events-bot-new/docs/llm/smart-update-lollipop-mixed-phase-prompts.md)

Core invariants:

- baseline `Smart Update` stays untouched;
- facts are accumulated first;
- every meaningful grounded fact must survive;
- baseline infoblock logic and baseline list rendering are reused;
- `Gemma` handles multi-step preparation;
- `4o` is used once, only at the final public-writing step.

## Canonical Family Chain

```text
source.scope
  -> scope.extract
  -> scope.select

facts.extract
  -> baseline_fact_extractor
  -> facts.extract.subject
  -> facts.extract.card
  -> facts.extract.agenda
  -> facts.extract.support
  -> facts.extract.performer
  -> facts.extract.participation
  -> facts.extract.stage.tightened
  -> facts.extract.theme.challenger

facts.dedup
  -> facts.dedup.baseline_diff
  -> facts.dedup.cross_enrich
  -> facts.dedup.audit

facts.merge
  -> facts.merge.bucket
  -> facts.merge.link
  -> facts.merge.resolve
  -> facts.merge.emit

facts.prioritize
  -> facts.prioritize.weight
  -> facts.prioritize.lead

editorial.hooks
  -> hooks.seed
  -> hooks.select

editorial.patterns
  -> pattern.route

editorial.layout
  -> layout.plan

writer_pack.compose
  -> writer_pack.compose

writer_pack.select
  -> pack.select

writer.final_4o
  -> writer.final_4o
```

## Current Status

| Family | Status | Current source of truth |
|---|---|---|
| `source.scope` | planned | baseline Smart Update scoping behavior + `2.16.x` design notes |
| `facts.extract` | active, first bank built | extract family-lab + evidence pack + consultation |
| `facts.dedup` | active, merge-ready | dedup iter3 lab + consultation synthesis |
| `facts.merge` | active, iter6 residual subset cleared / full sanity rerun pending | merge link song-list consultation synthesis + merge iter6 lab |
| `facts.prioritize` | active, iter2 lab complete | `facts.prioritize iter2` lab + downstream consultation synthesis |
| `editorial.hooks` | planned after spine | hook/angle carries from `2.15.2/2.15.3` and downstream consultation |
| `editorial.patterns` | planned after spine | macro-pattern carries from `2.15.2` and downstream consultation |
| `editorial.layout` | planned inside first spine | baseline formatting + `2.15.9` block separation carry + downstream consultation |
| `writer_pack.compose` | planned as deterministic assembler | baseline lists/infoblock + downstream consultation |
| `writer_pack.select` | planned as deterministic validator | downstream consultation |
| `writer.final_4o` | planned as final public writer | one final call only; constrained by writer pack |

Important limitation:

- current machine-readable `stage bank` only covers `facts.extract`;
- the rest of the funnel is still documented canonically here and in family reports, but not yet registered as full multi-version banks.

## Historical Carry-Over By Family

This is the current salvage map of useful prior work beyond `facts.extract`.

### `source.scope`

Useful carries:

- baseline Smart Update single-source processing and multi-source fact accumulation spine
- baseline multi-event separation behavior from live runtime
- `source_scope_extract` requirement from `2.16.x` design work

Current state:

- no stable Gemma prompt has been promoted yet;
- scope remains a planned family, not a banked one.
- the first explicit source-risk probe is now fixed in the casebook:
  - mixed-phase series post with future anchor
  - source URL: `vk.com/wall-179910542_11821`
  - current recommended `v1` interceptor:
    - `scope.extract.phase_map`
    - `scope.select.target_phase`
    - `facts.extract.phase_scoped`

### `facts.extract`

Current active shortlist:

- `baseline_fact_extractor`
- `facts.extract.subject`
- `facts.extract.card`
- `facts.extract.agenda`
- `facts.extract.support`
- `facts.extract.performer`
- `facts.extract.participation`
- `facts.extract.stage.tightened`
- `facts.extract.theme.challenger`

Historical sources:

- `2.15.5` through `2.15.8`
- baseline fact extractor

### `facts.dedup`

Useful carries:

- `baseline_diff.v1 -> v2 -> v3(id-anchor)` from the recent `lollipop` rounds
- `cross_enrich.v1`
- deterministic audit pass

Current stable shape:

- `baseline_diff.v3(id-anchor)` replaced brittle exact-text anchoring
- `facts.dedup` is now treated as a quality-uplift layer, not a prose layer
- canonical active chain is `baseline_diff -> cross_enrich -> audit`

### `facts.merge`

Useful carries already visible in history:

- `normalize_fact_floor` from `ice-cream` as a positive carry for canonical fact-floor shaping
- `facts.type_buckets.v1` from `ice-cream` / seed-bank work
- `pack.separate_blocks.v1` from `2.15.9` for strict narrative vs logistics separation
- baseline infoblock logic as `infoblock.baseline.v1`
- baseline list preservation as `lists.baseline.v1`
- `screening.metadata_guard.v1` from `2.15.10` as a narrow guarded carry

What is not carried:

- `ice-cream` public Gemma writer mainline
- `generate -> audit -> repair` as main flow

### `facts.prioritize`

Useful carries:

- `plan_v1_basic` from `2.15.5` only as structural split idea
- `plan_lead` lessons from `2.15.5`, `2.15.6`, `2.15.7`
- `pattern_and_format_plan` structural lessons from `2.15.2`
- baseline narrative vs infoblock split
- downstream consultation conclusion:
  - keep only `prioritize.weight` and `prioritize.lead`
  - do not keep `infoblock_split` as a separate stage
  - `prioritize` must remain JSON-only and non-prose

Important note:

- this family must not become prose generation;
- it should only assign roles to already merged facts.
- first real run now uses `fact_id`-based prompting (`weight -> lead`) to keep fact text immutable and make downstream audit deterministic.

### `editorial.hooks`

Useful carries:

- `2.15.2` lead anti-pattern examples
- `2.15.2` pattern lead hints
- seed-bank consultation conclusion that hooks should be logical angle seeds, not drafted paragraphs
- `ice-cream` negative knowledge around anti-cliche / anti-expansion

Working interpretation:

- likely closer to `angle.discovery` than to free-form hook drafting

### `editorial.patterns`

Useful carries:

- `2.15.2` macro-pattern work
- `2.15.2` structural contract rewrite for pattern-specific output behavior
- the recommendation to collapse to a smaller number of structurally distinct patterns
- seed-bank consultation advice to keep pattern choice logical and fact-density-driven, not aesthetic
- downstream consultation conclusion:
  - do not start with Gemma `patterns.seed/select`
  - first implementation should use deterministic `pattern.route`

### `editorial.layout`

Useful carries:

- baseline headings, lists, blockquote, and Telegraph-safe formatting
- `2.15.9` strict block separation
- baseline rule that infoblock is rendered separately and reliably

### `writer_pack.compose`

Useful carries:

- baseline infoblock rendering
- baseline list preservation
- baseline markdown-safe formatting
- `2.15.9` block separation idea
- seed-bank consultation requirement that `pack` be a structured writer payload, not a narrative draft
- downstream consultation correction:
  - `writer_pack.compose` should be deterministic hydration/assembly, not a Gemma rewriting stage

### `writer.final_4o`

Useful carries:

- single final `4o` call only
- `writer.final_4o.spec` idea from seed-bank consultation
- negative knowledge bank from `ice-cream`
- downstream consultation correction:
  - no separate `writer.spec` stage; the writer pack itself is the spec

Hard constraint:

- `4o` must receive a writer-ready pack, not raw extraction noise

## Downstream Consultation Result

The original downstream `14`-stage sketch should not be implemented literally.

Current canonical downstream spine after `Opus + Gemini` consultation:

```text
facts.prioritize.weight     (Gemma)
facts.prioritize.lead       (Gemma)
editorial.layout.plan       (Gemma)
writer_pack.compose         (deterministic)
writer_pack.select          (deterministic)
writer.final_4o             (4o)
```

Variation layers should be added only after this spine is stable:

```text
editorial.hooks.seed        (Gemma)
editorial.hooks.select      (deterministic)
editorial.patterns.route    (deterministic)
```

Rationale:

- `pack.compose` is a fact-hydration problem, not a semantic-writing problem;
- `pattern.route` is better treated as a rules engine than as a Gemma creativity task;
- `writer.spec` and `facts.prioritize.infoblock_split` are redundant.

## Current Facts.Merge State

`facts.merge` is the current active family.

It will not write prose. It will convert deduped claims into one canonical merged event pack.

### Inputs

- baseline fact floor
- `facts.dedup` outputs from shortlisted extraction branches
- baseline list and infoblock logic
- provenance for every surviving fact

### `facts.merge.bucket`

Goal:

- place each surviving fact into the correct semantic bucket

Initial target buckets:

- `event_core`
- `program_list`
- `people_and_roles`
- `forward_looking`
- `logistics_infoblock`
- `support_context`
- `uncertain`

Carry-over to use:

- `facts.type_buckets.v1`
- baseline infoblock/list logic
- `screening.metadata_guard.v1` for risky metadata-like facts

Current state:

- `bucket.v2` is good enough to keep as the current classifier;
- after `iter3`, the main unresolved issue was no longer semantic bucketing alone but pre-resolve cluster topology.

### `facts.merge.link`

Goal:

- link same-bucket records that describe the same underlying fact cluster before canonical resolution

Rules:

- any record type may link to any other record type inside one bucket;
- different named entities must stay separate;
- different semantic claims must stay separate;
- same literal list with different lead-in phrasing should link;
- overlapping baseline facts are allowed to link.

Current state after `iter4`:

- `facts.merge.link.v1` is now a first-class stage;
- it fixed the real duplicate-heavy cases on `2734` and `2731`;
- remaining open issue `2759` suggests a narrower overlap/link containment problem, not a reason to remove linkage.

Current state after `iter5`:

- `facts.merge.link.v2` fixed `2759` by collapsing the generic exhibition-theme line with the richer “through documents and objects” line;
- `audit.v3` removed the false-positive duplicate flags on `2673` and `2447`;
- one residual blocker remains on `2731`, where the model explains that identical song-list lines should merge but still returns two different `cluster_id` values.

Current state after `iter6`:

- `facts.merge.link.v3` plus the narrow literal-list cleaner removed the residual `2731` song-list blocker;
- the targeted regression subset `2731 + 2673 + 2447 + 2759` finished with `events_with_flags = 0`;
- `facts.merge` looks ready to unblock `facts.prioritize`, but a fresh full `12`-event sanity rerun is still desirable before treating the family as fully closed.
- `facts.prioritize iter2` is now live on top of the current full `facts.merge iter5` casebook and clears the first `12`-event audit with `events_with_flags = 0`; screening/program edge-cases are handled through a safe secondary lead fallback (`program_list` / `people_and_roles`) when `event_core` and `forward_looking` are empty.

### `facts.merge.resolve`

Goal:

- merge complementary facts inside a bucket without losing any grounded detail

Rules:

- union-first, not compression-first
- do not convert support or logistics into `event_core`
- do not flatten literal lists into vague prose
- keep provenance
- keep uncertain conflicts explicit instead of silently collapsing them

Useful prior carries:

- `normalize_fact_floor`
- `pack.separate_blocks.v1`

Current state after `iter4`:

- compact algorithmic `resolve.v3` remains acceptable as the canonical resolver;
- once `facts.merge.link.v1` was inserted, `2734` and `2731` stopped being real `resolve` failures;
- the remaining open cases are now:
  - `2759` as overlapping baseline-fact containment/linkage;
  - `2673` and `2447` as likely duplicate-audit false positives.

### `facts.merge.emit`

Goal:

- emit a single canonical merged fact pack for downstream families

Target output shape:

```json
{
  "event_core": [],
  "program_list": [],
  "people_and_roles": [],
  "forward_looking": [],
  "logistics_infoblock": [],
  "support_context": [],
  "uncertain": [],
  "provenance": []
}
```

## What Happens After Facts.Merge

Once `facts.merge` is stable, the next layers should behave like this:

- `facts.prioritize`: decide `lead_candidate`, `body_core`, `list_only`, `infoblock_only`, `support_only`
- `editorial.hooks`: derive angle seeds from merged facts
- `editorial.patterns`: choose a structural pattern
- `editorial.layout`: define heading/list/block plan
- `writer_pack.compose`: build one or more structured writer payloads
- `writer_pack.select`: pick the final payload
- `writer.final_4o`: render the public event text once

## Current Canonical Documents

- extract raw JSON: `/workspaces/events-bot-new/artifacts/codex/smart_update_lollipop_facts_extract_family_v2_16_2_2026-03-09.json`
- extract consultation packet: `/workspaces/events-bot-new/artifacts/codex/smart_update_lollipop_facts_extract_family_v2_16_2_2026-03-09_consultation_packet.json`
- extract prompt inventory: `/workspaces/events-bot-new/artifacts/codex/smart_update_lollipop_facts_extract_family_v2_16_2_2026-03-09/prompt_inventory.json`
- dedup iter3 lab: `/workspaces/events-bot-new/docs/reports/smart-update-gemma-event-copy-v2-16-2-lollipop-family-facts-dedup-lab-iter3-2026-03-09.md`
- dedup iter3 synthesis: `/workspaces/events-bot-new/docs/reports/smart-update-gemma-event-copy-v2-16-2-lollipop-family-facts-dedup-iter3-consultation-synthesis-2026-03-09.md`
- merge linkage consultation synthesis: `/workspaces/events-bot-new/docs/reports/smart-update-gemma-event-copy-v2-16-2-lollipop-family-facts-merge-linkage-consultation-synthesis-2026-03-09.md`
- merge iter4 lab: `/workspaces/events-bot-new/docs/reports/smart-update-gemma-event-copy-v2-16-2-lollipop-family-facts-merge-lab-iter4-2026-03-09.md`
- merge iter5 lab: `/workspaces/events-bot-new/docs/reports/smart-update-gemma-event-copy-v2-16-2-lollipop-family-facts-merge-lab-iter5-2026-03-09.md`
- merge link song-list consultation synthesis: `/workspaces/events-bot-new/docs/reports/smart-update-gemma-event-copy-v2-16-2-lollipop-family-facts-merge-link-songlist-consultation-synthesis-2026-03-09.md`
- merge iter6 lab: `/workspaces/events-bot-new/docs/reports/smart-update-gemma-event-copy-v2-16-2-lollipop-family-facts-merge-lab-iter6-2026-03-09.md`
- machine-readable stage bank: `/workspaces/events-bot-new/artifacts/codex/stage_bank/smart_update_lollipop_stage_bank_v2_16_2_2026-03-09.json`

## Current Routing State

Right now `lollipop` documentation is partially routed:

- `facts.extract`, `facts.dedup`, and current `facts.merge` reports are routed in `docs/routes.yml`
- this funnel file is the canonical cross-family route entry

This file is now the canonical route entry for the whole `lollipop` funnel.
