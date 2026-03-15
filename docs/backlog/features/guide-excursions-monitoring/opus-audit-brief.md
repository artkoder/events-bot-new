# Guide Excursions Monitoring Opus Audit Brief

> **Status:** Ready for external audit  
> **Audience:** `Claude Opus` or another strong architecture/prompt-review model  
> **Intent:** получить не generic strategy advice, а критичный implementation-grade audit по уже подготовленному excursion stack.

## 1. Goal

We have a new feature in design: **monitoring guide excursions from Telegram channels**, extracting structured facts, building a digest, and publishing it into Telegram.

The architecture is already largely designed.  
We do **not** want a greenfield redesign unless you believe the current direction is fundamentally wrong.

We want a **critical engineering audit** of the current plan:

- architecture completeness;
- LLM-first stage design;
- prompt-family shape;
- facts/data model;
- digest UX and public-surface logic;
- frozen evaluation pack;
- implementation sequencing.

This audit should help us decide whether the current package is ready to move into implementation and live E2E debugging.

## 2. What Exists Already

Current feature docs:

- `docs/backlog/features/guide-excursions-monitoring/README.md`
- `docs/backlog/features/guide-excursions-monitoring/architecture.md`
- `docs/backlog/features/guide-excursions-monitoring/mvp.md`
- `docs/backlog/features/guide-excursions-monitoring/digest-spec.md`
- `docs/backlog/features/guide-excursions-monitoring/llm-first.md`
- `docs/backlog/features/guide-excursions-monitoring/eval-pack.md`
- `docs/backlog/features/guide-excursions-monitoring/casebook.md`
- `docs/backlog/features/guide-excursions-monitoring/e2e.md`

Existing implemented baseline systems that this feature should reuse:

- `docs/features/telegram-monitoring/README.md`
- `docs/features/smart-event-update/README.md`
- `docs/features/digests/README.md`
- `docs/features/llm-gateway/README.md`
- `docs/llm/request-guide.md`
- `docs/llm/smart-update-lollipop-funnel.md`

## 3. Non-Negotiable Constraints

These are current project constraints unless you believe one of them is a true blocker.

### 3.1. Execution boundary

- Raw Telegram scanning must happen in a **Kaggle notebook**.
- The notebook should **maximally reuse** the existing `TelegramMonitor` stack.
- The bot runtime must **not** perform Telethon source scanning.
- The only Telethon user session stays on the Kaggle side.

### 3.2. MVP scope

- Telegram only
- no VK yet
- no public pages yet
- no Telegraph pages for excursions in MVP
- digest publication only to `@keniggpt`
- source onboarding via migration/seed, no UI yet

### 3.3. Domain/product constraints

- `on_request` / `private-group-only` / school-group offers should **not** go to public digests by default
- `audience_fit` is first-class and must be accumulated early
- the system should distinguish:
  - `guide_personal`
  - `guide_project`
  - `excursion_operator`
  - `organization_with_tours`
  - `aggregator`
- already **past occurrences should not be stored for now**
- future booking-click tracking is needed later, so the schema now reserves tracking fields

### 3.4. Media constraint

- bot runtime cannot depend on Telethon for media retrieval
- temporary media path is:
  - `forward source message -> extract file_id -> delete relay -> sendMediaGroup`
- Kaggle staging is fallback only

### 3.5. UX constraint

- operator flow must stay simple:
  - `/guide_excursions`
  - `Run light scan`
  - `Run full scan`
  - `Preview new digest`
  - `Preview last call`
  - `Publish to @keniggpt`
  - `Send test report`

## 4. Current Intended Architecture

Very short summary:

- `Trail Scout v1`
  - Kaggle notebook
  - Telethon fetch, grouped albums, OCR, prefilter, Gemma extraction
- `Route Weaver v1`
  - facts-first merge on server side
  - `guide / template / occurrence` materialization
- `Lollipop Trails v1`
  - short digest copy from fact pack
- `Trail Digest v1`
  - ranking, grouping, preview, publish
- `Media Bridge v1`
  - temporary `forward -> file_id`
- `Guide Atlas v1`
  - admin surfaces and `/general_stats`

Core domain entities:

- `GuideProfile`
- `ExcursionTemplate`
- `ExcursionOccurrence`
- `GuideFactClaim`

Current LLM chain:

```text
deterministic prefilter
-> trail_scout.screen.v1
-> trail_scout.{announce|status|template}_extract.v1
-> deterministic shortlist
-> route_weaver.occurrence_match.v1 (only if ambiguous)
-> deterministic materialize
-> lollipop_trails.digest_card.v1
```

## 5. Real Casebook Context

This is not a synthetic exercise.

The feature is based on real Telegram posts from:

- `tanja_from_koenigsberg`
- `gid_zelenogradsk`
- `katimartihobby`
- `amber_fringilla`
- `alev701`
- `vkaliningrade`
- `ruin_keepers`
- `twometerguide`
- `valeravezet`
- `excursions_profitour`

Important recurring source patterns:

- one post often contains **multiple excursion occurrences**
- there are many **status-only updates**: few seats, sold out, meeting point, reschedule
- many posts are **mixed** with lectures, gastronomy, audioquests, lifestyle or recap prose
- some channels are dominated by **template-only on-request offers**
- grouped albums matter because the linked `message_id` can have no caption text
- the same route can appear in:
  - original guide channel
  - collaborator channel
  - organization channel
  - aggregator

## 6. Frozen Evaluation Pack

There is already a frozen eval set in:

- `docs/backlog/features/guide-excursions-monitoring/eval-pack.md`

It includes expected handling for cases such as:

- multi-announce original guide post
- mixed post with lectures and excursions
- status-only post with one freed seat
- reschedule post
- sparse schedule post
- on-request school-group operator post
- aggregator meeting-point reminder
- grouped album without caption on the linked message

The eval pack also defines pre-implementation metrics and acceptance targets.

## 7. What We Need From You

We want a **critical audit** with concrete implementation implications.

Please focus on the following:

### 7.1. Architecture review

- Is the stage split correct?
- Are any responsibilities in the wrong place?
- Is any important stage still missing?
- Is facts-first applied correctly here, or do you see leakage back toward raw-post logic?
- Is the Kaggle boundary well-chosen?

### 7.2. LLM-first / prompt review

- Are the current prompt families the right minimum set?
- Are any of them too broad, too vague, or too optimistic for `Gemma`?
- Should any stage be split or collapsed?
- Are the schemas too large or too small?
- Are there any places where deterministic precompute should replace prompt burden?
- If any prompt family should be rewritten, provide a concrete replacement.

### 7.3. Data/facts model review

- Is `guide / template / occurrence / fact_claim` sufficient for MVP?
- Is `audience_fit` modeled early enough and concretely enough?
- Is the current handling of `on_request` coherent?
- Is the “do not store past occurrences” rule acceptable for MVP, or likely to cause operational problems?
- Is the future booking-click tracking reservation sufficient?

### 7.4. Digest review

- Is the digest layout understandable and information-dense enough?
- Is the `new_occurrences` vs `last_call` split the right MVP cut?
- Is the current omission policy correct?
- Is the public digest too complex, too dense, or still missing important signals?
- Does the current treatment of operator/group offers look safe?

### 7.5. Eval pack review

- Is the frozen eval set strong enough?
- What important cases are still missing?
- Are the proposed metrics useful and realistic?
- What would you add, remove, or tighten before implementation begins?

### 7.6. Implementation sequencing

- What is the correct order of implementation?
- What would you cut if the current MVP is still too ambitious?
- What should be deferred without harming the value of the first release?

## 8. Important Nuances To Challenge

Please do **not** just affirm the design.

Challenge these choices if needed:

- storing no past occurrences
- having only one server-side LLM writer stage for digest copy
- relying on `forward -> file_id` media bridge
- treating some operator scheduled posts as `digest_eligible=yes`
- not doing guide/template pages yet
- using one SQLite DB with separate `guide_*` tables

If any of these look like actual blockers, say so directly.

## 9. What We Do Not Want

Please avoid:

- generic product brainstorming
- “it depends” without concrete recommendation
- total architecture replacement unless truly necessary
- giant new pipeline proposals detached from the existing bot
- advice that ignores the already implemented `Telegram monitoring -> Smart Update -> /digest` baseline

## 10. Required Deliverable Format

Please answer in **Markdown** using this exact structure:

### 1. Executive verdict

One of:

- `GO`
- `GO WITH CHANGES`
- `NO-GO`

With a short reason.

### 2. Blocking issues

List only truly blocking issues.

### 3. Non-blocking issues

List important but non-fatal issues.

### 4. Architecture audit

Short, concrete, component-by-component review.

### 5. Prompt-family audit

For each stage:

- `keep`
- `revise`
- `split`
- `drop`

And explain why.

If you recommend a change, include the revised prompt or revised schema.

Stages to review:

- `trail_scout.screen.v1`
- `trail_scout.announce_extract.v1`
- `trail_scout.status_extract.v1`
- `trail_scout.template_extract.v1`
- `route_weaver.occurrence_match.v1`
- `lollipop_trails.digest_card.v1`

### 6. Facts/data model audit

Review:

- `GuideProfile`
- `ExcursionTemplate`
- `ExcursionOccurrence`
- `GuideFactClaim`
- booking tracking reservation
- no-past-occurrence rule

### 7. Digest UX audit

Review:

- `new_occurrences`
- `last_call`
- card layout
- omission policy
- operator/group-offer exclusion
- media bundle logic

### 8. Eval pack audit

Review the frozen cases and metrics.  
Add missing cases if needed.

### 9. Recommended implementation order

Ordered, practical, minimal path to first working release.

### 10. Final recommendation

One short final paragraph:

- what is ready now
- what must change before coding starts
- what can safely wait

## 11. Preferred Audit Style

Please be direct.

We prefer:

- concrete criticism
- prompt-level corrections
- implementation-grade advice
- explicit prioritization

If the current design is overcomplicated, say where.
If it is underspecified, say where.
If it is mostly ready, say what exactly still needs tightening.
