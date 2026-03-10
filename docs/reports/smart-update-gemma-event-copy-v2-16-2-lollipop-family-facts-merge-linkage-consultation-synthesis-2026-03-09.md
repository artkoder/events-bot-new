# Smart Update Gemma Event Copy V2.16.2 Lollipop Facts.Merge Linkage Consultation Synthesis

Дата: 2026-03-09

## 1. Scope

- family: `facts.merge`
- focus: post-`iter3` bottleneck after `bucket.v2 -> resolve.v3 -> emit.v1`
- consultation mode: single-launch `Gemini 3.1 Pro`
- question: is the next issue still in `resolve`, or is a separate linkage stage needed before `resolve`

## 2. Consultation Input

`Gemini` received:

- exact current `bucket.v2` prompt;
- the current `merge_target_id` / anchored-record limitation;
- representative fail-cases `2734`, `2731`, `2759`;
- likely audit false positives `2673`, `2447`;
- two competing designs:
  - overload `bucket.v3`;
  - add a new `facts.merge.link.v1` stage.

## 3. Consultation Verdict

Main conclusion:

- the next real bottleneck is `bucket/linkage`, not `resolve`;
- overloading `bucket` further is not recommended;
- the safer Gemma-friendly move is a dedicated `facts.merge.link.v1` stage.

Concrete guidance:

1. Keep `bucket` focused on semantic bucket choice.
2. Add `facts.merge.link.v1` after `bucket` and before `resolve`.
3. Run linkage per bucket, not across the full event.
4. Allow any record type to cluster with any other record type inside one bucket.
5. Keep strict guards:
   - different named entities => do not cluster;
   - different semantic claims => do not cluster;
   - same literal list with different lead-in phrasing => cluster;
   - overlapping baseline facts must be allowed to cluster.

`Gemini` also supplied a full candidate prompt for `facts.merge.link.v1` in the raw consultation output.

## 4. Implementation Decision

The consultation was accepted as the next `facts.merge` structural change.

Chosen move:

- keep `bucket.v2`;
- insert `facts.merge.link.v1`;
- keep `resolve.v3` downstream;
- re-run the same `12`-event casebook before changing `facts.prioritize`.

Rejected move:

- no `bucket.v3` overload round before trying linkage.

## 5. Outcome After Iter4

The subsequent `iter4` run confirmed the consultation direction.

Aggregate change:

- `events_with_flags`: `5 -> 3`
- `events_with_duplicate_suspects`: `5 -> 3`

Resolved true duplicate-heavy cases:

- `2734`: duplicate performer/award lines collapsed into one cluster;
- `2731`: duplicate song-list lines collapsed into one `program_list` cluster.

Still open:

- `2759`: overlapping baseline exhibition facts still survive separately;
- `2673` and `2447`: remaining flags still look closer to duplicate-audit false positives than to linkage failures.

## 6. Current Decision

- `facts.merge.link.v1`: keep
- `facts.merge.resolve.v3`: keep
- `facts.prioritize`: still blocked

Next likely round:

- either `facts.merge.link.v2` for overlapping baseline fact containment like `2759`;
- or a narrower duplicate-audit retune if `2759` is the only remaining real merge issue.
