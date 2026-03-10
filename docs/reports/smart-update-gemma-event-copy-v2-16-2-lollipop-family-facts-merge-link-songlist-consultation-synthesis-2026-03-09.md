# Smart Update Gemma Event Copy V2.16.2 Lollipop Facts.Merge Link Song-List Consultation Synthesis

Дата: `2026-03-09`

## Scope

- family: `facts.merge`
- narrow issue: residual `2731` duplicate in `program_list`
- consultation model: `gemini-3.1-pro-preview`
- target stage: `facts.merge.link`

## Problem

After `facts.merge iter5`, only one blocker remained:

- two `program_list` records in `2731` contained the same literal song list;
- `facts.merge.link.v2` reasoning said they should merge;
- but the returned `cluster_id` values stayed different;
- `resolve.v3` therefore kept two separate `program_list` items.

## Gemini Verdict

`Gemini` recommended `Option C`:

1. strengthen the prompt contract;
2. add a narrow deterministic cleaner guard.

Key recommendation:

- the prompt must define merge mechanically, not only semantically;
- if two records are merged, they must literally return the same `cluster_id`;
- for `program_list` + `list_item_literal`, a deterministic guard is safe when quoted-item sets are exactly equal.

## Implemented Carry

Accepted into `facts.merge.link.v3`:

- explicit schema rule that merged rows must share the same `cluster_id`;
- explicit example for identical literal song lists with one shared root id;
- narrow cleaner guard:
  - only in `program_list`
  - only for `list_item_literal`
  - only when quoted-item sets are non-empty and exactly equal.

## Outcome In Iter6

`iter6` was run as a targeted regression subset:

- `2731`
- `2673`
- `2447`
- `2759`

Result:

- `events_with_flags = 0`
- `events_with_duplicate_suspects = 0`
- `2731` is now clean
- guard-cases `2673`, `2447`, and `2759` stayed clean

## Current Decision

- `facts.merge.link.v3`: keep
- `facts.merge.resolve.v3`: keep
- `facts.merge` residual regression subset: cleared

Remaining caution:

- `iter6` was a targeted subset run, not a fresh full `12`-event rerun;
- before treating `facts.merge` as fully closed, a full sanity rerun is still desirable.
