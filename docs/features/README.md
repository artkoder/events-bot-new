# Features (Index)

Feature documentation is the canonical place to understand **current behavior**,
the **UX / commands**, and the **code entrypoints** for each capability.

## Implemented

- `docs/features/digests/README.md` — digests generation and publishing.
- `docs/features/tourist-label/README.md` — “туристам интересно/неинтересно” labeling workflow.

## Adding a new feature doc

Create `docs/features/<feature>/README.md` and keep related materials inside the
same folder. Prefer small focused files over a single huge README.

Recommended file set:
- `README.md` — overview + links (required)
- `design.md` — constraints/decisions and invariants
- `testing.md` — unit/e2e strategy and how to run
- `ops.md` — operational notes, metrics, alerts

