# Documentation Map

This repo uses a **feature-oriented documentation** approach:
each major capability has a home under `docs/features/`, and cross-cutting
topics (ops, architecture, pipelines, reference) live in their own sections.

## Start here

- `README.md` — what the bot does + quick start.
- `docs/features/README.md` — feature index (where specs and behavior live).
- `docs/operations/commands.md` — operator/admin commands.
- `docs/architecture/overview.md` — system overview.

## Sections (where to look)

| What | Where | Notes |
|---|---|---|
| Implemented feature behavior | `docs/features/<feature>/` | Source of truth for UX + invariants + entrypoints |
| Ops / runbooks | `docs/operations/` | “How to run / debug / deploy” |
| Architecture | `docs/architecture/` | “How it works” (high level) |
| Pipelines & parsers | `docs/pipelines/` | Venue- and pipeline-specific notes |
| LLM prompts & policies | `docs/llm/` | Prompts, request format, topic logic |
| Reference data | `docs/reference/` | Locations, holidays, templates |
| Backlog & specs | `docs/backlog/` | Not implemented yet; treat as design notes |
| Reports & plans | `docs/reports/` | Human context, reviews, retrospectives |
| Tooling notes | `docs/tools/` | Local workflows and CLI cheat-sheets |

## Where to put new docs (rules)

### 1) Feature docs (default choice)
If the task changes user-visible behavior or adds a capability:
- Create/update `docs/features/<feature>/README.md`.
- Add extra files under the same folder (e.g. `design.md`, `ux.md`, `data.md`, `testing.md`).

**Minimum sections** for a feature README:
1. Goal + non-goals
2. User flows / commands / UI
3. Entry points in code (modules/files)
4. Config/env vars
5. Data model (tables/fields if relevant)
6. Testing (unit + e2e) and how to run
7. Operational notes (alerts, failure modes)

### 2) Backlog/specs (not implemented)
If the task is a plan/spec only:
- Put it in `docs/backlog/` (prefer `docs/backlog/linear/<ID>-<slug>.md` for Linear tasks).
- Link it from the relevant feature folder (or create the feature folder as a placeholder).

### 3) Reports (analysis you want to preserve)
If you produce a review, investigation, or implementation plan:
- Put it in `docs/reports/` and link it from the feature/backlog doc.
- Use a date prefix: `docs/reports/YYYY-MM-DD-<topic>.md` (or keep existing historical names, but link them from an index).

### 4) Raw outputs / artifacts (never in repo root)
Anything generated (logs, snapshots, screenshots, test dumps) goes to:
- `artifacts/` (see `artifacts/README.md`)
