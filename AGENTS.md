# Agent Guide (Repository Navigation)

This file is the fast entrypoint for any AI-agent or new contributor.

## Required reading (in order)

1. `CODEX.md` — mandatory engineering rules + definition of done.
2. `docs/README.md` — documentation map + where to put new docs.
3. `docs/operations/commands.md` — bot commands and operator flows.
4. `docs/architecture/overview.md` — high-level architecture.

## Where things live

- **Feature docs (source of truth):** `docs/features/<feature>/README.md`
- **Ops/runbooks (how to run):** `docs/operations/`
- **Architecture (how it works):** `docs/architecture/`
- **Pipelines/parsers:** `docs/pipelines/`
- **LLM integration:** `docs/llm/`
- **Reference lists used by prompts:** `docs/reference/`
- **Backlog/specs (not implemented):** `docs/backlog/`
- **Human reports/plans (decision context):** `docs/reports/`
- **Generated/temporary artifacts (never in repo root):** `artifacts/`
- **Codex run logs / agent checkpoints:** `.codex/reports/` (treat as scratch unless explicitly curated into `docs/reports/`)

## Adding documentation for a new task (feature-oriented)

1. Create or update the feature folder: `docs/features/<feature>/README.md`.
2. If the task is not implemented yet, keep the spec in `docs/backlog/` (and link to it from the feature README).
3. If you produce an analysis/plan that should be preserved, put it in `docs/reports/` and link it from the relevant feature/backlog doc.
4. Put raw outputs (logs, test dumps, db snapshots, screenshots) under `artifacts/` and link to them from the report if needed.

