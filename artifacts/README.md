# Artifacts (Generated Outputs)

This directory is the home for **generated** and **temporary** files that should
not clutter the repository root:

- logs
- manual test outputs
- E2E run outputs (JSON reports, screenshots)
- database snapshots and scratch DBs

## Rules

1. Do not put artifacts in the repository root.
2. Prefer creating a dated subfolder: `artifacts/<type>/YYYY-MM-DD/`.
3. Link from `docs/reports/` when an artifact matters for future readers.

