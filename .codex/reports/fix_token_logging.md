Updated `log_token_usage` to wait for the Supabase insert instead of fire‑and‑forget, so token rows aren’t dropped on shutdown or short‑lived runs; also adjusted the debug log message to match the new behavior in `main.py`. Phase reports are written to `.codex/reports/PHASE-1.md`, `.codex/reports/PHASE-2.md`, `.codex/reports/PHASE-3.md`, `.codex/reports/PHASE-4.md`.

Tests: `pytest tests/ -x -q` timed out twice (120s and 300s), so the full run didn’t complete.

Limitations: couldn’t check prod logs because `flyctl` isn’t installed here and network access is restricted.

Next steps:
1) Re-run `pytest tests/ -x -q` with a longer timeout or in your local environment.
2) Run `flyctl logs -a events-bot-new-wngqia --no-tail` on your machine to confirm `log_token_usage` outcomes.