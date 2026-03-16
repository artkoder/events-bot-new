# CLAUDE

## Session Defaults
- Use Claude Opus only in this repository.
- Keep effort at `high` for every session and delegated task.
- For difficult consultations, architecture review, deep debugging, or major redesign, you may temporarily raise effort to `max`.
- Keep extended thinking enabled by default.
- Do not switch to Sonnet or Haiku unless the user explicitly changes project policy.
- Built-in non-project delegations are blocked in shared settings; use the project `Opus` alias for delegation.

## Opus Alias
- The project provides a dedicated subagent alias: `Opus`.
- Use `Opus` for consultation, architecture review, prompt critique, and substantial rework.
- When the user asks for "Opus", "consultation", "second opinion", or "доработай через Opus", delegate to the `Opus` subagent instead of changing the main session model ad hoc.
- For LLM-quality tasks, prefer asking `Opus` for concrete prompt-family edits, schema tightening, and `lollipop`-style stage decomposition rather than broad high-level architecture commentary.

## Working Rules
- Start with the canonical project docs: `AGENTS.md`, `docs/README.md`, and `docs/routes.yml`.
- For behavior changes, keep canonical docs in `docs/` updated and add a concise entry to `CHANGELOG.md` under `[Unreleased]`.
- Treat `AGENTS.md` as the repository-wide routing and workflow contract.
- Respect Telegram session boundaries exactly as written in `AGENTS.md`: `TELEGRAM_AUTH_BUNDLE_S22` is for Kaggle/remote monitoring only, while `TELEGRAM_AUTH_BUNDLE_E2E` (or `TELEGRAM_SESSION`) is for local live E2E only.
- Do not substitute one auth bundle for another without explicit user permission, even as a temporary debugging shortcut.
