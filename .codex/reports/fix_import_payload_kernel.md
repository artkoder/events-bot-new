Added kernel selection gating for payload import and restart so renders only start after a kernel is chosen, and wired imported payload data into the kernel-selection callback flow. `video_announce/scenario.py` now stores imported payload JSON until a kernel is picked, starts import renders via a dedicated path, and resets failed sessions to SELECTED before showing the kernel picker; `video_announce/handlers.py` only replies when the import flow returns a message; `CHANGELOG.md` updated.

Tests:
- `pytest tests/test_video_announce_payload_import.py`

Notes:
- Reviewed `CODEX.md`.
- No lint config found in the repo.

Commit: `3645b8c` on `dev`.

Next steps:
1. Run the full test suite if you want broader coverage.
2. Run your project’s lint command if you have one configured locally.