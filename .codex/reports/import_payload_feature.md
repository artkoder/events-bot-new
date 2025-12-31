Added a payload‑import path for video announce renders so operators can upload `payload.json` and kick off a new Kaggle render without the event selection stage. The profile channel menu now exposes the new action, and the render pipeline accepts payload‑only JSON when launching.

- Added the `📥 Импортировать payload` button and `vidimport:{profile_key}` callback to the profile channel UI in `video_announce/scenario.py`, plus a pending‑upload prompt for JSON.
- Implemented payload JSON validation, default kernel selection, and a new import‑and‑render flow in `video_announce/scenario.py` that creates a rendering session immediately.
- Added a payload upload handler and pending state wiring in `video_announce/handlers.py`, and registered the message handler in `main_part2.py`.
- Updated `CHANGELOG.md` and added unit tests for payload parsing + default kernel selection in `tests/test_video_announce_payload_import.py`.

**Phase Results**
- Phase 1: Reviewed `CODEX.md` and inspected the video announce flow/pending state pattern and render pipeline.
- Phase 2: Implemented UI + callback, pending upload handling, payload parsing, default kernel selection, and payload‑only render path.
- Phase 3: Added tests, updated changelog, ran tests (smoke failed due to missing `bpy`), committed to `dev`.

**Tests**
- `pytest tests/test_video_announce_payload_import.py` (pass).
- `pytest -m smoke` (failed during collection: `ModuleNotFoundError: No module named 'bpy'` in `3d_intro/*`).

**Notes**
- README/user docs not updated because this is an operator‑only bot UI action; CHANGELOG updated.
- Untracked files present and left untouched: `.codex/reports/fix_video_afisha_mask.md`, `.codex/reports/fix_video_afisha_notebook.md`, `.codex/reports/fix_video_mask_error.md`, `scripts/reset_session_53.py`. Want me to do anything with these?

**Commit**
- `143306b Add payload import for video announce renders`

Next steps:
1) Re‑run `pytest -m smoke` in an environment with Blender (`bpy`) installed.
2) If Blender isn’t available here, decide whether to mark `3d_intro` tests as skipped in CI for non‑Blender environments.