**Findings**
- High (fixed): /3di stopped after payload generation, so steps 4–8 never executed; added default Kaggle dataset/kernel/poll/output + DB update pipeline and wired it into /3di (`preview_3d/handlers.py:109`, `preview_3d/handlers.py:427`).
- High (fixed): fresh SQLite init missed `preview_3d_url`, causing runtime failures when updating results; added bootstrap column creation (`db.py:222`).

**Checks**
- GPU enablement OK (`kaggle/Preview3D/kernel-metadata.json`).
- CPU fallback OK in Blender script (Cycles GPU → CPU) (`kaggle/Preview3D/preview_3d.ipynb`).
- 7‑image cap enforced in payload + downloader (`preview_3d/handlers.py:452`, `kaggle/Preview3D/preview_3d.ipynb`).
- Monthly pages use 3D preview as primary image when present (`main_part2.py:160`).

**Open Questions**
- Should the Kaggle dataset slug stay fixed (`preview3d-dataset`)? Parallel /3di runs would overwrite payload; consider per‑session slug or queueing.

**Change Summary**
- Implemented Kaggle orchestration + result parsing in `preview_3d/handlers.py`.
- Added `preview_3d_url` bootstrap column and aligned tests in `db.py` and `tests/test_preview_3d.py`.
- Tests not run (not requested).

**Next Steps**
1. `pytest tests/test_preview_3d.py`
2. Run `/3di` on a small month to validate Kaggle credentials and kernel runtime.