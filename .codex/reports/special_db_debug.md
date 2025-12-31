Captured the root cause and fix options in a phase‑by‑phase report at `.codex/reports/special_db_debug.md`.

Key points: the `exec()` runs `main_part2.py` into the `globals()` of whatever module loaded `main.py`; when started as a script that module is `__main__`, while the handler later imports `main` as a separate module, so `db` stays `None` in that second module.

If you want me to proceed, I can:
1) Patch `handlers/special_cmd.py` to use `runtime.require_main_attr` (preferred, already available).  
2) Add the `sys.modules["main"]` alias in `main.py` for script execution.