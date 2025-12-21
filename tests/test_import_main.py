"""Lightweight import smoke test for the main entrypoint."""

def test_main_imports_and_exports():
    import importlib

    importlib.invalidate_caches()
    import main

    assert hasattr(main, "format_day_pretty"), "format_day_pretty should be available from main"
