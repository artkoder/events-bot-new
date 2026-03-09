from __future__ import annotations

import json
from pathlib import Path


def _load_notebook_cell_source(*, startswith: str) -> str:
    nb = json.loads(Path("kaggle/TelegramMonitor/telegram_monitor.ipynb").read_text())
    for cell in nb.get("cells", []):
        src = "".join(cell.get("source", []))
        if src.startswith(startswith):
            return src
    raise AssertionError(f"Notebook cell not found: {startswith!r}")


def test_tg_monitor_notebook_declares_recent_rescan_envs():
    src = _load_notebook_cell_source(startswith="KAGGLE_INPUT = Path('/kaggle/input')")
    assert "RECENT_RESCAN_ENABLED = os.getenv('TG_MONITORING_RECENT_RESCAN_ENABLED', '1') == '1'" in src
    assert "RECENT_RESCAN_LIMIT = max(" in src
    assert "TG_MONITORING_RECENT_RESCAN_LIMIT" in src


def test_tg_monitor_notebook_scans_recent_messages_without_new_posts():
    src = _load_notebook_cell_source(
        startswith="async def scan_source(client: TelegramClient, source: dict) -> dict:"
    )
    assert "source.skip reason=no_new_messages" not in src
    assert "recent_event_message_ids = source.get('recent_event_message_ids') or []" in src
    assert "recent_rescan_enabled = RECENT_RESCAN_ENABLED and bool(recent_event_message_ids)" in src
    assert "source.recent_rescan_only" in src
    assert "async def _append_metrics_only_message(msg) -> bool:" in src
    assert "'metrics_only_refresh': True" in src
    assert "client.get_messages" in src
    assert "ids=batch_ids" in src
    assert "recent_get_messages" in src
    assert "source.recent_rescan username=%s processed=%s limit=%s known_event_posts=%s" in src
