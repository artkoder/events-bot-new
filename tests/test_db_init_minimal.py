import sqlite3

import pytest

from db import Database


@pytest.mark.asyncio
async def test_db_init_minimal_creates_core_tables_only(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_INIT_MINIMAL", "1")
    db_path = str(tmp_path / "db.sqlite")
    db = Database(db_path)
    await db.init()

    conn = sqlite3.connect(db_path)
    try:
        names = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
    finally:
        conn.close()

    assert "event" in names
    assert "event_source" in names
    assert "event_source_fact" in names
    assert "eventposter" in names
    assert "telegram_source" in names
    assert "telegram_scanned_message" in names
    # Optional navigation/publication tables are skipped in minimal mode.
    assert "monthpage" not in names

