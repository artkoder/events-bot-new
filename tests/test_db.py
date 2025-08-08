import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pytest
from sqlalchemy import text

from main import Database


@pytest.mark.asyncio
async def test_journal_mode_wal(tmp_path):
    db_path = tmp_path / "test.sqlite"
    db = Database(str(db_path))
    await db.init()
    async with db.engine.connect() as conn:
        result = await conn.execute(text("PRAGMA journal_mode"))
        mode = result.scalar()
    await db.engine.dispose()
    assert mode.lower() == "wal"


@pytest.mark.asyncio
async def test_festival_has_source_text(tmp_path):
    db_path = tmp_path / "test.sqlite"
    db = Database(str(db_path))
    await db.init()
    async with db.engine.connect() as conn:
        result = await conn.execute(text("PRAGMA table_info(festival)"))
        cols = [r[1] for r in result.fetchall()]
    await db.engine.dispose()
    assert "source_text" in cols

