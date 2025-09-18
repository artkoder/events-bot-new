import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import json

import pytest
from sqlalchemy import text

from main import Database
from models import Event


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


@pytest.mark.asyncio
async def test_page_section_cache_exists(tmp_path):
    db_path = tmp_path / "test.sqlite"
    db = Database(str(db_path))
    await db.init()
    async with db.engine.connect() as conn:
        result = await conn.execute(text("PRAGMA table_info(page_section_cache)"))
        cols = [r[1] for r in result.fetchall()]
    await db.engine.dispose()
    assert {"page_key", "section_key", "hash", "updated_at"} <= set(cols)


@pytest.mark.asyncio
async def test_event_topics_columns(tmp_path):
    db_path = tmp_path / "test.sqlite"
    db = Database(str(db_path))
    await db.init()

    async with db.engine.connect() as conn:
        result = await conn.execute(text("PRAGMA table_info(event)"))
        cols = result.fetchall()

    await db.engine.dispose()

    col_map = {row[1]: row for row in cols}
    assert "topics" in col_map
    assert "topics_manual" in col_map
    assert col_map["topics"][2].upper() == "TEXT"
    assert col_map["topics"][4] == "'[]'"
    assert col_map["topics_manual"][2].upper() == "BOOLEAN"
    assert col_map["topics_manual"][4] in ("0", 0)


@pytest.mark.asyncio
async def test_event_topics_roundtrip(tmp_path):
    db_path = tmp_path / "test.sqlite"
    db = Database(str(db_path))
    await db.init()

    async with db.get_session() as session:
        ev = Event(
            title="T",
            description="D",
            festival=None,
            date="2025-01-02",
            time="10:00",
            location_name="Loc",
            source_text="Src",
            topics=["ART"],
            topics_manual=True,
        )
        session.add(ev)
        await session.commit()
        event_id = ev.id

    async with db.raw_conn() as conn:
        cursor = await conn.execute(
            "SELECT topics, topics_manual FROM event WHERE id=?", (event_id,)
        )
        row = await cursor.fetchone()
    assert json.loads(row[0]) == ["ART"]
    assert row[1] in (1, True)

    async with db.get_session() as session:
        stored = await session.get(Event, event_id)
        assert stored is not None
        assert stored.topics == ["ART"]
        assert stored.topics_manual is True

