from __future__ import annotations

from pathlib import Path

import pytest

from db import Database
from linked_events import recompute_linked_event_ids
from models import Event


@pytest.mark.asyncio
async def test_recompute_linked_event_ids_links_and_unlinks(tmp_path: Path) -> None:
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    async with db.get_session() as session:
        e1 = Event(
            title="Мастер-класс по живописи маслом",
            description="d1",
            date="2026-03-07",
            time="11:00",
            location_name="Студия",
            source_text="s1",
        )
        e2 = Event(
            title="Мастер-класс по живописи маслом",
            description="d2",
            date="2026-03-12",
            time="10:30",
            location_name="Студия",
            source_text="s2",
        )
        e3 = Event(
            title="Другая встреча",
            description="d3",
            date="2026-03-12",
            time="10:30",
            location_name="Студия",
            source_text="s3",
        )
        session.add(e1)
        session.add(e2)
        session.add(e3)
        await session.commit()
        await session.refresh(e1)
        await session.refresh(e2)
        await session.refresh(e3)

    res = await recompute_linked_event_ids(db, int(e1.id or 0))
    assert set(res.group_event_ids) == {int(e1.id or 0), int(e2.id or 0)}

    async with db.get_session() as session:
        e1_db = await session.get(Event, int(e1.id or 0))
        e2_db = await session.get(Event, int(e2.id or 0))
        e3_db = await session.get(Event, int(e3.id or 0))

    assert e1_db is not None
    assert e2_db is not None
    assert e3_db is not None
    assert int(e2.id or 0) in (e1_db.linked_event_ids or [])
    assert int(e1.id or 0) in (e2_db.linked_event_ids or [])
    assert (e3_db.linked_event_ids or []) == []

    # Rename one occurrence: links must be removed from both sides.
    async with db.get_session() as session:
        e2_db2 = await session.get(Event, int(e2.id or 0))
        assert e2_db2 is not None
        e2_db2.title = "Мастер-класс по керамике"
        await session.commit()

    await recompute_linked_event_ids(db, int(e2.id or 0))

    async with db.get_session() as session:
        e1_after = await session.get(Event, int(e1.id or 0))
        e2_after = await session.get(Event, int(e2.id or 0))
    assert e1_after is not None
    assert e2_after is not None
    assert (e1_after.linked_event_ids or []) == []
    assert (e2_after.linked_event_ids or []) == []

