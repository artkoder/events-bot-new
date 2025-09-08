import pytest
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

from main import Database, Event
from digests import (
    build_lectures_digest_candidates,
    make_intro,
    aggregate_digest_topics,
)


@pytest.mark.asyncio
async def test_build_lectures_digest_candidates_expand_to_14(tmp_path):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    now = datetime(2025, 5, 1, 12, 0)

    async with db.get_session() as session:
        def add(offset_days: int, time: str, title: str):
            dt = now + timedelta(days=offset_days)
            ev = Event(
                title=title,
                description="d",
                date=dt.strftime("%Y-%m-%d"),
                time=time,
                location_name="x",
                source_text="s",
                event_type="лекция",
            )
            session.add(ev)

        # Event starting in less than 2h -> should be ignored
        add(0, "13:00", "early")
        # Five events within 7 days
        add(0, "15:00", "d0")
        add(1, "12:00", "d1")
        add(2, "12:00", "d2")
        add(3, "12:00", "d3")
        add(4, "12:00", "d4")
        # Two more beyond 7 days
        add(8, "12:00", "d8")
        add(9, "12:00", "d9")
        await session.commit()

    events, horizon = await build_lectures_digest_candidates(db, now)
    titles = [e.title for e in events]

    assert horizon == 14
    assert titles == ["d0", "d1", "d2", "d3", "d4", "d8", "d9"]


@pytest.mark.asyncio
async def test_build_lectures_digest_candidates_limit(tmp_path):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    now = datetime(2025, 5, 1, 12, 0)

    async with db.get_session() as session:
        def add(offset_days: int, hour: int, title: str):
            dt = now + timedelta(days=offset_days)
            ev = Event(
                title=title,
                description="d",
                date=dt.strftime("%Y-%m-%d"),
                time=f"{hour:02d}:00",
                location_name="x",
                source_text="s",
                event_type="лекция",
            )
            session.add(ev)

        add(0, 13, "early")  # filtered by +2h rule
        idx = 0
        for day in range(7):
            for h in (15, 16):
                idx += 1
                add(day, h, f"e{idx}")
        await session.commit()

    events, horizon = await build_lectures_digest_candidates(db, now)

    assert horizon == 7
    assert len(events) == 9
    assert events[0].title == "e1"


def test_make_intro_morphology():
    assert "интересную лекцию" in make_intro(1, 7, [])
    assert "интересных лекций" in make_intro(7, 7, [])
    assert "интересных лекций" in make_intro(9, 7, [])
    assert "ближайших двух недель" in make_intro(3, 14, [])


def test_aggregate_topics():
    events = [
        SimpleNamespace(topics=["искусство", "культура"]),
        SimpleNamespace(topics=["история россии", "российская история"]),
        SimpleNamespace(topics=["ит", "тех"]),
        SimpleNamespace(topics=["музыка"]),
        SimpleNamespace(topics=["культура"]),
    ]
    assert aggregate_digest_topics(events) == [
        "искусство",
        "история россии",
        "музыка",
    ]
