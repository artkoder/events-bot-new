import pytest
from datetime import datetime

from digests import parse_start_time, build_lectures_digest_candidates
from main import Database, Event


def test_parse_start_time_basic_cases():
    assert parse_start_time("18:30") == (18, 30)
    assert parse_start_time("18.30") == (18, 30)
    assert parse_start_time("18:30–20:00") == (18, 30)
    assert parse_start_time("18:30-20:00") == (18, 30)
    assert parse_start_time("18:30.15:30") == (18, 30)
    assert parse_start_time(" ") is None
    assert parse_start_time("24:99") == (23, 59)


@pytest.mark.asyncio
async def test_today_filter_with_dirty_time(tmp_path):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    now = datetime(2025, 5, 1, 16, 25)

    async with db.get_session() as session:
        date = now.date().isoformat()
        later = Event(
            title="later",
            description="d",
            date=date,
            time="18.30",
            location_name="x",
            source_text="s",
            event_type="лекция",
        )
        early = Event(
            title="early",
            description="d",
            date=date,
            time="17:00",
            location_name="x",
            source_text="s",
            event_type="лекция",
        )
        session.add_all([later, early])
        await session.commit()

    events, _ = await build_lectures_digest_candidates(db, now)
    titles = [e.title for e in events]
    assert titles == ["later"]
