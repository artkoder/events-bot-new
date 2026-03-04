import pytest

from sqlalchemy import select

from db import Database
from models import Event
from smart_event_update import EventCandidate, smart_event_update
import smart_event_update as su


@pytest.mark.asyncio
async def test_time_is_default_does_not_block_matching_and_is_overridden(tmp_path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    async def _no_merge(*_args, **_kwargs):  # noqa: ANN001 - test helper
        return None

    async def _no_rewrite(*_args, **_kwargs):  # noqa: ANN001 - test helper
        return None

    async def _no_digest(*_args, **_kwargs):  # noqa: ANN001 - test helper
        return None

    async def _no_facts(*_args, **_kwargs):  # noqa: ANN001 - test helper
        return []

    async def _no_topics(*_args, **_kwargs):  # noqa: ANN001 - test helper
        return None

    monkeypatch.setattr(su, "_llm_merge_event", _no_merge)
    monkeypatch.setattr(su, "_rewrite_description_journalistic", _no_rewrite)
    monkeypatch.setattr(su, "_llm_build_search_digest", _no_digest)
    monkeypatch.setattr(su, "_llm_extract_candidate_facts", _no_facts)
    monkeypatch.setattr(su, "_classify_topics", _no_topics)

    async with db.get_session() as session:
        session.add(
            Event(
                id=1,
                title="Гараж",
                description="",
                date="2026-04-04",
                time="19:00",
                time_is_default=True,
                location_name="Драмтеатр",
                city="Калининград",
                source_text="VK пост без явного времени",
            )
        )
        await session.commit()

    candidate = EventCandidate(
        source_type="tg",
        source_url="https://t.me/dramteatr39/3869",
        source_text="Начало в 18:00.",
        raw_excerpt="",
        title="Гараж",
        date="2026-04-04",
        time="18:00",
        location_name="Драмтеатр",
        city="Калининград",
        trust_level="medium",
    )

    res = await smart_event_update(db, candidate, check_source_url=False, schedule_tasks=False)
    assert res.status == "merged"
    assert res.event_id == 1

    async with db.get_session() as session:
        rows = (await session.execute(select(Event))).scalars().all()
        assert len(rows) == 1
        ev = await session.get(Event, 1)
        assert ev is not None
        assert (ev.time or "").strip() == "18:00"
        assert bool(getattr(ev, "time_is_default", False)) is False

