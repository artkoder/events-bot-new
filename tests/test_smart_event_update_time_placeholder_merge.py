import pytest

from sqlalchemy import select

from db import Database
from models import Event, EventSource
from smart_event_update import EventCandidate, smart_event_update
import smart_event_update as su


@pytest.mark.asyncio
async def test_tg_candidate_fills_placeholder_time_and_merges(tmp_path, monkeypatch):
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
                title="EURODANCE'90",
                description="",
                date="2026-03-09",
                time="00:00",  # legacy placeholder
                location_name="Янтарь-холл",
                city="Калининград",
                source_text="src",
            )
        )
        await session.commit()

    candidate = EventCandidate(
        source_type="tg",
        source_url="https://t.me/yantarholl/4214",
        source_text="Начало в 19:00.",
        raw_excerpt="",
        title="EURODANCE'90",
        date="2026-03-09",
        time="19:00",
        location_name="Янтарь холл, Ленина 11, Светлогорск",
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
        assert (ev.time or "").strip() == "19:00"


@pytest.mark.asyncio
async def test_missing_time_candidate_merges_into_existing_explicit_time(tmp_path, monkeypatch):
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
                title="Мельница: Крапива",
                description="",
                date="2026-04-03",
                time="19:00",
                location_name="Янтарь-холл",
                city="Калининград",
                source_text="src",
            )
        )
        await session.commit()

    candidate = EventCandidate(
        source_type="tg",
        source_url="https://t.me/yantarholl/4222",
        source_text="",
        raw_excerpt="",
        title="Мельница: Крапива",
        date="2026-04-03",
        time=None,
        location_name="Янтарь холл, Ленина 11, Светлогорск",
        city="Калининград",
        trust_level="medium",
    )

    res = await smart_event_update(db, candidate, check_source_url=False, schedule_tasks=False)
    assert res.event_id == 1
    assert res.status in {"merged", "skipped_nochange"}

    async with db.get_session() as session:
        rows = (await session.execute(select(Event))).scalars().all()
        assert len(rows) == 1
        ev = await session.get(Event, 1)
        assert ev is not None
        assert (ev.time or "").strip() == "19:00"


@pytest.mark.asyncio
async def test_check_source_url_false_reuses_event_by_event_source_url(tmp_path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    async def _no_match(*_args, **_kwargs):  # noqa: ANN001 - test helper
        return None, 0.0, "no_match_for_test"

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

    monkeypatch.setattr(su, "_llm_match_event", _no_match)
    monkeypatch.setattr(su, "_llm_merge_event", _no_merge)
    monkeypatch.setattr(su, "_rewrite_description_journalistic", _no_rewrite)
    monkeypatch.setattr(su, "_llm_build_search_digest", _no_digest)
    monkeypatch.setattr(su, "_llm_extract_candidate_facts", _no_facts)
    monkeypatch.setattr(su, "_classify_topics", _no_topics)

    source_url = "https://t.me/meowafisha/6743"
    async with db.get_session() as session:
        session.add(
            Event(
                id=1,
                title="Мохообразные и их место на планете",
                description="",
                date="2026-02-20",
                time="18:00",
                location_name="Сигнал",
                city="Калининград",
                source_text="src",
            )
        )
        session.add(
            EventSource(
                event_id=1,
                source_type="tg",
                source_url=source_url,
                source_text="src",
                trust_level="medium",
            )
        )
        await session.commit()

    candidate = EventCandidate(
        source_type="tg",
        source_url=source_url,
        source_text="Начало в 18:00.",
        raw_excerpt="",
        title="Мохообразные и их место на планете",
        date="2026-02-20",
        time="18:00",
        location_name="Сигнал",
        city="Калининград",
        trust_level="medium",
    )

    res = await smart_event_update(db, candidate, check_source_url=False, schedule_tasks=False)
    assert res.event_id == 1
    assert res.status in {"merged", "skipped_nochange"}

    async with db.get_session() as session:
        evs = (await session.execute(select(Event))).scalars().all()
        assert len(evs) == 1


@pytest.mark.asyncio
async def test_city_filter_keeps_empty_city_rows_to_prevent_duplicates(tmp_path, monkeypatch):
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
                title='Сергей Маковецкий представляет "Скрипка Ротшильда"',
                description="",
                date="2026-07-11",
                time="",
                location_name="Драмтеатр",
                city="",  # legacy/partial import without city
                source_text="Анонс спектакля.",
            )
        )
        await session.commit()

    candidate = EventCandidate(
        source_type="vk",
        source_url="https://vk.com/wall-100137391_163915",
        source_text='11 июля в 19:00 Сергей Маковецкий представляет "Скрипка Ротшильда".',
        raw_excerpt='Сергей Маковецкий представляет "Скрипка Ротшильда".',
        title='Сергей Маковецкий представляет "Скрипка Ротшильда"',
        date="2026-07-11",
        time="19:00",
        location_name="Драмтеатр",
        city="Калининград",
        trust_level="medium",
    )

    res = await smart_event_update(db, candidate, check_source_url=True, schedule_tasks=False)
    assert res.status == "merged"
    assert res.event_id == 1

    async with db.get_session() as session:
        rows = (await session.execute(select(Event))).scalars().all()
        assert len(rows) == 1
        ev = await session.get(Event, 1)
        assert ev is not None
        assert (ev.time or "").strip() == "19:00"
