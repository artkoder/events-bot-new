import pytest

from db import Database
from models import Event
import smart_event_update as su
from smart_event_update import EventCandidate, PosterCandidate, smart_event_update


async def _no_topics(*args, **kwargs):  # noqa: ANN001 - test helper
    return None


@pytest.mark.asyncio
async def test_create_uses_bundle_title_when_provided(tmp_path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    monkeypatch.setattr(su, "_classify_topics", _no_topics)
    monkeypatch.setattr(su, "SMART_UPDATE_LLM_DISABLED", True)

    async def _bundle(*args, **kwargs):  # noqa: ANN001 - test helper
        return {
            "title": "Пушистая Масленица",
            "description": "Описание.",
            "facts": ["Факт 1"],
            "search_digest": "Дайджест.",
        }

    monkeypatch.setattr(su, "_llm_create_description_facts_and_digest", _bundle)

    candidate = EventCandidate(
        source_type="telegram",
        source_url="https://t.me/castleneuhausen/3034",
        source_text="На этой неделе отмечаем Масленицу!",
        raw_excerpt="",
        title="Экскурсия в замок Нойхаузен",
        date="2026-02-18",
        time="",
        location_name="Замок Нойхаузен",
        city="Калининград",
        posters=[PosterCandidate(ocr_title="ПУШИСТАЯ МАСЛЕНИЦА")],
    )

    res = await smart_event_update(db, candidate, check_source_url=True, schedule_tasks=False)
    assert res.status == "created"
    assert res.event_id is not None

    async with db.get_session() as session:
        ev = await session.get(Event, int(res.event_id))
        assert ev is not None
        assert (ev.title or "").strip() == "Пушистая Масленица"


@pytest.mark.asyncio
async def test_create_allows_bundle_title_override_for_generic_event_type_venue_titles(
    tmp_path, monkeypatch
):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    monkeypatch.setattr(su, "_classify_topics", _no_topics)
    monkeypatch.setattr(su, "SMART_UPDATE_LLM_DISABLED", True)

    async def _bundle(*args, **kwargs):  # noqa: ANN001 - test helper
        return {
            "title": "EвроДэнс'90",
            "description": "Описание.",
            "facts": ["Факт 1"],
            "search_digest": "Дайджест.",
        }

    monkeypatch.setattr(su, "_llm_create_description_facts_and_digest", _bundle)

    candidate = EventCandidate(
        source_type="vk",
        source_url="https://vk.com/wall-100137391_164044",
        source_text='В проекте «EвроДэнс\'90» выступят Natasha Wright и Kevin McCoy.',
        raw_excerpt="",
        title="Концерт — Янтарь холл",
        date="2026-03-09",
        time="",
        location_name="Янтарь холл, Ленина 11, Светлогорск",
        location_address="Ленина 11",
        city="Светлогорск",
        event_type="концерт",
    )

    res = await smart_event_update(db, candidate, check_source_url=True, schedule_tasks=False)
    assert res.status == "created"
    assert res.event_id is not None

    async with db.get_session() as session:
        ev = await session.get(Event, int(res.event_id))
        assert ev is not None
        assert (ev.title or "").strip() == "EвроДэнс'90"


@pytest.mark.asyncio
async def test_merge_allows_title_update_from_generic_existing_title_when_grounded(
    tmp_path, monkeypatch
):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    monkeypatch.setattr(su, "_classify_topics", _no_topics)
    monkeypatch.setattr(su, "SMART_UPDATE_LLM_DISABLED", True)

    async with db.get_session() as session:
        session.add(
            Event(
                title="Концерт — Янтарь холл",
                description="Описание.",
                date="2026-03-09",
                time="",
                location_name="Янтарь холл, Ленина 11, Светлогорск",
                location_address="Ленина 11",
                city="Светлогорск",
                event_type="концерт",
                source_text="",
            )
        )
        await session.commit()

    async def _merge(*args, **kwargs):  # noqa: ANN001 - test helper
        return {
            "title": "EвроДэнс'90",
            "description": "Описание.",
            "search_digest": None,
            "ticket_link": None,
            "ticket_price_min": None,
            "ticket_price_max": None,
            "ticket_status": None,
            "added_facts": [],
            "duplicate_facts": [],
            "conflict_facts": [],
            "skipped_conflicts": [],
        }

    monkeypatch.setattr(su, "_llm_merge_event", _merge)

    candidate = EventCandidate(
        source_type="telegram",
        source_url="https://t.me/test/123",
        source_text='Проект «EвроДэнс\'90» пройдет в "Янтарь-холле".',
        raw_excerpt="",
        title="Концерт — Янтарь холл",
        date="2026-03-09",
        time="",
        location_name="Янтарь холл, Ленина 11, Светлогорск",
        location_address="Ленина 11",
        city="Светлогорск",
        event_type="концерт",
    )

    res = await smart_event_update(db, candidate, check_source_url=True, schedule_tasks=False)
    assert res.status == "merged"
    assert res.event_id is not None

    async with db.get_session() as session:
        ev = await session.get(Event, int(res.event_id))
        assert ev is not None
        assert (ev.title or "").strip() == "EвроДэнс'90"
