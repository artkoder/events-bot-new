from __future__ import annotations

import pytest

import smart_event_update as su
from db import Database
from smart_event_update import EventCandidate, PosterCandidate, smart_event_update


async def _patch_smart_update_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _merge(*args, **kwargs):  # noqa: ANN001 - test helper
        return {
            "title": None,
            "description": "Описание события.",
            "ticket_link": None,
            "ticket_price_min": None,
            "ticket_price_max": None,
            "ticket_status": None,
            "added_facts": [],
            "duplicate_facts": [],
            "conflict_facts": [],
            "skipped_conflicts": [],
        }

    async def _bundle(*args, **kwargs):  # noqa: ANN001 - test helper
        return {
            "title": "Тестовое событие",
            "description": "Описание события.",
            "facts": ["Факт 1"],
            "search_digest": "Короткий дайджест.",
        }

    async def _no_text(*args, **kwargs):  # noqa: ANN001 - test helper
        return None

    monkeypatch.setattr(su, "_llm_merge_event", _merge)
    monkeypatch.setattr(su, "_llm_create_description_facts_and_digest", _bundle)
    monkeypatch.setattr(su, "_rewrite_description_journalistic", _no_text)
    monkeypatch.setattr(su, "_llm_build_search_digest", _no_text)
    monkeypatch.setattr(su, "_classify_topics", _no_text)
    monkeypatch.setattr(su, "_ask_gemma_text", _no_text)


@pytest.mark.asyncio
async def test_smart_update_silences_far_future_event_when_poster_date_conflicts(tmp_path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    await _patch_smart_update_offline(monkeypatch)

    candidate = EventCandidate(
        source_type="vk",
        source_url="https://vk.com/wall-1_1",
        source_text="Анонс события.",
        raw_excerpt="",
        title="Джаз с характером: Илья Пищенко",
        # Far future on purpose: must trigger the guard regardless of the current date.
        date="2099-11-19",
        time="20:30",
        location_name="Сигнал",
        city="Калининград",
        trust_level="medium",
        posters=[
            PosterCandidate(
                catbox_url="https://example.com/poster.jpg",
                ocr_text="17/02\n20/30",
            )
        ],
    )

    res = await smart_event_update(db, candidate, check_source_url=False, schedule_tasks=False)
    assert res.status == "created"
    assert res.event_id is not None

    async with db.get_session() as session:
        ev = await session.get(su.Event, int(res.event_id))
        assert ev is not None
        assert ev.silent is True


@pytest.mark.asyncio
async def test_smart_update_does_not_silence_when_poster_date_matches(tmp_path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    await _patch_smart_update_offline(monkeypatch)

    candidate = EventCandidate(
        source_type="vk",
        source_url="https://vk.com/wall-1_2",
        source_text="Анонс события.",
        raw_excerpt="",
        title="Джаз с характером: Илья Пищенко",
        date="2099-02-17",
        time="20:30",
        location_name="Сигнал",
        city="Калининград",
        trust_level="medium",
        posters=[
            PosterCandidate(
                catbox_url="https://example.com/poster.jpg",
                ocr_text="17/02\n20/30",
            )
        ],
    )

    res = await smart_event_update(db, candidate, check_source_url=False, schedule_tasks=False)
    assert res.status == "created"
    assert res.event_id is not None

    async with db.get_session() as session:
        ev = await session.get(su.Event, int(res.event_id))
        assert ev is not None
        assert ev.silent is False

