from __future__ import annotations

import sys
import types

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

    async def _false_async(*args, **kwargs):  # noqa: ANN001 - test helper
        return False

    festival_queue_stub = types.ModuleType("festival_queue")

    class _FestivalDecision:
        context = "none"
        festival = None
        festival_full = None
        dedup_links = []
        signals = {}

    async def _festival_enqueue(*args, **kwargs):  # noqa: ANN001 - test helper
        return None

    linked_events_stub = types.ModuleType("linked_events")

    class _LinkedEventsResult:
        changed_event_ids: list[int] = []

    async def _recompute_linked_event_ids(*args, **kwargs):  # noqa: ANN001 - test helper
        return _LinkedEventsResult()

    festival_queue_stub.detect_festival_context = lambda **kwargs: _FestivalDecision()  # noqa: E731
    festival_queue_stub.enqueue_festival_source = _festival_enqueue
    linked_events_stub.recompute_linked_event_ids = _recompute_linked_event_ids

    monkeypatch.setenv("REGION_FILTER_ENABLED", "0")
    monkeypatch.setitem(sys.modules, "festival_queue", festival_queue_stub)
    monkeypatch.setitem(sys.modules, "linked_events", linked_events_stub)
    monkeypatch.setattr(su, "_llm_merge_event", _merge)
    monkeypatch.setattr(su, "_llm_create_description_facts_and_digest", _bundle)
    monkeypatch.setattr(su, "_rewrite_description_journalistic", _no_text)
    monkeypatch.setattr(su, "_llm_build_search_digest", _no_text)
    monkeypatch.setattr(su, "_classify_topics", _no_text)
    monkeypatch.setattr(su, "_ask_gemma_text", _no_text)
    monkeypatch.setattr(su, "_apply_holiday_festival_mapping", _false_async)


@pytest.mark.asyncio
async def test_smart_update_silences_far_future_event_when_poster_date_conflicts(tmp_path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    try:
        await _patch_smart_update_offline(monkeypatch)
        monkeypatch.setattr(su, "SMART_UPDATE_FAR_FUTURE_CREATE_MONTHS", 2000)

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
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_smart_update_does_not_silence_when_poster_date_matches(tmp_path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    try:
        await _patch_smart_update_offline(monkeypatch)
        monkeypatch.setattr(su, "SMART_UPDATE_FAR_FUTURE_CREATE_MONTHS", 2000)

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
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_smart_update_rejects_far_future_event_without_strong_signals_even_when_poster_matches(
    tmp_path,
    monkeypatch,
):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    try:
        await _patch_smart_update_offline(monkeypatch)

        candidate = EventCandidate(
            source_type="telegram",
            source_url="https://t.me/test/1",
            source_text="4 марта прошла встреча клуба исследователей нейросетей.",
            raw_excerpt="Прошла встреча клуба.",
            title="Клуб исследователей нейросетей: создание мультфильмов",
            date="2099-03-04",
            time="19:00",
            location_name="Сигнал",
            city="Калининград",
            posters=[
                PosterCandidate(
                    catbox_url="https://example.com/poster.jpg",
                    ocr_text="04 марта • 19:00",
                )
            ],
        )

        res = await smart_event_update(db, candidate, check_source_url=False, schedule_tasks=False)
        assert res.status == "rejected_far_future_low_confidence"
        assert res.reason in {"far_future:recap_context", "far_future:no_strong_signal"}
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_smart_update_rejects_far_future_event_on_poster_conflict_even_with_specific_ticket(
    tmp_path,
    monkeypatch,
):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    try:
        await _patch_smart_update_offline(monkeypatch)

        candidate = EventCandidate(
            source_type="vk",
            source_url="https://vk.com/wall-1_3",
            source_text=(
                "17/02\n20:30\n\n"
                "19 ноября в 20:30 концерт.\n"
                "Билеты: https://signalcommunity.timepad.ru/event/3806838/"
            ),
            raw_excerpt="19 ноября в 20:30 концерт.",
            title="Джаз с характером: Илья Пищенко",
            date="2099-11-19",
            time="20:30",
            location_name="Сигнал",
            city="Калининград",
            ticket_link="https://signalcommunity.timepad.ru/event/3806838/",
            posters=[
                PosterCandidate(
                    catbox_url="https://example.com/poster.jpg",
                    ocr_text="17/02\n20/30",
                )
            ],
        )

        res = await smart_event_update(db, candidate, check_source_url=False, schedule_tasks=False)
        assert res.status == "rejected_far_future_low_confidence"
        assert res.reason is not None
        assert "poster_date_mismatch" in res.reason
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_smart_update_rejects_far_future_event_with_specific_ticket_but_without_explicit_year(
    tmp_path,
    monkeypatch,
):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    try:
        await _patch_smart_update_offline(monkeypatch)

        candidate = EventCandidate(
            source_type="vk",
            source_url="https://vk.com/wall-1_4",
            source_text=(
                "19 ноября в 20:30 концерт.\n"
                "Билеты: https://signalcommunity.timepad.ru/event/3806838/"
            ),
            raw_excerpt="19 ноября в 20:30 концерт.",
            title="Джаз с характером: Илья Пищенко",
            date="2099-11-19",
            time="20:30",
            location_name="Сигнал",
            city="Калининград",
            ticket_link="https://signalcommunity.timepad.ru/event/3806838/",
            posters=[
                PosterCandidate(
                    catbox_url="https://example.com/poster.jpg",
                    ocr_text="19/11\n20:30",
                )
            ],
        )

        res = await smart_event_update(db, candidate, check_source_url=False, schedule_tasks=False)
        assert res.status == "rejected_far_future_low_confidence"
        assert res.reason == "far_future:low_grounding_score:1"
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_smart_update_allows_far_future_event_with_explicit_year_signal(tmp_path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    try:
        await _patch_smart_update_offline(monkeypatch)

        candidate = EventCandidate(
            source_type="telegram",
            source_url="https://t.me/test/2",
            source_text=(
                "19 ноября 2099 в 20:30 состоится концерт Ильи Пищенко.\n"
                "Билеты: https://signalcommunity.timepad.ru/event/9999999/"
            ),
            raw_excerpt="19 ноября 2099 в 20:30 состоится концерт.",
            title="Джаз с характером: Илья Пищенко",
            date="2099-11-19",
            time="20:30",
            location_name="Сигнал",
            city="Калининград",
            ticket_link="https://signalcommunity.timepad.ru/event/9999999/",
            posters=[
                PosterCandidate(
                    catbox_url="https://example.com/poster.jpg",
                    ocr_text="19/11\n20:30",
                )
            ],
        )

        res = await smart_event_update(db, candidate, check_source_url=False, schedule_tasks=False)
        assert res.status == "created"
        assert res.event_id is not None
    finally:
        await db.close()
