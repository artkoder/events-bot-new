from __future__ import annotations

import pytest
from sqlalchemy import select

from db import Database
from models import TicketSiteQueueItem
import smart_event_update as su
from smart_event_update import EventCandidate, smart_event_update


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
async def test_smart_update_enqueues_ticket_site_url_from_source_text_with_event_id(tmp_path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    await _patch_smart_update_offline(monkeypatch)

    candidate = EventCandidate(
        source_type="bot",
        source_url="https://example.com/post/1",
        source_text=(
            "Билеты: https://pyramida.info/tickets/kino-tribyut_55746464). "
            "Ждём вас на концерте."
        ),
        raw_excerpt="",
        title="Трибьют-концерт группы Кино",
        date="2026-02-21",
        time="20:00",
        location_name="Калининград",
        city="Калининград",
        trust_level="high",
    )

    result = await smart_event_update(db, candidate, check_source_url=True, schedule_tasks=False)
    assert result.status == "created"
    assert result.event_id is not None

    normalized_url = "https://pyramida.info/tickets/kino-tribyut_55746464"
    async with db.get_session() as session:
        row = (
            await session.execute(
                select(TicketSiteQueueItem).where(TicketSiteQueueItem.url == normalized_url)
            )
        ).scalar_one_or_none()
        assert row is not None
        assert str(row.site_kind) == "pyramida"
        assert int(row.event_id or 0) == int(result.event_id)
        assert str(row.source_post_url or "") == candidate.source_url


@pytest.mark.asyncio
async def test_smart_update_enqueues_ticket_site_url_from_links_payload(tmp_path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    await _patch_smart_update_offline(monkeypatch)

    qtickets_url = "https://qtickets.events/event/12345"
    candidate = EventCandidate(
        source_type="vk",
        source_url="https://vk.com/wall-1_42",
        source_text="Анонс события.",
        raw_excerpt="",
        title="Музыкальный вечер",
        date="2026-03-05",
        time="19:00",
        location_name="Дом искусств",
        city="Калининград",
        trust_level="medium",
        links_payload=[{"url": qtickets_url}],
    )

    result = await smart_event_update(db, candidate, check_source_url=True, schedule_tasks=False)
    assert result.status == "created"
    assert result.event_id is not None

    async with db.get_session() as session:
        row = (
            await session.execute(
                select(TicketSiteQueueItem).where(TicketSiteQueueItem.url == qtickets_url)
            )
        ).scalar_one_or_none()
        assert row is not None
        assert str(row.site_kind) == "qtickets"
        assert int(row.event_id or 0) == int(result.event_id)
        assert str(row.source_post_url or "") == candidate.source_url
