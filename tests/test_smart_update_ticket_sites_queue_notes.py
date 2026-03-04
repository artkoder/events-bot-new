from __future__ import annotations

import pytest
from sqlalchemy import select

from db import Database
from models import TicketSiteQueueItem
import smart_event_update as su
from smart_event_update import EventCandidate, smart_event_update


async def _no_topics(*args, **kwargs):  # noqa: ANN001 - test helper
    return None


@pytest.mark.asyncio
async def test_smart_update_enqueues_ticket_site_url_and_returns_queue_notes(tmp_path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    monkeypatch.setattr(su, "_classify_topics", _no_topics)

    async def _bundle(*args, **kwargs):  # noqa: ANN001 - test helper
        return {
            "title": "Трибьют-концерт группы Кино",
            "description": "Описание события.",
            "facts": ["Факт 1"],
            "search_digest": "Дайджест.",
        }

    monkeypatch.setattr(su, "_llm_create_description_facts_and_digest", _bundle)

    url = "https://pyramida.info/tickets/kino-tribyut_55746464"
    candidate = EventCandidate(
        source_type="bot",
        source_url="https://example.com/source/1",
        source_text=f"Билеты: {url}",
        raw_excerpt="",
        title="Трибьют-концерт группы Кино",
        date="2026-02-21",
        time="20:00",
        location_name="Калининград",
        city="Калининград",
        ticket_link=url,
        trust_level="high",
    )

    res = await smart_event_update(db, candidate, check_source_url=True, schedule_tasks=False)
    assert res.status == "created"
    assert res.event_id is not None
    assert any("очередь мониторинга билетных сайтов" in (n or "") for n in (res.queue_notes or []))
    assert any(url in (n or "") for n in (res.queue_notes or []))

    async with db.get_session() as session:
        row = (
            await session.execute(select(TicketSiteQueueItem).where(TicketSiteQueueItem.url == url))
        ).scalar_one_or_none()
        assert row is not None
        assert str(row.site_kind) == "pyramida"
        assert int(row.event_id or 0) == int(res.event_id)
