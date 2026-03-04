from __future__ import annotations

import pytest
from sqlalchemy import select

from db import Database
from models import Event, EventSource
import smart_event_update as su
from smart_event_update import EventCandidate, smart_event_update


async def _no_topics(*args, **kwargs):  # noqa: ANN001 - test helper
    return None


@pytest.mark.asyncio
async def test_rescue_bundle_title_prevents_duplicate_when_candidate_title_is_weak(
    tmp_path, monkeypatch
):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    # Keep the test deterministic (no network calls).
    monkeypatch.setenv("REGION_FILTER_ENABLED", "0")
    monkeypatch.setattr(su, "SMART_UPDATE_LLM_DISABLED", True)
    monkeypatch.setattr(su, "_classify_topics", _no_topics)

    async def _combo(*args, **kwargs):  # noqa: ANN001 - test helper
        return {
            "action": "create",
            "match_event_id": None,
            "confidence": 0.2,
            "reason_short": "create_for_test",
            "bundle": {
                "title": "Королева Луиза: идеал или красивая легенда?",
                "description": "Описание.",
                "facts": [],
                "search_digest": None,
                "short_description": None,
            },
        }

    async def _merge(*args, **kwargs):  # noqa: ANN001 - test helper
        return {
            "title": None,
            "description": None,
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

    monkeypatch.setattr(su, "_llm_match_or_create_bundle", _combo)
    monkeypatch.setattr(su, "_llm_merge_event", _merge)

    async with db.get_session() as session:
        session.add(
            Event(
                id=1,
                title="Выставка «Королева Луиза: идеал или красивая легенда?»",
                description="Описание.",
                date="2026-03-10",
                time="10:00",
                end_date="2026-06-15",
                location_name="Музей Мирового океана",
                location_address="наб. Петра Великого, 5",
                city="Калининград",
                event_type="выставка",
                source_text="Исходный текст.",
            )
        )
        await session.commit()

    candidate = EventCandidate(
        source_type="vk",
        source_url="https://vk.com/wall-53460968_11193",
        source_text="10 марта откроют выставку «Королева Луиза: идеал или красивая легенда?»",
        raw_excerpt="",
        # Weak/garbled title: forces match/create step to choose create in this test.
        title="Выставка",
        date="2026-03-10",
        time="",
        location_name="Музей Мирового океана",
        location_address="наб. Петра Великого, 5",
        city="Калининград",
        event_type="выставка",
    )

    res = await smart_event_update(db, candidate, check_source_url=False, schedule_tasks=False)
    assert res.status == "merged"
    assert res.event_id == 1

    async with db.get_session() as session:
        events = (await session.execute(select(Event))).scalars().all()
        assert len(events) == 1
        sources = (
            await session.execute(select(EventSource).where(EventSource.event_id == 1))
        ).scalars().all()
        assert any(s.source_url == candidate.source_url for s in sources)

