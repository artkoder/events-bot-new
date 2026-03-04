from __future__ import annotations

import pytest

from db import Database
from models import Event
import smart_event_update as su
from smart_event_update import EventCandidate, smart_event_update


async def _no_topics(*args, **kwargs):  # noqa: ANN001 - test helper
    return None


@pytest.mark.asyncio
async def test_create_shrinks_overexpanded_description_to_source_budget(tmp_path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    monkeypatch.setattr(su, "_classify_topics", _no_topics)
    monkeypatch.setattr(su, "SMART_UPDATE_FACT_FIRST", False)

    async def _overlong_bundle(*args, **kwargs):  # noqa: ANN001 - test helper
        long_desc = "\n\n".join(
            [f"Абзац {i}: подробности события без воды." for i in range(1, 180)]
        )
        return {
            "title": "Тестовое событие",
            "description": long_desc,
            "facts": ["Факт 1", "Факт 2"],
            "search_digest": "Короткий дайджест.",
            "short_description": "Нейтральный анонс о событии с программой и участниками, без лишних деталей, коротко.",
        }

    called = {"shrink": 0}

    async def _fake_shrink(**kwargs):  # noqa: ANN001 - test helper
        called["shrink"] += 1
        return "Короткое описание без воды."

    monkeypatch.setattr(su, "_llm_create_description_facts_and_digest", _overlong_bundle)
    monkeypatch.setattr(su, "_llm_shrink_description_to_budget", _fake_shrink)

    candidate = EventCandidate(
        source_type="telegram",
        source_url="https://t.me/test/1",
        source_text=("Короткий исходник. " * 12).strip(),
        raw_excerpt="",
        title="Событие",
        date="2026-02-20",
        time="19:00",
        location_name="Дом культуры",
        city="Калининград",
    )

    res = await smart_event_update(db, candidate, check_source_url=True, schedule_tasks=False)
    assert res.status == "created"
    assert res.event_id is not None
    assert called["shrink"] == 1

    async with db.get_session() as session:
        ev = await session.get(Event, int(res.event_id))
        assert ev is not None
        assert (ev.description or "").strip() == "Короткое описание без воды."
