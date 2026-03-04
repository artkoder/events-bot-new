import pytest

from sqlalchemy import select

from db import Database
from models import Event
from smart_event_update import EventCandidate, smart_event_update
import smart_event_update as su


@pytest.mark.asyncio
async def test_parser_related_title_match_uses_date_time_location_before_llm(
    tmp_path, monkeypatch
):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    monkeypatch.setenv("REGION_FILTER_ENABLED", "0")
    monkeypatch.setattr(su, "SMART_UPDATE_LLM_DISABLED", True)

    async def _no_merge(*_args, **_kwargs):  # noqa: ANN001 - test helper
        return None

    async def _no_rewrite(*_args, **_kwargs):  # noqa: ANN001 - test helper
        return None

    async def _no_digest(*_args, **_kwargs):  # noqa: ANN001 - test helper
        return None

    async def _no_short(*_args, **_kwargs):  # noqa: ANN001 - test helper
        return None

    async def _no_topics(*_args, **_kwargs):  # noqa: ANN001 - test helper
        return None

    monkeypatch.setattr(su, "_llm_merge_event", _no_merge)
    monkeypatch.setattr(su, "_rewrite_description_journalistic", _no_rewrite)
    monkeypatch.setattr(su, "_llm_build_search_digest", _no_digest)
    monkeypatch.setattr(su, "_llm_build_short_description", _no_short)
    monkeypatch.setattr(su, "_classify_topics", _no_topics)

    async with db.get_session() as session:
        session.add(
            Event(
                id=1,
                title="Гегель",
                description="Лекция о философии истории.",
                date="2026-03-03",
                time="18:30",
                location_name="Дом китобоя",
                city="Калининград",
                source_text="Исходный анонс лекции.",
            )
        )
        session.add(
            Event(
                id=2,
                title="Другая лекция",
                description="Нерелевантное событие.",
                date="2026-03-03",
                time="20:00",
                location_name="Дом китобоя",
                city="Калининград",
                source_text="Другое событие на той же площадке.",
            )
        )
        await session.commit()

    candidate = EventCandidate(
        source_type="parser:domkitoboya",
        source_url="https://example.com/hegel",
        source_text="Андрей Тесля расскажет о философии истории Гегеля.",
        raw_excerpt="Лекция о философии истории.",
        title="Гегель: философия истории",
        date="2026-03-03",
        time="18:30",
        location_name="Дом китобоя",
        city="Калининград",
        trust_level="high",
    )

    result = await smart_event_update(
        db,
        candidate,
        check_source_url=True,
        schedule_tasks=False,
    )

    assert result.status == "merged"
    assert result.event_id == 1

    async with db.get_session() as session:
        rows = (await session.execute(select(Event).order_by(Event.id))).scalars().all()
        assert [int(row.id) for row in rows] == [1, 2]
        merged = await session.get(Event, 1)
        assert merged is not None
        assert merged.source_post_url == "https://example.com/hegel"
