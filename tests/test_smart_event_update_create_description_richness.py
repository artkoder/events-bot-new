import pytest

from db import Database
from models import Event
import smart_event_update as su
from smart_event_update import EventCandidate, PosterCandidate, smart_event_update


async def _no_topics(*args, **kwargs):  # noqa: ANN001 - test helper
    return None


@pytest.mark.asyncio
async def test_create_rewrites_full_when_description_too_short_vs_rich_source_text(
    tmp_path, monkeypatch
):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    monkeypatch.setattr(su, "_classify_topics", _no_topics)
    monkeypatch.setattr(su, "SMART_UPDATE_FACT_FIRST", False)

    async def _short_create_bundle(*args, **kwargs):  # noqa: ANN001 - test helper
        return {
            "description": "Коротко.",
            "facts": ["Факт 1"],
            "search_digest": "Развёрнутое событие с насыщенной программой и участниками.",
            "short_description": "Нейтральный анонс о событии с программой и участниками, без лишних деталей, коротко.",
        }

    called = {"full": 0}

    async def _full_rewrite(*args, **kwargs):  # noqa: ANN001 - test helper
        called["full"] += 1
        # Must be long enough to pass the min_expected * 0.75 guard.
        return ("Это подробное описание события. " * 40).strip()

    monkeypatch.setattr(su, "_llm_create_description_facts_and_digest", _short_create_bundle)
    monkeypatch.setattr(su, "_rewrite_description_full_from_sources", _full_rewrite)

    candidate = EventCandidate(
        source_type="vk",
        source_url="https://vk.com/wall-1_1",
        source_text=("Очень длинный исходный текст. " * 120).strip(),
        raw_excerpt="",
        title="Большое событие",
        date="2026-02-20",
        time="19:00",
        location_name="Дом культуры",
        city="Калининград",
    )

    res = await smart_event_update(db, candidate, check_source_url=True, schedule_tasks=False)
    assert res.status == "created"
    assert res.event_id is not None
    assert called["full"] == 1

    async with db.get_session() as session:
        ev = await session.get(Event, int(res.event_id))
        assert ev is not None
        assert len((ev.description or "").strip()) >= 500
        assert "Коротко" not in (ev.description or "")


@pytest.mark.asyncio
async def test_create_can_use_poster_ocr_as_rich_source_for_full_rewrite(
    tmp_path, monkeypatch
):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    monkeypatch.setattr(su, "_classify_topics", _no_topics)
    monkeypatch.setattr(su, "SMART_UPDATE_FACT_FIRST", False)

    async def _short_create_bundle(*args, **kwargs):  # noqa: ANN001 - test helper
        return {
            "description": "Коротко.",
            "facts": ["Факт 1"],
            "search_digest": "Событие по материалам афиши.",
            "short_description": "Нейтральный анонс о событии с программой и участниками, без лишних деталей, коротко.",
        }

    called = {"full": 0}

    async def _full_rewrite(*args, **kwargs):  # noqa: ANN001 - test helper
        called["full"] += 1
        return ("Подробности по афише. " * 35).strip()

    monkeypatch.setattr(su, "_llm_create_description_facts_and_digest", _short_create_bundle)
    monkeypatch.setattr(su, "_rewrite_description_full_from_sources", _full_rewrite)

    candidate = EventCandidate(
        source_type="vk",
        source_url="https://vk.com/wall-1_2",
        source_text="",
        raw_excerpt="",
        title="Событие с афиши",
        date="2026-03-02",
        time="20:00",
        location_name="Театр",
        city="Калининград",
        posters=[
            PosterCandidate(
                ocr_text=("Текст афиши с программой и деталями. " * 90).strip(),
            )
        ],
    )

    res = await smart_event_update(db, candidate, check_source_url=True, schedule_tasks=False)
    assert res.status == "created"
    assert res.event_id is not None
    assert called["full"] == 1

    async with db.get_session() as session:
        ev = await session.get(Event, int(res.event_id))
        assert ev is not None
        assert len((ev.description or "").strip()) >= 400


@pytest.mark.asyncio
async def test_rewrite_journalistic_runs_when_only_poster_ocr_present(monkeypatch):
    async def _fake_ask(*args, **kwargs):  # noqa: ANN001 - test helper
        return "Нормальный текст."

    monkeypatch.setattr(su, "_ask_gemma_text", _fake_ask)

    candidate = EventCandidate(
        source_type="vk",
        source_url="test://poster-only",
        source_text="",
        raw_excerpt="",
        title="Событие",
        date="2026-03-02",
        time="20:00",
        location_name="Театр",
        city="Калининград",
        posters=[PosterCandidate(ocr_text="Афиша: подробное описание. " * 10)],
    )

    out = await su._rewrite_description_journalistic(candidate)
    assert (out or "").strip() == "Нормальный текст."
