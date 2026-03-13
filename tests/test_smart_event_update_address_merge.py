import pytest

from sqlalchemy import select

from db import Database
from models import Event
from smart_event_update import EventCandidate, smart_event_update
import smart_event_update as su


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


@pytest.mark.asyncio
async def test_exact_title_merge_uses_matching_address_when_venue_alias_differs(
    tmp_path, monkeypatch
):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    monkeypatch.setenv("REGION_FILTER_ENABLED", "0")
    monkeypatch.setattr(su, "SMART_UPDATE_LLM_DISABLED", True)
    monkeypatch.setattr(su, "_llm_merge_event", _no_merge)
    monkeypatch.setattr(su, "_rewrite_description_journalistic", _no_rewrite)
    monkeypatch.setattr(su, "_llm_build_search_digest", _no_digest)
    monkeypatch.setattr(su, "_llm_extract_candidate_facts", _no_facts)
    monkeypatch.setattr(su, "_classify_topics", _no_topics)

    async with db.get_session() as session:
        session.add(
            Event(
                id=1,
                title="В поисках Нарнии",
                description="Телеграм-источник.",
                date="2026-03-13",
                time="18:00",
                location_name="Гусевский историко-краеведческий музей имени А.М. Иванова",
                location_address="ул. Московская, 36А",
                city="Гусев",
                source_text="Первый анонс.",
                source_post_url="https://t.me/zamokinsterburg/5434",
                source_message_id=5434,
            )
        )
        await session.commit()

    candidate = EventCandidate(
        source_type="vk",
        source_url="https://vk.com/wall-218733829_2846",
        source_text="ВК-пост о том же показе.",
        raw_excerpt="Благотворительное представление.",
        title="В поисках Нарнии",
        date="2026-03-13",
        time="18:00",
        location_name="Гусевский музей",
        location_address="Московская 36а",
        city="Гусев",
        trust_level="high",
    )

    res = await smart_event_update(db, candidate, check_source_url=False, schedule_tasks=False)

    assert res.status == "merged"
    assert res.event_id == 1

    async with db.get_session() as session:
        rows = (await session.execute(select(Event))).scalars().all()
        assert len(rows) == 1
        merged = await session.get(Event, 1)
        assert merged is not None
        assert merged.location_name == "Гусевский историко-краеведческий музей имени А.М. Иванова"
