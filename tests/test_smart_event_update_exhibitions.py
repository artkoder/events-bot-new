from __future__ import annotations

import pytest
from sqlalchemy import select

from db import Database
from models import Event, EventSource, EventSourceFact
import smart_event_update as su
from smart_event_update import EventCandidate, smart_event_update


def _base_event(**overrides: object) -> Event:
    payload = {
        "title": "Выставка TEST",
        "description": "Описание выставки.",
        "date": "2026-02-01",
        "time": "",
        "end_date": "2026-03-01",
        "location_name": "Галерея TEST",
        "city": "Калининград",
        "event_type": "выставка",
        "source_text": "Базовый текст выставки.",
    }
    payload.update(overrides)
    return Event(**payload)


async def _patch_llm_minimal(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _merge(*args, **kwargs):  # noqa: ANN001 - test helper
        return {
            "title": None,
            "description": "Обновлённое описание выставки.",
            "ticket_link": None,
            "ticket_price_min": None,
            "ticket_price_max": None,
            "ticket_status": None,
            "added_facts": ["Добавлен факт о выставке."],
            "duplicate_facts": [],
            "conflict_facts": [],
            "skipped_conflicts": [],
        }

    async def _no_rewrite(*args, **kwargs):  # noqa: ANN001 - test helper
        return None

    async def _no_digest(*args, **kwargs):  # noqa: ANN001 - test helper
        return None

    async def _no_topics(*args, **kwargs):  # noqa: ANN001 - test helper
        return None

    monkeypatch.setattr(su, "_llm_merge_event", _merge)
    monkeypatch.setattr(su, "_rewrite_description_journalistic", _no_rewrite)
    monkeypatch.setattr(su, "_llm_build_search_digest", _no_digest)
    monkeypatch.setattr(su, "_classify_topics", _no_topics)


@pytest.mark.asyncio
async def test_exhibition_merges_when_candidate_date_is_inside_existing_period(
    tmp_path, monkeypatch
):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    await _patch_llm_minimal(monkeypatch)

    async with db.get_session() as session:
        session.add(
            _base_event(
                id=1,
                title="Выставка Фигаро",
                date="2026-01-15",
                end_date="2026-03-01",
                source_text="Открытие выставки Фигаро.",
            )
        )
        await session.commit()

    # Candidate date is later than the original start date, but within end_date range.
    # Smart Update must merge into the existing exhibition instead of creating a duplicate.
    candidate = EventCandidate(
        source_type="telegram",
        source_url="https://t.me/vorotagallery/21",
        source_text="Выставка продолжается до конца марта. Добавлены новые работы.",
        raw_excerpt="Выставка продолжается.",
        title="Выставка Фигаро",
        date="2026-02-20",
        end_date="2026-03-31",
        time="",
        location_name="Галерея TEST",
        city="Калининград",
        event_type="выставка",
        trust_level="medium",
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
        events = (await session.execute(select(Event))).scalars().all()
        assert len(events) == 1
        merged = events[0]
        assert merged.end_date == "2026-03-31"
        sources = (
            await session.execute(
                select(EventSource).where(EventSource.event_id == int(merged.id or 0))
            )
        ).scalars().all()
        assert len(sources) == 1
        assert sources[0].source_url == "https://t.me/vorotagallery/21"


@pytest.mark.asyncio
async def test_exhibition_end_date_extension_respects_trust_priority(
    tmp_path, monkeypatch
):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    await _patch_llm_minimal(monkeypatch)

    async with db.get_session() as session:
        session.add(
            _base_event(
                id=1,
                title="Выставка Длительная",
                date="2026-03-01",
                end_date="2026-04-30",
                source_text="Официальный анонс выставки.",
            )
        )
        session.add(
            EventSource(
                event_id=1,
                source_type="telegram",
                source_url="https://t.me/source/high",
                source_text="Официальный анонс выставки.",
                trust_level="high",
            )
        )
        await session.commit()

    candidate = EventCandidate(
        source_type="vk",
        source_url="https://vk.ru/wall-1_100",
        source_text="Выставка продлена до конца мая.",
        raw_excerpt="Выставка продлена.",
        title="Выставка Длительная",
        date="2026-04-10",
        end_date="2026-05-31",
        time="",
        location_name="Галерея TEST",
        city="Калининград",
        event_type="выставка",
        trust_level="low",
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
        event = await session.get(Event, 1)
        assert event is not None
        # Lower-trust source must not override high-trust end_date.
        assert event.end_date == "2026-04-30"
        conflicts = (
            await session.execute(
                select(EventSourceFact).where(
                    EventSourceFact.event_id == 1,
                    EventSourceFact.status == "conflict",
                )
            )
        ).scalars().all()
        assert any("Дата окончания:" in str(row.fact or "") for row in conflicts)


@pytest.mark.asyncio
async def test_exhibition_without_end_date_gets_default_one_month_period(
    tmp_path, monkeypatch
):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    await _patch_llm_minimal(monkeypatch)

    created = EventCandidate(
        source_type="telegram",
        source_url="https://t.me/vorotagallery/21",
        source_text="Открытие выставки современного искусства.",
        raw_excerpt="Открытие выставки.",
        title="Выставка TEST default end date",
        date="2026-01-10",
        end_date=None,
        time="",
        location_name="Галерея TEST",
        city="Калининград",
        event_type="выставка",
        trust_level="medium",
    )

    create_result = await smart_event_update(
        db,
        created,
        check_source_url=True,
        schedule_tasks=False,
    )
    assert create_result.status == "created"

    async with db.get_session() as session:
        event = await session.get(Event, int(create_result.event_id or 0))
        assert event is not None
        assert event.end_date == "2026-02-10"

    # Later source with explicit end_date must update the same exhibition, not create a duplicate.
    extended = EventCandidate(
        source_type="telegram",
        source_url="https://t.me/vorotagallery/22",
        source_text="Выставка продлена до конца марта.",
        raw_excerpt="Выставка продлена.",
        title="Выставка TEST default end date",
        date="2026-01-10",
        end_date="2026-03-31",
        time="",
        location_name="Галерея TEST",
        city="Калининград",
        event_type="выставка",
        trust_level="medium",
    )
    merge_result = await smart_event_update(
        db,
        extended,
        check_source_url=True,
        schedule_tasks=False,
    )
    assert merge_result.status == "merged"
    assert merge_result.event_id == create_result.event_id

    async with db.get_session() as session:
        events = (await session.execute(select(Event))).scalars().all()
        assert len(events) == 1
        assert events[0].end_date == "2026-03-31"


@pytest.mark.asyncio
async def test_single_day_event_misclassified_as_exhibition_does_not_get_default_end_date(
    tmp_path, monkeypatch
):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    await _patch_llm_minimal(monkeypatch)

    candidate = EventCandidate(
        source_type="vk",
        source_url="https://vk.com/wall-1_1",
        source_text="Модный показ. 20 февраля 2026. Начало в 19:00.",
        raw_excerpt="Модный показ 20 февраля.",
        title="Модный показ",
        date="2026-02-20",
        end_date=None,
        time="19:00",
        location_name="Галерея TEST",
        city="Калининград",
        # Upstream LLM bug: misclassified as 'выставка' should NOT trigger +1 month.
        event_type="выставка",
        trust_level="medium",
    )

    result = await smart_event_update(
        db,
        candidate,
        check_source_url=True,
        schedule_tasks=False,
    )
    assert result.status == "created"
    assert result.event_id is not None

    async with db.get_session() as session:
        event = await session.get(Event, int(result.event_id or 0))
        assert event is not None
        assert event.end_date in (None, "")


@pytest.mark.asyncio
async def test_exhibition_with_event_anchors_is_not_skipped_as_promo(
    tmp_path, monkeypatch
):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    await _patch_llm_minimal(monkeypatch)

    candidate = EventCandidate(
        source_type="telegram",
        source_url="https://t.me/vorotagallery/21",
        source_text="Выставка. Акция открытия: вход свободный в первый день.",
        raw_excerpt="Выставка в галерее.",
        title="Выставка TEST promo guard",
        date="2026-02-15",
        end_date=None,
        time="",
        location_name="Галерея TEST",
        city="Калининград",
        event_type="выставка",
        trust_level="medium",
    )

    result = await smart_event_update(
        db,
        candidate,
        check_source_url=True,
        schedule_tasks=False,
    )
    assert result.status == "created"
    assert result.event_id is not None


@pytest.mark.asyncio
async def test_merge_normalizes_existing_english_event_type_alias(
    tmp_path, monkeypatch
):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    await _patch_llm_minimal(monkeypatch)
    async def _force_match(*args, **kwargs):  # noqa: ANN001 - test helper
        return 1, 1.0, "forced_for_test"
    monkeypatch.setattr(su, "_llm_match_event", _force_match)

    async with db.get_session() as session:
        session.add(
            _base_event(
                id=1,
                title="Пять веков русского искусства",
                date="2026-02-03",
                end_date="2026-04-03",
                location_name="Третьяковская галерея",
                event_type="exhibition",
                source_text="Исходный английский alias event_type.",
            )
        )
        await session.commit()

    candidate = EventCandidate(
        source_type="telegram",
        source_url="https://t.me/tretyakovka_kaliningrad/2391",
        source_text="Остается пара месяцев до окончания выставки.",
        raw_excerpt="Выставка продолжается.",
        title="Пять веков русского искусства",
        date="2026-02-03",
        end_date="2026-04-03",
        time="",
        location_name="Третьяковская галерея",
        city="Калининград",
        event_type="выставка",
        trust_level="medium",
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
        event = await session.get(Event, 1)
        assert event is not None
        assert event.event_type == "выставка"


@pytest.mark.asyncio
async def test_source_facts_are_replaced_for_same_event_and_source_url(tmp_path):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    async with db.get_session() as session:
        session.add(
            Event(
                id=1,
                title="TEST source facts replace",
                description="Описание",
                date="2026-02-11",
                time="19:00",
                location_name="Локация TEST",
                city="Калининград",
                source_text="Текст источника",
            )
        )
        session.add(
            EventSource(
                event_id=1,
                source_type="telegram",
                source_url="https://t.me/example/100",
                source_text="Пост 100",
            )
        )
        await session.commit()

    candidate = EventCandidate(
        source_type="telegram",
        source_url="https://t.me/example/100",
        source_text="Пост 100",
        raw_excerpt="Пост 100",
        title="TEST source facts replace",
        date="2026-02-11",
        end_date=None,
        time="19:00",
        location_name="Локация TEST",
        city="Калининград",
        event_type="выставка",
        trust_level="medium",
    )

    async with db.get_session() as session:
        first_added = await su._record_source_facts(
            session,
            1,
            candidate,
            [("Факт A", "added"), ("Факт B", "duplicate")],
        )
        await session.commit()
    assert first_added == 2

    async with db.get_session() as session:
        second_added = await su._record_source_facts(
            session,
            1,
            candidate,
            [("Факт C", "added")],
        )
        await session.commit()
    assert second_added == 1

    async with db.get_session() as session:
        rows = (
            await session.execute(
                select(EventSourceFact.fact, EventSourceFact.status).where(
                    EventSourceFact.event_id == 1
                )
            )
        ).all()
    assert rows == [("Факт C", "added")]


@pytest.mark.asyncio
async def test_single_candidate_title_mismatch_does_not_merge_long_event(
    tmp_path, monkeypatch
):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    async def _no_match(*args, **kwargs):  # noqa: ANN001 - test helper
        return None, 0.0, "no_match_for_test"

    async def _no_rewrite(*args, **kwargs):  # noqa: ANN001 - test helper
        return None

    async def _no_digest(*args, **kwargs):  # noqa: ANN001 - test helper
        return None

    async def _no_facts(*args, **kwargs):  # noqa: ANN001 - test helper
        return []

    async def _no_topics(*args, **kwargs):  # noqa: ANN001 - test helper
        return None

    monkeypatch.setattr(su, "_llm_match_event", _no_match)
    monkeypatch.setattr(su, "_rewrite_description_journalistic", _no_rewrite)
    monkeypatch.setattr(su, "_llm_build_search_digest", _no_digest)
    monkeypatch.setattr(su, "_llm_extract_candidate_facts", _no_facts)
    monkeypatch.setattr(su, "_classify_topics", _no_topics)

    async with db.get_session() as session:
        session.add(
            Event(
                id=1,
                title="Космос красного",
                description="Исходная выставка музея.",
                date="2026-01-30",
                time="00:00",
                end_date="2026-04-05",
                location_name="Калининградский музей изобразительных искусств",
                location_address="Ленинский проспект 83",
                city="Калининград",
                event_type="выставка",
                source_text="Исходный текст.",
            )
        )
        session.add(
            EventSource(
                event_id=1,
                source_type="site",
                source_url="https://vk.com/wall-9118984_23257",
                source_text="Исходный текст.",
                trust_level="high",
            )
        )
        await session.commit()

    candidate = EventCandidate(
        source_type="telegram",
        source_url="https://t.me/meowafisha/6657",
        source_text="1 марта творческий вечер «Оттепель». Начало в 17:00.",
        raw_excerpt="Творческий вечер «Оттепель».",
        title="Оттепель",
        date="2026-03-01",
        end_date=None,
        time="17:00",
        location_name="Калининград",
        city="Калининград",
        event_type=None,
        trust_level="low",
    )

    result = await smart_event_update(
        db,
        candidate,
        check_source_url=False,
        schedule_tasks=False,
    )

    assert result.status == "created"
    assert result.event_id is not None
    assert int(result.event_id) != 1

    async with db.get_session() as session:
        original = await session.get(Event, 1)
        assert original is not None
        assert original.title == "Космос красного"
        events = (await session.execute(select(Event))).scalars().all()
        assert len(events) == 2
