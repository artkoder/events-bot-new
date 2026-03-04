from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest
from sqlalchemy import select

from db import Database
from models import Event, EventSource, EventSourceFact
import smart_event_update as su
from smart_event_update import EventCandidate, smart_event_update


@pytest.mark.asyncio
async def test_fact_first_merge_backfills_legacy_description_facts(tmp_path, monkeypatch) -> None:
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    event_date = (date.today() + timedelta(days=10)).isoformat()
    legacy_description = (
        "Старое описание события (из базы, до Smart Update). "
        "Это достаточно длинный текст, чтобы безопасно извлечь базовые факты из legacy-описания.\n\n"
        "Деталь А.\n"
        "Деталь Б.\n"
        "Ещё одна нейтральная фраза для длины, без новых утверждений."
    )

    async with db.get_session() as session:
        session.add(
            Event(
                id=1,
                title="Событие",
                description=legacy_description,
                date=event_date,
                time="19:00",
                location_name="Venue",
                city="Калининград",
                source_text="legacy source",
            )
        )
        # Simulate pre-fact-first backfill: legacy source exists, but only a note fact is stored.
        legacy_source = EventSource(
            event_id=1,
            source_type="legacy",
            source_url="legacy:event_description:1",
            source_text=legacy_description,
            imported_at=datetime.now(timezone.utc),
            trust_level="high",
        )
        session.add(legacy_source)
        await session.flush()
        session.add(
            EventSourceFact(
                event_id=1,
                source_id=int(legacy_source.id or 0),
                fact="Снапшот описания до Smart Update сохранён",
                status="note",
                created_at=datetime.now(timezone.utc),
            )
        )
        await session.commit()

    monkeypatch.setattr(su, "SMART_UPDATE_LLM_DISABLED", False)
    monkeypatch.setattr(su, "SMART_UPDATE_FACT_FIRST", True)

    async def _fake_extract(candidate, *, text_for_facts=None):  # noqa: ANN001
        assert getattr(candidate, "source_type", None) == "legacy"
        assert "Старое описание" in str(text_for_facts or "")
        return ["Деталь А", "Деталь Б"]

    captured: dict[str, object] = {}

    async def _fake_fact_first_desc(**kwargs):  # noqa: ANN003 - test helper
        captured["facts_text_clean"] = list(kwargs.get("facts_text_clean") or [])
        return "FF_DESC"

    async def _fake_merge(*_args, **_kwargs):  # noqa: ANN001
        return {
            "title": None,
            "description": "ignored by fact-first",
            "added_facts": ["Новый факт"],
            "duplicate_facts": [],
            "conflict_facts": [],
            "skipped_conflicts": [],
        }

    async def _no_topics(*_args, **_kwargs):  # noqa: ANN001
        return None

    async def _no_holidays(*_args, **_kwargs):  # noqa: ANN001
        return False

    async def _fake_short(*_args, **_kwargs):  # noqa: ANN001
        return "Короткое описание события без логистики, чтобы тест прошёл корректно сейчас всегда точно."

    async def _fake_digest(*_args, **_kwargs):  # noqa: ANN001
        return "Краткий дайджест для поиска."

    monkeypatch.setattr(su, "_llm_extract_candidate_facts", _fake_extract)
    monkeypatch.setattr(su, "_llm_fact_first_description_md", _fake_fact_first_desc)
    monkeypatch.setattr(su, "_llm_merge_event", _fake_merge)
    monkeypatch.setattr(su, "_classify_topics", _no_topics)
    monkeypatch.setattr(su, "_apply_holiday_festival_mapping", _no_holidays)
    monkeypatch.setattr(su, "_llm_build_short_description", _fake_short)
    monkeypatch.setattr(su, "_llm_build_search_digest", _fake_digest)

    candidate = EventCandidate(
        source_type="telegram",
        source_url="test://tg/1",
        source_text=(
            "Новая дополнительная деталь, которой точно не было в прошлом описании события."
        ),
        raw_excerpt="",
        title="Событие",
        date=event_date,
        time="19:00",
        location_name="Venue",
        city="Калининград",
    )

    res = await smart_event_update(
        db,
        candidate,
        check_source_url=True,
        schedule_tasks=False,
    )
    assert res.status == "merged"
    assert res.event_id == 1

    async with db.get_session() as session:
        ev = await session.get(Event, 1)
        assert ev is not None
        assert (ev.description or "").strip() == "FF_DESC"

        legacy_source = (
            await session.execute(
                select(EventSource).where(
                    EventSource.event_id == 1,
                    EventSource.source_type == "legacy",
                )
            )
        ).scalar_one_or_none()
        assert legacy_source is not None

        rows = (
            await session.execute(
                select(EventSourceFact.fact, EventSourceFact.status).where(
                    EventSourceFact.source_id == int(legacy_source.id or 0)
                )
            )
        ).all()
        added = {str(f).strip() for f, st in rows if str(st or "").strip().lower() in {"added", "duplicate"}}
        assert "Деталь А" in added
        assert "Деталь Б" in added

    facts_used = [str(x).strip() for x in (captured.get("facts_text_clean") or []) if str(x).strip()]
    assert "Деталь А" in facts_used
    assert "Новый факт" in facts_used


@pytest.mark.asyncio
async def test_fact_first_backfills_legacy_facts_even_with_existing_non_legacy_facts(
    tmp_path, monkeypatch
) -> None:
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    event_date = (date.today() + timedelta(days=10)).isoformat()
    legacy_description = (
        "Старое описание события (из базы, до Smart Update). "
        "Это достаточно длинный текст, чтобы безопасно извлечь базовые факты из legacy-описания.\n\n"
        "Деталь А.\n"
        "Деталь Б.\n"
        "Ещё одна нейтральная фраза для длины, без новых утверждений."
    )

    async with db.get_session() as session:
        session.add(
            Event(
                id=1,
                title="Событие",
                # Simulate an already-merged state where the public description has changed.
                description="Текущее описание (уже после Smart Update).",
                date=event_date,
                time="19:00",
                location_name="Venue",
                city="Калининград",
                source_text="legacy source",
            )
        )
        non_legacy_source = EventSource(
            event_id=1,
            source_type="telegram",
            source_url="test://tg/old",
            source_text="старый текст источника",
            imported_at=datetime.now(timezone.utc),
            trust_level="high",
        )
        session.add(non_legacy_source)
        await session.flush()
        session.add(
            EventSourceFact(
                event_id=1,
                source_id=int(non_legacy_source.id or 0),
                fact="Старый факт из TG",
                status="added",
                created_at=datetime.now(timezone.utc),
            )
        )

        legacy_source = EventSource(
            event_id=1,
            source_type="legacy",
            source_url="legacy:event_description:1",
            source_text=legacy_description,
            imported_at=datetime.now(timezone.utc),
            trust_level="high",
        )
        session.add(legacy_source)
        await session.flush()
        session.add(
            EventSourceFact(
                event_id=1,
                source_id=int(legacy_source.id or 0),
                fact="Снапшот описания до Smart Update сохранён",
                status="note",
                created_at=datetime.now(timezone.utc),
            )
        )
        await session.commit()

    monkeypatch.setattr(su, "SMART_UPDATE_LLM_DISABLED", False)
    monkeypatch.setattr(su, "SMART_UPDATE_FACT_FIRST", True)

    async def _fake_extract(candidate, *, text_for_facts=None):  # noqa: ANN001
        assert getattr(candidate, "source_type", None) == "legacy"
        assert "Старое описание" in str(text_for_facts or "")
        return ["Деталь А", "Деталь Б"]

    captured: dict[str, object] = {}

    async def _fake_fact_first_desc(**kwargs):  # noqa: ANN003 - test helper
        captured["facts_text_clean"] = list(kwargs.get("facts_text_clean") or [])
        return "FF_DESC"

    async def _fake_merge(*_args, **_kwargs):  # noqa: ANN001
        return {
            "title": None,
            "description": "ignored by fact-first",
            "added_facts": ["Новый факт"],
            "duplicate_facts": [],
            "conflict_facts": [],
            "skipped_conflicts": [],
        }

    async def _no_topics(*_args, **_kwargs):  # noqa: ANN001
        return None

    async def _no_holidays(*_args, **_kwargs):  # noqa: ANN001
        return False

    async def _fake_short(*_args, **_kwargs):  # noqa: ANN001
        return "Короткое описание события без логистики, чтобы тест прошёл корректно сейчас всегда точно."

    async def _fake_digest(*_args, **_kwargs):  # noqa: ANN001
        return "Краткий дайджест для поиска."

    monkeypatch.setattr(su, "_llm_extract_candidate_facts", _fake_extract)
    monkeypatch.setattr(su, "_llm_fact_first_description_md", _fake_fact_first_desc)
    monkeypatch.setattr(su, "_llm_merge_event", _fake_merge)
    monkeypatch.setattr(su, "_classify_topics", _no_topics)
    monkeypatch.setattr(su, "_apply_holiday_festival_mapping", _no_holidays)
    monkeypatch.setattr(su, "_llm_build_short_description", _fake_short)
    monkeypatch.setattr(su, "_llm_build_search_digest", _fake_digest)

    candidate = EventCandidate(
        source_type="telegram",
        source_url="test://tg/1",
        source_text=(
            "Новая дополнительная деталь, которой точно не было в прошлом описании события."
        ),
        raw_excerpt="",
        title="Событие",
        date=event_date,
        time="19:00",
        location_name="Venue",
        city="Калининград",
    )

    res = await smart_event_update(
        db,
        candidate,
        check_source_url=True,
        schedule_tasks=False,
    )
    assert res.status == "merged"
    assert res.event_id == 1

    async with db.get_session() as session:
        legacy_source = (
            await session.execute(
                select(EventSource).where(
                    EventSource.event_id == 1,
                    EventSource.source_type == "legacy",
                )
            )
        ).scalar_one_or_none()
        assert legacy_source is not None
        rows = (
            await session.execute(
                select(EventSourceFact.fact, EventSourceFact.status).where(
                    EventSourceFact.source_id == int(legacy_source.id or 0)
                )
            )
        ).all()
        added = {str(f).strip() for f, st in rows if str(st or "").strip().lower() in {"added", "duplicate"}}
        assert "Деталь А" in added
        assert "Деталь Б" in added

    facts_used = [str(x).strip() for x in (captured.get("facts_text_clean") or []) if str(x).strip()]
    assert "Деталь А" in facts_used
    assert "Новый факт" in facts_used
    assert "Старый факт из TG" in facts_used
