import pytest

from sqlalchemy import select

from db import Database
from models import Event
from smart_event_update import EventCandidate, smart_event_update
import smart_event_update as su


@pytest.mark.asyncio
async def test_smart_update_skips_ticket_giveaway_with_only_deadline_date(tmp_path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    monkeypatch.setenv("REGION_FILTER_ENABLED", "0")
    monkeypatch.setattr(su, "SMART_UPDATE_LLM_DISABLED", True)

    candidate = EventCandidate(
        source_type="telegram",
        source_url="test://smart-update/giveaway/deadline-only",
        source_text=(
            "Розыгрыш билетов на матч «Балтика» — «ЦСКА».\n"
            "Итоги 14.03.\n"
            "Условия: подпишись и сделай репост."
        ),
        raw_excerpt="Розыгрыш билетов на матч «Балтика» — «ЦСКА». Итоги 14.03.",
        title="Розыгрыш билетов на матч «Балтика» — «ЦСКА»",
        date="2026-03-14",
        time="",
        location_name="Тестовая площадка",
        city="Калининград",
        trust_level="medium",
    )

    result = await smart_event_update(
        db,
        candidate,
        check_source_url=True,
        schedule_tasks=False,
    )

    assert result.status == "skipped_giveaway"
    assert result.reason == "giveaway_no_event"

    async with db.get_session() as session:
        rows = (await session.execute(select(Event))).scalars().all()
        assert rows == []


@pytest.mark.asyncio
async def test_smart_update_skips_ticket_giveaway_when_event_is_only_prize_reference(tmp_path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    monkeypatch.setenv("REGION_FILTER_ENABLED", "0")
    monkeypatch.setattr(su, "SMART_UPDATE_LLM_DISABLED", True)

    candidate = EventCandidate(
        source_type="vk",
        source_url="https://vk.com/wall-86702629_7354",
        source_text=(
            "РОЗЫГРЫШ билетов на матч «Балтика» — «ЦСКА».\n"
            "Главный приз — два билета на матч, который состоится 14 марта на Ростех Арене.\n"
            "Для участия подпишись на сообщество, поставь лайк и напиши комментарий.\n"
            "Итоги подведём 10 марта."
        ),
        raw_excerpt=(
            "Розыгрыш билетов на матч «Балтика» — «ЦСКА». "
            "Главный приз — два билета на матч, который состоится 14 марта на Ростех Арене."
        ),
        title="Розыгрыш билетов на матч «Балтика» — «ЦСКА»",
        date="2026-03-14",
        time="",
        location_name="Ростех Арена",
        city="Калининград",
        trust_level="medium",
    )

    result = await smart_event_update(
        db,
        candidate,
        check_source_url=True,
        schedule_tasks=False,
    )

    assert result.status == "skipped_giveaway"
    assert result.reason == "giveaway_no_event"

    async with db.get_session() as session:
        rows = (await session.execute(select(Event))).scalars().all()
        assert rows == []
