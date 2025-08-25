import logging
from datetime import date, timedelta

import pytest

import main
from models import Festival


@pytest.mark.asyncio
async def test_build_festivals_nav_block_caches_and_detects_change(tmp_path):
    db = main.Database(str(tmp_path / "db.sqlite"))
    await db.init()
    today = date.today().isoformat()

    # create first festival
    async with db.get_session() as session:
        session.add(
            Festival(
                name="Fest1",
                telegraph_path="p1",
                start_date=today,
                end_date=today,
            )
        )
        await session.commit()

    html1, lines1, changed1 = await main.build_festivals_nav_block(db)
    assert changed1 is True
    assert "Fest1" in html1

    # second call without changes should reuse cached html
    html2, lines2, changed2 = await main.build_festivals_nav_block(db)
    assert changed2 is False
    assert html2 == html1
    assert lines2 == lines1

    # add another festival to trigger change
    async with db.get_session() as session:
        session.add(
            Festival(
                name="Fest2",
                telegraph_path="p2",
                start_date=today,
                end_date=today,
            )
        )
        await session.commit()

    html3, _, changed3 = await main.build_festivals_nav_block(db)
    assert changed3 is True
    assert "Fest2" in html3


@pytest.mark.asyncio
async def test_upcoming_festivals_filters_past_and_logs(tmp_path, caplog):
    db = main.Database(str(tmp_path / "db.sqlite"))
    await db.init()
    today = date.today()
    yesterday = today - timedelta(days=1)
    tomorrow = today + timedelta(days=1)

    async with db.get_session() as session:
        session.add(
            Festival(
                name="Past",
                start_date=yesterday.isoformat(),
                end_date=yesterday.isoformat(),
            )
        )
        session.add(
            Festival(
                name="Future",
                start_date=tomorrow.isoformat(),
                end_date=tomorrow.isoformat(),
            )
        )
        await session.commit()

    with caplog.at_level(logging.DEBUG):
        items = await main.upcoming_festivals(db, today=today)

    names = [fest.name for _, _, fest in items]
    assert "Past" not in names
    assert "Future" in names
    assert any("db upcoming_festivals took" in r.message for r in caplog.records)

