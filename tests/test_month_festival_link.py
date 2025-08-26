import pytest
from pathlib import Path
from datetime import date, datetime
import logging

import main
from db import Database
from models import Event, Festival
from telegraph.utils import nodes_to_html


@pytest.mark.asyncio
async def test_month_page_links_festival(tmp_path: Path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    async with db.get_session() as session:
        fest = Festival(name="Fest", telegraph_path="fest")
        session.add(fest)
        session.add(
            Event(
                title="E",
                description="d",
                source_text="s",
                date="2025-07-16",
                time="18:00",
                location_name="Hall",
                festival=fest.name,
            )
        )
        await session.commit()

    class FakeDate(date):
        @classmethod
        def today(cls):
            return date(2025, 7, 10)

    class FakeDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2025, 7, 10, 12, 0, tzinfo=tz)

    monkeypatch.setattr(main, "date", FakeDate)
    monkeypatch.setattr(main, "datetime", FakeDatetime)

    _, content, _ = await main.build_month_page_content(db, "2025-07")
    html = nodes_to_html(content)
    assert '<a href="https://telegra.ph/fest">Fest</a>' in html


@pytest.mark.asyncio
async def test_month_render_fest_link_logged(tmp_path: Path, monkeypatch, caplog):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    async with db.get_session() as session:
        fest = Festival(name="Fest", telegraph_path="fest")
        ev = Event(
            title="E",
            description="d",
            source_text="s",
            date="2025-07-16",
            time="18:00",
            location_name="Hall",
            festival=fest.name,
        )
        session.add_all([fest, ev])
        await session.commit()
        eid = ev.id

    class FakeDate(date):
        @classmethod
        def today(cls):
            return date(2025, 7, 10)

    class FakeDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2025, 7, 10, 12, 0, tzinfo=tz)

    monkeypatch.setattr(main, "date", FakeDate)
    monkeypatch.setattr(main, "datetime", FakeDatetime)

    with caplog.at_level(logging.INFO):
        await main.build_month_page_content(db, "2025-07")

    rec = next(r for r in caplog.records if r.message == "month_render_fest_link")
    assert rec.event_id == eid
    assert rec.festival == "Fest"
    assert rec.has_url is False
    assert rec.has_path is True
    assert rec.href_used == "https://telegra.ph/fest"
