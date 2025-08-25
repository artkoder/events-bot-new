import pytest
from datetime import date, datetime
from pathlib import Path

import main
from db import Database
from models import Festival, Event
from telegraph.utils import nodes_to_html


@pytest.mark.asyncio
async def test_sync_festivals_index_page(tmp_path: Path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    today = date.today().isoformat()
    async with db.get_session() as session:
        session.add(Festival(name="Fest", start_date=today, end_date=today))
        await session.commit()

    stored = {}

    class DummyTelegraph:
        def __init__(self, *args, **kwargs):
            pass

        def create_page(self, title, html_content):
            stored["html"] = html_content
            return {"url": "https://telegra.ph/fests", "path": "fests"}

        def edit_page(self, path, title, html_content):
            stored["edited"] = html_content
            return {}

    async def fake_telegraph_call(func, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(main, "Telegraph", DummyTelegraph)
    monkeypatch.setattr(main, "telegraph_call", fake_telegraph_call)

    async def fake_create_page(tg, *a, **k):
        return tg.create_page(*a, **k)

    monkeypatch.setattr(main, "telegraph_create_page", fake_create_page)
    monkeypatch.setattr(main, "get_telegraph_token", lambda: "token")

    await main.sync_festivals_index_page(db)

    assert "Ближайшие фестивали" in stored["html"]
    url = await main.get_setting_value(db, "fest_index_url")
    path = await main.get_setting_value(db, "fest_index_path")
    assert url == "https://telegra.ph/fests"
    assert path == "fests"


@pytest.mark.asyncio
async def test_month_page_has_festivals_link(tmp_path: Path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    async with db.get_session() as session:
        session.add(
            Event(
                title="E",
                description="d",
                source_text="s",
                date="2025-07-16",
                time="18:00",
                location_name="Hall",
            )
        )
        await session.commit()

    await main.set_setting_value(db, "fest_index_url", "https://telegra.ph/fests")

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
    assert '<a href="https://telegra.ph/fests">🎪 Все фестивали региона</a>' in html
