import pytest
from pathlib import Path
from datetime import date, datetime

import main
from db import Database
from models import Event, Festival
from telegraph.utils import nodes_to_html


@pytest.mark.asyncio
async def test_weekend_page_links_festival(tmp_path: Path, monkeypatch):
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
                date="2025-07-12",
                time="18:00",
                location_name="Hall",
                festival=fest.name,
            )
        )
        await session.commit()

    async def fake_create_page(tg, *args, **kwargs):
        return {"path": "p", "url": "http://t.me/p"}

    async def fake_edit_page(tg, path, **kwargs):
        return None

    async def fake_build(*a, **k):
        return "<p>src</p>", [], 0

    monkeypatch.setattr(main, "telegraph_create_page", fake_create_page)
    monkeypatch.setattr(main, "telegraph_edit_page", fake_edit_page)
    monkeypatch.setattr(main, "build_source_page_content", fake_build)
    monkeypatch.setattr(main, "Telegraph", lambda access_token=None, domain=None: object())
    monkeypatch.setattr(main, "get_telegraph_token", lambda: "t")

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

    _, content, _ = await main.build_weekend_page_content(db, "2025-07-12")
    html = nodes_to_html(content)
    assert '<a href="https://telegra.ph/fest">Fest</a>' in html
