import logging
import pytest
from datetime import date, datetime, timedelta
from pathlib import Path

import main
from db import Database
from models import Festival, Event
from telegraph.utils import nodes_to_html
from markup import FEST_INDEX_INTRO_START, FEST_INDEX_INTRO_END


@pytest.mark.asyncio
async def test_sync_festivals_index_page_created(tmp_path: Path, monkeypatch, caplog):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    today = date.today().isoformat()
    async with db.get_session() as session:
        session.add_all(
            [
                Festival(
                    name="WithImg",
                    start_date=today,
                    end_date=today,
                    photo_url="https://example.com/i.jpg",
                    telegraph_path="withimg",
                ),
                Festival(
                    name="NoImg",
                    start_date=today,
                    end_date=today,
                    telegraph_path="noimg",
                ),
            ]
        )
        await session.commit()

    stored = {}

    class DummyTelegraph:
        def __init__(self, *args, **kwargs):
            pass

        def create_page(self, title, html_content):
            stored["html"] = html_content
            return {"url": "https://telegra.ph/fests", "path": "fests"}

        def edit_page(self, path, title, html_content, **kwargs):
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

    with caplog.at_level(logging.INFO):
        await main.sync_festivals_index_page(db)

    html = stored["html"]
    assert "Все фестивали Калининградской области" not in html
    assert html.count("<h3>") == 2
    assert (
        '<h3><a href="https://telegra.ph/withimg">WithImg</a></h3>' in html
    )
    assert (
        '<h3><a href="https://telegra.ph/noimg">NoImg</a></h3>' in html
    )
    assert (
        '<figure><a href="https://telegra.ph/withimg"><img src="https://example.com/i.jpg"/></a>'
        in html
    )
    assert html.count('<p dir="auto">\u200b</p>') == 1
    assert html.count(FEST_INDEX_INTRO_START) == 1
    assert html.count(FEST_INDEX_INTRO_END) == 1
    assert html.count("https://t.me/kenigevents") >= 2
    url = await main.get_setting_value(db, "fest_index_url")
    path = await main.get_setting_value(db, "fest_index_path")
    assert url == "https://telegra.ph/fests"
    assert path == "fests"
    rec = next(r for r in caplog.records if getattr(r, "action", None) == "created")
    assert rec.target == "tg"
    assert rec.path == "fests"
    assert rec.url == "https://telegra.ph/fests"


@pytest.mark.asyncio
async def test_sync_festivals_index_page_updated(tmp_path: Path, monkeypatch, caplog):
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

        def edit_page(self, path, title, html_content, **kwargs):
            stored["edited"] = html_content
            return {}

    async def fake_telegraph_call(func, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(main, "Telegraph", DummyTelegraph)
    monkeypatch.setattr(main, "telegraph_call", fake_telegraph_call)
    monkeypatch.setattr(main, "get_telegraph_token", lambda: "token")
    await main.set_setting_value(db, "fest_index_path", "fests")

    with caplog.at_level(logging.INFO):
        await main.sync_festivals_index_page(db)

    rec = next(r for r in caplog.records if getattr(r, "action", None) == "edited")
    assert rec.target == "tg"
    assert rec.path == "fests"
    html = stored["edited"]
    assert "Все фестивали Калининградской области" not in html
    assert html.count(FEST_INDEX_INTRO_START) == 1
    assert html.count(FEST_INDEX_INTRO_END) == 1


@pytest.mark.asyncio
async def test_sync_festivals_index_page_sorted(tmp_path: Path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    today = date.today()
    soon = today + timedelta(days=1)
    later = today + timedelta(days=2)
    async with db.get_session() as session:
        session.add_all(
            [
                Festival(
                    name="Later",
                    start_date=later.isoformat(),
                    end_date=later.isoformat(),
                ),
                Festival(
                    name="Soon",
                    start_date=soon.isoformat(),
                    end_date=soon.isoformat(),
                ),
            ]
        )
        await session.commit()

    stored = {}

    class DummyTelegraph:
        def __init__(self, *a, **k):
            pass

        def create_page(self, title, html_content):
            stored["html"] = html_content
            return {"url": "https://telegra.ph/fests", "path": "fests"}

        def edit_page(self, path, title, html_content, **kwargs):
            stored["html"] = html_content
            return {}

    monkeypatch.setattr(main, "Telegraph", DummyTelegraph)
    monkeypatch.setattr(main, "telegraph_call", lambda func, *a, **k: func(*a, **k))

    async def fake_create_page(tg, *a, **k):
        return tg.create_page(*a, **k)

    monkeypatch.setattr(main, "telegraph_create_page", fake_create_page)
    monkeypatch.setattr(main, "get_telegraph_token", lambda: "token")

    await main.sync_festivals_index_page(db)
    html = stored["html"]
    assert html.index("Soon") < html.index("Later")


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
    assert '<a href="https://telegra.ph/fests">🎪 Все фестивали Калининградской области</a>' in html


def test_sanitize_telegraph_html_rewrites_and_checks():
    html = "<h1>X</h1><h5>Y</h5><p>Z</p>"
    assert main.sanitize_telegraph_html(html) == "<h3>X</h3><h3>Y</h3><p>Z</p>"
    with pytest.raises(ValueError):
        main.sanitize_telegraph_html("<p>ok</p><script>bad</script>")


@pytest.mark.asyncio
async def test_rebuild_festivals_index_force_updates(tmp_path: Path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    today = date.today().isoformat()
    async with db.get_session() as session:
        session.add(Festival(name="Fest", start_date=today, end_date=today))
        await session.commit()

    calls = {"edited": 0}

    class DummyTelegraph:
        def __init__(self, *a, **k):
            pass

        def edit_page(self, path, title, html_content, **kwargs):
            calls["edited"] += 1
            return {}

        def create_page(self, title, html_content):
            return {"url": "https://telegra.ph/fests", "path": "fests"}

    async def fake_call(func, *a, **k):
        return func(*a, **k)

    async def fake_create_page(tg, *a, **k):
        return tg.create_page(*a, **k)

    monkeypatch.setattr(main, "Telegraph", DummyTelegraph)
    monkeypatch.setattr(main, "telegraph_call", fake_call)
    monkeypatch.setattr(main, "telegraph_create_page", fake_create_page)
    monkeypatch.setattr(main, "get_telegraph_token", lambda: "token")

    await main.rebuild_festivals_index_if_needed(db)
    status, _ = await main.rebuild_festivals_index_if_needed(db)
    assert status == "nochange"
    before = calls["edited"]

    status, _ = await main.rebuild_festivals_index_if_needed(db, force=True)
    assert status == "updated"
    assert calls["edited"] == before + 1
