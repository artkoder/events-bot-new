import logging
import pytest
from datetime import date, datetime, timedelta
from pathlib import Path

import main
import vk_intake
from db import Database
from models import Festival, Event
from telegraph.utils import nodes_to_html
from markup import FEST_INDEX_INTRO_START, FEST_INDEX_INTRO_END
from sqlalchemy import select


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

    await main.set_setting_value(
        db, "festivals_index_cover", "https://example.com/cover.jpg"
    )

    class DummyTelegraph:
        def __init__(self, *args, **kwargs):
            pass

        def create_page(self, title, html_content, **_):
            stored["html"] = html_content
            return {"url": "https://telegra.ph/fests", "path": "fests"}

        def edit_page(self, path, title, html_content, **kwargs):
            stored["edited"] = html_content
            return {}

        def get_page(self, path, return_html=True):
            return {"content_html": stored.get("html")}

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
    assert html.startswith('<figure><img src="https://example.com/cover.jpg"/></figure>')
    assert '<figure><img src="https://example.com/cover.jpg"/></figure>' in html
    assert html.index(FEST_INDEX_INTRO_START) > html.index(
        '<figure><img src="https://example.com/cover.jpg"/></figure>'
    )
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
    assert (
        html.index(
            '<figure><a href="https://telegra.ph/withimg"><img src="https://example.com/i.jpg"/></a>'
        )
        < html.index('<h3><a href="https://telegra.ph/withimg">WithImg</a></h3>')
    )
    assert html.count('<p dir="auto">\u200b</p>') == 1
    assert html.count(FEST_INDEX_INTRO_START) == 1
    assert html.count(FEST_INDEX_INTRO_END) == 1
    assert html.count("https://t.me/kenigevents") >= 2
    url = await main.get_setting_value(db, "fest_index_url")
    path = await main.get_setting_value(db, "fest_index_path")
    assert url == "https://telegra.ph/fests"
    assert path == "fests"
    rec = next(
        r for r in caplog.records if getattr(r, "action", None) in {"created", "edited"}
    )
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

        def get_page(self, path, return_html=True):
            return {"content_html": stored.get("edited")}

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

        def create_page(self, title, html_content, **_):
            stored["html"] = html_content
            return {"url": "https://telegra.ph/fests", "path": "fests"}

        def edit_page(self, path, title, html_content, **kwargs):
            stored["html"] = html_content
            return {}

        def get_page(self, path, return_html=True):
            return {"content_html": stored.get("html")}

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
async def test_persist_event_updates_festival_range(tmp_path: Path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    async with db.get_session() as session:
        session.add(
            Festival(
                name="Fest",
                start_date="2025-06-01",
                end_date="2025-06-05",
            )
        )
        await session.commit()

    async def fake_assign_event_topics(event):
        return [], 0, None, False

    async def fake_schedule_event_update_tasks(*args, **kwargs):
        return {}

    nav_calls = {"rebuild": 0}

    async def fake_rebuild_fest_nav_if_changed(db_obj):
        nav_calls["rebuild"] += 1
        return True

    monkeypatch.setattr(main, "assign_event_topics", fake_assign_event_topics)
    monkeypatch.setattr(main, "schedule_event_update_tasks", fake_schedule_event_update_tasks)
    monkeypatch.setattr(main, "rebuild_fest_nav_if_changed", fake_rebuild_fest_nav_if_changed)

    draft = vk_intake.EventDraft(
        title="New Event",
        date="2025-06-10",
        time="10:00",
        festival="Fest",
        end_date="2025-06-12",
        source_text="New Event",
    )

    await vk_intake.persist_event_and_pages(draft, [], db)

    async with db.get_session() as session:
        fest = (
            await session.execute(select(Festival).where(Festival.name == "Fest"))
        ).scalar_one()

    assert fest.start_date == "2025-06-01"
    assert fest.end_date == "2025-06-12"
    assert nav_calls["rebuild"] == 1

    items = await main.upcoming_festivals(db, today=date(2025, 6, 6))
    assert any(f.name == "Fest" for _, _, f in items)


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
    assert '<h3><a href="https://telegra.ph/fests">Фестивали</a></h3>' in html


@pytest.mark.asyncio
async def test_weekend_page_has_festivals_link(tmp_path: Path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    saturday = date(2025, 7, 12)
    async with db.get_session() as session:
        session.add(
            Event(
                title="E",
                description="d",
                source_text="s",
                date=saturday.isoformat(),
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

    _, content, _ = await main.build_weekend_page_content(db, saturday.isoformat())
    html = nodes_to_html(content)
    assert '<h3><a href="https://telegra.ph/fests">Фестивали</a></h3>' in html


def test_sanitize_telegraph_html_rewrites_and_checks():
    html = "<h1>X</h1><h5>Y</h5><p>Z</p>"
    assert main.sanitize_telegraph_html(html) == "<h3>X</h3><h3>Y</h3><p>Z</p>"
    with pytest.raises(ValueError):
        main.sanitize_telegraph_html("<p>ok</p><script>bad</script>")


def test_sanitize_telegraph_html_keeps_linked_images():
    html = (
        '<figure><a href="https://example.com"><img src="https://e.com/i.jpg"/></a>'
        "<figcaption>X</figcaption></figure>"
    )
    assert main.sanitize_telegraph_html(html) == html


def test_build_festival_card_html_preserves_link():
    fest = Festival(name="Fest", telegraph_path="fest", photo_url="https://example.com/i.jpg")
    nodes, used_img, _ = main.build_festival_card_nodes(
        fest, None, None, with_image=True, add_spacer=False
    )
    html = nodes_to_html(nodes)
    assert (
        '<a href="https://telegra.ph/fest"><img src="https://example.com/i.jpg"/></a>'
        in html
    )
    assert html.index("<figure>") < html.index("<h3>")
    assert main.sanitize_telegraph_html(html) == html


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

        def create_page(self, title, html_content, **_):
            return {"url": "https://telegra.ph/fests", "path": "fests"}

        def get_page(self, path, return_html=True):
            return {"content_html": ""}

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


@pytest.mark.asyncio
async def test_weekendimg_photo_updates_festivals_cover(monkeypatch):
    user_id = 123
    prev_wait = dict(main.weekend_img_wait.items())
    main.weekend_img_wait.clear()
    main.weekend_img_wait[user_id] = main.FESTIVALS_INDEX_MARKER

    recorded: dict[str, object] = {}

    async def fake_extract(message, bot):
        return [(b"data", "cover.jpg")]

    async def fake_upload(images, limit=1, force=True, event_hint=None):
        return ["https://catbox.moe/cover.jpg"], ""

    async def fake_set(db_obj, key, value):
        recorded["set"] = (db_obj, key, value)

    async def fake_sync(db_obj):
        recorded["sync"] = db_obj

    async def fake_rebuild(db_obj, force=False):
        recorded["rebuild"] = force
        return "updated", "https://telegra.ph/fests"

    monkeypatch.setattr(main, "extract_images", fake_extract)
    monkeypatch.setattr(main, "upload_images", fake_upload)
    monkeypatch.setattr(main, "set_setting_value", fake_set)
    monkeypatch.setattr(main, "sync_festivals_index_page", fake_sync)
    monkeypatch.setattr(main, "rebuild_festivals_index_if_needed", fake_rebuild)

    class DummyUser:
        def __init__(self, ident: int) -> None:
            self.id = ident

    class DummyMessage:
        def __init__(self, ident: int) -> None:
            self.from_user = DummyUser(ident)
            self.replies: list[str] = []

        async def reply(self, text: str) -> None:
            self.replies.append(text)

    message = DummyMessage(user_id)
    db_obj = object()

    try:
        await main.handle_weekendimg_photo(message, db_obj, bot=None)
        cleared = user_id not in main.weekend_img_wait
    finally:
        main.weekend_img_wait.clear()
        main.weekend_img_wait.update(prev_wait)

    assert recorded["set"] == (db_obj, "festivals_index_cover", "https://catbox.moe/cover.jpg")
    assert recorded["sync"] is db_obj
    assert recorded["rebuild"] is True
    assert message.replies and "https://telegra.ph/fests" in message.replies[-1]
    assert cleared
