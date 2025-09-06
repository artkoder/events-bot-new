import importlib
from datetime import date

import pytest

import main as orig_main


class FakeTG:
    def __init__(self, access_token=None):
        pass
    def get_page(self, *a, **k):
        return {}


async def fake_call(func, *a, **k):
    return {}


@pytest.mark.asyncio
async def test_ensure_event_telegraph_link_pure(tmp_path, monkeypatch):
    m = importlib.reload(orig_main)
    db = m.Database(str(tmp_path / "db.sqlite"))
    await db.init()
    async with db.get_session() as session:
        ev = m.Event(
            title="T",
            description="D",
            date="2025-09-01",
            time="12:00",
            location_name="Loc",
            source_text="SRC",
            telegraph_path="abc",
            source_post_url="https://example.com",
        )
        session.add(ev)
        await session.commit()
        eid = ev.id
    async with db.get_session() as session:
        ev = await session.get(m.Event, eid)
    called = False
    async def fake_update(*a, **k):
        nonlocal called
        called = True
    monkeypatch.setattr(m, "update_telegraph_event_page", fake_update)
    await m.ensure_event_telegraph_link(ev, None, db)
    assert not called
    assert ev.telegraph_url == "https://telegra.ph/abc"
    async with db.get_session() as session:
        refreshed = await session.get(m.Event, eid)
        assert refreshed.telegraph_url == "https://telegra.ph/abc"


@pytest.mark.asyncio
async def test_update_event_page_edits_without_create(tmp_path, monkeypatch):
    m = importlib.reload(orig_main)
    db = m.Database(str(tmp_path / "db.sqlite"))
    await db.init()
    async with db.get_session() as session:
        ev = m.Event(
            title="T",
            description="D",
            date="2025-09-01",
            time="12:00",
            location_name="Loc",
            source_text="SRC",
            telegraph_path="abc",
            telegraph_url="https://telegra.ph/abc",
        )
        session.add(ev)
        await session.commit()
        eid = ev.id
    async def fake_bspc(*a, **k):
        return "<p>x</p>", "", ""
    monkeypatch.setattr(m, "build_source_page_content", fake_bspc)
    monkeypatch.setattr(m, "get_telegraph_token", lambda: "t")
    monkeypatch.setattr(m, "Telegraph", FakeTG)
    monkeypatch.setattr(m, "telegraph_call", fake_call)
    create_calls = []
    edit_calls = []
    async def fake_create(*a, **k):
        create_calls.append(1)
        return {"url": "https://telegra.ph/new", "path": "new"}
    async def fake_edit(tg, path, **k):
        edit_calls.append(path)
        return {}
    monkeypatch.setattr(m, "telegraph_create_page", fake_create)
    monkeypatch.setattr(m, "telegraph_edit_page", fake_edit)
    await m.update_telegraph_event_page(eid, db, None)
    assert create_calls == []
    assert edit_calls == ["abc"]


@pytest.mark.asyncio
async def test_navigation_builds_do_not_touch_events(tmp_path, monkeypatch):
    m = importlib.reload(orig_main)
    db = m.Database(str(tmp_path / "db.sqlite"))
    await db.init()
    async with db.get_session() as session:
        ev = m.Event(
            title="E",
            description="D",
            date="2025-09-05",
            time="12:00",
            location_name="Loc",
            source_text="TXT",
        )
        session.add(ev)
        await session.commit()
        eid = ev.id
    async def fake_bspc2(*a, **k):
        return "<p>x</p>", "", ""
    monkeypatch.setattr(m, "build_source_page_content", fake_bspc2)
    monkeypatch.setattr(m, "get_telegraph_token", lambda: "t")
    monkeypatch.setattr(m, "Telegraph", FakeTG)
    async def fake_month(*a, **k):
        return "T", [], 0
    async def fake_weekend(*a, **k):
        return "W", [], 0
    monkeypatch.setattr(m, "build_month_page_content", fake_month)
    monkeypatch.setattr(m, "build_weekend_page_content", fake_weekend)
    monkeypatch.setattr(m, "telegraph_call", fake_call)
    m.DISABLE_EVENT_PAGE_UPDATES = False
    create_calls = []
    edit_calls = []
    async def fake_create(tg, *a, caller="event_pipeline", eid=None, **k):
        create_calls.append((caller, eid))
        return {"url": "https://tg/x", "path": "x"}
    async def fake_edit(tg, path, *, caller="event_pipeline", eid=None, **k):
        edit_calls.append((caller, eid, path))
        return {}
    monkeypatch.setattr(m, "telegraph_create_page", fake_create)
    monkeypatch.setattr(m, "telegraph_edit_page", fake_edit)
    await m.update_telegraph_event_page(eid, db, None)
    await m.sync_month_page(db, "2025-09", update_links=False)
    await m.sync_weekend_page(db, "2025-09-06", update_links=False, post_vk=False)
    async with db.get_session() as session:
        ev = await session.get(m.Event, eid)
        ev.title = "New"
        session.add(ev)
        await session.commit()
    await m.update_telegraph_event_page(eid, db, None)
    assert all(c[0] == "event_pipeline" for c in create_calls if c[1] == eid)
    assert all(c[0] == "event_pipeline" for c in edit_calls if c[1] == eid)
    await m.rebuild_pages(db, ["2025-09"], ["2025-09-06"])
    assert all(c[0] == "event_pipeline" for c in create_calls if c[1] == eid)
    assert all(c[0] == "event_pipeline" for c in edit_calls if c[1] == eid)
