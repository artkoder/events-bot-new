from types import SimpleNamespace
from datetime import date, timedelta
import re

import pytest

import main
from db import Database
from models import Event, MonthPage


@pytest.mark.asyncio
async def test_pages_rebuild_buttons_include_future_months(tmp_path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    async with db.get_session() as session:
        session.add(
            Event(
                title="A",
                description="D",
                date="2025-09-10",
                time="10:00",
                location_name="loc",
                source_text="src",
            )
        )
        session.add(
            Event(
                title="B",
                description="D",
                date="2025-12-01",
                time="10:00",
                location_name="loc",
                source_text="src",
            )
        )
        await session.commit()

    class FixedDate(date):
        @classmethod
        def today(cls):
            return cls(2025, 9, 1)

    monkeypatch.setattr(main, "date", FixedDate)

    class Button:
        def __init__(self, text, callback_data):
            self.text = text
            self.callback_data = callback_data

    class Markup:
        def __init__(self, inline_keyboard):
            self.inline_keyboard = inline_keyboard

    monkeypatch.setattr(
        main,
        "types",
        SimpleNamespace(InlineKeyboardButton=Button, InlineKeyboardMarkup=Markup),
    )

    sent = {}

    class Bot:
        async def send_message(self, chat_id, text, reply_markup=None):
            sent["markup"] = reply_markup

    message = SimpleNamespace(chat=SimpleNamespace(id=1), text="/pages_rebuild")
    await main.handle_pages_rebuild(message, db, Bot())
    months = [row[0].text for row in sent["markup"].inline_keyboard[:-1]]
    assert months == ["2025-09", "2025-12"]


@pytest.mark.asyncio
async def test_pages_rebuild_cb_all(monkeypatch, tmp_path):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    async with db.get_session() as session:
        session.add(
            Event(
                title="A",
                description="D",
                date="2025-09-10",
                time="10:00",
                location_name="loc",
                source_text="src",
            )
        )
        session.add(
            Event(
                title="B",
                description="D",
                date="2025-12-01",
                time="10:00",
                location_name="loc",
                source_text="src",
            )
        )
        await session.commit()

    class FixedDate(date):
        @classmethod
        def today(cls):
            return cls(2025, 9, 1)

    monkeypatch.setattr(main, "date", FixedDate)

    captured = {}

    async def fake_perform(db_obj, months, force=True):
        captured["months"] = months
        return "ok"

    monkeypatch.setattr(main, "_perform_pages_rebuild", fake_perform)

    class Bot:
        async def send_message(self, chat_id, text):
            pass

    class Callback:
        data = "pages_rebuild:ALL"
        message = SimpleNamespace(chat=SimpleNamespace(id=1))

        async def answer(self):
            pass

    await main.handle_pages_rebuild_cb(Callback(), db, Bot())
    assert captured["months"] == ["2025-09", "2025-12"]


@pytest.mark.asyncio
async def test_pages_rebuild_ignores_invalid_dates(tmp_path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    async with db.get_session() as session:
        session.add(
            Event(
                title="Valid",
                description="D",
                date="2025-10-05",
                time="10:00",
                location_name="loc",
                source_text="src",
            )
        )
        session.add(
            Event(
                title="Bad1",
                description="D",
                date="21.07.24",
                time="10:00",
                location_name="loc",
                source_text="src",
            )
        )
        session.add(
            Event(
                title="Bad2",
                description="D",
                date="лекция",
                time="10:00",
                location_name="loc",
                source_text="src",
            )
        )
        session.add(
            Event(
                title="Past",
                description="D",
                date="2024-01-01",
                time="10:00",
                location_name="loc",
                source_text="src",
            )
        )
        await session.commit()

    class FixedDate(date):
        @classmethod
        def today(cls):
            return cls(2025, 9, 1)

    monkeypatch.setattr(main, "date", FixedDate)

    class Button:
        def __init__(self, text, callback_data):
            self.text = text
            self.callback_data = callback_data

    class Markup:
        def __init__(self, inline_keyboard):
            self.inline_keyboard = inline_keyboard

    monkeypatch.setattr(
        main,
        "types",
        SimpleNamespace(InlineKeyboardButton=Button, InlineKeyboardMarkup=Markup),
    )

    sent = {}

    class Bot:
        async def send_message(self, chat_id, text, reply_markup=None):
            sent["markup"] = reply_markup

    message = SimpleNamespace(chat=SimpleNamespace(id=1), text="/pages_rebuild")
    await main.handle_pages_rebuild(message, db, Bot())
    months = [row[0].text for row in sent["markup"].inline_keyboard[:-1]]
    assert months == ["2025-10"]


@pytest.mark.asyncio
async def test_pages_rebuild_split_keeps_day_boundaries(tmp_path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    month = "2025-10"
    base_day = date(2025, 10, 1)
    long_text = "Лонгтекст " * 200

    async with db.get_session() as session:
        for day_offset in range(3):
            day = base_day + timedelta(days=day_offset)
            for idx in range(3):
                session.add(
                    Event(
                        title=f"Event {day_offset}-{idx}",
                        description=long_text,
                        date=day.isoformat(),
                        time=f"{10 + idx:02d}:00",
                        location_name="loc",
                        source_text="src",
                    )
                )
        await session.commit()

    monkeypatch.setattr(main, "TELEGRAPH_LIMIT", 500)
    monkeypatch.setattr(main, "get_telegraph_token", lambda: "token")

    class DummyTelegraph:
        def __init__(self, *args, **kwargs):
            pass

        async def get_page(self, *args, **kwargs):
            return {}

    monkeypatch.setattr(main, "Telegraph", DummyTelegraph)

    html_by_path: dict[str, str] = {}

    async def fake_create_page(tg, *, title, html_content, caller="event_pipeline", **kwargs):
        path = f"path{len(html_by_path) + 1}"
        html_by_path[path] = html_content
        return {"url": f"https://telegra.ph/{path}", "path": path}

    async def fake_edit_page(
        tg, path, *, title, html_content, caller="event_pipeline", **kwargs
    ):
        html_by_path[path] = html_content
        return {"url": f"https://telegra.ph/{path}", "path": path}

    async def fake_check_month_page_markers(tg, path):
        return None

    monkeypatch.setattr(main, "telegraph_create_page", fake_create_page)
    monkeypatch.setattr(main, "telegraph_edit_page", fake_edit_page)
    monkeypatch.setattr(main, "check_month_page_markers", fake_check_month_page_markers)

    message = SimpleNamespace(chat=SimpleNamespace(id=1), text=f"/pages_rebuild {month}")

    class Bot:
        async def send_message(self, chat_id, text, reply_markup=None):
            pass

    await main.handle_pages_rebuild(message, db, Bot())

    async with db.get_session() as session:
        page = await session.get(MonthPage, month)

    assert page is not None
    assert page.path and page.path2
    assert page.path in html_by_path
    assert page.path2 in html_by_path

    def days_from_html(html: str) -> set[str]:
        return set(re.findall(r"<!--DAY:(\d{4}-\d{2}-\d{2}) START-->", html))

    days1 = days_from_html(html_by_path[page.path])
    days2 = days_from_html(html_by_path[page.path2])

    expected_days = {(base_day + timedelta(days=i)).isoformat() for i in range(3)}

    assert days1
    assert days2
    assert days1.isdisjoint(days2)
    assert days1 | days2 == expected_days


@pytest.mark.asyncio
async def test_month_nav_update_falls_back_to_full_rebuild(tmp_path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    month = "2025-10"

    async with db.get_session() as session:
        session.add(
            MonthPage(
                month=month,
                url="https://telegra.ph/path1",
                path="path1",
                content_hash="old",
            )
        )
        await session.commit()

    monkeypatch.setattr(main, "get_telegraph_token", lambda: "token")

    class DummyTelegraph:
        def __init__(self, *args, **kwargs):
            pass

    async def dummy_get_page(*args, **kwargs):
        return {}

    DummyTelegraph.get_page = dummy_get_page

    monkeypatch.setattr(main, "Telegraph", DummyTelegraph)

    async def fake_telegraph_call(func, path, return_html=True):
        return {"content_html": "<p>old</p>", "title": "Old"}

    monkeypatch.setattr(main, "telegraph_call", fake_telegraph_call)

    async def fake_build_month_nav_block(db_obj, month_str):
        return "<nav>links</nav>"

    monkeypatch.setattr(main, "build_month_nav_block", fake_build_month_nav_block)

    monkeypatch.setattr(
        main,
        "ensure_footer_nav_with_hr",
        lambda html, nav_block, month, page: html + nav_block,
    )

    async def fake_get_month_data(db_obj, month_str):
        return [], []

    monkeypatch.setattr(main, "get_month_data", fake_get_month_data)

    full_render_called = False

    async def fake_build_month_page_content(db_obj, month_str, events, exhibitions):
        nonlocal full_render_called
        full_render_called = True
        return ("Title", [{"tag": "p", "children": ["new"]}], 0)

    monkeypatch.setattr(main, "build_month_page_content", fake_build_month_page_content)

    edit_calls: list[dict] = []

    async def fake_edit_page(
        tg,
        path,
        *,
        title,
        html_content,
        caller="event_pipeline",
        **kwargs,
    ):
        call_index = len(edit_calls)
        edit_calls.append({"path": path, "html": html_content})
        if call_index == 0:
            raise main.TelegraphException("CONTENT TOO BIG")
        return {"url": f"https://telegra.ph/{path}", "path": path}

    monkeypatch.setattr(main, "telegraph_edit_page", fake_edit_page)

    async def fake_check_month_page_markers(tg, path):
        return None

    monkeypatch.setattr(main, "check_month_page_markers", fake_check_month_page_markers)

    await main._sync_month_page_inner(db, month, update_links=True)

    assert full_render_called
    assert len(edit_calls) == 2

    async with db.get_session() as session:
        page = await session.get(MonthPage, month)

    assert page.content_hash == main.content_hash(edit_calls[-1]["html"])

