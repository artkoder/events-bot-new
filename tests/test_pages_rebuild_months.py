from types import SimpleNamespace
from datetime import date

import pytest

import main
from db import Database
from models import Event


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

