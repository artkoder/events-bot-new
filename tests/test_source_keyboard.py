import pytest
from aiogram import Bot
from aiogram.exceptions import TelegramBadRequest
from datetime import datetime

import main
from main import Database, Event, MonthPage, update_source_post_keyboard


class DummyBot(Bot):
    def __init__(self, token: str):
        super().__init__(token)
        self.edits = []
        self.sent = []

    async def edit_message_reply_markup(self, chat_id, message_id, reply_markup=None, **kwargs):
        self.edits.append((chat_id, message_id, reply_markup))
        from types import SimpleNamespace
        return SimpleNamespace()

    async def send_message(self, chat_id, text, reply_markup=None, **kwargs):
        from types import SimpleNamespace
        self.sent.append((chat_id, text, reply_markup))
        return SimpleNamespace(message_id=99, chat=SimpleNamespace(id=chat_id))

    async def get_me(self):
        from types import SimpleNamespace
        return SimpleNamespace(id=42)

    async def get_chat_member(self, chat_id, user_id):
        from types import SimpleNamespace
        return SimpleNamespace(can_edit_messages=True, status="administrator")


@pytest.mark.asyncio
async def test_update_keyboard_with_ics(tmp_path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    bot = DummyBot("123:abc")

    class FakeDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2025, 7, 2, tzinfo=tz)

    monkeypatch.setattr(main, "datetime", FakeDatetime)
    async with db.get_session() as session:
        session.add_all([
            MonthPage(month="2025-07", url="https://t.me/c/m1", path="p1"),
            MonthPage(month="2025-08", url="https://t.me/c/m2", path="p2"),
            Event(
                id=1,
                title="A",
                description="d",
                source_text="s",
                date="2025-07-18",
                time="19:00",
                location_name="Hall",
                city="Town",
                ics_post_url="https://t.me/c/asset/1",
                source_chat_id=-100,
                source_message_id=10,
            ),
        ])
        await session.commit()
    await update_source_post_keyboard(1, db, bot)
    assert bot.edits
    markup = bot.edits[0][2]
    assert len(markup.inline_keyboard) == 2
    assert markup.inline_keyboard[0][0].url == "https://t.me/c/asset/1"
    assert "Добавить в календарь" == markup.inline_keyboard[0][0].text
    assert len(markup.inline_keyboard[1]) == 2
    assert markup.inline_keyboard[1][0].url == "https://t.me/c/m1"
    assert markup.inline_keyboard[1][1].url == "https://t.me/c/m2"


@pytest.mark.asyncio
async def test_update_keyboard_without_ics(tmp_path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    bot = DummyBot("123:abc")

    class FakeDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2025, 7, 2, tzinfo=tz)

    monkeypatch.setattr(main, "datetime", FakeDatetime)
    async with db.get_session() as session:
        session.add_all([
            MonthPage(month="2025-07", url="https://t.me/c/m1", path="p1"),
            Event(
                id=1,
                title="A",
                description="d",
                source_text="s",
                date="2025-07-18",
                time="19:00",
                location_name="Hall",
                city="Town",
                source_chat_id=-100,
                source_message_id=10,
            ),
        ])
        await session.commit()
    await update_source_post_keyboard(1, db, bot)
    assert bot.edits
    markup = bot.edits[0][2]
    assert len(markup.inline_keyboard) == 1
    assert len(markup.inline_keyboard[0]) == 1
    assert markup.inline_keyboard[0][0].url == "https://t.me/c/m1"


@pytest.mark.asyncio
async def test_update_keyboard_future_event_uses_current_month(tmp_path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    bot = DummyBot("123:abc")

    class FakeDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2025, 8, 5, tzinfo=tz)

    monkeypatch.setattr(main, "datetime", FakeDatetime)
    async with db.get_session() as session:
        session.add_all(
            [
                MonthPage(month="2025-08", url="https://t.me/c/m1", path="p1"),
                MonthPage(month="2025-09", url="https://t.me/c/m2", path="p2"),
                Event(
                    id=1,
                    title="A",
                    description="d",
                    source_text="s",
                    date="2025-11-18",
                    time="19:00",
                    location_name="Hall",
                    city="Town",
                    source_chat_id=-100,
                    source_message_id=10,
                ),
            ]
        )
        await session.commit()
    await update_source_post_keyboard(1, db, bot)
    assert bot.edits
    markup = bot.edits[0][2]
    assert len(markup.inline_keyboard) == 1
    assert len(markup.inline_keyboard[0]) == 2
    assert markup.inline_keyboard[0][0].url == "https://t.me/c/m1"
    assert markup.inline_keyboard[0][1].url == "https://t.me/c/m2"


@pytest.mark.asyncio
async def test_update_keyboard_fallback_service_message(tmp_path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    bot = DummyBot("123:abc")

    class FakeDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2025, 7, 2, tzinfo=tz)

    monkeypatch.setattr(main, "datetime", FakeDatetime)

    async def failing_edit(*a, **k):
        raise TelegramBadRequest(
            method="editMessageReplyMarkup", message="message can't be edited"
        )

    bot.edit_message_reply_markup = failing_edit  # type: ignore

    async with db.get_session() as session:
        session.add_all([
            MonthPage(month="2025-07", url="https://t.me/c/m1", path="p1"),
            Event(
                id=1,
                title="A",
                description="d",
                source_text="s",
                date="2025-07-18",
                time="19:00",
                location_name="Hall",
                city="Town",
                ics_post_url="https://t.me/c/asset/1",
                source_chat_id=-100,
                source_message_id=10,
                creator_id=1,
            ),
        ])
        await session.commit()

    await update_source_post_keyboard(1, db, bot)

    assert bot.sent
    chat_id, text, markup = bot.sent[0]
    assert chat_id == 1
    assert text == "Добавить в календарь/Навигация по месяцам"
    assert markup.inline_keyboard[0][0].url == "https://t.me/c/asset/1"
    async with db.get_session() as session:
        ev = await session.get(Event, 1)
        assert ev.source_message_id == 10


@pytest.mark.asyncio
async def test_update_keyboard_fallback_private_chat(tmp_path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    bot = DummyBot("123:abc")

    class FakeDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2025, 7, 2, tzinfo=tz)

    monkeypatch.setattr(main, "datetime", FakeDatetime)

    async def failing_edit(*a, **k):
        raise TelegramBadRequest(
            method="editMessageReplyMarkup", message="message can't be edited"
        )

    bot.edit_message_reply_markup = failing_edit  # type: ignore

    async with db.get_session() as session:
        session.add_all([
            MonthPage(month="2025-07", url="https://t.me/c/m1", path="p1"),
            Event(
                id=1,
                title="A",
                description="d",
                source_text="s",
                date="2025-07-18",
                time="19:00",
                location_name="Hall",
                city="Town",
                ics_post_url="https://t.me/c/asset/1",
                source_chat_id=1,
                source_message_id=10,
            ),
        ])
        await session.commit()

    await update_source_post_keyboard(1, db, bot)

    assert bot.sent
    chat_id, text, markup = bot.sent[0]
    assert chat_id == 1
    assert text == "Добавить в календарь/Навигация по месяцам"
    assert markup.inline_keyboard[0][0].url == "https://t.me/c/asset/1"
    async with db.get_session() as session:
        ev = await session.get(Event, 1)
        assert ev.source_message_id == 99


@pytest.mark.asyncio
async def test_update_keyboard_restores_chat_id_from_url(tmp_path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    bot = DummyBot("123:abc")

    class FakeDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2025, 7, 2, tzinfo=tz)

    monkeypatch.setattr(main, "datetime", FakeDatetime)

    async with db.get_session() as session:
        session.add_all([
            MonthPage(month="2025-07", url="https://t.me/c/m1", path="p1"),
            Event(
                id=1,
                title="A",
                description="d",
                source_text="s",
                date="2025-07-18",
                time="19:00",
                location_name="Hall",
                city="Town",
                ics_post_url="https://t.me/c/asset/1",
                source_chat_id=1,
                source_message_id=10,
                source_post_url="https://t.me/c/123/10",
            ),
        ])
        await session.commit()

    await update_source_post_keyboard(1, db, bot)

    assert bot.edits
    chat_id, msg_id, _ = bot.edits[0]
    assert chat_id == -100123
    assert msg_id == 10
    async with db.get_session() as session:
        ev = await session.get(Event, 1)
        assert ev.source_chat_id == -100123
