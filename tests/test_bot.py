import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from pathlib import Path

import pytest
from aiogram import Bot, types
from sqlmodel import select
from datetime import date, timedelta, timezone, datetime, time
from typing import Any
import asyncio
import main
from telegraph.api import json_dumps
from telegraph import TelegraphException


from main import (
    Database,
    PendingUser,
    Setting,
    User,
    Event,
    Festival,
    MonthPage,
    WeekendPage,
    create_app,
    handle_register,
    handle_start,
    handle_tz,
    handle_requests,
    handle_partner_info_message,
    handle_add_event_raw,
    handle_add_event,
    handle_ask_4o,
    handle_events,
    handle_exhibitions,
    handle_stats,
    handle_edit_message,
    process_request,
    parse_event_via_4o,
    telegraph_test,
    get_telegraph_token,
    editing_sessions,
    festival_edit_sessions,
    festival_dates,
    send_festival_poll,
    notify_inactive_partners,

)


@pytest.fixture(autouse=True)
def _mock_sync_vk_source_post(monkeypatch):
    async def fake_sync(*args, **kwargs):
        return "https://vk.com/source"
    monkeypatch.setattr(main, "sync_vk_source_post", fake_sync)

FUTURE_DATE = (date.today() + timedelta(days=10)).isoformat()


class DummyBot(Bot):
    def __init__(self, token: str):
        super().__init__(token)
        self.messages = []
        self.edits = []

    async def send_message(self, chat_id, text, **kwargs):
        self.messages.append((chat_id, text, kwargs))

    async def edit_message_reply_markup(
        self, chat_id: int | None = None, message_id: int | None = None, **kwargs
    ):
        self.edits.append((chat_id, message_id, kwargs))

    async def download(self, file_id, destination):
        destination.write(b"img")


class DummyChat:
    def __init__(self, id, title, username=None, type="channel"):
        self.id = id
        self.title = title
        self.username = username
        self.type = type


class DummyMember:
    def __init__(self, status):
        self.status = status


class DummyUpdate:
    def __init__(self, chat_id, title, status="administrator"):
        self.chat = DummyChat(chat_id, title)
        self.new_chat_member = DummyMember(status)


@pytest.mark.asyncio
async def test_registration_limit(tmp_path: Path):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    bot = DummyBot("123:abc")

    for i in range(1, 11):
        msg = types.Message.model_validate(
            {
                "message_id": i,
                "date": 0,
                "chat": {"id": i, "type": "private"},
                "from": {"id": i, "is_bot": False, "first_name": "U"},
                "text": "/register",
            }
        )
        await handle_register(msg, db, bot)

    msg_over = types.Message.model_validate(
        {
            "message_id": 11,
            "date": 0,
            "chat": {"id": 11, "type": "private"},
            "from": {"id": 11, "is_bot": False, "first_name": "U"},
            "text": "/register",
        }
    )
    await handle_register(msg_over, db, bot)

    async with db.get_session() as session:
        result = await session.execute(select(PendingUser))
        count = len(result.scalars().all())
    assert count == 10


@pytest.mark.asyncio
async def test_tz_setting(tmp_path: Path):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    bot = DummyBot("123:abc")

    start_msg = types.Message.model_validate(
        {
            "message_id": 1,
            "date": 0,
            "chat": {"id": 1, "type": "private"},
            "from": {"id": 1, "is_bot": False, "first_name": "A"},
            "text": "/start",
        }
    )
    await handle_start(start_msg, db, bot)

    tz_msg = types.Message.model_validate(
        {
            "message_id": 2,
            "date": 0,
            "chat": {"id": 1, "type": "private"},
            "from": {"id": 1, "is_bot": False, "first_name": "A"},
            "text": "/tz +05:00",
        }
    )
    await handle_tz(tz_msg, db, bot)

    async with db.get_session() as session:
        setting = await session.get(Setting, "tz_offset")
    assert setting and setting.value == "+05:00"


@pytest.mark.asyncio
async def test_start_superadmin(tmp_path: Path):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    bot = DummyBot("123:abc")

    start_msg = types.Message.model_validate(
        {
            "message_id": 1,
            "date": 0,
            "chat": {"id": 1, "type": "private"},
            "from": {"id": 1, "is_bot": False, "first_name": "A"},
            "text": "/start",
        }
    )
    await handle_start(start_msg, db, bot)

    async with db.get_session() as session:
        user = await session.get(User, 1)
    assert user and user.is_superadmin


@pytest.mark.asyncio
async def test_partner_registration(tmp_path: Path):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    bot = DummyBot("123:abc")

    # superadmin becomes user 1
    start_msg = types.Message.model_validate(
        {
            "message_id": 1,
            "date": 0,
            "chat": {"id": 1, "type": "private"},
            "from": {"id": 1, "is_bot": False, "first_name": "S"},
            "text": "/start",
        }
    )
    await handle_start(start_msg, db, bot)

    # user 2 registers
    reg_msg = types.Message.model_validate(
        {
            "message_id": 2,
            "date": 0,
            "chat": {"id": 2, "type": "private"},
            "from": {"id": 2, "is_bot": False, "first_name": "U"},
            "text": "/register",
        }
    )
    await handle_register(reg_msg, db, bot)

    # superadmin requests and selects partner
    req_msg = types.Message.model_validate(
        {
            "message_id": 3,
            "date": 0,
            "chat": {"id": 1, "type": "private"},
            "from": {"id": 1, "is_bot": False, "first_name": "S"},
            "text": "/requests",
        }
    )
    await handle_requests(req_msg, db, bot)

    cb = types.CallbackQuery.model_validate(
        {
            "id": "c1",
            "from": {"id": 1, "is_bot": False, "first_name": "S"},
            "chat_instance": "1",
            "data": "partner:2",
            "message": {"message_id": 3, "date": 0, "chat": {"id": 1, "type": "private"}},
        }
    ).as_(bot)
    async def dummy_answer(*args, **kwargs):
        return None

    object.__setattr__(cb, "answer", dummy_answer)
    object.__setattr__(cb.message, "answer", dummy_answer)
    await process_request(cb, db, bot)

    info_msg = types.Message.model_validate(
        {
            "message_id": 4,
            "date": 0,
            "chat": {"id": 1, "type": "private"},
            "from": {"id": 1, "is_bot": False, "first_name": "S"},
            "text": "Org, Loc",
        }
    )
    await handle_partner_info_message(info_msg, db, bot)

    async with db.get_session() as session:
        user2 = await session.get(User, 2)
    assert user2 and user2.is_partner
    assert user2.organization == "Org"
    assert user2.location == "Loc"
    # check messages to user and admin
    assert any("approved" in m[1] for m in bot.messages if m[0] == 2)
    assert any("approved" in m[1] for m in bot.messages if m[0] == 1)


def test_create_app_requires_webhook_url(monkeypatch):
    monkeypatch.delenv("WEBHOOK_URL", raising=False)
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:abc")

    with pytest.raises(RuntimeError, match="WEBHOOK_URL is missing"):
        create_app()


@pytest.mark.asyncio
async def test_add_event_raw(tmp_path: Path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    bot = DummyBot("123:abc")

    async def fake_create(title, text, source, html_text=None, media=None, ics_url=None, db=None, **kwargs):
        return "https://t.me/test", "path"

    monkeypatch.setattr("main.create_source_page", fake_create)

    msg = types.Message.model_validate(
        {
            "message_id": 1,
            "date": 0,
            "chat": {"id": 1, "type": "private"},
            "from": {"id": 1, "is_bot": False, "first_name": "M"},
            "text": f"/addevent_raw Party|{FUTURE_DATE}|18:00|Club",
        }
    )

    await handle_add_event_raw(msg, db, bot)

    async with db.get_session() as session:
        events = (await session.execute(select(Event))).scalars().all()

    assert len(events) == 1
    assert events[0].title == "Party"
    assert events[0].telegraph_url == "https://t.me/test"


@pytest.mark.asyncio
async def test_month_page_sync(tmp_path: Path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    bot = DummyBot("123:abc")

    async def fake_create(title, text, source, html_text=None, media=None, ics_url=None, db=None, **kwargs):
        return "https://t.me/test", "path"

    called = {}

    async def fake_sync(db_obj, month):
        called["month"] = month

    monkeypatch.setattr("main.create_source_page", fake_create)
    monkeypatch.setattr("main.sync_month_page", fake_sync)

    msg = types.Message.model_validate(
        {
            "message_id": 1,
            "date": 0,
            "chat": {"id": 1, "type": "private"},
            "from": {"id": 1, "is_bot": False, "first_name": "M"},
            "text": "/addevent_raw Party|2025-07-16|18:00|Club",
        }
    )

    await handle_add_event_raw(msg, db, bot)

    assert called.get("month") == "2025-07"


@pytest.mark.asyncio
async def test_weekend_page_sync(tmp_path: Path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    bot = DummyBot("123:abc")

    async def fake_create(title, text, source, html_text=None, media=None, ics_url=None, db=None, **kwargs):
        return "url", "p"

    called = {}

    async def fake_month(db_obj, month):
        called["month"] = month

    async def fake_weekend(db_obj, start, update_links=True):
        called["weekend"] = start

    monkeypatch.setattr("main.create_source_page", fake_create)
    monkeypatch.setattr("main.sync_month_page", fake_month)
    monkeypatch.setattr("main.sync_weekend_page", fake_weekend)

    msg = types.Message.model_validate(
        {
            "message_id": 1,
            "date": 0,
            "chat": {"id": 1, "type": "private"},
            "from": {"id": 1, "is_bot": False, "first_name": "M"},
            "text": "/addevent_raw Party|2025-07-12|18:00|Club",
        }
    )

    await handle_add_event_raw(msg, db, bot)

    assert called.get("weekend") == "2025-07-12"


@pytest.mark.asyncio
async def test_add_event_raw_update(tmp_path: Path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    bot = DummyBot("123:abc")

    async def fake_create(title, text, source, html_text=None, media=None, ics_url=None, db=None, **kwargs):
        return "https://t.me/test", "path"

    monkeypatch.setattr("main.create_source_page", fake_create)

    msg1 = types.Message.model_validate(
        {
            "message_id": 1,
            "date": 0,
            "chat": {"id": 1, "type": "private"},
            "from": {"id": 1, "is_bot": False, "first_name": "M"},
            "text": "/addevent_raw Party|2025-07-16|18:00|Club",
        }
    )
    await handle_add_event_raw(msg1, db, bot)

    msg2 = types.Message.model_validate(
        {
            "message_id": 2,
            "date": 0,
            "chat": {"id": 1, "type": "private"},
            "from": {"id": 1, "is_bot": False, "first_name": "M"},
            "text": "/addevent_raw Party show|2025-07-16|18:00|Club",
        }
    )
    await handle_add_event_raw(msg2, db, bot)

    async with db.get_session() as session:
        events = (await session.execute(select(Event))).scalars().all()

    assert len(events) == 1
    assert events[0].title == "Party show"


@pytest.mark.asyncio
async def test_edit_event(tmp_path: Path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    bot = DummyBot("123:abc")

    async def fake_create(title, text, source, html_text=None, media=None, ics_url=None, db=None, **kwargs):
        return "https://t.me/test", "path"

    monkeypatch.setattr("main.create_source_page", fake_create)

    msg = types.Message.model_validate(
        {
            "message_id": 1,
            "date": 0,
            "chat": {"id": 1, "type": "private"},
            "from": {"id": 1, "is_bot": False, "first_name": "M"},
            "text": "/addevent_raw Party|2025-07-16|18:00|Club",
        }
    )
    await handle_add_event_raw(msg, db, bot)

    async with db.get_session() as session:
        event = (await session.execute(select(Event))).scalars().first()

    editing_sessions[1] = (event.id, "title")
    edit_msg = types.Message.model_validate(
        {
            "message_id": 2,
            "date": 0,
            "chat": {"id": 1, "type": "private"},
            "from": {"id": 1, "is_bot": False, "first_name": "M"},
            "text": "New Title",
        }
    )
    await handle_edit_message(edit_msg, db, bot)

    async with db.get_session() as session:
        updated = await session.get(Event, event.id)
    assert updated.title == "New Title"


@pytest.mark.asyncio
async def test_edit_remove_ticket_link(tmp_path: Path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    bot = DummyBot("123:abc")

    async def fake_create(title, text, source, html_text=None, media=None, ics_url=None, db=None, **kwargs):
        return "https://t.me/test", "path"

    monkeypatch.setattr("main.create_source_page", fake_create)

    msg = types.Message.model_validate(
        {
            "message_id": 1,
            "date": 0,
            "chat": {"id": 1, "type": "private"},
            "from": {"id": 1, "is_bot": False, "first_name": "M"},
            "text": "/addevent_raw Party|2025-07-16|18:00|Club",
        }
    )
    await handle_add_event_raw(msg, db, bot)

    async with db.get_session() as session:
        event = (await session.execute(select(Event))).scalars().first()
        event.ticket_link = "https://reg"
        await session.commit()

    editing_sessions[1] = (event.id, "ticket_link")
    edit_msg = types.Message.model_validate(
        {
            "message_id": 2,
            "date": 0,
            "chat": {"id": 1, "type": "private"},
            "from": {"id": 1, "is_bot": False, "first_name": "M"},
            "text": "-",
        }
    )
    await handle_edit_message(edit_msg, db, bot)

    async with db.get_session() as session:
        updated = await session.get(Event, event.id)
    assert updated.ticket_link is None


@pytest.mark.asyncio
async def test_edit_event_forwarded(tmp_path: Path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    bot = DummyBot("123:abc")

    async def fake_create(title, text, source, html_text=None, media=None, ics_url=None, db=None, **kwargs):
        return "https://t.me/test", "path"

    monkeypatch.setattr("main.create_source_page", fake_create)

    msg = types.Message.model_validate(
        {
            "message_id": 1,
            "date": 0,
            "chat": {"id": 1, "type": "private"},
            "from": {"id": 1, "is_bot": False, "first_name": "M"},
            "text": "/addevent_raw Party|2025-07-16|18:00|Club",
        }
    )
    await handle_add_event_raw(msg, db, bot)

    async with db.get_session() as session:
        event = (await session.execute(select(Event))).scalars().first()

    editing_sessions[1] = (event.id, "title")
    edit_msg = types.Message.model_validate(
        {
            "message_id": 2,
            "date": 0,
            "chat": {"id": 1, "type": "private"},
            "from": {"id": 1, "is_bot": False, "first_name": "M"},
            "caption": "Forwarded Title",
            "forward_from_chat": {"id": -100123, "type": "channel"},
            "forward_from_message_id": 5,
        }
    )
    await handle_edit_message(edit_msg, db, bot)

    async with db.get_session() as session:
        updated = await session.get(Event, event.id)
    assert updated.title == "Forwarded Title"


@pytest.mark.asyncio
async def test_edit_boolean_fields(tmp_path: Path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    bot = DummyBot("123:abc")

    async def fake_create(title, text, source, html_text=None, media=None, ics_url=None, db=None, **kwargs):
        return "https://t.me/test", "path"

    monkeypatch.setattr("main.create_source_page", fake_create)

    msg = types.Message.model_validate(
        {
            "message_id": 1,
            "date": 0,
            "chat": {"id": 1, "type": "private"},
            "from": {"id": 1, "is_bot": False, "first_name": "M"},
            "text": "/addevent_raw Party|2025-07-16|18:00|Club",
        }
    )
    await handle_add_event_raw(msg, db, bot)

    async with db.get_session() as session:
        event = (await session.execute(select(Event))).scalars().first()

    editing_sessions[1] = (event.id, "is_free")
    edit_msg = types.Message.model_validate(
        {
            "message_id": 2,
            "date": 0,
            "chat": {"id": 1, "type": "private"},
            "from": {"id": 1, "is_bot": False, "first_name": "M"},
            "text": "yes",
        }
    )
    await handle_edit_message(edit_msg, db, bot)

    editing_sessions[1] = (event.id, "pushkin_card")
    edit_msg2 = types.Message.model_validate(
        {
            "message_id": 3,
            "date": 0,
            "chat": {"id": 1, "type": "private"},
            "from": {"id": 1, "is_bot": False, "first_name": "M"},
            "text": "true",
        }
    )
    await handle_edit_message(edit_msg2, db, bot)

    async with db.get_session() as session:
        updated = await session.get(Event, event.id)

    assert updated.is_free is True
    assert updated.pushkin_card is True


@pytest.mark.asyncio
async def test_events_list(tmp_path: Path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    bot = DummyBot("123:abc")

    async def fake_create(title, text, source, html_text=None, media=None, ics_url=None, db=None, **kwargs):
        return "https://t.me/test", "path"

    monkeypatch.setattr("main.create_source_page", fake_create)

    start_msg = types.Message.model_validate(
        {
            "message_id": 1,
            "date": 0,
            "chat": {"id": 1, "type": "private"},
            "from": {"id": 1, "is_bot": False, "first_name": "A"},
            "text": "/start",
        }
    )
    await handle_start(start_msg, db, bot)

    add_msg = types.Message.model_validate(
        {
            "message_id": 2,
            "date": 0,
            "chat": {"id": 1, "type": "private"},
            "from": {"id": 1, "is_bot": False, "first_name": "A"},
            "text": f"/addevent_raw Party|{FUTURE_DATE}|18:00|Club",
        }
    )
    await handle_add_event_raw(add_msg, db, bot)

    bot.messages.clear()
    list_msg = types.Message.model_validate(
        {
            "message_id": 3,
            "date": 0,
            "chat": {"id": 1, "type": "private"},
            "from": {"id": 1, "is_bot": False, "first_name": "A"},
            "text": f"/events {FUTURE_DATE}",
        }
    )

    await handle_events(list_msg, db, bot)

    assert bot.messages
    text = bot.messages[-1][1]
    expected_date = date.fromisoformat(FUTURE_DATE).strftime("%d.%m.%Y")
    assert f"Events on {expected_date}" in text
    assert "1. Party" in text
    assert "18:00 Club" in text  # location no city
    assert "исходное: https://t.me/test" in text


@pytest.mark.asyncio
async def test_events_russian_date_current_year(tmp_path: Path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    bot = DummyBot("123:abc")

    async def fake_create(title, text, source, html_text=None, media=None, ics_url=None, db=None, **kwargs):
        return "u", "p"

    monkeypatch.setattr("main.create_source_page", fake_create)

    start_msg = types.Message.model_validate(
        {
            "message_id": 1,
            "date": 0,
            "chat": {"id": 1, "type": "private"},
            "from": {"id": 1, "is_bot": False, "first_name": "A"},
            "text": "/start",
        }
    )
    await handle_start(start_msg, db, bot)

    add_msg = types.Message.model_validate(
        {
            "message_id": 2,
            "date": 0,
            "chat": {"id": 1, "type": "private"},
            "from": {"id": 1, "is_bot": False, "first_name": "A"},
            "text": "/addevent_raw Party|2025-08-02|18:00|Club",
        }
    )
    await handle_add_event_raw(add_msg, db, bot)

    class FakeDate(date):
        @classmethod
        def today(cls):
            return date(2025, 7, 15)

    class FakeDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2025, 7, 15, 12, 0, tzinfo=tz)

    monkeypatch.setattr(main, "date", FakeDate)
    monkeypatch.setattr(main, "datetime", FakeDatetime)

    bot.messages.clear()
    list_msg = types.Message.model_validate(
        {
            "message_id": 3,
            "date": 0,
            "chat": {"id": 1, "type": "private"},
            "from": {"id": 1, "is_bot": False, "first_name": "A"},
            "text": "/events 2 августа",
        }
    )

    await handle_events(list_msg, db, bot)

    assert bot.messages
    text = bot.messages[-1][1]
    assert "02.08.2025" in text


@pytest.mark.asyncio
async def test_events_russian_date_next_year(tmp_path: Path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    bot = DummyBot("123:abc")

    async def fake_create(title, text, source, html_text=None, media=None, ics_url=None, db=None, **kwargs):
        return "u", "p"

    monkeypatch.setattr("main.create_source_page", fake_create)

    start_msg = types.Message.model_validate(
        {
            "message_id": 1,
            "date": 0,
            "chat": {"id": 1, "type": "private"},
            "from": {"id": 1, "is_bot": False, "first_name": "A"},
            "text": "/start",
        }
    )
    await handle_start(start_msg, db, bot)

    add_msg = types.Message.model_validate(
        {
            "message_id": 2,
            "date": 0,
            "chat": {"id": 1, "type": "private"},
            "from": {"id": 1, "is_bot": False, "first_name": "A"},
            "text": "/addevent_raw Party|2026-09-05|18:00|Club",
        }
    )
    await handle_add_event_raw(add_msg, db, bot)

    class FakeDate(date):
        @classmethod
        def today(cls):
            return date(2025, 10, 10)

    class FakeDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2025, 10, 10, 12, 0, tzinfo=tz)

    monkeypatch.setattr(main, "date", FakeDate)
    monkeypatch.setattr(main, "datetime", FakeDatetime)

    bot.messages.clear()
    list_msg = types.Message.model_validate(
        {
            "message_id": 3,
            "date": 0,
            "chat": {"id": 1, "type": "private"},
            "from": {"id": 1, "is_bot": False, "first_name": "A"},
            "text": "/events 5 сентября",
        }
    )

    await handle_events(list_msg, db, bot)

    assert bot.messages
    text = bot.messages[-1][1]
    assert "05.09.2026" in text


@pytest.mark.asyncio
async def test_ask4o_admin(tmp_path: Path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    bot = DummyBot("123:abc")

    start_msg = types.Message.model_validate(
        {
            "message_id": 1,
            "date": 0,
            "chat": {"id": 1, "type": "private"},
            "from": {"id": 1, "is_bot": False, "first_name": "A"},
            "text": "/start",
        }
    )
    await handle_start(start_msg, db, bot)

    called = {}

    async def fake_ask(text: str) -> str:
        called["text"] = text
        return "ok"

    monkeypatch.setattr("main.ask_4o", fake_ask)

    msg = types.Message.model_validate(
        {
            "message_id": 2,
            "date": 0,
            "chat": {"id": 1, "type": "private"},
            "from": {"id": 1, "is_bot": False, "first_name": "A"},
            "text": "/ask4o hello",
        }
    )

    await handle_ask_4o(msg, db, bot)

    assert called.get("text") == "hello"


@pytest.mark.asyncio
async def test_ask4o_not_admin(tmp_path: Path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    bot = DummyBot("123:abc")

    called = False

    async def fake_ask(text: str) -> str:
        nonlocal called
        called = True
        return "ok"

    monkeypatch.setattr("main.ask_4o", fake_ask)

    msg = types.Message.model_validate(
        {
            "message_id": 1,
            "date": 0,
            "chat": {"id": 2, "type": "private"},
            "from": {"id": 2, "is_bot": False, "first_name": "B"},
            "text": "/ask4o hi",
        }
    )

    await handle_ask_4o(msg, db, bot)

    assert called is False


@pytest.mark.asyncio
async def test_parse_event_includes_date(monkeypatch):
    called = {}

    class DummySession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            pass

        async def post(self, url, json=None, headers=None):
            called["payload"] = json

            class Resp:
                def raise_for_status(self):
                    pass

                async def json(self):
                    return {"choices": [{"message": {"content": "{}"}}]}

            return Resp()

    monkeypatch.setenv("FOUR_O_TOKEN", "x")
    monkeypatch.setattr("main.ClientSession", DummySession)

    await parse_event_via_4o("text")

    assert "Today is" in called["payload"]["messages"][1]["content"]


@pytest.mark.asyncio
async def test_add_events_from_text_channel_title(tmp_path: Path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    captured = {}

    async def fake_parse(text: str):
        captured["text"] = text
        return [
            {
                "title": "T",
                "short_description": "d",
                "date": FUTURE_DATE,
                "time": "18:00",
                "location_name": "Hall",
            }
        ]

    async def fake_create(title, text, source, html_text=None, media=None, ics_url=None, db=None, **kwargs):
        return "u", "p"

    monkeypatch.setattr("main.parse_event_via_4o", fake_parse)
    monkeypatch.setattr("main.create_source_page", fake_create)

    await main.add_events_from_text(db, "info", None, None, None, channel_title="Chan")

    assert "Chan" in captured["text"]


@pytest.mark.asyncio
async def test_telegraph_test(monkeypatch, capsys):
    class DummyTG:
        def __init__(self, access_token=None):
            self.access_token = access_token

        def create_page(self, title, html_content=None, **_):
            return {"url": "https://telegra.ph/test", "path": "test"}

        def edit_page(self, path, title, html_content):
            pass

    monkeypatch.setenv("TELEGRAPH_TOKEN", "t")
    monkeypatch.setattr(
        "main.Telegraph",
        lambda access_token=None, domain=None: DummyTG(access_token),
    )

    await telegraph_test()
    captured = capsys.readouterr()
    assert "Created https://telegra.ph/test" in captured.out
    assert "Edited https://telegra.ph/test" in captured.out


@pytest.mark.asyncio
async def test_telegraph_call_timeout(monkeypatch):
    monkeypatch.setattr(main, "TELEGRAPH_TIMEOUT", 0.05)

    def slow():
        import time as time_module
        time_module.sleep(0.2)

    with pytest.raises(TelegraphException):
        await main.telegraph_call(slow)


@pytest.mark.asyncio
async def test_create_source_page_photo(monkeypatch):
    class DummyTG:
        def __init__(self, access_token=None):
            self.access_token = access_token
            self.upload_called = False

        def upload_file(self, f):
            self.upload_called = True

        def create_page(self, title, html_content=None, **_):
            assert "<img" not in html_content
            return {"url": "https://telegra.ph/test", "path": "test"}

    monkeypatch.setenv("TELEGRAPH_TOKEN", "t")
    monkeypatch.setattr(
        "main.Telegraph", lambda access_token=None, domain=None: DummyTG(access_token)
    )

    res = await main.create_source_page(
        "Title", "text", None, media=(b"img", "photo.jpg")
    )
    assert res == ("https://telegra.ph/test", "test", "disabled", 0)


@pytest.mark.asyncio
async def test_create_source_page_photo_catbox(monkeypatch):
    class DummyTG:
        def __init__(self, access_token=None):
            self.access_token = access_token

        def create_page(self, title, html_content=None, **_):
            assert "<img" in html_content
            return {"url": "https://telegra.ph/test", "path": "test"}

    class DummyResp:
        status = 200

        async def text(self):
            return "https://files.catbox.moe/img.jpg"

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

    class DummySession:
        def __init__(self, *_, **__):
            self.post_called = False

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def post(self, url, data=None):
            self.post_called = True
            return DummyResp()

    monkeypatch.setenv("TELEGRAPH_TOKEN", "t")
    monkeypatch.setattr(
        "main.Telegraph", lambda access_token=None, domain=None: DummyTG(access_token)
    )
    monkeypatch.setattr(main, "ClientSession", DummySession)
    monkeypatch.setattr(main, "CATBOX_ENABLED", True)
    monkeypatch.setattr(main, "imghdr", type("X", (), {"what": lambda *a, **k: "jpeg"}))

    res = await main.create_source_page(
        "Title", "text", None, media=(b"img", "photo.jpg")
    )
    assert res == ("https://telegra.ph/test", "test", "ok", 1)


@pytest.mark.asyncio
async def test_create_source_page_reuse_urls(monkeypatch):
    class DummyTG:
        def create_page(self, title, html_content=None, **_):
            assert "https://files.catbox.moe/img.jpg" in html_content
            return {"url": "https://telegra.ph/test", "path": "test"}

    class DummySession:
        def __init__(self, *_, **__):
            raise AssertionError("should not be called")

    monkeypatch.setenv("TELEGRAPH_TOKEN", "t")
    monkeypatch.setattr(
        "main.Telegraph", lambda access_token=None, domain=None: DummyTG()
    )
    monkeypatch.setattr(main, "ClientSession", DummySession)
    monkeypatch.setattr(main, "CATBOX_ENABLED", True)

    res = await main.create_source_page(
        "Title",
        "text",
        None,
        media=(b"img", "photo.jpg"),
        catbox_urls=["https://files.catbox.moe/img.jpg"],
    )
    assert res == ("https://telegra.ph/test", "test", "", 1)


@pytest.mark.asyncio
async def test_create_source_page_normalizes_hashtags(monkeypatch):
    class DummyTG:
        def __init__(self, access_token=None):
            self.access_token = access_token

        def create_page(self, title, html_content=None, **_):
            assert "#1_августа" not in html_content
            assert "1 августа" in html_content
            return {"url": "https://telegra.ph/test", "path": "test"}

    monkeypatch.setenv("TELEGRAPH_TOKEN", "t")
    monkeypatch.setattr(
        "main.Telegraph", lambda access_token=None, domain=None: DummyTG(access_token)
    )

    res = await main.create_source_page("Title", "#1_августа text", None)
    assert res == ("https://telegra.ph/test", "test", "", 0)


def test_get_telegraph_token_creates(tmp_path, monkeypatch):
    class DummyTG:
        def create_account(self, short_name):
            return {"access_token": "abc"}

    monkeypatch.delenv("TELEGRAPH_TOKEN", raising=False)
    monkeypatch.setattr(main, "Telegraph", lambda: DummyTG())
    monkeypatch.setattr(main, "TELEGRAPH_TOKEN_FILE", str(tmp_path / "token.txt"))

    token = get_telegraph_token()
    assert token == "abc"
    assert (tmp_path / "token.txt").read_text() == "abc"


def test_get_telegraph_token_env(monkeypatch):
    monkeypatch.setenv("TELEGRAPH_TOKEN", "zzz")
    token = get_telegraph_token()
    assert token == "zzz"


@pytest.mark.asyncio
async def test_addevent_caption_photo(tmp_path: Path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    bot = DummyBot("123:abc")

    async def fake_parse(text: str, source_channel: str | None = None) -> list[dict]:
        return [
            {
                "title": "T",
                "short_description": "d",
                "date": FUTURE_DATE,
                "time": "18:00",
                "location_name": "Hall",
            }
        ]

    captured = {}

    async def fake_create(title, text, source, html_text=None, media=None, ics_url=None, db=None, **kwargs):
        captured["media"] = media
        captured["urls"] = kwargs.get("catbox_urls")
        return "u", "p"

    monkeypatch.setattr("main.parse_event_via_4o", fake_parse)
    monkeypatch.setattr("main.create_source_page", fake_create)

    msg = types.Message.model_validate(
        {
            "message_id": 1,
            "date": 0,
            "chat": {"id": 1, "type": "private"},
            "from": {"id": 1, "is_bot": False, "first_name": "A"},
            "caption": "/addevent text",
            "photo": [
                {
                    "file_id": "f1",
                    "file_unique_id": "u1",
                    "width": 100,
                    "height": 100,
                }
            ],
        }
    )

    await handle_add_event(msg, db, bot)

    assert captured["media"] is None
    assert captured["urls"] == []


@pytest.mark.asyncio
async def test_addevent_strips_command(tmp_path: Path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    bot = DummyBot("123:abc")

    async def fake_parse(text: str, source_channel: str | None = None) -> list[dict]:
        return [
            {
                "title": "T",
                "short_description": "d",
                "date": FUTURE_DATE,
                "time": "18:00",
                "location_name": "Hall",
            }
        ]

    captured = {}

    async def fake_create(title, text, source, html_text=None, media=None, ics_url=None, db=None, **kwargs):
        captured["text"] = text
        captured["html"] = html_text
        return "u", "p"

    monkeypatch.setattr("main.parse_event_via_4o", fake_parse)
    monkeypatch.setattr("main.create_source_page", fake_create)

    msg = types.Message.model_validate(
        {
            "message_id": 1,
            "date": 0,
            "chat": {"id": 1, "type": "private"},
            "from": {"id": 1, "is_bot": False, "first_name": "A"},
            "text": "/addevent\nSome info",
        }
    )

    await handle_add_event(msg, db, bot)

    assert captured["text"] == "Some info"
    assert captured["html"] == "Some info"


@pytest.mark.asyncio
async def test_addevent_vk_wall_link(tmp_path: Path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    bot = DummyBot("123:abc")

    captured = {}

    async def fake_parse(text: str) -> list[dict]:
        captured["text"] = text
        return [
            {
                "title": "T",
                "short_description": "d",
                "date": FUTURE_DATE,
                "time": "18:00",
                "location_name": "Hall",
            }
        ]

    async def fake_create(title, text, source, html_text=None, media=None, ics_url=None, db=None, **kwargs):
        captured["source"] = source
        captured["display"] = kwargs.get("display_link")
        return "u", "p"

    monkeypatch.setattr("main.parse_event_via_4o", fake_parse)
    monkeypatch.setattr("main.create_source_page", fake_create)

    msg = types.Message.model_validate(
        {
            "message_id": 1,
            "date": 0,
            "chat": {"id": 1, "type": "private"},
            "from": {"id": 1, "is_bot": False, "first_name": "A"},
            "text": "/addevent https://vk.com/wall-1_2\nSome info",
        }
    )

    await handle_add_event(msg, db, bot)

    async with db.get_session() as session:
        ev = (await session.execute(select(Event))).scalars().first()

    assert ev.source_post_url == "https://vk.com/wall-1_2"
    assert captured["text"] == "Some info"
    assert captured["source"] == "https://vk.com/wall-1_2"
    assert captured.get("display") is False


@pytest.mark.asyncio
async def test_addevent_vk_wall_link_query(tmp_path: Path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    bot = DummyBot("123:abc")

    captured = {}

    async def fake_parse(text: str) -> list[dict]:
        captured["text"] = text
        return [
            {
                "title": "T",
                "short_description": "d",
                "date": FUTURE_DATE,
                "time": "18:00",
                "location_name": "Hall",
            }
        ]

    async def fake_create(title, text, source, html_text=None, media=None, ics_url=None, db=None, **kwargs):
        captured["source"] = source
        captured["display"] = kwargs.get("display_link")
        return "u", "p"

    monkeypatch.setattr("main.parse_event_via_4o", fake_parse)
    monkeypatch.setattr("main.create_source_page", fake_create)

    msg = types.Message.model_validate(
        {
            "message_id": 1,
            "date": 0,
            "chat": {"id": 1, "type": "private"},
            "from": {"id": 1, "is_bot": False, "first_name": "A"},
            "text": "/addevent https://vk.com/page?w=wall-1_2\nSome info",
        }
    )

    await handle_add_event(msg, db, bot)

    async with db.get_session() as session:
        ev = (await session.execute(select(Event))).scalars().first()

    assert ev.source_post_url == "https://vk.com/page?w=wall-1_2"
    assert captured["text"] == "Some info"
    assert captured["source"] == "https://vk.com/page?w=wall-1_2"
    assert captured.get("display") is False


@pytest.mark.asyncio
async def test_forward_add_event(tmp_path: Path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    bot = DummyBot("123:abc")

    async def fake_parse(text: str, source_channel: str | None = None) -> list[dict]:
        return [
            {
                "title": "Forwarded",
                "short_description": "desc",
                "date": "2025-07-16",
                "time": "18:00",
                "location_name": "Club",
            }
        ]

    async def fake_create(title, text, source, html_text=None, media=None, ics_url=None, db=None, **kwargs):
        return "https://t.me/page", "p"

    monkeypatch.setattr("main.parse_event_via_4o", fake_parse)
    monkeypatch.setattr("main.create_source_page", fake_create)

    start_msg = types.Message.model_validate(
        {
            "message_id": 1,
            "date": 0,
            "chat": {"id": 1, "type": "private"},
            "from": {"id": 1, "is_bot": False, "first_name": "A"},
            "text": "/start",
        }
    )
    await handle_start(start_msg, db, bot)

    upd = DummyUpdate(-100123, "Chan")
    await main.handle_my_chat_member(upd, db)

    async with db.get_session() as session:
        ch = await session.get(main.Channel, -100123)
        ch.is_registered = True
        await session.commit()

    fwd_msg = types.Message.model_validate(
        {
            "message_id": 3,
            "date": 0,
            "forward_date": 0,
            "forward_from_chat": {"id": -100123, "type": "channel", "username": "chan"},
            "forward_from_message_id": 10,
            "chat": {"id": 1, "type": "private"},
            "from": {"id": 1, "is_bot": False, "first_name": "A"},
            "text": "Some text",
        }
    )

    await main.handle_forwarded(fwd_msg, db, bot)

    async with db.get_session() as session:
        ev = (await session.execute(select(Event))).scalars().first()

    assert ev.source_post_url == "https://t.me/chan/10"


@pytest.mark.asyncio
async def test_forward_passes_channel_name(tmp_path: Path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    bot = DummyBot("123:abc")

    captured = {}

    async def fake_parse(text: str, source_channel: str | None = None) -> list[dict]:
        captured["chan"] = source_channel
        return [
            {
                "title": "Forwarded",
                "short_description": "desc",
                "date": FUTURE_DATE,
                "time": "18:00",
                "location_name": "Club",
            }
        ]

    monkeypatch.setattr("main.parse_event_via_4o", fake_parse)

    async def fake_create(*args, **kwargs):
        return "u", "p"

    monkeypatch.setattr("main.create_source_page", fake_create)

    start_msg = types.Message.model_validate(
        {
            "message_id": 1,
            "date": 0,
            "chat": {"id": 1, "type": "private"},
            "from": {"id": 1, "is_bot": False, "first_name": "A"},
            "text": "/start",
        }
    )
    await handle_start(start_msg, db, bot)

    upd = DummyUpdate(-100123, "Chan")
    await main.handle_my_chat_member(upd, db)

    async with db.get_session() as session:
        ch = await session.get(main.Channel, -100123)
        ch.is_registered = True
        await session.commit()

    fwd_msg = types.Message.model_validate(
        {
            "message_id": 3,
            "date": 0,
            "forward_date": 0,
            "forward_from_chat": {"id": -100123, "type": "channel", "title": "Chan"},
            "forward_from_message_id": 10,
            "chat": {"id": 1, "type": "private"},
            "from": {"id": 1, "is_bot": False, "first_name": "A"},
            "text": "Some text",
        }
    )

    await main.handle_forwarded(fwd_msg, db, bot)

    assert captured["chan"] == "Chan"


@pytest.mark.asyncio
async def test_parse_event_alias_channel_title(monkeypatch):
    seen = {}

    class DummySession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            pass

        async def post(self, url, json=None, headers=None):
            seen["payload"] = json

            class Resp:
                def raise_for_status(self):
                    pass

                async def json(self):
                    return {"choices": [{"message": {"content": "{}"}}]}

            return Resp()

    monkeypatch.setenv("FOUR_O_TOKEN", "x")
    monkeypatch.setattr("main.ClientSession", DummySession)

    await main.parse_event_via_4o("t", channel_title="Name")

    assert "Name" in seen["payload"]["messages"][1]["content"]


@pytest.mark.asyncio
async def test_forward_add_event_origin(tmp_path: Path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    bot = DummyBot("123:abc")

    async def fake_parse(text: str, source_channel: str | None = None) -> list[dict]:
        return [
            {
                "title": "Forwarded",
                "short_description": "desc",
                "date": "2025-07-16",
                "time": "18:00",
                "location_name": "Club",
            }
        ]

    async def fake_create(title, text, source, html_text=None, media=None, ics_url=None, db=None, **kwargs):
        return "https://t.me/page", "p"

    monkeypatch.setattr("main.parse_event_via_4o", fake_parse)
    monkeypatch.setattr("main.create_source_page", fake_create)

    start_msg = types.Message.model_validate(
        {
            "message_id": 1,
            "date": 0,
            "chat": {"id": 1, "type": "private"},
            "from": {"id": 1, "is_bot": False, "first_name": "A"},
            "text": "/start",
        }
    )
    await handle_start(start_msg, db, bot)

    upd = DummyUpdate(-100123, "Chan")
    await main.handle_my_chat_member(upd, db)

    async with db.get_session() as session:
        ch = await session.get(main.Channel, -100123)
        ch.is_registered = True
        await session.commit()

    fwd_msg = types.Message.model_validate(
        {
            "message_id": 3,
            "date": 0,
            "forward_origin": {
                "type": "channel",
                "chat": {"id": -100123, "type": "channel", "username": "chan"},
                "message_id": 10,
                "date": 0,
            },
            "chat": {"id": 1, "type": "private"},
            "from": {"id": 1, "is_bot": False, "first_name": "A"},
            "text": "Some text",
        }
    )

    await main.handle_forwarded(fwd_msg, db, bot)

    async with db.get_session() as session:
        ev = (await session.execute(select(Event))).scalars().first()

    assert ev.source_post_url == "https://t.me/chan/10"


@pytest.mark.asyncio
async def test_forward_add_event_photo(tmp_path: Path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    bot = DummyBot("123:abc")

    async def fake_parse(text: str, source_channel: str | None = None) -> list[dict]:
        return [
            {
                "title": "Forwarded",
                "short_description": "desc",
                "date": FUTURE_DATE,
                "time": "18:00",
                "location_name": "Club",
            }
        ]

    captured = {}

    async def fake_add(db2, text, source_link, html_text=None, media=None, **kwargs):
        captured["media"] = media
        return []

    monkeypatch.setattr("main.parse_event_via_4o", fake_parse)
    monkeypatch.setattr("main.add_events_from_text", fake_add)

    start_msg = types.Message.model_validate(
        {
            "message_id": 1,
            "date": 0,
            "chat": {"id": 1, "type": "private"},
            "from": {"id": 1, "is_bot": False, "first_name": "A"},
            "text": "/start",
        }
    )
    await handle_start(start_msg, db, bot)

    upd = DummyUpdate(-100123, "Chan")
    await main.handle_my_chat_member(upd, db)

    async with db.get_session() as session:
        ch = await session.get(main.Channel, -100123)
        ch.is_registered = True
        await session.commit()

    fwd_msg = types.Message.model_validate(
        {
            "message_id": 3,
            "date": 0,
            "forward_date": 0,
            "forward_from_chat": {"id": -100123, "type": "channel", "username": "chan"},
            "forward_from_message_id": 10,
            "chat": {"id": 1, "type": "private"},
            "from": {"id": 1, "is_bot": False, "first_name": "A"},
            "text": "Some text",
            "photo": [
                {
                    "file_id": "f2",
                    "file_unique_id": "u2",
                    "width": 50,
                    "height": 50,
                }
            ],
        }
    )

    await main.handle_forwarded(fwd_msg, db, bot)

    assert captured["media"] == [(b"img", "photo.jpg")]


@pytest.mark.asyncio
async def test_forward_unregistered(tmp_path: Path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    bot = DummyBot("123:abc")

    async def fake_parse(text: str, source_channel: str | None = None) -> list[dict]:
        return [
            {
                "title": "Fwd",
                "short_description": "d",
                "date": "2025-07-16",
                "time": "18:00",
                "location_name": "Club",
            }
        ]

    async def fake_create(title, text, source, html_text=None, media=None, ics_url=None, db=None, **kwargs):
        return "https://t.me/page", "p"

    monkeypatch.setattr("main.parse_event_via_4o", fake_parse)
    monkeypatch.setattr("main.create_source_page", fake_create)

    start_msg = types.Message.model_validate(
        {
            "message_id": 1,
            "date": 0,
            "chat": {"id": 1, "type": "private"},
            "from": {"id": 1, "is_bot": False, "first_name": "A"},
            "text": "/start",
        }
    )
    await handle_start(start_msg, db, bot)

    upd = DummyUpdate(-100123, "Chan")
    await main.handle_my_chat_member(upd, db)

    fwd_msg = types.Message.model_validate(
        {
            "message_id": 2,
            "date": 0,
            "forward_date": 0,
            "forward_from_chat": {"id": -100123, "type": "channel", "username": "chan"},
            "forward_from_message_id": 10,
            "chat": {"id": 1, "type": "private"},
            "from": {"id": 1, "is_bot": False, "first_name": "A"},
            "text": "Some text",
        }
    )

    await main.handle_forwarded(fwd_msg, db, bot)

    async with db.get_session() as session:
        ev = (await session.execute(select(Event))).scalars().first()

    assert ev.source_post_url is None


@pytest.mark.asyncio
async def test_media_group_caption_first(tmp_path: Path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    bot = DummyBot("123:abc")

    async def fake_parse(text: str, source_channel: str | None = None) -> list[dict]:
        return [
            {
                "title": "MG",
                "short_description": "d",
                "date": "2025-07-16",
                "time": "18:00",
                "location_name": "Club",
            }
        ]

    async def fake_create(title, text, source, html_text=None, media=None, ics_url=None, db=None, **kwargs):
        return "https://t.me/page", "p"

    monkeypatch.setattr("main.parse_event_via_4o", fake_parse)
    monkeypatch.setattr("main.create_source_page", fake_create)

    start_msg = types.Message.model_validate(
        {
            "message_id": 1,
            "date": 0,
            "chat": {"id": 1, "type": "private"},
            "from": {"id": 1, "is_bot": False, "first_name": "A"},
            "text": "/start",
        }
    )
    await handle_start(start_msg, db, bot)

    upd = DummyUpdate(-100123, "Chan")
    await main.handle_my_chat_member(upd, db)

    async with db.get_session() as session:
        ch = await session.get(main.Channel, -100123)
        ch.is_registered = True
        await session.commit()

    msg1 = types.Message.model_validate(
        {
            "message_id": 2,
            "date": 0,
            "forward_date": 0,
            "media_group_id": "g1",
            "forward_from_chat": {"id": -100123, "type": "channel", "username": "chan"},
            "forward_from_message_id": 10,
            "chat": {"id": 1, "type": "private"},
            "from": {"id": 1, "is_bot": False, "first_name": "A"},
            "caption": "Announce",
        }
    )
    await main.handle_forwarded(msg1, db, bot)

    msg2 = types.Message.model_validate(
        {
            "message_id": 3,
            "date": 0,
            "forward_date": 0,
            "media_group_id": "g1",
            "forward_from_chat": {"id": -100123, "type": "channel", "username": "chan"},
            "forward_from_message_id": 11,
            "chat": {"id": 1, "type": "private"},
            "from": {"id": 1, "is_bot": False, "first_name": "A"},
        }
    )
    await main.handle_forwarded(msg2, db, bot)

    async with db.get_session() as session:
        ev = (await session.execute(select(Event))).scalars().first()

    assert ev.title == "MG"
    assert ev.source_post_url == "https://t.me/chan/10"


@pytest.mark.asyncio
async def test_media_group_caption_last(tmp_path: Path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    bot = DummyBot("123:abc")

    async def fake_parse(text: str, source_channel: str | None = None) -> list[dict]:
        return [
            {
                "title": "MG",
                "short_description": "d",
                "date": "2025-07-16",
                "time": "18:00",
                "location_name": "Club",
            }
        ]

    async def fake_create(title, text, source, html_text=None, media=None, ics_url=None, db=None, **kwargs):
        return "https://t.me/page", "p"

    monkeypatch.setattr("main.parse_event_via_4o", fake_parse)
    monkeypatch.setattr("main.create_source_page", fake_create)

    start_msg = types.Message.model_validate(
        {
            "message_id": 1,
            "date": 0,
            "chat": {"id": 1, "type": "private"},
            "from": {"id": 1, "is_bot": False, "first_name": "A"},
            "text": "/start",
        }
    )
    await handle_start(start_msg, db, bot)

    upd = DummyUpdate(-100123, "Chan")
    await main.handle_my_chat_member(upd, db)

    async with db.get_session() as session:
        ch = await session.get(main.Channel, -100123)
        ch.is_registered = True
        await session.commit()

    msg1 = types.Message.model_validate(
        {
            "message_id": 2,
            "date": 0,
            "forward_date": 0,
            "media_group_id": "g2",
            "forward_from_chat": {"id": -100123, "type": "channel", "username": "chan"},
            "forward_from_message_id": 10,
            "chat": {"id": 1, "type": "private"},
            "from": {"id": 1, "is_bot": False, "first_name": "A"},
        }
    )
    await main.handle_forwarded(msg1, db, bot)

    msg2 = types.Message.model_validate(
        {
            "message_id": 3,
            "date": 0,
            "forward_date": 0,
            "media_group_id": "g2",
            "forward_from_chat": {"id": -100123, "type": "channel", "username": "chan"},
            "forward_from_message_id": 11,
            "chat": {"id": 1, "type": "private"},
            "from": {"id": 1, "is_bot": False, "first_name": "A"},
            "caption": "Announce",
        }
    )
    await main.handle_forwarded(msg2, db, bot)

    async with db.get_session() as session:
        evs = (await session.execute(select(Event))).scalars().all()

    assert len(evs) == 1
    assert evs[0].source_post_url == "https://t.me/chan/11"


@pytest.mark.asyncio
async def test_mark_free(tmp_path: Path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    bot = DummyBot("123:abc")

    async def fake_create(title, text, source, html_text=None, media=None, ics_url=None, db=None, **kwargs):
        return "https://t.me/test", "path"

    monkeypatch.setattr("main.create_source_page", fake_create)

    msg = types.Message.model_validate(
        {
            "message_id": 1,
            "date": 0,
            "chat": {"id": 1, "type": "private"},
            "from": {"id": 1, "is_bot": False, "first_name": "M"},
            "text": "/addevent_raw Party|2025-07-16|18:00|Club",
        }
    )
    await handle_add_event_raw(msg, db, bot)

    async with db.get_session() as session:
        event = (await session.execute(select(Event))).scalars().first()

    cb = types.CallbackQuery.model_validate(
        {
            "id": "c1",
            "from": {"id": 1, "is_bot": False, "first_name": "M"},
            "chat_instance": "1",
            "data": f"markfree:{event.id}",
            "message": {
                "message_id": 2,
                "date": 0,
                "chat": {"id": 1, "type": "private"},
            },
        }
    ).as_(bot)

    async def dummy_answer(*args, **kwargs):
        return None

    object.__setattr__(cb, "answer", dummy_answer)
    await process_request(cb, db, bot)

    async with db.get_session() as session:
        updated = await session.get(Event, event.id)
    assert updated.is_free is True
    assert bot.edits
    btn = bot.edits[-1][2]["reply_markup"].inline_keyboard[0][0]
    assert btn.text == "\u2705 Бесплатное мероприятие"


@pytest.mark.asyncio
async def test_toggle_silent(tmp_path: Path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    bot = DummyBot("123:abc")

    async def fake_create(title, text, source, html_text=None, media=None, ics_url=None, db=None, **kwargs):
        return "https://t.me/test", "path"

    monkeypatch.setattr("main.create_source_page", fake_create)

    msg = types.Message.model_validate(
        {
            "message_id": 1,
            "date": 0,
            "chat": {"id": 1, "type": "private"},
            "from": {"id": 1, "is_bot": False, "first_name": "M"},
            "text": "/addevent_raw Party|2025-07-16|18:00|Club",
        }
    )
    await handle_add_event_raw(msg, db, bot)

    async with db.get_session() as session:
        event = (await session.execute(select(Event))).scalars().first()

    cb = types.CallbackQuery.model_validate(
        {
            "id": "c2",
            "from": {"id": 1, "is_bot": False, "first_name": "M"},
            "chat_instance": "1",
            "data": f"togglesilent:{event.id}",
            "message": {
                "message_id": 2,
                "date": 0,
                "chat": {"id": 1, "type": "private"},
            },
        }
    ).as_(bot)

    async def dummy_answer(*args, **kwargs):
        return None

    object.__setattr__(cb, "answer", dummy_answer)
    await process_request(cb, db, bot)

    async with db.get_session() as session:
        updated = await session.get(Event, event.id)
    assert updated.silent is True
    assert bot.edits
    btn = bot.edits[-1][2]["reply_markup"].inline_keyboard[0][0]
    assert "Тихий" in btn.text


@pytest.mark.asyncio
async def test_exhibition_listing(tmp_path: Path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    bot = DummyBot("123:abc")

    start_msg = types.Message.model_validate(
        {
            "message_id": 1,
            "date": 0,
            "chat": {"id": 1, "type": "private"},
            "from": {"id": 1, "is_bot": False, "first_name": "A"},
            "text": "/start",
        }
    )
    await handle_start(start_msg, db, bot)

    async def fake_parse(text: str, source_channel: str | None = None) -> list[dict]:
        return [
            {
                "title": "Expo",
                "short_description": "desc",
                "festival": "",
                "date": "2025-07-10",
                "end_date": "2025-07-20",
                "time": "",
                "location_name": "Hall",
                "location_address": "Addr",
                "city": "Калининград",
                "ticket_price_min": None,
                "ticket_price_max": None,
                "ticket_link": None,
                "event_type": "выставка",
                "emoji": None,
                "is_free": True,
            }
        ]

    monkeypatch.setattr("main.parse_event_via_4o", fake_parse)

    async def fake_create(title, text, source, html_text=None, media=None, ics_url=None, db=None, **kwargs):
        return "url", "p"

    monkeypatch.setattr("main.create_source_page", fake_create)

    add_msg = types.Message.model_validate(
        {
            "message_id": 2,
            "date": 0,
            "chat": {"id": 1, "type": "private"},
            "from": {"id": 1, "is_bot": False, "first_name": "A"},
            "text": "/addevent anything",
        }
    )
    await handle_add_event(add_msg, db, bot)

    evt_msg = types.Message.model_validate(
        {
            "message_id": 3,
            "date": 0,
            "chat": {"id": 1, "type": "private"},
            "from": {"id": 1, "is_bot": False, "first_name": "A"},
            "text": "/events 2025-07-10",
        }
    )
    await handle_events(evt_msg, db, bot)
    assert "(Открытие) Expo" in bot.messages[-1][1]

    evt_msg2 = types.Message.model_validate(
        {
            "message_id": 4,
            "date": 0,
            "chat": {"id": 1, "type": "private"},
            "from": {"id": 1, "is_bot": False, "first_name": "A"},
            "text": "/events 2025-07-20",
        }
    )
    await handle_events(evt_msg2, db, bot)
    assert "(Закрытие) Expo" in bot.messages[-1][1]

    exh_msg = types.Message.model_validate(
        {
            "message_id": 5,
            "date": 0,
            "chat": {"id": 1, "type": "private"},
            "from": {"id": 1, "is_bot": False, "first_name": "A"},
            "text": "/exhibitions",
        }
    )
    await handle_exhibitions(exh_msg, db, bot)
    assert "c 10 июля по 20 июля" in bot.messages[-1][1]


@pytest.mark.asyncio
async def test_multiple_events(tmp_path: Path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    bot = DummyBot("123:abc")

    start_msg = types.Message.model_validate(
        {
            "message_id": 1,
            "date": 0,
            "chat": {"id": 1, "type": "private"},
            "from": {"id": 1, "is_bot": False, "first_name": "A"},
            "text": "/start",
        }
    )
    await handle_start(start_msg, db, bot)

    async def fake_parse(text: str, source_channel: str | None = None) -> list[dict]:
        return [
            {
                "title": "One",
                "short_description": "d1",
                "date": "2025-07-10",
                "time": "18:00",
                "location_name": "Hall",
            },
            {
                "title": "Two",
                "short_description": "d2",
                "date": "2025-07-11",
                "time": "20:00",
                "location_name": "Hall",
            },
        ]

    async def fake_create(title, text, source, html_text=None, media=None, ics_url=None, db=None, **kwargs):
        return f"url/{title}", title

    monkeypatch.setattr("main.parse_event_via_4o", fake_parse)
    monkeypatch.setattr("main.create_source_page", fake_create)

    add_msg = types.Message.model_validate(
        {
            "message_id": 2,
            "date": 0,
            "chat": {"id": 1, "type": "private"},
            "from": {"id": 1, "is_bot": False, "first_name": "A"},
            "text": "/addevent multi",
        }
    )
    await handle_add_event(add_msg, db, bot)

    async with db.get_session() as session:
        events = (await session.execute(select(Event))).scalars().all()

    assert len(events) == 2
    assert any(e.title == "One" for e in events)
    assert any(e.title == "Two" for e in events)


@pytest.mark.asyncio
async def test_months_command(tmp_path: Path):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    bot = DummyBot("123:abc")

    async with db.get_session() as session:
        session.add(main.MonthPage(month="2025-07", url="https://t.me/p", path="p"))
        await session.commit()

    start_msg = types.Message.model_validate(
        {
            "message_id": 1,
            "date": 0,
            "chat": {"id": 1, "type": "private"},
            "from": {"id": 1, "is_bot": False, "first_name": "A"},
            "text": "/start",
        }
    )
    await handle_start(start_msg, db, bot)

    msg = types.Message.model_validate(
        {
            "message_id": 2,
            "date": 0,
            "chat": {"id": 1, "type": "private"},
            "from": {"id": 1, "is_bot": False, "first_name": "A"},
            "text": "/pages",
        }
    )

    await main.handle_pages(msg, db, bot)
    assert "2025-07" in bot.messages[-1][1]


@pytest.mark.asyncio
async def test_stats_pages(tmp_path: Path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    bot = DummyBot("123:abc")

    prev_month = (date.today().replace(day=1) - timedelta(days=1)).strftime("%Y-%m")
    prev_weekend = main.next_weekend_start(date.today() - timedelta(days=7))

    cur_month = date.today().strftime("%Y-%m")
    next_month = main.next_month(cur_month)
    cur_weekend = main.next_weekend_start(date.today())
    next_weekend = main.next_weekend_start(cur_weekend + timedelta(days=1))

    async with db.get_session() as session:
        session.add(main.MonthPage(month=prev_month, url="u", path="mp_prev"))
        session.add(main.MonthPage(month=cur_month, url="u2", path="mp_cur"))
        session.add(main.MonthPage(month=next_month, url="u3", path="mp_next"))
        session.add(main.WeekendPage(start=prev_weekend.isoformat(), url="w1", path="wp_prev"))
        session.add(main.WeekendPage(start=cur_weekend.isoformat(), url="w2", path="wp_cur"))
        session.add(main.WeekendPage(start=next_weekend.isoformat(), url="w3", path="wp_next"))

        await session.commit()

    class DummyTG:
        def __init__(self, access_token=None):
            self.access_token = access_token

        def get_views(self, path, **_):

            views = {
                "mp_prev": {"views": 100},
                "mp_cur": {"views": 200},
                "mp_next": {"views": 300},
                "wp_prev": {"views": 10},
                "wp_cur": {"views": 20},
                "wp_next": {"views": 30},
            }
            return views[path]


    monkeypatch.setenv("TELEGRAPH_TOKEN", "t")
    monkeypatch.setattr(
        "main.Telegraph", lambda access_token=None, domain=None: DummyTG(access_token)
    )


    start_msg = types.Message.model_validate({
        "message_id": 1,
        "date": 0,
        "chat": {"id": 1, "type": "private"},
        "from": {"id": 1, "is_bot": False, "first_name": "A"},
        "text": "/start",
    })
    await handle_start(start_msg, db, bot)

    msg = types.Message.model_validate({
        "message_id": 2,
        "date": 0,
        "chat": {"id": 1, "type": "private"},
        "from": {"id": 1, "is_bot": False, "first_name": "A"},
        "text": "/stats",
    })
    await handle_stats(msg, db, bot)


    lines = bot.messages[-1][1].splitlines()
    assert any("100" in l for l in lines)  # previous month
    assert any("10" in l for l in lines)   # previous weekend
    assert any("20" in l for l in lines)   # current weekend
    assert any("300" in l for l in lines)  # future month



@pytest.mark.asyncio
async def test_stats_events(tmp_path: Path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    bot = DummyBot("123:abc")

    prev_month_start = (date.today().replace(day=1) - timedelta(days=1)).replace(day=1)
    event_date = prev_month_start + timedelta(days=1)

    async with db.get_session() as session:
        session.add(
            Event(
                title="A",
                description="d",
                source_text="s",
                date=event_date.isoformat(),
                time="10:00",
                location_name="Hall",
                telegraph_url="http://a",
                telegraph_path="pa",
            )
        )
        session.add(
            Event(
                title="B",
                description="d",
                source_text="s",
                date=event_date.isoformat(),
                time="11:00",
                location_name="Hall",
                telegraph_url="http://b",
                telegraph_path="pb",
            )
        )
        await session.commit()

    class DummyTG:
        def __init__(self, access_token=None):
            pass

        def get_views(self, path, **_):
            return {"pa": {"views": 5}, "pb": {"views": 10}}[path]

    monkeypatch.setenv("TELEGRAPH_TOKEN", "t")

    monkeypatch.setattr(
        "main.Telegraph", lambda access_token=None, domain=None: DummyTG(access_token)
    )


    start_msg = types.Message.model_validate({
        "message_id": 1,
        "date": 0,
        "chat": {"id": 1, "type": "private"},
        "from": {"id": 1, "is_bot": False, "first_name": "A"},
        "text": "/start",
    })
    await handle_start(start_msg, db, bot)

    msg = types.Message.model_validate({
        "message_id": 2,
        "date": 0,
        "chat": {"id": 1, "type": "private"},
        "from": {"id": 1, "is_bot": False, "first_name": "A"},
        "text": "/stats events",
    })
    await handle_stats(msg, db, bot)

    lines = bot.messages[-1][1].splitlines()
    assert lines[0].startswith("http://b")
    assert "10" in lines[0]
    assert "5" in lines[1]


@pytest.mark.asyncio
async def test_stats_festivals(tmp_path: Path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    bot = DummyBot("123:abc")

    async with db.get_session() as session:
        session.add(
            main.Festival(
                name="Fest",
                telegraph_url="http://fest",
                telegraph_path="fp",
                vk_post_url="https://vk.com/wall-1_2",
            )
        )
        await session.commit()

    class DummyTG:
        def __init__(self, access_token=None, domain=None):
            pass

        def get_views(self, path, **_):
            return {"fp": {"views": 50}}[path]

    monkeypatch.setenv("TELEGRAPH_TOKEN", "t")
    monkeypatch.setattr(
        "main.Telegraph", lambda access_token=None, domain=None: DummyTG(access_token)
    )

    async def fake_vk_api(method, params, db=None, bot=None):
        if method == "wall.getById":
            return {"response": [{"views": {"count": 70}}]}
        if method == "stats.getPostReach":
            return {"response": [{"reach_total": 40}]}
        raise AssertionError(method)

    monkeypatch.setattr(main, "_vk_api", fake_vk_api)

    start_msg = types.Message.model_validate(
        {
            "message_id": 1,
            "date": 0,
            "chat": {"id": 1, "type": "private"},
            "from": {"id": 1, "is_bot": False, "first_name": "A"},
            "text": "/start",
        }
    )
    await handle_start(start_msg, db, bot)

    msg = types.Message.model_validate(
        {
            "message_id": 2,
            "date": 0,
            "chat": {"id": 1, "type": "private"},
            "from": {"id": 1, "is_bot": False, "first_name": "A"},
            "text": "/stats",
        }
    )
    await handle_stats(msg, db, bot)

    lines = bot.messages[-1][1].splitlines()
    assert "Фестивали (телеграм)" in lines
    assert any("Fest" in l and "50" in l for l in lines)
    assert any("Fest" in l and "70" in l and "40" in l for l in lines)


@pytest.mark.asyncio
async def test_build_month_page_content(tmp_path: Path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    async with db.get_session() as session:
        session.add(
            Event(
                title="T",
                description="d",
                source_text="s",
                date="2025-07-16",
                time="18:00",
                location_name="Hall",
                is_free=True,
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


    title, content = await main.build_month_page_content(db, "2025-07")
    assert "июле 2025" in title
    assert "Полюбить Калининград Анонсы" in title
    assert any(n.get("tag") == "br" for n in content)


@pytest.mark.asyncio
async def test_build_weekend_page_content(tmp_path: Path):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    saturday = main.next_weekend_start(date.today())
    async with db.get_session() as session:
        session.add(
            Event(
                title="W",
                description="d",
                source_text="s",
                date=saturday.isoformat(),
                time="18:00",
                location_name="Hall",
            )
        )
        await session.commit()

    title, content = await main.build_weekend_page_content(db, saturday.isoformat())
    assert "выходных" in title
    assert any(n.get("tag") == "h4" for n in content)
    intro = content[0]
    assert intro.get("tag") == "p"
    link = next(
        c
        for c in intro["children"]
        if isinstance(c, dict) and c.get("tag") == "a"
    )
    assert link.get("attrs", {}).get("href") == "https://t.me/kenigevents"
    assert str(saturday.day) in title

    cross = date(2025, 1, 31)
    async with db.get_session() as session:
        session.add(
            Event(
                title="C",
                description="d",
                source_text="s",
                date=cross.isoformat(),
                time="18:00",
                location_name="Hall",
            )
        )
        await session.commit()

    title2, _ = await main.build_weekend_page_content(db, cross.isoformat())
    assert "31 января" in title2 and "1 февраля" in title2


@pytest.mark.asyncio
async def test_weekend_nav_and_exhibitions(tmp_path: Path):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    saturday = date(2025, 7, 12)
    next_sat = saturday + timedelta(days=7)
    async with db.get_session() as session:
        session.add(WeekendPage(start=saturday.isoformat(), url="u1", path="p1"))
        session.add(WeekendPage(start=next_sat.isoformat(), url="u2", path="p2"))
        session.add(MonthPage(month="2025-07", url="m1", path="mp1"))
        session.add(MonthPage(month="2025-08", url="m2", path="mp2"))
        session.add(
            Event(
                title="Expo",
                description="d",
                source_text="s",
                date=(saturday - timedelta(days=1)).isoformat(),
                end_date=(saturday + timedelta(days=10)).isoformat(),
                time="10:00",
                location_name="Hall",
                event_type="выставка",
            )
        )
        await session.commit()

    _, content = await main.build_weekend_page_content(db, saturday.isoformat())
    nav_blocks = [
        n
        for n in content
        if n.get("tag") == "h4"
        and any(
            isinstance(c, dict) and c.get("attrs", {}).get("href") == "u2"
            for c in n.get("children", [])
        )
    ]
    assert len(nav_blocks) == 2
    first_block_children = nav_blocks[0]["children"]
    assert not isinstance(first_block_children[0], dict)

    month_link_present = any(
        n.get("tag") == "h4"
        and any(
            isinstance(c, dict) and c.get("attrs", {}).get("href") == "m1"
            for c in n.get("children", [])
        )
        for n in content
    )
    assert month_link_present

    idx_exh = next(
        i
        for i, n in enumerate(content)
        if n.get("tag") == "h3" and "Постоянные" in "".join(n.get("children", []))
    )
    assert content[idx_exh - 1].get("tag") == "p"
    assert content[idx_exh - 2].get("tag") == "br"


@pytest.mark.asyncio
async def test_month_nav_and_exhibitions(tmp_path: Path):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    async with db.get_session() as session:
        session.add(MonthPage(month="2025-07", url="m1", path="p1"))
        session.add(MonthPage(month="2025-08", url="m2", path="p2"))
        session.add(
            Event(
                title="Expo",
                description="d",
                source_text="s",
                date="2025-07-05",
                end_date="2025-07-20",
                time="10:00",
                location_name="Hall",
                event_type="выставка",
            )
        )
        await session.commit()

    _, content = await main.build_month_page_content(db, "2025-07")
    nav_blocks = [
        n
        for n in content
        if n.get("tag") == "h4"
        and any(
            isinstance(c, dict) and c.get("attrs", {}).get("href") == "m2"
            for c in n.get("children", [])
        )
    ]
    assert len(nav_blocks) == 2
    first_block_children = nav_blocks[0]["children"]
    assert not isinstance(first_block_children[0], dict)

    idx_exh = next(
        i
        for i, n in enumerate(content)
        if n.get("tag") == "h3" and "Постоянные" in "".join(n.get("children", []))
    )
    assert content[idx_exh - 1].get("tag") == "p"
    assert content[idx_exh - 2].get("tag") == "br"


@pytest.mark.asyncio
async def test_sync_weekend_page_first_creation_includes_nav(
    tmp_path: Path, monkeypatch
):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    saturday = date(2025, 7, 12)
    next_sat = saturday + timedelta(days=7)
    updates: list[list[dict]] = []

    class DummyTG:
        def create_page(self, title, content):
            return {"url": "u1", "path": "p1"}

        def edit_page(self, path, title=None, content=None):
            updates.append(content)

    monkeypatch.setattr("main.get_telegraph_token", lambda: "t")
    monkeypatch.setattr(
        "main.Telegraph", lambda access_token=None, domain=None: DummyTG()
    )

    async with db.get_session() as session:
        session.add(WeekendPage(start=next_sat.isoformat(), url="u2", path="p2"))
        session.add(MonthPage(month="2025-07", url="m1", path="mp1"))
        session.add(MonthPage(month="2025-08", url="m2", path="mp2"))
        session.add(
            Event(
                title="Expo",
                description="d",
                source_text="s",
                date=(saturday - timedelta(days=1)).isoformat(),
                end_date=(saturday + timedelta(days=10)).isoformat(),
                time="10:00",
                location_name="Hall",
                event_type="выставка",
            )
        )
        await session.commit()

    await main.sync_weekend_page(db, saturday.isoformat())
    assert updates
    content = updates[0]
    found_weekend = any(
        isinstance(n, dict)
        and n.get("tag") == "h4"
        and any(
            isinstance(c, dict) and c.get("attrs", {}).get("href") == "u2"
            for c in n.get("children", [])
        )
        for n in content
    )
    found_exh = any(
        isinstance(n, dict)
        and n.get("tag") == "h3"
        and "Постоянные" in "".join(n.get("children", []))
        for n in content
    )
    assert found_weekend
    assert found_exh


@pytest.mark.asyncio
async def test_sync_weekend_page_updates_other_pages(tmp_path: Path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    saturday = date(2025, 7, 12)
    next_sat = saturday + timedelta(days=7)

    edits: list[tuple[str, str]] = []

    class DummyTG:
        def create_page(self, title, content):
            edits.append(("create", "p1"))
            return {"url": "u1", "path": "p1"}

        def edit_page(self, path, title=None, content=None):
            edits.append(("edit", path))

    monkeypatch.setattr("main.get_telegraph_token", lambda: "t")
    monkeypatch.setattr(
        "main.Telegraph", lambda access_token=None, domain=None: DummyTG()
    )

    async with db.get_session() as session:
        session.add(WeekendPage(start=next_sat.isoformat(), url="u2", path="p2"))
        await session.commit()

    await main.sync_weekend_page(db, saturday.isoformat())

    assert ("edit", "p2") in edits


@pytest.mark.asyncio
async def test_missing_added_at(tmp_path: Path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    async with db.get_session() as session:
        session.add(
            Event(
                title="T",
                description="d",
                source_text="s",
                date="2025-07-16",
                time="18:00",
                location_name="Hall",
                is_free=True,
                added_at=None,
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


    title, content = await main.build_month_page_content(db, "2025-07")
    assert any(n.get("tag") == "h4" for n in content)


@pytest.mark.asyncio
async def test_event_title_link(tmp_path: Path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    async with db.get_session() as session:
        session.add(
            Event(
                title="Party",
                description="d",
                source_text="s",
                date="2025-07-16",
                time="18:00",
                location_name="Hall",
                source_post_url="https://t.me/chan/1",
                emoji="🎉",
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


    _, content = await main.build_month_page_content(db, FUTURE_DATE[:7])
    h4 = next(n for n in content if n.get("tag") == "h4")
    children = h4["children"]
    assert any(isinstance(c, dict) and c.get("tag") == "a" for c in children)
    anchor = next(c for c in children if isinstance(c, dict) and c.get("tag") == "a")
    assert anchor["attrs"]["href"] == "https://t.me/chan/1"
    assert anchor["children"] == ["Party"]


@pytest.mark.asyncio
async def test_emoji_not_duplicated(tmp_path: Path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    async with db.get_session() as session:
        session.add(
            Event(
                title="🎉 Party",
                description="d",
                source_text="s",
                date="2025-07-16",
                time="18:00",
                location_name="Hall",
                emoji="🎉",
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


    _, content = await main.build_month_page_content(db, FUTURE_DATE[:7])
    h4 = next(n for n in content if n.get("tag") == "h4")
    text = "".join(
        c if isinstance(c, str) else "".join(c.get("children", []))
        for c in h4["children"]
    )
    assert text.count("🎉") == 1


@pytest.mark.asyncio
async def test_spacing_after_headers(tmp_path: Path):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    async with db.get_session() as session:
        session.add(
            Event(
                title="Weekend",
                description="d",
                source_text="s",
                date=FUTURE_DATE,
                time="18:00",
                location_name="Hall",
            )
        )
        session.add(
            Event(
                title="Expo",
                description="d",
                source_text="s",
                date=date.today().isoformat(),
                time="20:00",
                location_name="Hall",
                end_date=(date.today() + timedelta(days=8)).isoformat(),
                event_type="выставка",
            )
        )
        await session.commit()

    _, content = await main.build_month_page_content(db, FUTURE_DATE[:7])
    idx = next(
        i
        for i, n in enumerate(content)
        if n.get("tag") == "h3" and str(date.fromisoformat(FUTURE_DATE).day) in "".join(n.get("children", []))
    )
    assert content[idx + 1].get("tag") == "br"
    exh_idx = next(
        i
        for i, n in enumerate(content)
        if n.get("tag") == "h3" and "Постоянные" in "".join(n.get("children", []))
    )
    assert content[exh_idx + 1].get("tag") == "br"


@pytest.mark.asyncio
async def test_event_spacing(tmp_path: Path):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    async with db.get_session() as session:
        session.add(
            Event(
                title="One",
                description="d",
                source_text="s",
                date=FUTURE_DATE,
                time="18:00",
                location_name="Hall",
            )
        )
        session.add(
            Event(
                title="Two",
                description="d",
                source_text="s",
                date=FUTURE_DATE,
                time="19:00",
                location_name="Hall",
            )
        )
        await session.commit()

    _, content = await main.build_month_page_content(db, FUTURE_DATE[:7])
    indices = [i for i, n in enumerate(content) if n.get("tag") == "h4"]
    assert content[indices[0] + 1].get("tag") == "p"


def test_registration_link_formatting():
    e = Event(
        title="T",
        description="d",
        source_text="s",
        date="2025-07-10",
        time="18:00",
        location_name="Hall",
        is_free=True,
        ticket_link="https://reg",
    )
    md = main.format_event_md(e)
    assert "Бесплатно [по регистрации](https://reg)" in md


def test_format_event_no_city_dup():
    e = Event(
        title="T",
        description="d",
        source_text="s",
        date="2025-07-10",
        time="18:00",
        location_name="Hall",
        location_address="Addr, Калининград",
        city="Калининград",
    )
    md = main.format_event_md(e)
    assert md.count("Калининград") == 1


def test_pushkin_card_formatting():
    e = Event(
        title="T",
        description="d",
        source_text="s",
        date="2025-07-10",
        time="18:00",
        location_name="Hall",
        ticket_link="https://reg",
        pushkin_card=True,
    )
    md = main.format_event_md(e)
    lines = md.split("\n")
    assert "\u2705 Пушкинская карта" in lines
    # next line should mention tickets or registration
    assert any("Билеты" in l or "регистра" in l for l in lines[lines.index("\u2705 Пушкинская карта") + 1:])


@pytest.mark.asyncio
async def test_date_range_parsing(tmp_path: Path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    bot = DummyBot("123:abc")

    async def fake_parse(text: str, source_channel: str | None = None) -> list[dict]:
        return [
            {
                "title": "Expo",
                "short_description": "desc",
                "date": "2025-07-01..2025-07-17",
                "time": "18:00",
                "location_name": "Hall",
                "event_type": "выставка",
            }
        ]

    async def fake_create(title, text, source, html_text=None, media=None, ics_url=None, db=None, **kwargs):
        return "url", "p"

    monkeypatch.setattr("main.parse_event_via_4o", fake_parse)
    monkeypatch.setattr("main.create_source_page", fake_create)

    async def fake_sync(*args, **kwargs):
        return None

    monkeypatch.setattr("main.sync_month_page", fake_sync)

    msg = types.Message.model_validate(
        {
            "message_id": 1,
            "date": 0,
            "chat": {"id": 1, "type": "private"},
            "from": {"id": 1, "is_bot": False, "first_name": "A"},
            "text": "/addevent any",
        }
    )

    await handle_add_event(msg, db, bot)

    async with db.get_session() as session:
        ev = (await session.execute(select(Event))).scalars().first()

    assert ev.date == "2025-07-01"
    assert ev.end_date == "2025-07-17"


def test_md_to_html_sanitizes():
    md = "# T\nline\n<tg-emoji emoji-id='1'>R</tg-emoji>"
    html = main.md_to_html(md)
    assert "<h1>" not in html
    assert "tg-emoji" not in html
    assert "<h3>" in html
    assert "<br" in html


@pytest.mark.asyncio
async def test_sync_month_page_error(tmp_path: Path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    async with db.get_session() as session:
        session.add(
            Event(
                title="Party",
                description="desc",
                source_text="t",
                date="2025-07-16",
                time="18:00",
                location_name="Club",
            )
        )
        session.add(main.MonthPage(month="2025-07", url="u", path="p"))
        await session.commit()

    class DummyTG:
        def edit_page(self, *args, **kwargs):
            raise Exception("fail")

    monkeypatch.setattr("main.get_telegraph_token", lambda: "t")
    monkeypatch.setattr(
        "main.Telegraph", lambda access_token=None, domain=None: DummyTG()
    )

    # Should not raise
    await main.sync_month_page(db, "2025-07")


@pytest.mark.asyncio
async def test_sync_month_page_split(tmp_path: Path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    async with db.get_session() as session:
        for day in range(1, 4):
            session.add(
                Event(
                    title=f"E{day}",
                    description="d",
                    source_text="s",
                    date=f"2025-07-{day:02d}",
                    time="10:00",
                    location_name="L",
                )
            )
        await session.commit()

    calls = {"created": []}

    class DummyTG:
        def __init__(self, access_token=None):
            pass

        def create_page(self, title, content=None, **_):
            calls["created"].append(json_dumps(content))
            idx = len(calls["created"])
            return {"url": f"u{idx}", "path": f"p{idx}"}

        def edit_page(self, path, title=None, content=None):
            pass

    monkeypatch.setattr("main.get_telegraph_token", lambda: "t")
    monkeypatch.setattr("main.Telegraph", lambda access_token=None, domain=None: DummyTG())
    monkeypatch.setattr("main.TELEGRAPH_PAGE_LIMIT", 10)

    await main.sync_month_page(db, "2025-07")

    async with db.get_session() as session:
        page = await session.get(MonthPage, "2025-07")
    assert page.url2 is not None
    assert len(calls["created"]) == 2


@pytest.mark.asyncio
async def test_sync_month_page_split_on_error(tmp_path: Path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    async with db.get_session() as session:
        for day in range(1, 4):
            session.add(
                Event(
                    title=f"E{day}",
                    description="d",
                    source_text="s",
                    date=f"2025-07-{day:02d}",
                    time="10:00",
                    location_name="L",
                )
            )
        session.add(MonthPage(month="2025-07", url="u1", path="p1"))
        await session.commit()

    calls = {"created": [], "edited": 0}

    class DummyTG:
        def __init__(self, access_token=None):
            pass

        def create_page(self, title, content=None, **_):
            calls["created"].append(json_dumps(content))
            idx = len(calls["created"]) + 1
            return {"url": f"u{idx}", "path": f"p{idx}"}

        def edit_page(self, path, title=None, content=None):
            calls["edited"] += 1
            if path == "p1" and calls["edited"] == 1:
                raise TelegraphException("CONTENT_TOO_BIG")

    monkeypatch.setattr("main.get_telegraph_token", lambda: "t")
    monkeypatch.setattr("main.Telegraph", lambda access_token=None, domain=None: DummyTG())

    await main.sync_month_page(db, "2025-07")

    async with db.get_session() as session:
        page = await session.get(MonthPage, "2025-07")
    assert page.url == "u1"
    assert page.url2 is not None
    assert len(calls["created"]) == 1


@pytest.mark.asyncio

async def test_current_month_omits_past_events(tmp_path: Path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    async with db.get_session() as session:
        session.add(
            Event(
                title="Past",
                description="d",
                source_text="s",
                date="2025-07-10",
                time="10:00",
                location_name="Hall",
            )
        )
        session.add(
            Event(
                title="Future",
                description="d",
                source_text="s",
                date="2025-07-20",
                time="10:00",
                location_name="Hall",
            )
        )
        await session.commit()

    class FakeDate(date):
        @classmethod
        def today(cls):
            return date(2025, 7, 15)


    class FakeDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2025, 7, 15, 12, 0, tzinfo=tz)

    monkeypatch.setattr(main, "date", FakeDate)
    monkeypatch.setattr(main, "datetime", FakeDatetime)


    _, content = await main.build_month_page_content(db, "2025-07")
    titles = [
        c
        for n in content
        if n.get("tag") == "h4"
        for c in n.get("children", [])
        if isinstance(c, str)
    ]
    assert any("Future" in t for t in titles)
    assert not any("Past" in t for t in titles)


@pytest.mark.asyncio

async def test_month_page_split_filters_past_events(tmp_path: Path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    async with db.get_session() as session:
        for day in range(5, 8):
            session.add(
                Event(
                    title=f"P{day}",
                    description="d",
                    source_text="s",
                    date=f"2025-07-{day:02d}",
                    time="10:00",
                    location_name="L",
                )
            )
        for day in range(19, 23):
            session.add(
                Event(
                    title=f"F{day}",
                    description="d",
                    source_text="s",
                    date=f"2025-07-{day:02d}",
                    time="10:00",
                    location_name="L",
                )
            )
        await session.commit()

    class FakeDate(date):
        @classmethod
        def today(cls):
            return date(2025, 7, 19)


    class FakeDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2025, 7, 19, 12, 0, tzinfo=tz)


    created: list[list] = []

    class DummyTG:
        def __init__(self, access_token=None):
            pass

        def create_page(self, title, content=None, **_):
            created.append(content)
            idx = len(created)
            return {"url": f"u{idx}", "path": f"p{idx}"}

        def edit_page(self, path, title=None, content=None):
            created.append(content)

    monkeypatch.setattr(main, "date", FakeDate)

    monkeypatch.setattr(main, "datetime", FakeDatetime)

    monkeypatch.setattr(main, "get_telegraph_token", lambda: "t")
    monkeypatch.setattr(
        "main.Telegraph", lambda access_token=None, domain=None: DummyTG()
    )
    monkeypatch.setattr(main, "TELEGRAPH_PAGE_LIMIT", 10)

    await main.sync_month_page(db, "2025-07")

    assert len(created) == 2
    titles = [
        c
        for n in created[0]
        if n.get("tag") == "h4"
        for c in n.get("children", [])
        if isinstance(c, str)
    ]
    assert not any(t.startswith("P") for t in titles)


@pytest.mark.asyncio

async def test_update_source_page_uses_content(monkeypatch):
    events = {}

    class DummyTG:
        def get_page(self, path, return_html=True):
            return {"content": "<p>old</p>"}

        def edit_page(self, path, title, html_content):
            events["html"] = html_content

    monkeypatch.setattr("main.get_telegraph_token", lambda: "t")
    monkeypatch.setattr(
        "main.Telegraph", lambda access_token=None, domain=None: DummyTG()
    )

    await main.update_source_page("path", "Title", "new")
    html = events.get("html", "")
    assert "<p>old</p>" in html
    assert "new" in html
    assert main.CONTENT_SEPARATOR in html


@pytest.mark.asyncio
async def test_update_source_page_footer(monkeypatch):
    edited = {}

    class DummyTG:
        def get_page(self, path, return_html=True):
            return {"content": "<p>old</p>"}

        def edit_page(self, path, title, html_content):
            edited["html"] = html_content

    monkeypatch.setattr("main.get_telegraph_token", lambda: "t")
    monkeypatch.setattr(
        "main.Telegraph", lambda access_token=None, domain=None: DummyTG()
    )

    await main.update_source_page("p", "T", "text")
    html = edited.get("html", "")
    assert "Полюбить Калининград Анонсы" in html
    assert "&nbsp;" in html


@pytest.mark.asyncio
async def test_update_source_page_normalizes_hashtags(monkeypatch):
    class DummyTG:
        def get_page(self, path, return_html=True):
            return {"content": ""}

        def edit_page(self, path, title, html_content):
            assert "#1_августа" not in html_content
            assert "1 августа" in html_content

    monkeypatch.setattr("main.get_telegraph_token", lambda: "t")
    monkeypatch.setattr(
        "main.Telegraph", lambda access_token=None, domain=None: DummyTG()
    )

    await main.update_source_page("p", "T", "#1_августа event")


def test_apply_ics_link_insert_and_remove():
    html = "<p><strong>T</strong></p><p></p><p>body</p>"
    added = main.apply_ics_link(html, "http://x")
    assert "Добавить в календарь" in added
    removed = main.apply_ics_link(added, None)
    assert "Добавить в календарь" not in removed


@pytest.mark.asyncio
async def test_update_source_page_ics(monkeypatch):
    edited = {}

    class DummyTG:
        def get_page(self, path, return_html=True):
            return {"content": "<p>T</p><p></p><p>body</p>"}

        def edit_page(self, path, title, html_content):
            edited["html"] = html_content

    monkeypatch.setattr("main.get_telegraph_token", lambda: "t")
    monkeypatch.setattr(
        "main.Telegraph", lambda access_token=None, domain=None: DummyTG()
    )

    await main.update_source_page_ics("p", "T", "http://x")
    assert "Добавить в календарь" in edited.get("html", "")
    await main.update_source_page_ics("p", "T", None)
    assert "Добавить в календарь" not in edited.get("html", "")


@pytest.mark.asyncio
async def test_create_source_page_adds_nav(tmp_path: Path, monkeypatch):
    captured = {}

    class DummyTG:
        def create_page(self, title, html_content=None, **_):
            captured["html"] = html_content
            return {"url": "https://telegra.ph/test", "path": "p"}

    monkeypatch.setenv("TELEGRAPH_TOKEN", "t")
    monkeypatch.setattr(
        "main.Telegraph", lambda access_token=None, domain=None: DummyTG()
    )

    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    async with db.get_session() as session:
        session.add(MonthPage(month="2025-07", url="u1", path="p1"))
        session.add(MonthPage(month="2025-08", url="u2", path="p2"))
        await session.commit()

    res = await main.create_source_page("T", "text", None, db=db)
    assert "u1" in captured.get("html", "")
    assert res[0] == "https://telegra.ph/test"


@pytest.mark.asyncio
async def test_create_source_page_footer(monkeypatch):
    captured = {}

    class DummyTG:
        def create_page(self, title, html_content=None, **_):
            captured["html"] = html_content
            return {"url": "https://telegra.ph/test", "path": "p"}

    monkeypatch.setenv("TELEGRAPH_TOKEN", "t")
    monkeypatch.setattr(
        "main.Telegraph", lambda access_token=None, domain=None: DummyTG()
    )

    await main.create_source_page("T", "text", None)
    html = captured.get("html", "")
    assert "Полюбить Калининград Анонсы" in html
    assert "&nbsp;" in html


@pytest.mark.asyncio
async def test_update_event_description_from_telegraph(tmp_path: Path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    class DummyTG:
        def get_page(self, path, return_html=True):
            return {"content": f"<p>first</p><p>{main.CONTENT_SEPARATOR}</p><p>second</p>"}

    monkeypatch.setattr("main.get_telegraph_token", lambda: "t")
    monkeypatch.setattr("main.Telegraph", lambda access_token=None, domain=None: DummyTG())

    event = Event(
        title="T",
        description="old",
        source_text="s",
        date=FUTURE_DATE,
        time="18:00",
        location_name="Hall",
        telegraph_path="p",
    )
    async with db.get_session() as session:
        session.add(event)
        await session.commit()

    async def fake_parse(text: str, source_channel: str | None = None) -> list[dict]:
        assert "first" in text and "second" in text
        return [
            {
                "title": "T",
                "short_description": "combined",
                "date": FUTURE_DATE,
                "time": "18:00",
                "location_name": "Hall",
            }
        ]

    monkeypatch.setattr("main.parse_event_via_4o", fake_parse)

    await main.update_event_description(event, db)

    async with db.get_session() as session:
        updated = await session.get(Event, event.id)

    assert updated.description == "combined"


@pytest.mark.asyncio
async def test_nav_limits_past(tmp_path: Path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    class FakeDate(date):
        @classmethod
        def today(cls):
            return date(2025, 7, 15)

    class FakeDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2025, 7, 15, 12, 0, tzinfo=tz)

    monkeypatch.setattr(main, "date", FakeDate)
    monkeypatch.setattr(main, "datetime", FakeDatetime)

    today = FakeDate.today()
    async with db.get_session() as session:
        session.add(
            Event(
                title="T",
                description="d",
                source_text="t",
                date=today.isoformat(),
                time="10:00",
                location_name="Hall",
            )
        )
        await session.commit()

    text, markup = await main.build_events_message(db, today, timezone.utc)
    row = markup.inline_keyboard[-1]
    assert len(row) == 1
    assert row[0].text == "\u25b6"


@pytest.mark.asyncio
async def test_nav_future_has_prev(tmp_path: Path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    class FakeDate(date):
        @classmethod
        def today(cls):
            return date(2025, 7, 15)

    class FakeDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2025, 7, 15, 12, 0, tzinfo=tz)

    monkeypatch.setattr(main, "date", FakeDate)
    monkeypatch.setattr(main, "datetime", FakeDatetime)

    today = FakeDate.today()
    future = today + timedelta(days=1)
    async with db.get_session() as session:
        session.add(
            Event(
                title="T",
                description="d",
                source_text="t",
                date=future.isoformat(),
                time="10:00",
                location_name="Hall",
            )
        )
        await session.commit()

    text, markup = await main.build_events_message(db, future, timezone.utc)
    row = markup.inline_keyboard[-1]
    assert len(row) == 2
    assert row[0].text == "\u25c0"
    assert row[1].text == "\u25b6"


@pytest.mark.asyncio
async def test_delete_event_updates_month(tmp_path: Path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    bot = DummyBot("123:abc")

    async def fake_create(title, text, source, html_text=None, media=None, ics_url=None, db=None, **kwargs):
        return "url", "p"

    called = {}

    async def fake_sync(db_obj, month):
        called["month"] = month

    monkeypatch.setattr("main.create_source_page", fake_create)
    monkeypatch.setattr("main.sync_month_page", fake_sync)

    add_msg = types.Message.model_validate(
        {
            "message_id": 1,
            "date": 0,
            "chat": {"id": 1, "type": "private"},
            "from": {"id": 1, "is_bot": False, "first_name": "A"},
            "text": "/addevent_raw Party|2025-07-16|18:00|Club",
        }
    )

    await handle_add_event_raw(add_msg, db, bot)

    async with db.get_session() as session:
        event = (await session.execute(select(Event))).scalars().first()

    cb = types.CallbackQuery.model_validate(
        {
            "id": "c1",
            "from": {"id": 1, "is_bot": False, "first_name": "A"},
            "chat_instance": "1",
            "data": f"del:{event.id}:{event.date}",
            "message": {
                "message_id": 2,
                "date": 0,
                "chat": {"id": 1, "type": "private"},
            },
        }
    ).as_(bot)
    object.__setattr__(cb.message, "_bot", bot)

    async def dummy_edit(*args, **kwargs):
        return None

    object.__setattr__(cb.message, "edit_text", dummy_edit)

    async def dummy_answer(*args, **kwargs):
        return None

    object.__setattr__(cb, "answer", dummy_answer)

    await process_request(cb, db, bot)

    assert called.get("month") == "2025-07"


@pytest.mark.asyncio
async def test_delete_event_cleans_vk_posts(tmp_path: Path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    bot = DummyBot("123:abc")

    called = {}

    async def fake_sync_month_page(db_obj, month):
        called["month"] = month

    async def fake_sync_weekend_page(db_obj, start, update_links=False):
        called["weekend_page"] = start

    async def fake_sync_vk_weekend_post(db_obj, start, bot=None):
        called["weekend_post"] = start

    async def fake_delete_vk_post(url, db_obj=None, bot_obj=None):
        called["deleted"] = url

    monkeypatch.setattr(main, "sync_month_page", fake_sync_month_page)
    monkeypatch.setattr(main, "sync_weekend_page", fake_sync_weekend_page)
    monkeypatch.setattr(main, "sync_vk_weekend_post", fake_sync_vk_weekend_post)
    monkeypatch.setattr(main, "delete_vk_post", fake_delete_vk_post)

    saturday = date(2025, 7, 19)

    async with db.get_session() as session:
        ev = Event(
            title="Party",
            description="d",
            source_text="s",
            date=saturday.isoformat(),
            time="10:00",
            location_name="Club",
            source_vk_post_url="https://vk.com/wall-1_1",
        )
        session.add(ev)
        await session.commit()
        eid = ev.id

    cb = types.CallbackQuery.model_validate(
        {
            "id": "c1",
            "from": {"id": 1, "is_bot": False, "first_name": "A"},
            "chat_instance": "1",
            "data": f"del:{eid}:{saturday.isoformat()}",
            "message": {
                "message_id": 2,
                "date": 0,
                "chat": {"id": 1, "type": "private"},
            },
        }
    ).as_(bot)
    object.__setattr__(cb.message, "_bot", bot)

    async def dummy_edit(*args, **kwargs):
        return None

    object.__setattr__(cb.message, "edit_text", dummy_edit)

    async def dummy_answer(*args, **kwargs):
        return None

    object.__setattr__(cb, "answer", dummy_answer)

    await process_request(cb, db, bot)

    assert called.get("deleted") == "https://vk.com/wall-1_1"
    assert called.get("weekend_post") == saturday.isoformat()


@pytest.mark.asyncio
async def test_title_duplicate_update(tmp_path: Path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    bot = DummyBot("123:abc")

    async def fake_create(title, text, source, html_text=None, media=None, ics_url=None, db=None, **kwargs):
        return "url", "p"

    monkeypatch.setattr("main.create_source_page", fake_create)

    msg1 = types.Message.model_validate(
        {
            "message_id": 1,
            "date": 0,
            "chat": {"id": 1, "type": "private"},
            "from": {"id": 1, "is_bot": False, "first_name": "M"},
            "text": "/addevent_raw Movie|2025-07-16|20:00|Hall",
        }
    )
    await handle_add_event_raw(msg1, db, bot)

    msg2 = types.Message.model_validate(
        {
            "message_id": 2,
            "date": 0,
            "chat": {"id": 1, "type": "private"},
            "from": {"id": 1, "is_bot": False, "first_name": "M"},
            "text": "/addevent_raw Movie|2025-07-16|20:00|Another",
        }
    )
    await handle_add_event_raw(msg2, db, bot)

    async with db.get_session() as session:
        events = (await session.execute(select(Event))).scalars().all()

    assert len(events) == 1
    assert events[0].location_name == "Another"


@pytest.mark.asyncio
async def test_llm_duplicate_check(tmp_path: Path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    bot = DummyBot("123:abc")

    async def fake_create(title, text, source, html_text=None, media=None, ics_url=None, db=None, **kwargs):
        return "url", "p"

    called = {"cnt": 0}

    async def fake_check(ev, new):
        called["cnt"] += 1
        return True, "", ""

    monkeypatch.setattr("main.create_source_page", fake_create)
    monkeypatch.setattr("main.check_duplicate_via_4o", fake_check)

    msg1 = types.Message.model_validate(
        {
            "message_id": 1,
            "date": 0,
            "chat": {"id": 1, "type": "private"},
            "from": {"id": 1, "is_bot": False, "first_name": "M"},
            "text": "/addevent_raw Movie|2025-07-16|20:00|Hall",
        }
    )
    await handle_add_event_raw(msg1, db, bot)

    msg2 = types.Message.model_validate(
        {
            "message_id": 2,
            "date": 0,
            "chat": {"id": 1, "type": "private"},
            "from": {"id": 1, "is_bot": False, "first_name": "M"},
            "text": "/addevent_raw Premiere Movie|2025-07-16|20:00|Other",
        }
    )
    await handle_add_event_raw(msg2, db, bot)

    async with db.get_session() as session:
        events = (await session.execute(select(Event))).scalars().all()

    assert len(events) == 1
    assert called["cnt"] == 1


@pytest.mark.asyncio
async def test_extract_ticket_link(tmp_path: Path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    bot = DummyBot("123:abc")

    async def fake_parse(text: str, source_channel: str | None = None) -> list[dict]:
        return [
            {
                "title": "T",
                "short_description": "d",
                "date": FUTURE_DATE,
                "time": "18:00",
                "location_name": "Hall",
                "ticket_link": None,
                "event_type": "встреча",
                "emoji": None,
                "is_free": True,
            }
        ]

    async def fake_create(title, text, source, html_text=None, media=None, ics_url=None, db=None, **kwargs):
        return "url", "p"

    monkeypatch.setattr("main.parse_event_via_4o", fake_parse)
    monkeypatch.setattr("main.create_source_page", fake_create)

    html = "Регистрация <a href='https://reg'>по ссылке</a>"
    results = await main.add_events_from_text(db, "text", None, html, None)
    ev = results[0][0]
    assert ev.ticket_link == "https://reg"


@pytest.mark.asyncio
async def test_extract_ticket_link_near_word(tmp_path: Path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    bot = DummyBot("123:abc")

    async def fake_parse(text: str, source_channel: str | None = None) -> list[dict]:
        return [
            {
                "title": "T",
                "short_description": "d",
                "date": FUTURE_DATE,
                "time": "18:00",
                "location_name": "Hall",
                "ticket_link": None,
                "event_type": "встреча",
                "emoji": None,
                "is_free": True,
            }
        ]

    async def fake_create(title, text, source, html_text=None, media=None, ics_url=None, db=None, **kwargs):
        return "url", "p"

    monkeypatch.setattr("main.parse_event_via_4o", fake_parse)
    monkeypatch.setattr("main.create_source_page", fake_create)

    html = "Чтобы поучаствовать, нужна регистрация. <a href='https://reg2'>Жми</a>"
    results = await main.add_events_from_text(db, "text", None, html, None)
    ev = results[0][0]
    assert ev.ticket_link == "https://reg2"


@pytest.mark.asyncio
async def test_ticket_link_overrides_invalid(tmp_path: Path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    bot = DummyBot("123:abc")

    async def fake_parse(text: str, source_channel: str | None = None) -> list[dict]:
        return [
            {
                "title": "T",
                "short_description": "d",
                "date": FUTURE_DATE,
                "time": "18:00",
                "location_name": "Hall",
                "ticket_link": "Регистрация по ссылке",
                "event_type": "встреча",
                "emoji": None,
                "is_free": True,
            }
        ]

    async def fake_create(title, text, source, html_text=None, media=None, ics_url=None, db=None, **kwargs):
        return "url", "p"

    monkeypatch.setattr("main.parse_event_via_4o", fake_parse)
    monkeypatch.setattr("main.create_source_page", fake_create)

    html = "Регистрация <a href='https://real'>по ссылке</a>"
    results = await main.add_events_from_text(db, "text", None, html, None)
    ev = results[0][0]
    assert ev.ticket_link == "https://real"


@pytest.mark.asyncio
async def test_multiple_ticket_links(tmp_path: Path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    bot = DummyBot("123:abc")

    async def fake_parse(text: str, source_channel: str | None = None) -> list[dict]:
        return [
            {
                "title": "A",
                "short_description": "d1",
                "date": FUTURE_DATE,
                "time": "18:00",
                "location_name": "Hall",
                "ticket_link": None,
                "event_type": "концерт",
                "emoji": None,
                "is_free": True,
            },
            {
                "title": "B",
                "short_description": "d2",
                "date": FUTURE_DATE,
                "time": "19:00",
                "location_name": "Hall",
                "ticket_link": None,
                "event_type": "концерт",
                "emoji": None,
                "is_free": True,
            },
        ]

    async def fake_create(title, text, source, html_text=None, media=None, ics_url=None, db=None, **kwargs):
        return "url", "p"

    monkeypatch.setattr("main.parse_event_via_4o", fake_parse)
    monkeypatch.setattr("main.create_source_page", fake_create)

    html = (
        "Билеты <a href='https://l1'>купить</a>" 
        " и ещё один концерт. Билеты <a href='https://l2'>здесь</a>"
    )

    results = await main.add_events_from_text(db, "text", None, html, None)
    assert results[0][0].ticket_link == "https://l1"
    assert results[1][0].ticket_link == "https://l2"


@pytest.mark.asyncio
async def test_add_event_lines_include_vk_link(tmp_path: Path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    async def fake_parse(text: str) -> list[dict]:
        return [
            {
                "title": "T",
                "short_description": "d",
                "date": FUTURE_DATE,
                "time": "18:00",
                "location_name": "Hall",
            }
        ]

    async def fake_create(title, text, source, html_text=None, media=None, ics_url=None, db=None, **kwargs):
        return "https://t.me/page", "p"

    monkeypatch.setattr("main.parse_event_via_4o", fake_parse)
    monkeypatch.setattr("main.create_source_page", fake_create)

    results = await main.add_events_from_text(
        db, "text", "https://vk.com/wall-1_1", None, None
    )
    assert results
    lines = results[0][2]
    assert "telegraph: https://t.me/page" in lines
    idx = lines.index("telegraph: https://t.me/page")
    assert lines[idx + 1] == "vk_source: https://vk.com/source"
    assert lines[idx + 2] == "Vk: https://vk.com/wall-1_1"


@pytest.mark.asyncio
async def test_update_event_description_error_does_not_stop_sync(tmp_path: Path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    async def fake_parse(text: str, source_channel: str | None = None) -> list[dict]:
        return [
            {
                "title": "T",
                "short_description": "d",
                "date": "2025-08-09",
                "time": "14:00",
                "location_name": "Hall",
            }
        ]

    async def fake_create(*args, db=None, **kwargs):
        return "u", "p"

    called: dict[str, str] = {}

    async def fake_month(db_obj, month, update_links=True):
        called["month"] = month

    async def fake_weekend(db_obj, start, update_links=True):
        called["weekend"] = start

    async def boom(event, db_obj):
        raise RuntimeError("boom")

    monkeypatch.setattr("main.parse_event_via_4o", fake_parse)
    monkeypatch.setattr("main.create_source_page", fake_create)
    monkeypatch.setattr("main.sync_month_page", fake_month)
    monkeypatch.setattr("main.sync_weekend_page", fake_weekend)
    monkeypatch.setattr("main.update_event_description", boom)

    results = await main.add_events_from_text(db, "t", None, None, None)
    assert called.get("month") == "2025-08"
    assert called.get("weekend") == "2025-08-09"
    assert results and results[0][0].title == "T"


@pytest.mark.asyncio
async def test_add_event_strips_city_from_address(tmp_path: Path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    async def fake_parse(text: str, source_channel: str | None = None) -> list[dict]:
        return [
            {
                "title": "Show",
                "short_description": "d",
                "date": FUTURE_DATE,
                "time": "18:00",
                "location_name": "Hall",
                "location_address": "Addr, Калининград",
                "city": "Калининград",
            }
        ]

    async def fake_create(*args, db=None, **kwargs):
        return "u", "p"

    monkeypatch.setattr("main.parse_event_via_4o", fake_parse)
    monkeypatch.setattr("main.create_source_page", fake_create)

    results = await main.add_events_from_text(db, "t", None, None, None)
    ev = results[0][0]
    assert ev.location_address == "Addr"
    md = main.format_event_md(ev)
    assert md.count("Калининград") == 1


@pytest.mark.asyncio
async def test_festival_expands_dates(tmp_path: Path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    async def fake_parse(text: str, source_channel: str | None = None) -> list[dict]:
        return [
            {
                "title": "Jazz",
                "short_description": "desc",
                "date": "2025-08-01..2025-08-03",
                "time": "18:00",
                "location_name": "Park",
                "event_type": "концерт",
            }
        ]

    monkeypatch.setattr("main.parse_event_via_4o", fake_parse)

    async def fake_create(*args, db=None, **kwargs):
        return "u", "p"

    monkeypatch.setattr("main.create_source_page", fake_create)

    results = await main.add_events_from_text(db, "text", None, None, None)
    assert len(results) == 3
    async with db.get_session() as session:
        dates = sorted(
            (await session.execute(select(Event))).scalars(), key=lambda e: e.date
        )
        assert [e.date for e in dates] == ["2025-08-01", "2025-08-02", "2025-08-03"]


@pytest.mark.asyncio
async def test_exhibition_future_not_listed(tmp_path: Path):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    future_start = (date.today() + timedelta(days=10)).isoformat()
    future_end = (date.today() + timedelta(days=20)).isoformat()
    async with db.get_session() as session:
        session.add(
            Event(
                title="Expo",
                description="d",
                source_text="s",
                date=future_start,
                end_date=future_end,
                time="10:00",
                location_name="Hall",
                event_type="выставка",
            )
        )
        await session.commit()

    _, content = await main.build_month_page_content(db, future_start[:7])
    found_in_exh = False
    exh_section = False
    for n in content:
        if n.get("tag") == "h3" and "Постоянные" in "".join(n.get("children", [])):
            exh_section = True
        elif exh_section and isinstance(n, dict) and n.get("tag") == "h4":
            if any("Expo" in str(c) for c in n.get("children", [])):
                found_in_exh = True
    assert not found_in_exh


@pytest.mark.asyncio
async def test_past_exhibition_not_listed_in_events(tmp_path: Path):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    past_start = (date.today() - timedelta(days=6)).isoformat()
    future_end = (date.today() + timedelta(days=6)).isoformat()
    async with db.get_session() as session:
        session.add(
            Event(
                title="PastExpo",
                description="d",
                source_text="s",
                date=past_start,
                end_date=future_end,
                time="10:00",
                location_name="Hall",
                event_type="выставка",
            )
        )
        await session.commit()

    _, content = await main.build_month_page_content(db, past_start[:7])
    before_exh = True
    found = False
    for n in content:
        if n.get("tag") == "h3" and "Постоянные" in "".join(n.get("children", [])):
            before_exh = False
        if before_exh and isinstance(n, dict) and n.get("tag") == "h4":
            if any("PastExpo" in str(c) for c in n.get("children", [])):
                found = True
    assert not found


@pytest.mark.asyncio
async def test_exhibition_auto_year_end(tmp_path: Path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    async def fake_parse(text: str, source_channel: str | None = None) -> list[dict]:
        return [
            {
                "title": "AutoExpo",
                "short_description": "d",
                "location_name": "Hall",
                "event_type": "выставка",
            }
        ]

    async def fake_create(*args, db=None, **kwargs):
        return "u", "p"

    monkeypatch.setattr("main.parse_event_via_4o", fake_parse)
    monkeypatch.setattr("main.create_source_page", fake_create)

    class FakeDate(date):
        @classmethod
        def today(cls):
            return date(2025, 7, 15)

    class FakeDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2025, 7, 15, 12, 0, tzinfo=tz)

    monkeypatch.setattr(main, "date", FakeDate)
    monkeypatch.setattr(main, "datetime", FakeDatetime)

    results = await main.add_events_from_text(db, "text", None, None, None)
    assert results
    ev = results[0][0]
    today = FakeDate.today()
    assert ev.date == today.isoformat()
    assert ev.end_date == date(today.year, 12, 31).isoformat()

    _, content = await main.build_month_page_content(db, today.strftime("%Y-%m"))
    found = False
    exh_section = False
    for n in content:
        if n.get("tag") == "h3" and "Постоянные" in "".join(n.get("children", [])):
            exh_section = True
        elif exh_section and isinstance(n, dict) and n.get("tag") == "h4":
            if any("AutoExpo" in str(c) for c in n.get("children", [])):
                found = True
    assert found


@pytest.mark.asyncio
async def test_month_links_future(tmp_path: Path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    async with db.get_session() as session:
        session.add(MonthPage(month="2025-07", url="u1", path="p1"))
        session.add(MonthPage(month="2025-08", url="u2", path="p2"))
        session.add(MonthPage(month="2025-09", url="u3", path="p3"))
        await session.commit()

    class FakeDate(date):
        @classmethod
        def today(cls):
            return date(2025, 7, 15)

    class FakeDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2025, 7, 15, 12, 0, tzinfo=tz)

    monkeypatch.setattr(main, "date", FakeDate)
    monkeypatch.setattr(main, "datetime", FakeDatetime)
    title, content = await main.build_month_page_content(db, "2025-07")
    found = False
    for n in content:
        if (
            isinstance(n, dict)
            and n.get("tag") == "h4"
            and any("август" in str(c) for c in n.get("children", []))
        ):
            found = True
    assert found


@pytest.mark.asyncio
async def test_month_buttons_future(tmp_path: Path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    async with db.get_session() as session:
        session.add(MonthPage(month="2025-07", url="u1", path="p1"))
        session.add(MonthPage(month="2025-08", url="u2", path="p2"))
        session.add(MonthPage(month="2025-09", url="u3", path="p3"))
        session.add(MonthPage(month="2025-10", url="u4", path="p4"))
        await session.commit()

    class FakeDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2025, 8, 2, tzinfo=tz)

    monkeypatch.setattr(main, "datetime", FakeDatetime)
    buttons = await main.build_month_buttons(db)
    assert [b.text for b in buttons] == [
        "\U0001f4c5 август",
        "\U0001f4c5 сентябрь",
        "\U0001f4c5 октябрь",
    ]


@pytest.mark.asyncio
async def test_build_daily_posts(tmp_path: Path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    class FakeDate(date):
        @classmethod
        def today(cls):
            return date(2025, 7, 15)

    class FakeDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2025, 7, 15, 12, 0, tzinfo=tz)

    monkeypatch.setattr(main, "date", FakeDate)
    monkeypatch.setattr(main, "datetime", FakeDatetime)

    today = FakeDate.today()
    start = main.next_weekend_start(today)
    async with db.get_session() as session:
        session.add(
            Event(
                title="T",
                description="d",
                source_text="s",
                date=today.isoformat(),
                time="18:00",
                location_name="Hall",
            )
        )
        session.add(
            Event(
                title="S",
                description="d2",
                source_text="s2",
                date=today.isoformat(),
                time="19:00",
                location_name="Hall",
                silent=True,
            )
        )
        session.add(
            Event(
                title="W",
                description="weekend",
                source_text="s3",
                date=start.isoformat(),
                time="12:00",
                location_name="Hall",
                added_at=datetime.utcnow(),
            )
        )
        session.add(MonthPage(month=today.strftime("%Y-%m"), url="m1", path="p1"))
        session.add(
            MonthPage(
                month=main.next_month(today.strftime("%Y-%m")), url="m2", path="p2"
            )
        )
        session.add(WeekendPage(start=start.isoformat(), url="w", path="wp"))
        await session.commit()

    posts = await main.build_daily_posts(db, timezone.utc)
    assert posts
    text, markup = posts[0]
    assert "АНОНС" in text
    assert markup.inline_keyboard[0]
    assert text.count("\U0001f449") == 2
    first_btn = markup.inline_keyboard[0][0].text
    assert first_btn.startswith("(+1)")


@pytest.mark.asyncio
async def test_build_daily_posts_tomorrow(tmp_path: Path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    class FakeDate(date):
        @classmethod
        def today(cls):
            return date(2025, 7, 15)

    class FakeDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2025, 7, 15, 12, 0, tzinfo=tz)

    monkeypatch.setattr(main, "date", FakeDate)
    monkeypatch.setattr(main, "datetime", FakeDatetime)

    today = FakeDate.today()
    tomorrow = today + timedelta(days=1)
    async with db.get_session() as session:
        session.add(
            Event(
                title="T",
                description="d",
                source_text="s",
                date=tomorrow.isoformat(),
                time="18:00",
                location_name="Hall",
            )
        )
        await session.commit()

    now = FakeDatetime.now(timezone.utc) + timedelta(days=1)
    posts = await main.build_daily_posts(db, timezone.utc, now)
    assert posts and tomorrow.strftime("%d") in posts[0][0]



@pytest.mark.asyncio
async def test_daily_weekend_date_link(tmp_path: Path):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    today = date.today()
    saturday = main.next_weekend_start(today)
    async with db.get_session() as session:
        session.add(
            Event(
                title="W",
                description="weekend",
                source_text="s",
                date=saturday.isoformat(),
                time="12:00",
                location_name="Hall",
                added_at=datetime.utcnow(),
            )
        )
        session.add(WeekendPage(start=saturday.isoformat(), url="w", path="wp"))
        await session.commit()

    posts = await main.build_daily_posts(db, timezone.utc)
    text = posts[0][0]
    assert f'<a href="w">' in text



@pytest.mark.asyncio
async def test_send_daily_preview_disabled(tmp_path: Path):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    bot = DummyBot("123:abc")

    async with db.get_session() as session:
        session.add(
            main.Channel(channel_id=1, title="ch", is_admin=True, daily_time="08:00")
        )
        session.add(
            Event(
                title="T",
                description="d",
                source_text="s",
                date=date.today().isoformat(),
                time="18:00",
                location_name="Hall",
            )
        )
        await session.commit()

    await main.send_daily_announcement(db, bot, 1, timezone.utc)
    assert bot.messages
    assert bot.messages[-1][2].get("disable_web_page_preview") is True
    async with db.get_session() as session:
        ch = await session.get(main.Channel, 1)
    assert ch.last_daily == date.today().isoformat()


@pytest.mark.asyncio
async def test_daily_test_send_no_record(tmp_path: Path):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    bot = DummyBot("123:abc")

    async with db.get_session() as session:
        session.add(
            main.Channel(channel_id=1, title="ch", is_admin=True, daily_time="08:00")
        )
        await session.commit()

    await main.send_daily_announcement(db, bot, 1, timezone.utc, record=False)
    async with db.get_session() as session:
        ch = await session.get(main.Channel, 1)
    assert ch.last_daily is None


@pytest.mark.asyncio
async def test_build_daily_posts_split(tmp_path: Path):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    today = date.today()
    long_desc = "d" * 200
    async with db.get_session() as session:
        for i in range(50):
            session.add(
                Event(
                    title=f"T{i}",
                    description=long_desc,
                    source_text="s",
                    date=today.isoformat(),
                    time="18:00",
                    location_name="Hall",
                )
            )
        await session.commit()

    posts = await main.build_daily_posts(db, timezone.utc)
    assert len(posts) > 1
    for text, _ in posts:
        assert len(text) <= 4096


@pytest.mark.asyncio
async def test_daily_no_more_link(tmp_path: Path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    class FakeDate(date):
        @classmethod
        def today(cls):
            return date(2025, 7, 15)

    class FakeDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2025, 7, 15, 12, 0, tzinfo=tz)

    monkeypatch.setattr(main, "date", FakeDate)
    monkeypatch.setattr(main, "datetime", FakeDatetime)

    async with db.get_session() as session:
        session.add(
            Event(
                title="T",
                description="d, подробнее (https://t.me/test)",
                source_text="s",
                date=FakeDate.today().isoformat(),
                time="18:00",
                location_name="Hall",
            )
        )
        await session.commit()

    posts = await main.build_daily_posts(db, timezone.utc)
    text = posts[0][0]
    assert "подробнее" not in text


def test_format_event_vk_with_vk_link():
    e = Event(
        title="T",
        description="d",
        source_text="s",
        date="2025-07-10",
        time="18:00",
        location_name="Hall",
        source_post_url="https://vk.com/wall-1_1",
        telegraph_url="https://t.me/page",
    )
    text = main.format_event_vk(e)
    assert "[подробнее|https://vk.com/wall-1_1]" in text
    assert "t.me/page" not in text


def test_format_event_vk_fallback_link():
    e = Event(
        title="T",
        description="d",
        source_text="s",
        date="2025-07-10",
        time="18:00",
        location_name="Hall",
        source_post_url="https://vk.cc/abc",
        telegraph_url="https://t.me/page",
    )
    text = main.format_event_vk(e)
    assert "[подробнее|https://t.me/page]" in text


def test_format_event_vk_festival_link():
    e = Event(
        title="T",
        description="d",
        source_text="s",
        date="2025-07-10",
        time="18:00",
        location_name="Hall",
        festival="Jazz",
    )
    fest = main.Festival(name="Jazz", vk_post_url="https://vk.com/wall-1_1")
    text = main.format_event_vk(e, festival=fest)
    lines = text.splitlines()
    assert lines[1] == "✨ [https://vk.com/wall-1_1|Jazz]"


@pytest.mark.asyncio
async def test_daily_posts_festival_link(tmp_path: Path):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    today = date.today()
    async with db.get_session() as session:
        session.add(
            main.Festival(name="Jazz", telegraph_url="http://tg", vk_post_url="http://vk")
        )
        session.add(
            Event(
                title="T",
                description="d",
                source_text="s",
                date=today.isoformat(),
                time="18:00",
                location_name="Hall",
                festival="Jazz",
            )
        )
        await session.commit()

    posts = await main.build_daily_posts(db, timezone.utc)
    assert "http://tg" in posts[0][0]
    sec1, _ = await main.build_daily_sections_vk(db, timezone.utc)
    assert sec1


@pytest.mark.asyncio
async def test_handle_fest_list(tmp_path: Path):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    bot = DummyBot("123:abc")

    async with db.get_session() as session:
        session.add(User(user_id=1))
        session.add(main.Festival(name="Jazz"))
        await session.commit()

    msg = types.Message.model_validate(
        {
            "message_id": 1,
            "date": 0,
            "chat": {"id": 1, "type": "private"},
            "from": {"id": 1, "is_bot": False, "first_name": "A"},
            "text": "/fest",
        }
    )
    await main.handle_fest(msg, db, bot)
    assert "Jazz" in bot.messages[-1][1]


def test_event_to_nodes_festival_link():
    e = Event(
        title="T",
        description="d",
        source_text="s",
        date="2025-07-10",
        time="18:00",
        location_name="Hall",
        festival="Jazz",
    )
    fest = main.Festival(name="Jazz", telegraph_url="http://tg")
    nodes = main.event_to_nodes(e, fest)
    assert nodes[1]["children"][0]["attrs"]["href"] == "http://tg"
    assert sum(
        1
        for n in nodes
        if isinstance(n, dict)
        and any(
            isinstance(c, dict)
            and c.get("attrs", {}).get("href") == "http://tg"
            for c in n.get("children", [])
        )
    ) == 1


def test_event_to_nodes_festival_icon():
    e = Event(
        title="T",
        description="d",
        source_text="s",
        date="2025-07-10",
        time="18:00",
        location_name="Hall",
        festival="Jazz",
    )
    fest = main.Festival(name="Jazz", telegraph_url="http://tg")
    nodes = main.event_to_nodes(e, fest, fest_icon=True)
    assert nodes[1]["children"][0] == "✨ "



@pytest.mark.asyncio
async def test_daily_posts_festival_link(tmp_path: Path):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    today = date.today()
    async with db.get_session() as session:
        session.add(
            main.Festival(name="Jazz", telegraph_url="http://tg", vk_post_url="http://vk")
        )
        session.add(
            Event(
                title="T",
                description="d",
                source_text="s",
                date=today.isoformat(),
                time="18:00",
                location_name="Hall",
                festival="Jazz",
            )
        )
        await session.commit()

    posts = await main.build_daily_posts(db, timezone.utc)
    assert "http://tg" in posts[0][0]
    sec1, _ = await main.build_daily_sections_vk(db, timezone.utc)
    assert sec1


@pytest.mark.asyncio
async def test_festival_auto_page_creation(tmp_path: Path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    async def fake_parse(*args, **kwargs):
        return [
            {
                "title": "Jazz Day",
                "short_description": "desc",
                "date": FUTURE_DATE,
                "time": "18:00",
                "location_name": "Hall",
                "festival": "Jazz",
            }
        ]

    class DummyTG:
        def __init__(self, access_token=None):
            pass

        def create_page(self, title, content=None):
            return {"url": "http://tg", "path": "p"}

        def edit_page(self, path, title=None, content=None):
            pass

    monkeypatch.setenv("TELEGRAPH_TOKEN", "t")
    monkeypatch.setattr("main.Telegraph", lambda access_token=None: DummyTG())
    monkeypatch.setattr("main.parse_event_via_4o", fake_parse)

    async def fake_ask(text):
        return "Desc"

    monkeypatch.setattr("main.ask_4o", fake_ask)
    async def fake_create(*args, **kwargs):
        return "u", "p"

    monkeypatch.setattr("main.create_source_page", fake_create)

    await main.add_events_from_text(db, "t", None, None, None)

    async with db.get_session() as session:
        fest = (await session.execute(select(main.Festival))).scalars().first()
    assert fest and fest.telegraph_url == "http://tg"
    assert fest.description == "Desc"


@pytest.mark.asyncio
async def test_handle_fest_list(tmp_path: Path):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    bot = DummyBot("123:abc")

    async with db.get_session() as session:
        session.add(User(user_id=1))
        session.add(main.Festival(name="Jazz"))
        await session.commit()

    msg = types.Message.model_validate(
        {
            "message_id": 1,
            "date": 0,
            "chat": {"id": 1, "type": "private"},
            "from": {"id": 1, "is_bot": False, "first_name": "A"},
            "text": "/fest",
        }
    )
    await main.handle_fest(msg, db, bot)
    assert "Jazz" in bot.messages[-1][1]


@pytest.mark.asyncio
async def test_month_page_festival_link(tmp_path: Path):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    m = FUTURE_DATE[:7]
    async with db.get_session() as session:
        session.add(main.Festival(name="Jazz", telegraph_url="http://tg"))
        session.add(
            Event(
                title="T",
                description="d",
                source_text="s",
                date=FUTURE_DATE,
                time="18:00",
                location_name="Hall",
                festival="Jazz",
            )
        )
        await session.commit()

    title, content = await main.build_month_page_content(db, m)
    assert "http://tg" in json_dumps(content)


@pytest.mark.asyncio
async def test_daily_posts_festival_link(tmp_path: Path):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    today = date.today()
    async with db.get_session() as session:
        session.add(
            main.Festival(name="Jazz", telegraph_url="http://tg", vk_post_url="http://vk")
        )
        session.add(
            Event(
                title="T",
                description="d",
                source_text="s",
                date=today.isoformat(),
                time="18:00",
                location_name="Hall",
                festival="Jazz",
            )
        )
        await session.commit()

    posts = await main.build_daily_posts(db, timezone.utc)
    assert "http://tg" in posts[0][0]
    sec1, _ = await main.build_daily_sections_vk(db, timezone.utc)
    assert sec1


@pytest.mark.asyncio
async def test_festival_auto_page_creation(tmp_path: Path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    async def fake_parse(*args, **kwargs):
        return [
            {
                "title": "Jazz Day",
                "short_description": "desc",
                "date": FUTURE_DATE,
                "time": "18:00",
                "location_name": "Hall",
                "festival": "Jazz",
            }
        ]

    class DummyTG:
        def __init__(self, access_token=None):
            pass

        def create_page(self, title, content=None):
            return {"url": "http://tg", "path": "p"}

        def edit_page(self, path, title=None, content=None):
            pass

    monkeypatch.setenv("TELEGRAPH_TOKEN", "t")
    monkeypatch.setattr("main.Telegraph", lambda access_token=None: DummyTG())
    monkeypatch.setattr("main.parse_event_via_4o", fake_parse)

    async def fake_ask(text):
        return "Desc"

    monkeypatch.setattr("main.ask_4o", fake_ask)
    async def fake_create(*args, **kwargs):
        return "u", "p"

    monkeypatch.setattr("main.create_source_page", fake_create)

    await main.add_events_from_text(db, "t", None, None, None)

    async with db.get_session() as session:
        fest = (await session.execute(select(main.Festival))).scalars().first()
    assert fest and fest.telegraph_url == "http://tg"
    assert fest.description == "Desc"


@pytest.mark.asyncio
async def test_handle_fest_list(tmp_path: Path):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    bot = DummyBot("123:abc")

    async with db.get_session() as session:
        session.add(User(user_id=1))
        session.add(main.Festival(name="Jazz"))
        await session.commit()

    msg = types.Message.model_validate(
        {
            "message_id": 1,
            "date": 0,
            "chat": {"id": 1, "type": "private"},
            "from": {"id": 1, "is_bot": False, "first_name": "A"},
            "text": "/fest",
        }
    )
    await main.handle_fest(msg, db, bot)
    assert "Jazz" in bot.messages[-1][1]


@pytest.mark.asyncio
async def test_month_page_festival_link(tmp_path: Path):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    m = FUTURE_DATE[:7]
    async with db.get_session() as session:
        session.add(main.Festival(name="Jazz", telegraph_url="http://tg"))
        session.add(
            Event(
                title="T",
                description="d",
                source_text="s",
                date=FUTURE_DATE,
                time="18:00",
                location_name="Hall",
                festival="Jazz",
            )
        )
        await session.commit()

    title, content = await main.build_month_page_content(db, m)
    assert "http://tg" in json_dumps(content)


@pytest.mark.asyncio
async def test_daily_posts_festival_link(tmp_path: Path):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    today = date.today()
    async with db.get_session() as session:
        session.add(
            main.Festival(name="Jazz", telegraph_url="http://tg", vk_post_url="http://vk")
        )
        session.add(
            Event(
                title="T",
                description="d",
                source_text="s",
                date=today.isoformat(),
                time="18:00",
                location_name="Hall",
                festival="Jazz",
            )
        )
        await session.commit()

    posts = await main.build_daily_posts(db, timezone.utc)
    assert "http://tg" in posts[0][0]
    sec1, _ = await main.build_daily_sections_vk(db, timezone.utc)
    assert sec1


@pytest.mark.asyncio
async def test_festival_auto_page_creation(tmp_path: Path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    async def fake_parse(*args, **kwargs):
        return [
            {
                "title": "Jazz Day",
                "short_description": "desc",
                "date": FUTURE_DATE,
                "time": "18:00",
                "location_name": "Hall",
                "festival": "Jazz",
            }
        ]

    class DummyTG:
        def __init__(self, access_token=None):
            pass

        def create_page(self, title, content=None):
            return {"url": "http://tg", "path": "p"}

        def edit_page(self, path, title=None, content=None):
            pass

    monkeypatch.setenv("TELEGRAPH_TOKEN", "t")
    monkeypatch.setattr("main.Telegraph", lambda access_token=None: DummyTG())
    monkeypatch.setattr("main.parse_event_via_4o", fake_parse)

    async def fake_ask(text):
        return "Desc"

    monkeypatch.setattr("main.ask_4o", fake_ask)
    async def fake_create(*args, **kwargs):
        return "u", "p"

    monkeypatch.setattr("main.create_source_page", fake_create)

    await main.add_events_from_text(db, "t", None, None, None)

    async with db.get_session() as session:
        fest = (await session.execute(select(main.Festival))).scalars().first()
    assert fest and fest.telegraph_url == "http://tg"
    assert fest.description == "Desc"


@pytest.mark.asyncio
async def test_handle_fest_list(tmp_path: Path):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    bot = DummyBot("123:abc")

    async with db.get_session() as session:
        session.add(User(user_id=1))
        session.add(main.Festival(name="Jazz"))
        await session.commit()

    msg = types.Message.model_validate(
        {
            "message_id": 1,
            "date": 0,
            "chat": {"id": 1, "type": "private"},
            "from": {"id": 1, "is_bot": False, "first_name": "A"},
            "text": "/fest",
        }
    )
    await main.handle_fest(msg, db, bot)
    assert "Jazz" in bot.messages[-1][1]


@pytest.mark.asyncio
async def test_month_page_festival_link(tmp_path: Path):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    m = FUTURE_DATE[:7]
    async with db.get_session() as session:
        session.add(main.Festival(name="Jazz", telegraph_url="http://tg"))
        session.add(
            Event(
                title="T",
                description="d",
                source_text="s",
                date=FUTURE_DATE,
                time="18:00",
                location_name="Hall",
                festival="Jazz",
            )
        )
        await session.commit()

    title, content = await main.build_month_page_content(db, m)
    assert "http://tg" in json_dumps(content)


@pytest.mark.asyncio
async def test_upload_ics_content_type(tmp_path: Path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    event = Event(
        id=1,
        title="T",
        description="d",
        source_text="s",
        date=date.today().isoformat(),
        time="10:00",
        location_name="Hall",
    )

    class DummyBucket:
        def __init__(self):
            self.upload_args = None

        def upload(self, path, data, options):
            self.upload_args = (path, data, options)

        def get_public_url(self, path):
            return f"https://test/{path}"

        def remove(self, paths):
            pass

    class DummyStorage:
        def __init__(self):
            self.bucket = DummyBucket()

        def from_(self, bucket):
            return self.bucket

    class DummyClient:
        def __init__(self):
            self.storage = DummyStorage()

    dummy = DummyClient()
    monkeypatch.setattr(main, "get_supabase_client", lambda: dummy)

    url = await main.upload_ics(event, db)
    assert url.endswith(".ics")
    opts = dummy.storage.bucket.upload_args[2]
    assert opts["content-type"] == main.ICS_CONTENT_TYPE
    assert opts["content-disposition"].startswith("inline;")
    assert "filename=\"" in opts["content-disposition"]


@pytest.mark.asyncio
async def test_build_ics_content_headers(tmp_path: Path):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    event = Event(
        id=1,
        title="T",
        description="d",
        source_text="s",
        date=date.today().isoformat(),
        time="10:00",
        location_name="Hall",
    )

    content = await main.build_ics_content(db, event)
    assert content.endswith("\r\n")
    lines = content.split("\r\n")
    assert lines[0] == "BEGIN:VCALENDAR"
    assert lines[1] == "VERSION:2.0"
    assert lines[2].startswith("PRODID:")
    assert lines[3] == "CALSCALE:GREGORIAN"
    assert lines[4] == "METHOD:PUBLISH"
    assert lines[5].startswith("X-WR-CALNAME:")
    assert any(l.startswith("DTSTAMP:") for l in lines)
    assert lines.count("END:VCALENDAR") == 1


@pytest.mark.asyncio
async def test_build_ics_location_escape(tmp_path: Path):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    event = Event(
        id=1,
        title="T",
        description="d",
        source_text="s",
        date=date.today().isoformat(),
        time="10:00",
        location_address="Serg, 14",
        city="Kaliningrad",
    )
    content = await main.build_ics_content(db, event)
    assert "LOCATION:Serg\\,\\ 14\\,Kaliningrad" in content



def test_parse_time_range_dots():
    result = main.parse_time_range("10:30..18:00")
    assert result == (time(10, 30), time(18, 0))



@pytest.mark.asyncio
async def test_post_ics_asset_caption(tmp_path: Path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    bot = DummyBot("123:abc")

    ch = main.Channel(channel_id=-1002, title="Asset", is_admin=True, is_asset=True)
    async with db.get_session() as session:
        session.add(ch)
        await session.commit()

    event = Event(
        id=1,
        title="Concert",
        description="desc",
        source_text="s",
        date="2025-07-18",
        time="19:00",
        location_name="Сигнал",
        location_address="Леонова 22",
        city="Калининград",
    )

    async def fake_build(db2, ev):
        return "ICS"

    monkeypatch.setattr(main, "build_ics_content", fake_build)

    async def fake_send_document(self, chat_id, document, caption=None, parse_mode=None):
        self.messages.append((chat_id, caption))
        class Msg:
            message_id = 42
        return Msg()

    monkeypatch.setattr(DummyBot, "send_document", fake_send_document, raising=False)

    url, msg_id = await main.post_ics_asset(event, db, bot)
    assert msg_id == 42
    caption = bot.messages[0][1]
    day = main.format_day_pretty(date(2025, 7, 18))
    assert f"<b>Concert</b>" in caption
    assert f"<i>{day} 19:00 Сигнал, Леонова 22, #Калининград</i>" in caption


@pytest.mark.asyncio
async def test_forward_adds_calendar_button(tmp_path: Path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    bot = DummyBot("123:abc")

    start_msg = types.Message.model_validate(
        {
            "message_id": 1,
            "date": 0,
            "chat": {"id": 1, "type": "private"},
            "from": {"id": 1, "is_bot": False, "first_name": "A"},
            "text": "/start",
        }
    )
    await handle_start(start_msg, db, bot)

    ch_ann = main.Channel(channel_id=-1001, title="Ann", is_admin=True, is_registered=True)
    ch_asset = main.Channel(channel_id=-1002, title="Asset", is_admin=True, is_asset=True)
    async with db.get_session() as session:
        session.add(ch_ann)
        session.add(ch_asset)
        session.add(
            main.MonthPage(month="2025-07", url="m1", path="p1")
        )
        session.add(
            main.MonthPage(month="2025-08", url="m2", path="p2")
        )
        session.add(
            main.MonthPage(month="2025-10", url="m3", path="p3")
        )
        await session.commit()

    class FakeDate(date):
        @classmethod
        def today(cls):
            return date(2025, 7, 27)

    class FakeDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2025, 7, 27, 12, 0, tzinfo=tz)

    monkeypatch.setattr(main, "date", FakeDate)
    monkeypatch.setattr(main, "datetime", FakeDatetime)

    async def fake_build(db2, ev):
        return "ICS"

    monkeypatch.setattr(main, "build_ics_content", fake_build)
    async def fake_upload(ev, db2):
        return "https://x/ics"

    async def fake_create(*a, **k):
        return ("u", "p", "", 0)

    monkeypatch.setattr(main, "upload_ics", fake_upload)
    monkeypatch.setattr(main, "create_source_page", fake_create)
    monkeypatch.setattr(main, "update_source_page_ics", lambda *a, **k: None)

    async def fake_sync(*a, **k):
        return None

    monkeypatch.setattr(main, "sync_month_page", fake_sync)
    monkeypatch.setattr(main, "sync_weekend_page", fake_sync)

    async def fake_post(event, db2, b):
        return ("https://t.me/a/1", 55)

    monkeypatch.setattr(main, "post_ics_asset", fake_post)

    async def fake_send_document(self, chat_id, document, caption=None, parse_mode=None):
        class Msg:
            message_id = 77
        return Msg()

    monkeypatch.setattr(DummyBot, "send_document", fake_send_document, raising=False)

    async def fake_parse(text, source_channel=None):
        return [
            {
                "title": "T",
                "short_description": "d",
                "date": FUTURE_DATE,
                "time": "18:00",
                "location_name": "Club",
            }
        ]

    monkeypatch.setattr(main, "parse_event_via_4o", fake_parse)
    monkeypatch.setattr(main, "bot", bot, raising=False)

    fwd_msg = types.Message.model_validate(
        {
            "message_id": 2,
            "date": 0,
            "forward_date": 0,
            "forward_from_chat": {"id": -1001, "type": "channel", "username": "ann"},
            "forward_from_message_id": 10,
            "chat": {"id": 1, "type": "private"},
            "from": {"id": 1, "is_bot": False, "first_name": "A"},
            "text": "text",
        }
    )

    await main.handle_forwarded(fwd_msg, db, bot)

    assert bot.edits
    chat_id, msg_id, kwargs = bot.edits[0]
    assert chat_id == -1001
    assert msg_id == 10
    keyboard = kwargs["reply_markup"].inline_keyboard
    assert keyboard[0][0].text == "Добавить в календарь"
    row2 = keyboard[1]
    texts = [b.text for b in row2]
    assert texts == ["\U0001f4c5 июль", "\U0001f4c5 август", "\U0001f4c5 октябрь"]


@pytest.mark.asyncio
async def test_cleanup_old_events(tmp_path: Path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    bot = DummyBot("123:abc")

    async def nop(*args, **kwargs):
        pass

    monkeypatch.setattr(main, "delete_ics", nop)
    monkeypatch.setattr(main, "delete_asset_post", nop)
    monkeypatch.setattr(main, "remove_calendar_button", nop)

    old_date = (date.today() - timedelta(days=8)).isoformat()
    new_date = (date.today() + timedelta(days=1)).isoformat()

    async with db.get_session() as session:
        old = Event(
            title="Old",
            description="",
            date=old_date,
            time="18:00",
            location_name="P",
            source_text="",
        )
        new = Event(
            title="New",
            description="",
            date=new_date,
            time="18:00",
            location_name="P",
            source_text="",
        )
        session.add(old)
        session.add(new)
        await session.commit()
        old_id = old.id
        new_id = new.id

    await main.cleanup_old_events(db, bot)

    async with db.get_session() as session:
        old_ev = await session.get(Event, old_id)
        new_ev = await session.get(Event, new_id)

    assert old_ev is None
    assert new_ev is not None


@pytest.mark.asyncio
async def test_cleanup_scheduler_notifies_superadmin(tmp_path: Path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    bot = DummyBot("123:abc")

    start_msg = types.Message.model_validate(
        {
            "message_id": 1,
            "date": 0,
            "chat": {"id": 1, "type": "private"},
            "from": {"id": 1, "is_bot": False, "first_name": "A"},
            "text": "/start",
        }
    )
    await handle_start(start_msg, db, bot)

    async def nop(*args, **kwargs):
        pass

    monkeypatch.setattr(main, "delete_ics", nop)
    monkeypatch.setattr(main, "delete_asset_post", nop)
    monkeypatch.setattr(main, "remove_calendar_button", nop)

    old_date = (date.today() - timedelta(days=8)).isoformat()
    async with db.get_session() as session:
        session.add(
            Event(
                title="Old",
                description="",
                date=old_date,
                time="18:00",
                location_name="P",
                source_text="",
            )
        )
        await session.commit()

    class FakeDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime.combine(date.today(), time(3, 5), tzinfo=tz)

    monkeypatch.setattr(main, "datetime", FakeDatetime)

    async def fake_sleep(*args, **kwargs):
        raise asyncio.CancelledError

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    with pytest.raises(asyncio.CancelledError):
        await main.cleanup_scheduler(db, bot)

    assert any("Cleanup completed" in m[1] for m in bot.messages)


@pytest.mark.asyncio
async def test_page_update_scheduler(tmp_path: Path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    called = {}

    async def fake_month(db_obj, month):
        called["month"] = month

    async def fake_weekend(db_obj, start, update_links=True):
        called["weekend"] = start

    monkeypatch.setattr(main, "sync_month_page", fake_month)
    monkeypatch.setattr(main, "sync_weekend_page", fake_weekend)

    class FakeDate(date):
        @classmethod
        def today(cls):
            return date(2025, 7, 6)  # Sunday

    class FakeDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2025, 7, 6, 1, 5, tzinfo=tz)

    monkeypatch.setattr(main, "date", FakeDate)
    monkeypatch.setattr(main, "datetime", FakeDatetime)

    async def fake_sleep(*args, **kwargs):
        raise asyncio.CancelledError

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    with pytest.raises(asyncio.CancelledError):
        await main.page_update_scheduler(db)

    assert called.get("month") == "2025-07"
    assert called.get("weekend") == "2025-07-05"


@pytest.mark.asyncio
async def test_dumpdb(tmp_path: Path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    bot = DummyBot("123:abc")

    async def fake_send_document(self, chat_id, document, caption=None, parse_mode=None):
        self.sent = document
        self.messages.append((chat_id, caption))

    monkeypatch.setattr(DummyBot, "send_document", fake_send_document, raising=False)

    start_msg = types.Message.model_validate(
        {
            "message_id": 1,
            "date": 0,
            "chat": {"id": 1, "type": "private"},
            "from": {"id": 1, "is_bot": False, "first_name": "A"},
            "text": "/start",
        }
    )
    await handle_start(start_msg, db, bot)

    async with db.get_session() as session:
        session.add(main.Channel(channel_id=-100, title="Chan", is_registered=True))
        await session.commit()

    dump_msg = types.Message.model_validate(
        {
            "message_id": 2,
            "date": 0,
            "chat": {"id": 1, "type": "private"},
            "from": {"id": 1, "is_bot": False, "first_name": "A"},
            "text": "/dumpdb",
        }
    )

    await main.handle_dumpdb(dump_msg, db, bot)

    assert hasattr(bot, "sent")
    assert "Chan" in bot.messages[-1][1]
    assert "/restore" in bot.messages[-1][1]


@pytest.mark.asyncio
async def test_event_add_notifies_superadmin(tmp_path: Path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    bot = DummyBot("123:abc")

    async def fake_create(*args, **kwargs):
        return "u", "p"

    monkeypatch.setattr("main.create_source_page", fake_create)

    start_msg = types.Message.model_validate(
        {
            "message_id": 1,
            "date": 0,
            "chat": {"id": 1, "type": "private"},
            "from": {"id": 1, "is_bot": False, "first_name": "S"},
            "text": "/start",
        }
    )
    await handle_start(start_msg, db, bot)

    async with db.get_session() as session:
        session.add(User(user_id=2, username="u2"))
        await session.commit()

    add_msg = types.Message.model_validate(
        {
            "message_id": 2,
            "date": 0,
            "chat": {"id": 2, "type": "private"},
            "from": {"id": 2, "is_bot": False, "first_name": "U"},
            "text": f"/addevent_raw Party|{FUTURE_DATE}|18:00|Club",
        }
    )

    await handle_add_event_raw(add_msg, db, bot)

    assert any(
        "added event" in m[1] and "u2" in m[1] for m in bot.messages if m[0] == 1
    )


@pytest.mark.asyncio
async def test_partner_event_add_notifies_superadmin(tmp_path: Path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    bot = DummyBot("123:abc")

    async def fake_create(*args, **kwargs):
        return "u", "p"

    monkeypatch.setattr("main.create_source_page", fake_create)

    start_msg = types.Message.model_validate(
        {
            "message_id": 1,
            "date": 0,
            "chat": {"id": 1, "type": "private"},
            "from": {"id": 1, "is_bot": False, "first_name": "S"},
            "text": "/start",
        }
    )
    await handle_start(start_msg, db, bot)

    async with db.get_session() as session:
        session.add(User(user_id=3, username="p", is_partner=True))
        await session.commit()

    add_msg = types.Message.model_validate(
        {
            "message_id": 2,
            "date": 0,
            "chat": {"id": 3, "type": "private"},
            "from": {"id": 3, "is_bot": False, "first_name": "P"},
            "text": f"/addevent_raw Party|{FUTURE_DATE}|18:00|Club",
        }
    )

    await handle_add_event_raw(add_msg, db, bot)

    assert any(
        "partner" in m[1] and "added event" in m[1] for m in bot.messages if m[0] == 1
    )


@pytest.mark.asyncio
async def test_festival_poll_notifies_superadmin(tmp_path: Path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    bot = DummyBot("123:abc")

    start_msg = types.Message.model_validate(
        {
            "message_id": 1,
            "date": 0,
            "chat": {"id": 1, "type": "private"},
            "from": {"id": 1, "is_bot": False, "first_name": "S"},
            "text": "/start",
        }
    )
    await handle_start(start_msg, db, bot)

    async def fake_generate(fest):
        return "Q?"

    async def fake_post(group_id, question, options, db_obj, bot_obj):
        return "https://vk.com/poll1"

    monkeypatch.setattr(main, "generate_festival_poll_text", fake_generate)
    monkeypatch.setattr(main, "post_vk_poll", fake_post)

    async with db.get_session() as session:
        fest = Festival(name="Jazz")
        session.add(fest)
        await session.commit()
        await session.refresh(fest)
        fid = fest.id

    await send_festival_poll(db, fest, "-1", bot)

    async with db.get_session() as session:
        obj = await session.get(Festival, fid)

    assert obj and obj.vk_poll_url == "https://vk.com/poll1"
    assert any(
        "poll created" in m[1] and "https://vk.com/poll1" in m[1]
        for m in bot.messages
        if m[0] == 1
    )


@pytest.mark.asyncio
async def test_festival_description_dash(tmp_path: Path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    async def fake_parse(*args, **kwargs):
        return [
            {
                "title": "Jazz Day",
                "short_description": "desc",
                "date": FUTURE_DATE,
                "time": "18:00",
                "location_name": "Hall",
                "festival": "Jazz",
            }
        ]

    class DummyTG:
        def __init__(self, access_token=None):
            pass

        def create_page(self, title, content=None):
            return {"url": "http://tg", "path": "p"}

        def edit_page(self, path, title=None, content=None):
            pass

    monkeypatch.setenv("TELEGRAPH_TOKEN", "t")
    monkeypatch.setattr("main.Telegraph", lambda access_token=None: DummyTG())
    monkeypatch.setattr("main.parse_event_via_4o", fake_parse)

    async def fake_ask(text):
        return "Desc"

    monkeypatch.setattr("main.ask_4o", fake_ask)

    async def fake_create(*args, **kwargs):
        return "u", "p"

    monkeypatch.setattr("main.create_source_page", fake_create)

    async with db.get_session() as session:
        session.add(main.Festival(name="Jazz", description="-"))
        await session.commit()

    await main.add_events_from_text(db, "t", None, None, None)
    await main.sync_festival_page(db, "Jazz")

    async with db.get_session() as session:
        fest = (await session.execute(select(main.Festival))).scalars().first()

    assert fest and fest.description == "Desc"


@pytest.mark.asyncio
async def test_fest_list_includes_links(tmp_path: Path):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    bot = DummyBot("123:abc")

    async with db.get_session() as session:
        session.add(User(user_id=1))
        session.add(
            main.Festival(
                name="Jazz",
                telegraph_url="http://tg",
                website_url="https://jazz.ru",
                vk_url="https://vk.com/jazz",
                tg_url="https://t.me/jazz",
            )
        )
        await session.commit()

    msg = types.Message.model_validate(
        {
            "message_id": 1,
            "date": 0,
            "chat": {"id": 1, "type": "private"},
            "from": {"id": 1, "is_bot": False, "first_name": "A"},
            "text": "/fest",
        }
    )
    await main.handle_fest(msg, db, bot)
    text = bot.messages[-1][1]
    assert "http://tg" in text
    assert "https://jazz.ru" in text
    assert "https://vk.com/jazz" in text
    assert "https://t.me/jazz" in text


@pytest.mark.asyncio
async def test_edit_festival_contacts(tmp_path: Path):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    bot = DummyBot("123:abc")

    async with db.get_session() as session:
        session.add(User(user_id=1))
        fest = main.Festival(name="Jazz")
        session.add(fest)
        await session.commit()
        fid = fest.id

    festival_edit_sessions[1] = (fid, "site")

    msg = types.Message.model_validate(
        {
            "message_id": 2,
            "date": 0,
            "chat": {"id": 1, "type": "private"},
            "from": {"id": 1, "is_bot": False, "first_name": "A"},
            "text": "https://example.com",

        }
    )
    await main.handle_festival_edit_message(msg, db, bot)

    async with db.get_session() as session:
        fest = await session.get(main.Festival, fid)
        assert fest.website_url == "https://example.com"

    festival_edit_sessions[1] = (fid, "vk")

    msg2 = types.Message.model_validate(
        {
            "message_id": 3,
            "date": 0,
            "chat": {"id": 1, "type": "private"},
            "from": {"id": 1, "is_bot": False, "first_name": "A"},
            "text": "-",

        }
    )
    await main.handle_festival_edit_message(msg2, db, bot)

    async with db.get_session() as session:
        fest = await session.get(main.Festival, fid)
        assert fest.vk_url is None


@pytest.mark.asyncio
async def test_festival_page_contacts_and_dates(tmp_path: Path):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    async with db.get_session() as session:
        fest = main.Festival(
            name="Jazz",
            full_name="Jazz XVIII",
            website_url="https://jazz.ru",
            vk_url="https://vk.com/jazz",
            tg_url="https://t.me/jazz",
            photo_url="http://img",
        )
        session.add(fest)
        session.add(
            Event(
                title="A",
                description="d",
                source_text="s",
                date="2025-07-10",
                time="18:00",
                location_name="Hall",
                city="Калининград",
                festival="Jazz",
            )
        )
        session.add(
            Event(
                title="B",
                description="d",
                source_text="s",
                date="2025-07-12",
                time="19:00",
                location_name="Hall",
                city="Калининград",
                festival="Jazz",
            )
        )
        await session.commit()

    title, content = await main.build_festival_page_content(db, fest)
    assert title == "Jazz XVIII"
    dump = json_dumps(content)
    assert "Контакты фестиваля" in dump
    assert "Мероприятия фестиваля" in dump
    assert "\ud83d\udcc5" in dump or "📅" in dump
    assert "\ud83d\xdccd" in dump or "📍" in dump
    idx_contacts = next(
        i
        for i, n in enumerate(content)
        if n.get("tag") == "h3" and "Контакты" in "".join(n.get("children", []))
    )
    assert content[idx_contacts - 1].get("tag") == "p"
    assert content[idx_contacts - 2].get("tag") == "br"
    idx_events = next(
        i
        for i, n in enumerate(content)
        if n.get("tag") == "h3" and "Мероприятия" in "".join(n.get("children", []))
    )
    assert content[idx_events - 1].get("tag") == "p"
    assert content[idx_events - 2].get("tag") == "br"


@pytest.mark.asyncio
async def test_month_page_festival_star(tmp_path: Path):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    m = FUTURE_DATE[:7]
    async with db.get_session() as session:
        session.add(main.Festival(name="Jazz", telegraph_url="http://tg"))
        session.add(
            Event(
                title="T",
                description="d",
                source_text="s",
                date=FUTURE_DATE,
                time="18:00",
                location_name="Hall",
                festival="Jazz",
            )
        )
        await session.commit()

    _, content = await main.build_month_page_content(db, m)
    fest_line = next(
        n
        for n in content
        if isinstance(n, dict)
        and any(
            isinstance(c, dict) and c.get("attrs", {}).get("href") == "http://tg"
            for c in n.get("children", [])
        )
    )
    assert fest_line["children"][0] == "✨ "


@pytest.mark.asyncio
async def test_festival_vk_message_period_location(tmp_path: Path):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    async with db.get_session() as session:
        fest = main.Festival(name="Jazz", full_name="Jazz XVIII")
        session.add(fest)
        session.add(
            Event(
                title="A",
                description="d",
                source_text="s",
                date="2025-07-10",
                time="18:00",
                location_name="Hall",
                city="Калининград",
                festival="Jazz",
            )
        )
        session.add(
            Event(
                title="B",
                description="d",
                source_text="s",
                date="2025-07-12",
                time="19:00",
                location_name="Hall",
                city="Калининград",
                festival="Jazz",
            )
        )
        await session.commit()

    text = await main.build_festival_vk_message(db, fest)
    lines = text.splitlines()
    assert lines[0] == "Jazz XVIII"
    assert "\U0001f4c5" in text or "📅" in text
    assert "\U0001f4cd" in text or "📍" in text


@pytest.mark.asyncio
async def test_festival_page_lists_upcoming(tmp_path: Path):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    next_month_date = (date.fromisoformat(FUTURE_DATE) + timedelta(days=31)).isoformat()

    async with db.get_session() as session:
        fest1 = main.Festival(name="Jazz", telegraph_url="http://tg1")
        fest2 = main.Festival(name="Rock", telegraph_url="http://tg2")
        session.add(fest1)
        session.add(fest2)
        session.add(
            Event(
                title="A",
                description="d",
                source_text="s",
                date=FUTURE_DATE,
                time="18:00",
                location_name="Hall",
                festival="Jazz",
            )
        )
        session.add(
            Event(
                title="B",
                description="d",
                source_text="s",
                date=next_month_date,
                time="18:00",
                location_name="Park",
                festival="Rock",
            )
        )
        await session.commit()

    _, nodes = await main.build_festival_page_content(db, fest1)
    dump = json_dumps(nodes)
    assert "Ближайшие фестивали" in dump
    assert "Rock" in dump
    assert "Jazz" not in dump


@pytest.mark.asyncio
async def test_festival_vk_message_lists_upcoming(tmp_path: Path):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    next_month_date = (date.fromisoformat(FUTURE_DATE) + timedelta(days=31)).isoformat()

    async with db.get_session() as session:
        fest1 = main.Festival(name="Jazz", vk_post_url="http://vk1")
        fest2 = main.Festival(name="Rock", vk_post_url="http://vk2")
        session.add(fest1)
        session.add(fest2)
        session.add(
            Event(
                title="A",
                description="d",
                source_text="s",
                date=FUTURE_DATE,
                time="18:00",
                location_name="Hall",
                festival="Jazz",
            )
        )
        session.add(
            Event(
                title="B",
                description="d",
                source_text="s",
                date=next_month_date,
                time="18:00",
                location_name="Park",
                festival="Rock",
            )
        )
        await session.commit()

    text = await main.build_festival_vk_message(db, fest1)
    assert "Ближайшие фестивали" in text
    assert "[http://vk2|Rock]" in text
    assert "[http://vk1|Jazz]" not in text


@pytest.mark.asyncio
async def test_edit_vk_post_preserves_photos(monkeypatch):
    captured = {}

    async def fake_api(method, params, db=None, bot=None, token=None):
        if method == "wall.getById":
            assert params["posts"] == "-1_2"
            return {
                "response": [
                    {
                        "attachments": [
                            {
                                "type": "photo",
                                "photo": {"owner_id": -1, "id": 10},
                            }
                        ]
                    }
                ]
            }
        if method == "wall.edit":
            captured.update(params)
            return {"response": 1}
        raise AssertionError(method)

    monkeypatch.setattr(main, "_vk_api", fake_api)

    await main.edit_vk_post("https://vk.com/wall-1_2", "msg")

    assert captured.get("attachments") == "photo-1_10"


@pytest.mark.asyncio
async def test_edit_vk_post_add_photo(monkeypatch):
    captured = {}

    async def fake_api(method, params, db=None, bot=None, token=None):
        if method == "wall.getById":
            return {
                "response": [
                    {
                        "attachments": [
                            {
                                "type": "photo",
                                "photo": {"owner_id": -1, "id": 10},
                            }
                        ]
                    }
                ]
            }
        if method == "wall.edit":
            captured.update(params)
            return {"response": 1}
        raise AssertionError(method)

    monkeypatch.setattr(main, "_vk_api", fake_api)

    await main.edit_vk_post(
        "https://vk.com/wall-1_2",
        "msg",
        attachments=["photo-1_20"],
    )

    assert captured.get("attachments") == "photo-1_10,photo-1_20"


@pytest.mark.asyncio
async def test_partner_notification_scheduler(tmp_path: Path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    bot = DummyBot("123:abc")

    start_msg = types.Message.model_validate(
        {
            "message_id": 1,
            "date": 0,
            "chat": {"id": 1, "type": "private"},
            "from": {"id": 1, "is_bot": False, "first_name": "S"},
            "text": "/start",
        }
    )
    await handle_start(start_msg, db, bot)

    async with db.get_session() as session:
        session.add(User(user_id=2, username="p", is_partner=True))
        await session.commit()

    class FakeDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime.combine(date.today(), time(9, 5), tzinfo=tz)

    monkeypatch.setattr(main, "datetime", FakeDatetime)

    async def fake_sleep(*args, **kwargs):
        raise asyncio.CancelledError

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    with pytest.raises(asyncio.CancelledError):
        await main.partner_notification_scheduler(db, bot)

    assert any("неделе" in m[1] for m in bot.messages if m[0] == 2)
    assert any("p" in m[1] for m in bot.messages if m[0] == 1)


@pytest.mark.asyncio
async def test_partner_reminder_weekly(tmp_path: Path):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    bot = DummyBot("123:abc")

    async with db.get_session() as session:
        session.add(User(user_id=1, username="p", is_partner=True))
        await session.commit()

    tz = timezone.utc
    notified = await notify_inactive_partners(db, bot, tz)
    assert [u.user_id for u in notified] == [1]
    assert len(bot.messages) == 1

    notified = await notify_inactive_partners(db, bot, tz)
    assert notified == []
    assert len(bot.messages) == 1

    async with db.get_session() as session:
        user = await session.get(User, 1)
        user.last_partner_reminder = datetime.utcnow() - timedelta(days=8)
        await session.commit()

    notified = await notify_inactive_partners(db, bot, tz)
    assert [u.user_id for u in notified] == [1]
    assert len(bot.messages) == 2


@pytest.mark.asyncio
async def test_festival_dates_manual(tmp_path: Path):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    fest = Festival(name="Fest", start_date="2025-08-01", end_date="2025-08-03")
    async with db.get_session() as session:
        session.add(fest)
        await session.commit()
    start, end = festival_dates(fest, [])
    assert start == date(2025, 8, 1)
    assert end == date(2025, 8, 3)


