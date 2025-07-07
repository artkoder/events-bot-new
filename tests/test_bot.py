import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from pathlib import Path

import pytest
from aiogram import Bot, types
from sqlmodel import select
from datetime import date, timedelta, timezone
import main

from main import (
    Database,
    PendingUser,
    Setting,
    User,
    Event,
    MonthPage,
    create_app,
    handle_register,
    handle_start,
    handle_tz,
    handle_add_event_raw,
    handle_add_event,
    handle_ask_4o,
    handle_events,
    handle_exhibitions,
    handle_edit_message,
    process_request,
    parse_event_via_4o,
    telegraph_test,
    get_telegraph_token,
    editing_sessions,
)

FUTURE_DATE = (date.today() + timedelta(days=10)).isoformat()

class DummyBot(Bot):
    def __init__(self, token: str):
        super().__init__(token)
        self.messages = []
        self.edits = []

    async def send_message(self, chat_id, text, **kwargs):
        self.messages.append((chat_id, text))

    async def edit_message_reply_markup(
        self, chat_id: int | None = None, message_id: int | None = None, **kwargs
    ):
        self.edits.append((chat_id, message_id, kwargs))


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

    async def fake_create(title, text, source, html_text=None, media=None):
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

    async def fake_create(title, text, source, html_text=None, media=None):
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
async def test_add_event_raw_update(tmp_path: Path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    bot = DummyBot("123:abc")

    async def fake_create(title, text, source, html_text=None, media=None):
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

    async def fake_create(title, text, source, html_text=None, media=None):
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
async def test_events_list(tmp_path: Path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    bot = DummyBot("123:abc")

    async def fake_create(title, text, source, html_text=None, media=None):
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
async def test_ask4o_admin(tmp_path: Path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    bot = DummyBot("123:abc")

    start_msg = types.Message.model_validate({
        "message_id": 1,
        "date": 0,
        "chat": {"id": 1, "type": "private"},
        "from": {"id": 1, "is_bot": False, "first_name": "A"},
        "text": "/start",
    })
    await handle_start(start_msg, db, bot)

    called = {}

    async def fake_ask(text: str) -> str:
        called["text"] = text
        return "ok"

    monkeypatch.setattr("main.ask_4o", fake_ask)

    msg = types.Message.model_validate({
        "message_id": 2,
        "date": 0,
        "chat": {"id": 1, "type": "private"},
        "from": {"id": 1, "is_bot": False, "first_name": "A"},
        "text": "/ask4o hello",
    })

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

    msg = types.Message.model_validate({
        "message_id": 1,
        "date": 0,
        "chat": {"id": 2, "type": "private"},
        "from": {"id": 2, "is_bot": False, "first_name": "B"},
        "text": "/ask4o hi",
    })

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
async def test_telegraph_test(monkeypatch, capsys):
    class DummyTG:
        def __init__(self, access_token=None):
            self.access_token = access_token
        def create_page(self, title, html_content=None, **_):
            return {"url": "https://telegra.ph/test", "path": "test"}

        def edit_page(self, path, title, html_content):
            pass

    monkeypatch.setenv("TELEGRAPH_TOKEN", "t")
    monkeypatch.setattr("main.Telegraph", lambda access_token=None: DummyTG(access_token))

    await telegraph_test()
    captured = capsys.readouterr()
    assert "Created https://telegra.ph/test" in captured.out
    assert "Edited https://telegra.ph/test" in captured.out


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
    monkeypatch.setattr("main.Telegraph", lambda access_token=None: DummyTG(access_token))

    res = await main.create_source_page("Title", "text", None, media=(b"img", "photo.jpg"))
    assert res == ("https://telegra.ph/test", "test")


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
async def test_forward_add_event(tmp_path: Path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    bot = DummyBot("123:abc")

    async def fake_parse(text: str) -> list[dict]:
        return [{
            "title": "Forwarded",
            "short_description": "desc",
            "date": "2025-07-16",
            "time": "18:00",
            "location_name": "Club",
        }]

    async def fake_create(title, text, source, html_text=None, media=None):
        return "https://t.me/page", "p"

    monkeypatch.setattr("main.parse_event_via_4o", fake_parse)
    monkeypatch.setattr("main.create_source_page", fake_create)

    start_msg = types.Message.model_validate({
        "message_id": 1,
        "date": 0,
        "chat": {"id": 1, "type": "private"},
        "from": {"id": 1, "is_bot": False, "first_name": "A"},
        "text": "/start",
    })
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
async def test_forward_unregistered(tmp_path: Path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    bot = DummyBot("123:abc")

    async def fake_parse(text: str) -> list[dict]:
        return [{
            "title": "Fwd",
            "short_description": "d",
            "date": "2025-07-16",
            "time": "18:00",
            "location_name": "Club",
        }]

    async def fake_create(title, text, source, html_text=None, media=None):
        return "https://t.me/page", "p"

    monkeypatch.setattr("main.parse_event_via_4o", fake_parse)
    monkeypatch.setattr("main.create_source_page", fake_create)

    start_msg = types.Message.model_validate({
        "message_id": 1,
        "date": 0,
        "chat": {"id": 1, "type": "private"},
        "from": {"id": 1, "is_bot": False, "first_name": "A"},
        "text": "/start",
    })
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

    async def fake_parse(text: str) -> list[dict]:
        return [{
            "title": "MG",
            "short_description": "d",
            "date": "2025-07-16",
            "time": "18:00",
            "location_name": "Club",
        }]

    async def fake_create(title, text, source, html_text=None, media=None):
        return "https://t.me/page", "p"

    monkeypatch.setattr("main.parse_event_via_4o", fake_parse)
    monkeypatch.setattr("main.create_source_page", fake_create)

    start_msg = types.Message.model_validate({
        "message_id": 1,
        "date": 0,
        "chat": {"id": 1, "type": "private"},
        "from": {"id": 1, "is_bot": False, "first_name": "A"},
        "text": "/start",
    })
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

    async def fake_parse(text: str) -> list[dict]:
        return [{
            "title": "MG",
            "short_description": "d",
            "date": "2025-07-16",
            "time": "18:00",
            "location_name": "Club",
        }]

    async def fake_create(title, text, source, html_text=None, media=None):
        return "https://t.me/page", "p"

    monkeypatch.setattr("main.parse_event_via_4o", fake_parse)
    monkeypatch.setattr("main.create_source_page", fake_create)

    start_msg = types.Message.model_validate({
        "message_id": 1,
        "date": 0,
        "chat": {"id": 1, "type": "private"},
        "from": {"id": 1, "is_bot": False, "first_name": "A"},
        "text": "/start",
    })
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

    async def fake_create(title, text, source, html_text=None, media=None):
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

    async def fake_parse(text: str) -> list[dict]:
        return [{
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
        }]

    monkeypatch.setattr("main.parse_event_via_4o", fake_parse)

    async def fake_create(title, text, source, html_text=None, media=None):
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

    async def fake_parse(text: str) -> list[dict]:
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

    async def fake_create(title, text, source, html_text=None, media=None):
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
            "text": "/months",
        }
    )

    await main.handle_months(msg, db, bot)
    assert "2025-07" in bot.messages[-1][1]


@pytest.mark.asyncio
async def test_date_range_parsing(tmp_path: Path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    bot = DummyBot("123:abc")

    async def fake_parse(text: str) -> list[dict]:
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

    async def fake_create(title, text, source, html_text=None, media=None):
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
    monkeypatch.setattr("main.Telegraph", lambda access_token=None: DummyTG())

    # Should not raise
    await main.sync_month_page(db, "2025-07")


@pytest.mark.asyncio
async def test_update_source_page_uses_content(monkeypatch):
    events = {}

    class DummyTG:
        def get_page(self, path, return_html=True):
            return {"content": "<p>old</p>"}
        def edit_page(self, path, title, html_content):
            events["html"] = html_content

    monkeypatch.setattr("main.get_telegraph_token", lambda: "t")
    monkeypatch.setattr("main.Telegraph", lambda access_token=None: DummyTG())

    await main.update_source_page("path", "Title", "new")
    html = events.get("html", "")
    assert "<p>old</p>" in html
    assert "new" in html
    assert main.CONTENT_SEPARATOR in html


@pytest.mark.asyncio
async def test_nav_limits_past(tmp_path: Path):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    today = date.today()
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
    assert row[0].text == "\u25B6"


@pytest.mark.asyncio
async def test_nav_future_has_prev(tmp_path: Path):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    today = date.today()
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
    assert row[0].text == "\u25C0"
    assert row[1].text == "\u25B6"



@pytest.mark.asyncio
async def test_delete_event_updates_month(tmp_path: Path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    bot = DummyBot("123:abc")

    async def fake_create(title, text, source, html_text=None, media=None):
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

