import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from pathlib import Path

import pytest
from aiogram import Bot, types
from sqlmodel import select
import main

from main import (
    Database,
    PendingUser,
    Setting,
    User,
    Event,
    create_app,
    handle_register,
    handle_start,
    handle_tz,
    handle_add_event_raw,
    handle_ask_4o,
    handle_events,
    parse_event_via_4o,
    telegraph_test,
    get_telegraph_token,
)


class DummyBot(Bot):
    def __init__(self, token: str):
        super().__init__(token)
        self.messages = []

    async def send_message(self, chat_id, text, **kwargs):
        self.messages.append((chat_id, text))


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

    async def fake_create(title, text):
        return "https://t.me/test"

    monkeypatch.setattr("main.create_source_page", fake_create)

    msg = types.Message.model_validate(
        {
            "message_id": 1,
            "date": 0,
            "chat": {"id": 1, "type": "private"},
            "from": {"id": 1, "is_bot": False, "first_name": "M"},
            "text": "/addevent_raw Party|2025-01-01|18:00|Club",
        }
    )

    await handle_add_event_raw(msg, db, bot)

    async with db.get_session() as session:
        events = (await session.execute(select(Event))).scalars().all()

    assert len(events) == 1
    assert events[0].title == "Party"
    assert events[0].telegraph_url == "https://t.me/test"


@pytest.mark.asyncio
async def test_events_list(tmp_path: Path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    bot = DummyBot("123:abc")

    async def fake_create(title, text):
        return "https://t.me/test"

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
            "text": "/addevent_raw Party|2025-01-01|18:00|Club",
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
            "text": "/events 2025-01-01",
        }
    )

    await handle_events(list_msg, db, bot)

    assert bot.messages
    text = bot.messages[-1][1]
    assert "Events on 01.01.2025" in text
    assert "Party" in text
    assert "https://t.me/test" in text


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
        def __init__(self):
            self.access_token = None

        def create_page(self, title, html):
            return {"url": "https://telegra.ph/test", "path": "test"}

        def edit_page(self, path, title, html_content):
            pass

    monkeypatch.setenv("TELEGRAPH_TOKEN", "t")
    monkeypatch.setattr("main.Telegraph", lambda: DummyTG())

    await telegraph_test()
    captured = capsys.readouterr()
    assert "Created https://telegra.ph/test" in captured.out
    assert "Edited https://telegra.ph/test" in captured.out


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

