import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from pathlib import Path

import pytest
from aiogram import Bot, types
from sqlmodel import select

from main import (
    Database,
    PendingUser,
    Setting,
    User,

    create_app,

    handle_register,
    handle_start,
    handle_tz,
)


class DummyBot(Bot):
    async def send_message(self, *args, **kwargs):
        pass


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

