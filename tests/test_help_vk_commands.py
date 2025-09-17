import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pytest
from aiogram import types
from types import SimpleNamespace

import main
from main import Database, User


class DummyBot:
    def __init__(self):
        self.messages = []

    async def send_message(self, chat_id, text, **kwargs):
        msg = SimpleNamespace(
            message_id=len(self.messages) + 1,
            date=0,
            chat=SimpleNamespace(id=chat_id, type="private"),
            from_user=SimpleNamespace(id=0, is_bot=True, first_name="B"),
            text=text,
            reply_markup=kwargs.get("reply_markup"),
        )
        self.messages.append(msg)
        return msg


@pytest.mark.asyncio
async def test_help_superadmin_lists_vk_commands(tmp_path):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    async with db.get_session() as session:
        session.add(User(user_id=1, is_superadmin=True))
        await session.commit()
    msg = types.Message.model_validate(
        {
            "message_id": 1,
            "date": 0,
            "chat": {"id": 1, "type": "private"},
            "from": {"id": 1, "is_bot": False, "first_name": "U"},
            "text": "/help",
        }
    )
    bot = DummyBot()
    await main.handle_help(msg, db, bot)
    assert bot.messages, "no message sent"
    lines = bot.messages[0].text.splitlines()
    assert any(line.startswith("/vk ") for line in lines)
    assert any(line.startswith("/vk_queue") for line in lines)
    assert any(line.startswith("/vk_crawl_now") for line in lines)
    assert any("↪️ Репостнуть в Vk" in line for line in lines)
    assert any("✂️ Сокращённый рерайт" in line for line in lines)
    assert "/ocrtest — сравнить распознавание афиш" in lines


@pytest.mark.asyncio
async def test_help_user_hides_vk_queue_and_crawl(tmp_path):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    async with db.get_session() as session:
        session.add(User(user_id=2))
        await session.commit()
    msg = types.Message.model_validate(
        {
            "message_id": 1,
            "date": 0,
            "chat": {"id": 2, "type": "private"},
            "from": {"id": 2, "is_bot": False, "first_name": "U"},
            "text": "/help",
        }
    )
    bot = DummyBot()
    await main.handle_help(msg, db, bot)
    assert bot.messages, "no message sent"
    lines = bot.messages[0].text.splitlines()
    assert not any(line.startswith("/vk_queue") for line in lines)
    assert not any(line.startswith("/vk_crawl_now") for line in lines)
    assert any("↪️ Репостнуть в Vk" in line for line in lines)
    assert any("✂️ Сокращённый рерайт" in line for line in lines)

