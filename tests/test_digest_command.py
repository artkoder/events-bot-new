import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from datetime import datetime, timedelta
import pytest
from aiogram import types

import main
from main import Database, User, Event

class DummyBot:
    def __init__(self):
        self.messages = []

    async def send_message(self, chat_id, text, **kwargs):
        msg_id = len(self.messages) + 1
        data = {
            "message_id": msg_id,
            "date": 0,
            "chat": {"id": chat_id, "type": "private"},
            "from": {"id": 0, "is_bot": True, "first_name": "B"},
            "text": text,
        }
        if "reply_markup" in kwargs:
            data["reply_markup"] = kwargs["reply_markup"]
        msg = types.Message.model_validate(data)
        self.messages.append(msg)
        return msg


@pytest.mark.asyncio
async def test_handle_digest_sends_preview(tmp_path):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    async with db.get_session() as session:
        session.add(User(user_id=1, is_superadmin=True))
        dt = datetime.utcnow() + timedelta(days=1)
        ev = Event(
            title="L1",
            description="d",
            date=dt.strftime("%Y-%m-%d"),
            time="12:00",
            location_name="loc",
            source_text="s",
            event_type="лекция",
        )
        session.add(ev)
        await session.commit()
    msg = types.Message.model_validate({
        "message_id": 1,
        "date": 0,
        "chat": {"id": 1, "type": "private"},
        "from": {"id": 1, "is_bot": False, "first_name": "U"},
        "text": "/digest",
    })
    bot = DummyBot()
    await main.show_digest_menu(msg, db, bot)
    menu_msg = bot.messages[0]
    digest_id = menu_msg.reply_markup.inline_keyboard[0][0].callback_data.split(":")[-1]
    cb = types.CallbackQuery.model_validate(
        {
            "id": "1",
            "from": {"id": 1, "is_bot": False, "first_name": "U"},
            "message": menu_msg.model_dump(),
            "chat_instance": "1",
            "data": f"digest:select:lectures:{digest_id}",
        }
    )
    await main.handle_digest_select_lectures(cb, db, bot)
    assert any("Подобрали для вас" in m.text for m in bot.messages)


def test_help_contains_digest():
    assert any(cmd["usage"].startswith("/digest") for cmd in main.HELP_COMMANDS)
