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
        self.messages.append((chat_id, text))


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
    await main.handle_digest(msg, db, bot)
    assert bot.messages and "Подобрали для вас" in bot.messages[0][1]


def test_help_contains_digest():
    assert any(cmd["usage"].startswith("/digest") for cmd in main.HELP_COMMANDS)
