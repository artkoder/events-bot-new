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
async def test_handle_vk_queue_shows_counts_and_button(tmp_path):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    async with db.get_session() as session:
        session.add(User(user_id=1))
        await session.commit()
    async with db.raw_conn() as conn:
        rows = [
            (1, 1, 0, "t", None, 1, None, "pending"),
            (1, 2, 0, "t", None, 1, None, "pending"),
            (1, 3, 0, "t", None, 1, None, "locked"),
            (1, 4, 0, "t", None, 1, None, "skipped"),
            (1, 5, 0, "t", None, 1, None, "imported"),
            (1, 6, 0, "t", None, 1, None, "rejected"),
        ]
        await conn.executemany(
            "INSERT INTO vk_inbox(group_id, post_id, date, text, matched_kw, has_date, event_ts_hint, status) VALUES(?,?,?,?,?,?,?,?)",
            rows,
        )
        await conn.commit()
    msg = types.Message.model_validate(
        {
            "message_id": 1,
            "date": 0,
            "chat": {"id": 1, "type": "private"},
            "from": {"id": 1, "is_bot": False, "first_name": "U"},
            "text": "/vk_queue",
        }
    )
    bot = DummyBot()
    await main.handle_vk_queue(msg, db, bot)
    assert bot.messages, "no message sent"
    sent = bot.messages[0]
    assert "pending: 2" in sent.text
    assert "locked: 1" in sent.text
    assert sent.reply_markup.keyboard[0][0].text == main.VK_BTN_CHECK_EVENTS
